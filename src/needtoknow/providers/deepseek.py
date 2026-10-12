"""The DeepSeek adapter (T4.2): one chat completion, and what it cost.

`complete` posts the call the architecture fixes (section 11): the model from
`config/providers.toml`, `thinking` as a top level field, off unless asked or configured
otherwise, and JSON output when asked. Thinking is the API's default, and a raw JSON body
that wraps the field in `extra_body`, the OpenAI SDK's syntax, is silently ignored (N13).
It returns the text, the tokens as DeepSeek bills them, cache hits at a fiftieth of a
miss, the stop reason, the seconds, and the USD at the price in force when the call
began, since peak hours cost double. `max_tokens` and `temperature` are sent only when
given, so the API's own defaults apply otherwise.

429, 5xx and a dropped connection are tried again after 1, 2 and 4 seconds, as the API's
error page advises; 400, 401, 402 and 422 are not, and come back as a ProviderError that
says what went wrong and never holds the key. A reply that ends for want of the server's
resources is billed and unusable, so it is tried again and both are counted. The opener,
the clock and the sleep are parameters, so the tests never touch the network.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from needtoknow.providers.catalog import cost_usd, load_catalog
from needtoknow.providers.deepseek_account import HTTP_HINTS
from needtoknow.settings import setting

RETRIED = (429, 500, 502, 503, 504)
BACKOFF = (1.0, 2.0, 4.0)  # seconds before each retry: four attempts at most
HINTS = {400: "the request body is malformed", 422: "a parameter is invalid", **HTTP_HINTS}
EXPLAINED = (400, 422)  # the codes whose own message names the field at fault; no other reply's words are kept


class ProviderError(Exception):
    """DeepSeek did not complete the call; the message says why and never holds the key."""


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning: str  # the thinking DeepSeek returns beside the answer; empty with thinking off
    usage: dict  # input_hit, input_miss, output and reasoning tokens, over every billed attempt
    stop_reason: str  # DeepSeek's finish_reason: stop, length, content_filter, ...
    seconds: float
    cost_usd: float
    model: str  # the model that answered, as the reply names it
    attempts: int


def request_body(model: str, system: str, messages: list[dict], thinking: bool, as_json: bool,
                 max_tokens: int | None, temperature: float | None) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": system}, *messages],
            "thinking": {"type": "enabled" if thinking else "disabled"},  # top level, never inside extra_body
            "stream": False}
    if as_json:
        body["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    return body


def usage_of(reply: dict) -> dict:
    """A reply's tokens as DeepSeek prices them: input that hit the cache apart from input that missed it."""
    usage = reply.get("usage") or {}
    hit = int(usage.get("prompt_cache_hit_tokens", 0))
    miss = int(usage.get("prompt_cache_miss_tokens", int(usage.get("prompt_tokens", 0)) - hit))
    details = usage.get("completion_tokens_details") or {}
    return {"input_hit": hit, "input_miss": miss, "output": int(usage.get("completion_tokens", 0)),
            "reasoning": int(details.get("reasoning_tokens", 0))}


def refusal(error: urllib.error.HTTPError, key: str) -> str:
    """An HTTP error in the user's terms; DeepSeek's own words only where they name a parameter, without the key."""
    said = ""
    if error.code in EXPLAINED:
        try:
            said = str(json.loads(error.read() or b"{}").get("error", {}).get("message", ""))
        except (ValueError, AttributeError, OSError):
            said = ""
        said = said.replace(key, "(the key)")[:200]
    hint = HINTS.get(error.code)
    return f"HTTP {error.code} {error.reason}" + (f": {hint}" if hint else "") + (f" ({said})" if said else "")


def complete(system: str, messages: list[dict], thinking: bool | None = None, as_json: bool = False,
             max_tokens: int | None = None, temperature: float | None = None, config: dict | None = None,
             opener: Callable = urllib.request.urlopen, now: Callable[[], dt.datetime] | None = None,
             sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0) -> Completion:
    """One chat completion from DeepSeek, tried again on a busy or failing server, with its usage and cost."""
    deepseek = (config or load_catalog())["deepseek"]
    key = setting(deepseek["key_env"])
    if not key:
        raise ProviderError(f"no DeepSeek key: set {deepseek['key_env']} in the environment or in .env")
    if thinking is None:
        thinking = deepseek.get("thinking", "disabled") == "enabled"
    model = deepseek["model"]
    data = json.dumps(request_body(model, system, messages, thinking, as_json, max_tokens, temperature)).encode("utf-8")
    began_at = (now or (lambda: dt.datetime.now(dt.timezone.utc)))()
    began = time.perf_counter()
    billed = {"input_hit": 0, "input_miss": 0, "output": 0, "reasoning": 0}
    last = len(BACKOFF) + 1
    for attempt in range(1, last + 1):
        request = urllib.request.Request(
            deepseek["base_url"].rstrip("/") + "/chat/completions", data=data, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with opener(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRIED or attempt == last:
                raise ProviderError(f"DeepSeek refused the call: {refusal(error, key)}") from None
            sleep(BACKOFF[attempt - 1])
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == last:
                raise ProviderError(f"DeepSeek could not be reached: {type(error).__name__}") from None
            sleep(BACKOFF[attempt - 1])
            continue
        try:
            reply = json.loads(raw)
            choice = reply["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError(f"DeepSeek answered, but not with a completion: {raw[:60]!r}") from None
        usage = usage_of(reply)
        billed = {kind: billed[kind] + usage[kind] for kind in billed}
        stop = choice.get("finish_reason") or ""
        if stop == "insufficient_system_resource" and attempt < last:
            sleep(BACKOFF[attempt - 1])
            continue
        message = choice.get("message") or {}
        return Completion(text=message.get("content") or "", reasoning=message.get("reasoning_content") or "",
                          usage=billed, stop_reason=stop, seconds=time.perf_counter() - began,
                          cost_usd=cost_usd(deepseek["models"][model]["price"], billed, began_at, deepseek["peak"]),
                          model=str(reply.get("model") or model), attempts=attempt)
    raise ProviderError(f"DeepSeek gave no usable reply in {last} attempts")
