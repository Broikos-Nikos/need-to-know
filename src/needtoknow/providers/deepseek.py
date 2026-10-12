"""The DeepSeek adapter (T4.2): one chat completion, and what it cost.

`complete` posts the call the architecture fixes (section 11): the model from
`config/providers.toml`, `thinking` as a top level field, off unless asked or configured
otherwise, and JSON output when asked. Thinking is the API's default, and a raw JSON body
that wraps the field in `extra_body`, the OpenAI SDK's syntax, is silently ignored (N13).
It returns the text, the tokens as DeepSeek bills them, cache hits at a fiftieth of a
miss, the stop reason, the seconds, and the USD at the price in force when the call
began, since peak hours cost double. `max_tokens` and `temperature` are sent only when
given, so the API's own defaults apply otherwise.

Retries are `call.post`'s. A reply that ends for want of the server's resources is billed
and unusable, so it is tried again too, and every billed reply is counted. The opener,
the clock and the sleep are parameters, so the tests never touch the network.
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.request
from typing import Callable

from needtoknow.providers.call import Completion, ProviderError, post, priced, started
from needtoknow.providers.call import HINTS as COMMON_HINTS
from needtoknow.providers.catalog import load_catalog
from needtoknow.providers.deepseek_account import HTTP_HINTS
from needtoknow.settings import setting

HINTS = {**COMMON_HINTS, **HTTP_HINTS}
__all__ = ["Completion", "ProviderError", "complete", "request_body", "usage_of"]


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


def finish_of(reply: dict) -> str:
    choices = reply.get("choices") or [{}]
    return str((choices[0] or {}).get("finish_reason") or "") if isinstance(choices[0], dict) else ""


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
    began_at, began = started(now), time.perf_counter()
    replies, attempts = post(
        "DeepSeek", deepseek["base_url"].rstrip("/") + "/chat/completions", {"Authorization": "Bearer " + key},
        request_body(model, system, messages, thinking, as_json, max_tokens, temperature), key, opener, sleep, timeout,
        hints=HINTS, unusable=lambda reply: finish_of(reply) == "insufficient_system_resource")
    reply = replies[-1]
    choices = reply.get("choices")
    if not choices or not isinstance(choices[0], dict):
        raise ProviderError(f"DeepSeek answered, but not with a completion: {str(reply)[:60]}")
    billed = {kind: sum(usage_of(r)[kind] for r in replies) for kind in ("input_hit", "input_miss", "output", "reasoning")}
    message = choices[0].get("message") or {}
    return Completion(text=message.get("content") or "", reasoning=message.get("reasoning_content") or "",
                      usage=billed, stop_reason=finish_of(reply), seconds=time.perf_counter() - began,
                      cost_usd=priced(deepseek, model, billed, began_at),
                      model=str(reply.get("model") or model), attempts=attempts)
