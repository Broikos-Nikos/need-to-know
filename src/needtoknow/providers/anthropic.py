"""The Anthropic adapter (T4.3, N13): one message from the Messages API, tested on documented replies, never run.

The request is the Messages API's (`09-production/03`): the key in `x-api-key`, the version
in `anthropic-version`, the system rules as the top level `system`, and `max_tokens`,
which the API requires and which caps thinking and text together, so 4,096 when none is
given. Thinking is `{"type": "adaptive"}` when asked, `{"type": "disabled"}` when refused,
which some current models answer with a 400, and nothing when the caller leaves it to the
model, whose default on the current lineup is to think. JSON is the prompt's own rule:
the API enforces a shape only through a schema in `output_config.format`, which the
contract here does not carry. `temperature` is sent when given, and the current lineup
refuses it with a 400 too.

The reply's text is its `text` blocks and its reasoning its `thinking` blocks. Usage
reads `cache_read_input_tokens` as hits, `input_tokens` and `cache_creation_input_tokens`
as misses and `output_tokens` as output, thinking included and not counted apart, so
reasoning is None. `end_turn` and `stop_sequence` read as stop, `max_tokens` as length,
and `refusal`, which arrives as a 200 with no text, as refusal. No key exists for it
here, so it has never answered a question.
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.request
from typing import Callable

from needtoknow.providers.call import Completion, ProviderError, post, priced, started
from needtoknow.providers.catalog import load_catalog
from needtoknow.settings import setting

VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
STOPS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "refusal"}


def request_body(model: str, system: str, messages: list[dict], thinking: bool | None,
                 max_tokens: int | None, temperature: float | None) -> dict:
    body = {"model": model, "system": system, "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens}
    if thinking is not None:
        body["thinking"] = {"type": "adaptive" if thinking else "disabled"}
    if temperature is not None:
        body["temperature"] = temperature
    return body


def read(reply: dict) -> tuple[str, str, dict, str]:
    """The reply's text, its thinking, its usage and its stop reason."""
    blocks = reply.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    reasoning = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
    usage = reply.get("usage") or {}
    billed = {"input_hit": int(usage.get("cache_read_input_tokens") or 0),
              "input_miss": int(usage.get("input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0),
              "output": int(usage.get("output_tokens") or 0), "reasoning": None}
    stop = str(reply.get("stop_reason") or "")
    return text, reasoning, billed, STOPS.get(stop, stop)


def complete(system: str, messages: list[dict], thinking: bool | None = None, as_json: bool = False,
             max_tokens: int | None = None, temperature: float | None = None, config: dict | None = None,
             opener: Callable = urllib.request.urlopen, now: Callable[[], dt.datetime] | None = None,
             sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0) -> Completion:
    """One message from Anthropic, tried again on a busy, failing or overloaded server, with its usage. `as_json` is
    accepted for the common contract and changes nothing: the prompt asks for JSON."""
    anthropic = (config or load_catalog())["anthropic"]
    key, model = setting(anthropic["key_env"]), anthropic.get("model")
    if not model:
        raise ProviderError("no Anthropic model: set anthropic.model in config/providers.toml")
    if not key:
        raise ProviderError(f"no Anthropic key: set {anthropic['key_env']} in the environment or in .env")
    began_at, began = started(now), time.perf_counter()
    replies, attempts = post("Anthropic", anthropic["base_url"].rstrip("/") + "/v1/messages",
                             {"x-api-key": key, "anthropic-version": VERSION},
                             request_body(model, system, messages, thinking, max_tokens, temperature),
                             key, opener, sleep, timeout)
    text, reasoning, usage, stop = read(replies[-1])
    return Completion(text=text, reasoning=reasoning, usage=usage, stop_reason=stop,
                      seconds=time.perf_counter() - began, cost_usd=priced(anthropic, model, usage, began_at),
                      model=str(replies[-1].get("model") or model), attempts=attempts)
