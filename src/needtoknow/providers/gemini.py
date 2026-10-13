"""The Gemini adapter (T4.3, N13): one generateContent call, tested on documented replies, never run.

The request is the Gemini API's `models/{model}:generateContent` (`09-production/04`): the
key in `x-goog-api-key`, never in the URL, where an error would carry it; the system rules
as `systemInstruction`; the messages as `contents`, an assistant's turn under the role
`model`; and in `generationConfig`, `responseMimeType` for JSON, `maxOutputTokens` and
`temperature` when given. Thinking is `thinkingLevel`, named levels that replaced the
numeric budget: `high`, with the thoughts returned, when asked. Gemini 3 has no level that
turns thinking off, so refusing it is refused here, before any call, rather than sent and
ignored; left to the model, nothing is sent.

The reply's text is its first candidate's parts, the parts marked `thought` apart as the
reasoning. Usage reads `cachedContentTokenCount` as hits and the rest of
`promptTokenCount` as misses, and bills `candidatesTokenCount` and `thoughtsTokenCount`
together as output, the thoughts as reasoning. `STOP` reads as stop, `MAX_TOKENS` as
length, and the safety, recitation and blocklist reasons, or a prompt blocked with no
candidate, as content_filter. No key exists for it here, so it has never answered.
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.parse
import urllib.request
from typing import Callable

from needtoknow.providers.call import Completion, ProviderError, post, priced, record, records, started, tokens, words
from needtoknow.providers.catalog import load_catalog
from needtoknow.settings import setting

FILTERED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


def request_body(system: str, messages: list[dict], thinking: bool | None, as_json: bool,
                 max_tokens: int | None, temperature: float | None) -> dict:
    if thinking is False:
        raise ProviderError("Gemini 3 has no thinking level that turns it off: leave thinking to the model, or ask for it")
    config: dict = {}
    if as_json:
        config["responseMimeType"] = "application/json"
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    if temperature is not None:
        config["temperature"] = temperature
    if thinking:
        config["thinkingConfig"] = {"thinkingLevel": "high", "includeThoughts": True}
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages]
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents}
    if config:
        body["generationConfig"] = config
    return body


def read(reply: dict) -> tuple[str, str, dict, str]:
    """The first candidate's text and thoughts, the usage, and the stop reason."""
    candidates = records(reply.get("candidates"))
    parts = records(record(candidates[0].get("content")).get("parts")) if candidates else []
    text = "".join(words(p.get("text")) for p in parts if not p.get("thought"))
    reasoning = "\n".join(words(p.get("text")) for p in parts if p.get("thought"))
    usage = record(reply.get("usageMetadata"))
    cached, thoughts = tokens(usage.get("cachedContentTokenCount")), tokens(usage.get("thoughtsTokenCount"))
    billed = {"input_hit": cached, "input_miss": max(tokens(usage.get("promptTokenCount")) - cached, 0),
              "output": tokens(usage.get("candidatesTokenCount")) + thoughts, "reasoning": thoughts}
    finish = words(candidates[0].get("finishReason")) if candidates else "BLOCKED"
    if finish == "STOP":
        stop = "stop"
    elif finish == "MAX_TOKENS":
        stop = "length"
    elif finish in FILTERED or finish == "BLOCKED":
        stop = "content_filter"
    else:
        stop = finish.lower()
    return text, reasoning, billed, stop


def complete(system: str, messages: list[dict], thinking: bool | None = None, as_json: bool = False,
             max_tokens: int | None = None, temperature: float | None = None, config: dict | None = None,
             opener: Callable = urllib.request.urlopen, now: Callable[[], dt.datetime] | None = None,
             sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0) -> Completion:
    """One generateContent call to Gemini, tried again on a busy or failing server, with its usage."""
    gemini = (config or load_catalog())["gemini"]
    key, model = setting(gemini["key_env"]), gemini.get("model")
    if not model:
        raise ProviderError("no Gemini model: set gemini.model in config/providers.toml")
    if not key:
        raise ProviderError(f"no Gemini key: set {gemini['key_env']} in the environment or in .env")
    body = request_body(system, messages, thinking, as_json, max_tokens, temperature)
    url = f"{gemini['base_url'].rstrip('/')}/v1beta/models/{urllib.parse.quote(model)}:generateContent"
    began_at, began = started(now), time.perf_counter()
    replies, attempts = post("Gemini", url, {"x-goog-api-key": key}, body, key, opener, sleep, timeout)
    text, reasoning, usage, stop = read(replies[-1])
    return Completion(text=text, reasoning=reasoning, usage=usage, stop_reason=stop,
                      seconds=time.perf_counter() - began, cost_usd=priced(gemini, model, usage, began_at),
                      model=str(replies[-1].get("modelVersion") or model), attempts=attempts)
