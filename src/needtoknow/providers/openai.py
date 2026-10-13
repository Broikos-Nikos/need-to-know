"""The OpenAI adapter (T4.3, N13): one response from the Responses API, tested on documented replies, never run.

The Responses API is OpenAI's recommended surface for new work (`09-production/04`):
`instructions` carries the system rules, `input` the messages, `store` is false so the
call keeps no state, `text.format` asks for JSON, since `response_format` is silently
ignored there, and `max_output_tokens` caps the reply, reasoning included. Thinking is
`reasoning.effort`: `high` when asked, `none` when refused, which some models answer with
a 400, and nothing at all when the caller leaves it to the model.

The reply's text is its `output_text` parts; a `refusal` part ends it as a refusal. Usage
reads `input_tokens` with `input_tokens_details.cached_tokens` apart and `output_tokens`
with `output_tokens_details.reasoning_tokens` among them; an `incomplete` status gives
its reason, `max_output_tokens` read as length. No key exists for it here, so it has
never answered a question; its cost is None until config/providers.toml prices a model.
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.request
from typing import Callable

from needtoknow.providers.call import Completion, ProviderError, post, priced, record, records, started, tokens, words
from needtoknow.providers.catalog import load_catalog
from needtoknow.settings import setting

STOPS = {"max_output_tokens": "length", "content_filter": "content_filter"}


def request_body(model: str, system: str, messages: list[dict], thinking: bool | None, as_json: bool,
                 max_tokens: int | None, temperature: float | None) -> dict:
    body = {"model": model, "instructions": system, "input": messages, "store": False}
    if thinking is not None:
        body["reasoning"] = {"effort": "high" if thinking else "none"}
    if as_json:
        body["text"] = {"format": {"type": "json_object"}}
    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    return body


def read(reply: dict) -> tuple[str, str, dict, str]:
    """The reply's text, its reasoning summary, its usage and its stop reason."""
    text, reasoning, refused = [], [], False
    for item in records(reply.get("output")):
        if item.get("type") == "message":
            for part in records(item.get("content")):
                if part.get("type") == "output_text":
                    text.append(words(part.get("text")))
                elif part.get("type") == "refusal":
                    refused = True
        elif item.get("type") == "reasoning":
            reasoning += [words(s.get("text")) for s in records(item.get("summary"))]
    usage = record(reply.get("usage"))
    cached = tokens(record(usage.get("input_tokens_details")).get("cached_tokens"))
    billed = {"input_hit": cached, "input_miss": max(tokens(usage.get("input_tokens")) - cached, 0),
              "output": tokens(usage.get("output_tokens")),
              "reasoning": tokens(record(usage.get("output_tokens_details")).get("reasoning_tokens"))}
    if refused:
        stop = "refusal"
    elif reply.get("status") == "incomplete":
        reason = words(record(reply.get("incomplete_details")).get("reason")) or "incomplete"
        stop = STOPS.get(reason, reason)
    else:
        stop = "stop" if reply.get("status") == "completed" else str(reply.get("status") or "")
    return "".join(text), "\n".join(reasoning), billed, stop


def complete(system: str, messages: list[dict], thinking: bool | None = None, as_json: bool = False,
             max_tokens: int | None = None, temperature: float | None = None, config: dict | None = None,
             opener: Callable = urllib.request.urlopen, now: Callable[[], dt.datetime] | None = None,
             sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0) -> Completion:
    """One response from OpenAI, tried again on a busy or failing server, with its usage."""
    openai = (config or load_catalog())["openai"]
    key, model = setting(openai["key_env"]), openai.get("model")
    if not model:
        raise ProviderError("no OpenAI model: set openai.model in config/providers.toml")
    if not key:
        raise ProviderError(f"no OpenAI key: set {openai['key_env']} in the environment or in .env")
    began_at, began = started(now), time.perf_counter()
    replies, attempts = post("OpenAI", openai["base_url"].rstrip("/") + "/responses", {"Authorization": "Bearer " + key},
                             request_body(model, system, messages, thinking, as_json, max_tokens, temperature),
                             key, opener, sleep, timeout)
    text, reasoning, usage, stop = read(replies[-1])
    return Completion(text=text, reasoning=reasoning, usage=usage, stop_reason=stop,
                      seconds=time.perf_counter() - began, cost_usd=priced(openai, model, usage, began_at),
                      model=str(replies[-1].get("model") or model), attempts=attempts)
