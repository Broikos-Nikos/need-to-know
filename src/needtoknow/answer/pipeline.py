"""One question, end to end (T4.4): the role's search, the excerpts, the prompt, the provider or the answer cache,
and the answer read back, with each stage's seconds and what the call cost.

The provider's settings come from `config/providers.toml`, and the answer cache is keyed
by the prompt as sent (AC-F8). Every question appends one line to
NTK_DATA_DIR/logs/spend.jsonl: when, the role, the provider and model, the prompt's key,
whether the cache answered, the tokens as billed, the USD, the stop reason, the status
and each stage's seconds. The line holds numbers and hashes and never the text of a
question, an excerpt or an answer, so the log tells nobody what a role may read
(`specs/architecture.md` section 10, logs). `spend` sums it.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import time
from pathlib import Path

from needtoknow.answer import cache
from needtoknow.answer.prompt import build, parse, section_of
from needtoknow.index.expand import excerpts as expand
from needtoknow.index.search import search
from needtoknow.providers import anthropic, deepseek, gemini, openai
from needtoknow.providers.catalog import load_catalog
from needtoknow.settings import data_dir

ADAPTERS = {"deepseek": deepseek.complete, "openai": openai.complete, "anthropic": anthropic.complete,
            "gemini": gemini.complete}
KINDS = ("input_hit", "input_miss", "output", "reasoning")


def spend_path() -> Path:
    return data_dir() / "logs" / "spend.jsonl"


def settings_of(entry: dict) -> dict:
    """What the answer layer sends a provider with every question, from its entry in config/providers.toml."""
    thinking = entry.get("thinking")
    return {"thinking": None if thinking is None else thinking == "enabled", "as_json": True,
            "max_tokens": entry.get("max_tokens"), "temperature": entry.get("temperature")}


def ask(client, question: str, role: str, policy: dict | None = None, provider: str = "deepseek",
        use_cache: bool = True, config: dict | None = None, encode=None, rerank: str | None = None) -> dict:
    """The role's answer to the question: status, answer and citations as `parse` reads them, the excerpts shown,
    whether the cache answered, the model, the tokens and USD of the call, its stop reason and each stage's seconds."""
    catalog = config or load_catalog()
    entry = catalog[provider]
    seconds: dict[str, float] = {}
    began = clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        seconds[stage], clock = now - clock, now

    hits = search(client, question, role, policy, encode=encode, rerank=rerank)
    lap("search")
    shown = expand(hits)
    lap("expand")
    system, user = build(question, shown)
    messages = [{"role": "user", "content": user}]
    settings = settings_of(entry)
    entry_key = cache.key(provider, entry["model"], system, messages, settings)
    kept = None
    if use_cache:
        with cache.opened() as db:
            kept = cache.get(db, entry_key)
    if kept is None:
        done = ADAPTERS[provider](system, messages, config=catalog, **settings)
        if use_cache:
            with cache.opened() as db:
                cache.put(db, entry_key, provider, done)
        text, model, stop, usage, usd, attempts = (done.text, done.model, done.stop_reason, done.usage, done.cost_usd,
                                                   done.attempts)
    else:  # nothing billed now: the call that filled the entry was billed once, when it was made
        text, model, stop = kept["text"], kept["model"], kept["stop_reason"]
        usage, usd, attempts = {kind: 0 for kind in KINDS}, 0.0, 0
    lap("provider")
    answer = parse(text, shown, question)
    lap("parse")
    seconds["total"] = time.perf_counter() - began
    record = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "role": role, "provider": provider,
              "model": model, "prompt": entry_key[:16], "cached": kept is not None, "attempts": attempts,
              "usage": usage, "usd": usd, "stop_reason": stop, "status": answer["status"],
              "seconds": {stage: round(value, 3) for stage, value in seconds.items()}}
    log = spend_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")
    return {**answer, "excerpts": [{"doc_id": e["doc_id"], "section": section_of(e), "units": e["units"]} for e in shown],
            "cached": kept is not None, "model": model, "usage": usage, "cost_usd": usd, "stop_reason": stop,
            "attempts": attempts, "seconds": seconds, "prompt": entry_key}


def spend(path: Path | None = None) -> dict:
    """The spend log summed: questions, how many the cache answered, USD, tokens as billed, and the median and 95th
    percentile seconds of a question, by nearest rank."""
    path = path or spend_path()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []
    times = sorted(line["seconds"]["total"] for line in lines)
    return {"questions": len(lines), "cached": sum(line["cached"] for line in lines),
            "usd": sum(line["usd"] or 0.0 for line in lines), "unpriced": sum(line["usd"] is None for line in lines),
            "tokens": {kind: sum(line["usage"].get(kind) or 0 for line in lines) for kind in KINDS},
            "median_seconds": statistics.median(times) if times else None,
            "p95_seconds": times[math.ceil(0.95 * len(times)) - 1] if times else None}
