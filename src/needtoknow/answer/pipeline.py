"""One question, end to end (T4.4): the role's search, the excerpts, the prompt, the provider or the answer cache,
and the answer read back, with each stage's seconds and what the call cost.

The provider's settings come from `config/providers.toml`, and the answer cache is keyed
by the prompt as sent (AC-F8). Every question appends one line to
NTK_DATA_DIR/logs/spend.jsonl: when, the role, the provider and model, the prompt's key,
whether the cache answered, the tokens as billed, the USD, the stop reason, the status,
how many citations no excerpt could take (RS-F14) and each stage's seconds. The line holds numbers and hashes and never the text of a
question, an excerpt or an answer, so the log tells nobody what a role may read
(`specs/architecture.md` section 10, logs). `spend` sums it. A cache SQLite cannot read is
set aside, the question answered without it, and the line names where it went (HS-F17).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path

from needtoknow.answer import cache
from needtoknow.answer.prompt import build, build_guess, parse, reader_mark, section_of
from needtoknow.index.expand import excerpts as expand
from needtoknow.index.search import search
from needtoknow.policy import load_policy, readable
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


def warm(encode=None, rerank: str | None = None) -> float:
    """Load the encoder and the reranker a search will use, once a process, before any question is timed; the seconds
    it took, near nothing once they are loaded. Booked as search, a first question's load made the spend log's p95 a
    load time (PF-F8)."""
    began = time.perf_counter()
    from needtoknow.index import rerank as second
    if encode is None:
        from needtoknow.embed import model
        model.load()
    mode = rerank or second.load_config()["mode"]
    if mode == "cross":
        second.cross_encoder()
    elif mode == "multivector":
        second.colbert_head()
    return time.perf_counter() - began


def ask(client, question: str, role: str, policy: dict | None = None, provider: str = "deepseek",
        use_cache: bool = True, config: dict | None = None, encode=None, rerank: str | None = None,
        repeat: int = 0, closed_book: bool = False, guess: bool = False, trace: dict | None = None) -> dict:
    """The role's answer to the question: status, answer and citations as `parse` reads them, the excerpts shown,
    whether the cache answered, the model, the tokens and USD of the call, its stop reason and each stage's seconds.
    An evaluation's repeats pass their number, which the cache key carries (KB5-K13). A closed book question is asked
    with no excerpts and nothing searched, under the answer rules, which leave the model nothing but to refuse: the
    check that they hold with no documents. With `guess` it is asked instead under rules that want the model's best
    guess from its own knowledge, the control KB1-F7 asks for (DR-F27). `trace`, a dict, gets the search's stages
    (`index.search.search`) and the excerpts as shown, for the served path (T6.1)."""
    catalog = config or load_catalog()
    entry = catalog[provider]
    closed_book = closed_book or guess
    loaded = 0.0 if closed_book else warm(encode, rerank)  # before the clock: a question is timed, not a start
    seconds: dict[str, float] = {}
    began = clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        seconds[stage], clock = now - clock, now

    hits = [] if closed_book else search(client, question, role, policy, encode=encode, rerank=rerank, trace=trace)
    lap("search")
    shown = expand(hits) if hits else []
    lap("expand")
    if trace is not None:
        trace["shown"] = shown
    reader = reader_mark(readable(policy or load_policy(), role))
    system, user = build_guess(question, reader=reader) if guess else build(question, shown, reader=reader)
    messages = [{"role": "user", "content": user}]
    settings = settings_of(entry)
    entry_key = cache.key(provider, entry["model"], system, messages, settings, repeat)
    kept, aside = None, None
    if use_cache:
        try:
            with cache.opened() as db:
                kept = cache.get(db, entry_key, entry["model"])
        except sqlite3.DatabaseError:  # it ended the question in a traceback after both models loaded (HS-F17)
            aside = cache.set_aside()
    if kept is None:
        done = ADAPTERS[provider](system, messages, config=catalog, **settings)
        try:
            with cache.opened() as db:
                cache.served(db, entry["model"], done.model)
                if use_cache:
                    cache.put(db, entry_key, provider, done)
        except sqlite3.DatabaseError:
            aside = aside or cache.set_aside()
        text, model, stop, usage, usd, attempts = (done.text, done.model, done.stop_reason, done.usage, done.cost_usd,
                                                   done.attempts)
    else:  # nothing billed now: the call that filled the entry was billed once, when it was made
        text, model, stop = kept["text"], kept["model"], kept["stop_reason"]
        usage, usd, attempts = {kind: 0 for kind in KINDS}, 0.0, 0
    lap("provider")
    answer = parse(text, shown, question)
    lap("parse")
    seconds["total"] = time.perf_counter() - began
    seconds["load"] = loaded
    record = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "role": role, "provider": provider,
              "model": model, "prompt": entry_key[:16], "repeat": repeat, "cached": kept is not None,
              "attempts": attempts,
              "usage": usage, "usd": usd, "stop_reason": stop, "status": answer["status"],
              "dropped": len(answer["dropped"]),
              "seconds": {stage: round(value, 3) for stage, value in seconds.items()}}
    if aside:
        record["cache_set_aside"] = str(aside)
    log = spend_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")
    return {**answer, "excerpts": [{"doc_id": e["doc_id"], "section": section_of(e), "units": e["units"]} for e in shown],
            "cached": kept is not None, "model": model, "usage": usage, "cost_usd": usd, "stop_reason": stop,
            "attempts": attempts, "seconds": seconds, "prompt": entry_key,
            "cache_set_aside": str(aside) if aside else None}


def spend(path: Path | None = None) -> dict:
    """The spend log summed: questions, how many the cache answered, USD, tokens as billed, and the median and 95th
    percentile seconds of a question, by nearest rank. A question's seconds leave out the models' loading, which its
    line keeps apart since PF-F8; a line from before then counts its load in its search."""
    path = path or spend_path()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []
    times = sorted(line["seconds"]["total"] for line in lines)
    return {"questions": len(lines), "cached": sum(line["cached"] for line in lines),
            "usd": sum(line["usd"] or 0.0 for line in lines), "unpriced": sum(line["usd"] is None for line in lines),
            "tokens": {kind: sum(line["usage"].get(kind) or 0 for line in lines) for kind in KINDS},
            "median_seconds": statistics.median(times) if times else None,
            "p95_seconds": times[math.ceil(0.95 * len(times)) - 1] if times else None}
