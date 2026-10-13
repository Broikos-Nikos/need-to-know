"""The answer cache (T4.4, AC-F8): a provider's reply, kept under the prompt as sent.

The key is a hash of everything the provider reads, the rules, the mark of the asking
role's label set, the excerpts and the question, with the provider, the model and the
call's settings; never of the role's name. A narrowed, relabelled or edited section changes
the excerpts a role is shown, so it changes the key, and an answer cannot outlive the
change that should end it. A key made of the question and the role's readable labels alone
would not move when a section narrows, since the role's labels stay as they were, and it
would serve the old answer. Two roles share an entry only when they read the same labels
and were shown the same text, so a reply from the cache tells no role that someone with
other access asked (AC-F11). An evaluation's repeats put their number in the key, so each
is asked of the provider once and read back after (KB5-K13).

A model name is an alias the provider may move to another model (KB5-K14): each entry keeps
the model its reply named, every fresh reply records the model now behind its alias, and an
entry whose model is no longer that one is stale and asked again.

Only a reply that ended normally is kept: one cut short, filtered or refused is asked
again next time. The cache lives under NTK_DATA_DIR/llm-cache, never in the repository.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from needtoknow.providers.call import Completion
from needtoknow.settings import data_dir


def key(provider: str, model: str, system: str, messages: list[dict], settings: dict, repeat: int = 0) -> str:
    """The hash of what the provider reads and how it is asked, and of an evaluation's repeat after the first."""
    asked = {"provider": provider, "model": model, "system": system, "messages": messages, "settings": settings}
    if repeat:
        asked["repeat"] = repeat
    canonical = json.dumps(asked, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_path() -> Path:
    return data_dir() / "llm-cache" / "answers.db"


@contextmanager
def opened(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, provider TEXT NOT NULL, "
                   "model TEXT NOT NULL, text TEXT NOT NULL, reasoning TEXT NOT NULL, usage TEXT NOT NULL, "
                   "stop_reason TEXT NOT NULL, stored REAL NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS served (alias TEXT PRIMARY KEY, model TEXT NOT NULL, seen REAL NOT NULL)")
        yield db
        db.commit()
    finally:
        db.close()


def get(db: sqlite3.Connection, entry: str, alias: str | None = None) -> dict | None:
    """The entry's reply, or None when there is none or when the alias it was asked by now answers with another
    model."""
    row = db.execute("SELECT model, text, reasoning, usage, stop_reason FROM answers WHERE key = ?", (entry,)).fetchone()
    if row is None:
        return None
    model, text, reasoning, usage, stop = row
    now = db.execute("SELECT model FROM served WHERE alias = ?", (alias,)).fetchone() if alias else None
    if now is not None and now[0] != model:  # stale: the alias has moved since this reply (KB5-K14)
        return None
    return {"model": model, "text": text, "reasoning": reasoning, "usage": json.loads(usage), "stop_reason": stop}


def served(db: sqlite3.Connection, alias: str, model: str) -> None:
    """Record the model a fresh reply under this alias named."""
    db.execute("INSERT OR REPLACE INTO served VALUES (?, ?, ?)", (alias, model, time.time()))


def put(db: sqlite3.Connection, entry: str, provider: str, done: Completion) -> bool:
    """Keep a reply that ended normally; return whether it was kept."""
    if done.stop_reason != "stop":
        return False
    db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (entry, provider, done.model, done.text, done.reasoning, json.dumps(done.usage), done.stop_reason,
                time.time()))
    return True
