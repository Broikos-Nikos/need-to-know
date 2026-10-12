"""The answer cache (T4.4, AC-F8): a provider's reply, kept under the prompt as sent.

The key is a hash of everything the provider reads, the rules, the excerpts and the
question, with the provider, the model and the call's settings; never of who asked. A
narrowed, relabelled or edited section changes the excerpts a role is shown, so it changes
the key, and an answer cannot outlive the change that should end it. A key made of the
question and the role's readable labels would not move when a section narrows, since the
role's labels stay as they were, and it would serve the old answer. Two roles share an
entry only when they were shown the same text, which both may read.

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


def key(provider: str, model: str, system: str, messages: list[dict], settings: dict) -> str:
    """The hash of what the provider reads and how it is asked."""
    canonical = json.dumps({"provider": provider, "model": model, "system": system, "messages": messages,
                            "settings": settings}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
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
        yield db
        db.commit()
    finally:
        db.close()


def get(db: sqlite3.Connection, entry: str) -> dict | None:
    row = db.execute("SELECT model, text, reasoning, usage, stop_reason FROM answers WHERE key = ?", (entry,)).fetchone()
    if row is None:
        return None
    model, text, reasoning, usage, stop = row
    return {"model": model, "text": text, "reasoning": reasoning, "usage": json.loads(usage), "stop_reason": stop}


def put(db: sqlite3.Connection, entry: str, provider: str, done: Completion) -> bool:
    """Keep a reply that ended normally; return whether it was kept."""
    if done.stop_reason != "stop":
        return False
    db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (entry, provider, done.model, done.text, done.reasoning, json.dumps(done.usage), done.stop_reason,
                time.time()))
    return True
