"""The vector cache (T3.1): each passage's first token state, kept by the passage that made it.

A unit is embedded as its passage: its heading path, already cut to the headings every
one of its readers may read (AC-F4), on a line above its text. The cache is one SQLite
file per model under NTK_DATA_DIR/embed, named by the model's repository, revision,
dtype and pooling recipe, and keyed by the first 16 bytes of the passage's sha256, so a
passage met again, in a later run, after a relabel or for a rebuilt index, is never
embedded twice. It keeps the state in the model's own dtype, before normalising, so
storing it loses nothing, and `read` gives the same float32 rows from it every time.

`fill` embeds every manifest unit whose passage is missing, in shards committed one by
one: a run killed midway keeps what it finished, and the next run embeds only the rest.
A passage over the model's max_length is refused, naming its unit, before its shard
reaches the GPU.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from needtoknow.embed import model
from needtoknow.ingest.units import passage
from needtoknow.settings import data_dir

SCHEMA_VERSION = 1
# The pooling: the first token's last hidden state. A change to it is a new cache file.
RECIPE = "cls1"
# Passages per commit: about a minute of the GPU at the measured rate.
SHARD = 8192
SCHEMA = """
CREATE TABLE meta (model TEXT NOT NULL);
CREATE TABLE vectors (key BLOB NOT NULL UNIQUE, state BLOB NOT NULL);
"""


class CacheError(RuntimeError):
    """A cache file of another model or schema, or a stored state of the wrong size."""


def model_key(config: dict) -> str:
    return f"{config['repo']}@{config['revision']} {config['dtype']} {RECIPE}"


def cache_path(config: dict) -> Path:
    name = config["repo"].rsplit("/", 1)[-1]
    return data_dir() / "embed" / f"{name}-{config['revision'][:12]}-{config['dtype']}-{RECIPE}.db"


def passage_key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()[:16]


@contextmanager
def opened(config: dict | None = None, path: Path | None = None):
    """The model's cache, created on first use; a file made for another model or schema is refused."""
    config = config or model.load_config()
    path = path or cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        # States are 2 KB: at the default 4 KB page each row would take a page of its own.
        db.execute("PRAGMA page_size = 65536")
        if not db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchone():
            db.executescript(SCHEMA)
            db.execute("INSERT INTO meta (model) VALUES (?)", (model_key(config),))
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.commit()
        db.execute("PRAGMA journal_mode = WAL")
        try:
            found = db.execute("SELECT model FROM meta").fetchone()
        except sqlite3.OperationalError:
            found = None
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION or found != (model_key(config),):
            raise CacheError(f"{path} holds {found[0] if found else 'no model'} at schema {version}, "
                             f"not {model_key(config)} at schema {SCHEMA_VERSION}")
        yield db
        db.commit()
    finally:
        db.close()


def read(db: sqlite3.Connection, keys: list[bytes], config: dict | None = None):
    """Normalised float32 rows for the given passage keys, in their order; a missing or malformed one raises."""
    import numpy as np

    config = config or model.load_config()
    dtype = np.dtype(config["dtype"])
    size = config["dimensions"] * dtype.itemsize
    found: dict[bytes, bytes] = {}
    for start in range(0, len(keys), 900):
        part = keys[start:start + 900]
        found.update(db.execute(f"SELECT key, state FROM vectors WHERE key IN ({','.join('?' * len(part))})", part))
    missing = [k.hex() for k in keys if k not in found]
    if missing:
        raise CacheError(f"{len(missing)} passages are not in the cache, the first {missing[0]}")
    wrong = [k.hex() for k in keys if len(found[k]) != size]
    if wrong:
        raise CacheError(f"{len(wrong)} stored states are not {size} bytes, the first {wrong[0]}")
    states = np.frombuffer(b"".join(found[k] for k in keys), dtype=dtype).reshape(len(keys), config["dimensions"])
    return model.normalise(states)


def fill(manifest: Path | None = None, budget: int | None = None, shard: int = SHARD, report=None) -> dict:
    """Embed every unit of the manifest whose passage the cache lacks; a summary of the run."""
    from needtoknow.ingest.manifest import opened as manifest_opened

    config = model.load_config()
    with manifest_opened(manifest) as db:
        units = db.execute("SELECT unit_id, path, text FROM units ORDER BY unit_id").fetchall()
    with opened(config) as cache:
        have = {key for (key,) in cache.execute("SELECT key FROM vectors")}
        todo: dict[bytes, tuple[str, str]] = {}
        for unit_id, path, text in units:
            text = passage(json.loads(path), text)
            key = passage_key(text)
            if key not in have:
                todo.setdefault(key, (unit_id, text))
        items = list(todo.items())
        summary = {"units": len(units), "cached": len(have), "missing": len(items), "embedded": 0, "tokens": 0,
                   "seconds": 0.0, "model": model_key(config), "path": str(cache_path(config))}
        del units, todo
        started = time.perf_counter()
        for start in range(0, len(items), shard):
            part = items[start:start + shard]
            ids = model.tokenize([text for _, (_, text) in part])
            over = [(unit_id, len(x)) for (_, (unit_id, _)), x in zip(part, ids) if len(x) > config["max_length"]]
            if over:
                raise model.EmbedError(f"{len(over)} passages are longer than max_length {config['max_length']} "
                                       f"tokens, the first unit {over[0][0]} at {over[0][1]}: refused rather than cut")
            states = model.states(ids, budget)
            cache.executemany("INSERT INTO vectors (key, state) VALUES (?, ?)",
                              ((key, row.tobytes()) for (key, _), row in zip(part, states)))
            cache.commit()
            summary["embedded"] += len(part)
            summary["tokens"] += sum(map(len, ids))
            summary["seconds"] = time.perf_counter() - started
            if report:
                rate = summary["tokens"] / summary["seconds"]
                report(f"embedded {summary['embedded']} of {len(items)} passages, {summary['tokens']} tokens, "
                       f"{rate:,.0f} tokens/s")
    return summary
