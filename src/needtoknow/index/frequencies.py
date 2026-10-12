"""Document frequencies per label set (T3.12, N50): what a role's lexical weights are counted from.

The collection's sparse vectors hold BM25's term frequency part alone. A question's term
weights are BM25's IDF, ln(1 + (N-n+0.5)/(n+0.5)), Qdrant's own formula, with N the
passages and n the passages holding the term, both summed over the label sets the asking
role may read. So a word frequent only in text the role may not read weighs exactly
what it weighs in the role's own view (AC-F5, KB1-F4), and a change to the policy takes
effect at the next question, since views are summed then.

The sync keeps the table in step with the index: a point's terms are counted in when it
is upserted, and counted out, from the point itself, before it is deleted. `recount`
rebuilds the table from the index, for a crash between the two stores.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from needtoknow.settings import data_dir

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE meta (collection TEXT NOT NULL);
CREATE TABLE sets (id INTEGER PRIMARY KEY, label_set TEXT NOT NULL UNIQUE, passages INTEGER NOT NULL);
CREATE TABLE df (term INTEGER NOT NULL, set_id INTEGER NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (term, set_id)) WITHOUT ROWID;
"""


class FrequencyError(RuntimeError):
    """A frequency table kept for another collection or schema."""


def table_path() -> Path:
    return data_dir() / "index" / "frequencies.db"


@contextmanager
def opened(collection: str, path: Path | None = None):
    """The frequency table of a collection, created empty on first use."""
    path = path or table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        if not db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchone():
            db.executescript(SCHEMA)
            db.execute("INSERT INTO meta (collection) VALUES (?)", (collection,))
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.commit()
        db.execute("PRAGMA journal_mode = WAL")
        found = db.execute("SELECT collection FROM meta").fetchone()
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION or found != (collection,):
            raise FrequencyError(f"{path} counts {found[0] if found else 'nothing'}, not {collection}; "
                                 f"ntk index create rebuilds it with the collection")
        yield db
        db.commit()
    finally:
        db.close()


def count(db: sqlite3.Connection, points, sign: int) -> None:
    """Count points in (sign 1) or out (sign -1), each a label set and the term ids of its passage.

    A term counted out to 0 keeps its row, which weighs as nothing.
    """
    by_set: dict[str, Counter] = {}
    passages: Counter = Counter()
    for label_set, terms in points:
        by_set.setdefault(label_set, Counter()).update(set(terms))
        passages[label_set] += 1
    for label_set, tally in by_set.items():
        db.execute("INSERT INTO sets (label_set, passages) VALUES (?, ?) "
                   "ON CONFLICT(label_set) DO UPDATE SET passages = passages + excluded.passages",
                   (label_set, sign * passages[label_set]))
        set_id = db.execute("SELECT id FROM sets WHERE label_set = ?", (label_set,)).fetchone()[0]
        db.executemany("INSERT INTO df (term, set_id, n) VALUES (?, ?, ?) "
                       "ON CONFLICT(term, set_id) DO UPDATE SET n = n + excluded.n",
                       [(term, set_id, sign * n) for term, n in tally.items()])


def weights(db: sqlite3.Connection, terms: list[int], label_sets: list[str]) -> list[float]:
    """BM25's IDF of each term over the given label sets, the role's view."""
    marks = ",".join("?" * len(label_sets)) or "NULL"
    ids = [row[0] for row in db.execute(f"SELECT id FROM sets WHERE label_set IN ({marks})", label_sets)]
    total = db.execute(f"SELECT COALESCE(SUM(passages), 0) FROM sets WHERE id IN ({','.join('?' * len(ids)) or 'NULL'})",
                       ids).fetchone()[0]
    held = dict(db.execute(f"SELECT term, SUM(n) FROM df WHERE term IN ({','.join('?' * len(terms)) or 'NULL'}) "
                           f"AND set_id IN ({','.join('?' * len(ids)) or 'NULL'}) GROUP BY term", [*terms, *ids]))
    return [math.log(1 + (total - held.get(term, 0) + 0.5) / (held.get(term, 0) + 0.5)) for term in terms]


def recount(db: sqlite3.Connection, client, collection: str) -> int:
    """Rebuild the table from every point of the collection; the points counted."""
    from needtoknow.index.collection import SPARSE

    db.execute("DELETE FROM df")
    db.execute("DELETE FROM sets")
    counted, offset = 0, None
    while True:
        records, offset = client.scroll(collection, limit=2_000, offset=offset, with_payload=["label_set"],
                                        with_vectors=[SPARSE])
        count(db, [(r.payload["label_set"], r.vector[SPARSE].indices) for r in records], 1)
        counted += len(records)
        if offset is None:
            return counted
