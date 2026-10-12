"""The manifest (T2.7): every file and every unit ingestion has seen, in SQLite under NTK_DATA_DIR.

A file is keyed by its doc_id, which its sidecar gives and which stays when the file
moves; it keeps its path, size, mtime, the hash of its decoded text and the versions
of the steps that cut it. A unit is keyed by its content addressed id (architecture
section 3), so a run that finds nothing new changes no row. Each run records what it
saw and what it changed. `build` reads every file every time; the scan of T2.8 adds
the shortcuts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from needtoknow.corpus.mutate import corpus_files
from needtoknow.ingest.pipeline import VERSIONS, document_units
from needtoknow.ingest.units import Unit
from needtoknow.settings import data_dir

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE files (
    doc_id TEXT PRIMARY KEY, path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
    content_hash TEXT NOT NULL, format TEXT NOT NULL, encoding TEXT NOT NULL, language TEXT NOT NULL,
    versions TEXT NOT NULL, run_id INTEGER NOT NULL
);
CREATE TABLE units (
    unit_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES files(doc_id) ON DELETE CASCADE,
    section_key TEXT NOT NULL, layer TEXT NOT NULL, labels TEXT NOT NULL, path TEXT NOT NULL,
    text TEXT NOT NULL, start INTEGER NOT NULL, "end" INTEGER NOT NULL, language TEXT NOT NULL, run_id INTEGER NOT NULL
);
CREATE INDEX units_by_doc ON units(doc_id);
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, started TEXT NOT NULL, finished TEXT,
    files INTEGER, files_changed INTEGER, files_removed INTEGER, units_added INTEGER, units_removed INTEGER,
    units_total INTEGER
);
"""


class ManifestError(RuntimeError):
    """A manifest from another schema version, or one that cannot be opened."""


def manifest_path() -> Path:
    return data_dir() / "manifest.db"


@contextmanager
def opened(path: Path | None = None):
    """The manifest, created on first use, refused when its schema is another version."""
    path = path or manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version == 0 and not db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchone():
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif version != SCHEMA_VERSION:
            raise ManifestError(f"{path} has schema version {version}, this code reads {SCHEMA_VERSION}: delete it and build again")
        yield db
        db.commit()
    finally:
        db.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def unique_ids(found: list[Unit]) -> list[Unit]:
    """Two identical units in one section share an id; the second and later get the occurrence appended."""
    seen: dict[str, int] = {}
    out = []
    for u in found:
        n = seen.get(u.unit_id, 0)
        seen[u.unit_id] = n + 1
        out.append(u if n == 0 else replace(u, unit_id=f"{u.unit_id}#{n}"))
    return out


def record(db: sqlite3.Connection, run_id: int, file: Path, meta: dict, full: Path, text: str, found: list[Unit]) -> tuple[int, int]:
    """One file's row and units brought up to date; returns the units added and removed."""
    stat = file.stat()
    db.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET path = excluded.path, "
        "size = excluded.size, mtime_ns = excluded.mtime_ns, content_hash = excluded.content_hash, format = excluded.format, "
        "encoding = excluded.encoding, language = excluded.language, versions = excluded.versions, run_id = excluded.run_id",
        (meta["doc_id"], file.relative_to(full).as_posix(), stat.st_size, stat.st_mtime_ns,
         hashlib.sha256(text.encode("utf-8")).hexdigest(), meta["format"], meta["encoding"], meta["language"], VERSIONS, run_id))
    found = unique_ids(found)
    old = {row[0] for row in db.execute("SELECT unit_id FROM units WHERE doc_id = ?", (meta["doc_id"],))}
    new = {u.unit_id: u for u in found}
    removed = old - set(new)
    added = [u for uid, u in new.items() if uid not in old]
    db.executemany("DELETE FROM units WHERE unit_id = ?", [(uid,) for uid in removed])
    db.executemany("INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   [(u.unit_id, u.doc_id, u.section_key, u.layer, json.dumps(u.labels), json.dumps(u.path, ensure_ascii=False),
                     u.text, u.start, u.end, u.language, run_id) for u in added])
    return len(added), len(removed)


def build(full: Path | None = None, path: Path | None = None) -> dict:
    """Every file of the corpus read, cut and recorded; documents gone from disk leave the manifest."""
    full = full or data_dir() / "corpus" / "full"
    with opened(path) as db:
        run_id = db.execute("INSERT INTO runs (kind, started) VALUES ('build', ?)", (now(),)).lastrowid
        summary = {"run": run_id, "files": 0, "files_changed": 0, "files_removed": 0, "units_added": 0, "units_removed": 0}
        unknown: list[str] = []
        found = corpus_files(full)
        for doc_id, (file, _sidecar, meta) in sorted(found.items()):
            text, cut = document_units(file, meta, unknown)
            added, removed = record(db, run_id, file, meta, full, text, cut)
            summary["files"] += 1
            summary["files_changed"] += bool(added or removed)
            summary["units_added"] += added
            summary["units_removed"] += removed
        for (doc_id,) in db.execute("SELECT doc_id FROM files").fetchall():
            if doc_id not in found:
                summary["units_removed"] += db.execute("SELECT count(*) FROM units WHERE doc_id = ?", (doc_id,)).fetchone()[0]
                db.execute("DELETE FROM files WHERE doc_id = ?", (doc_id,))
                summary["files_removed"] += 1
        summary["units_total"] = db.execute("SELECT count(*) FROM units").fetchone()[0]
        summary["unknown_audiences"] = sorted(set(unknown))
        db.execute("UPDATE runs SET finished = ?, files = ?, files_changed = ?, files_removed = ?, units_added = ?, "
                   "units_removed = ?, units_total = ? WHERE run_id = ?",
                   (now(), summary["files"], summary["files_changed"], summary["files_removed"], summary["units_added"],
                    summary["units_removed"], summary["units_total"], run_id))
    return summary
