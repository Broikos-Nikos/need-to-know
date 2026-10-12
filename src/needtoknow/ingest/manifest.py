"""The manifest (T2.7) and the scan that keeps it current (T2.8), in SQLite under NTK_DATA_DIR.

A file is keyed by its doc_id, which its sidecar gives and which stays when the file
moves. It keeps its path, the size and mtime of the file and of its sidecar, a hash
of what its units are cut from (the decoded text and the sidecar's default
audience), and the versions of the steps that cut it. A unit is keyed by its content
addressed id (architecture section 3), so a run that finds nothing new changes no row.

The scan reads a file again only when its size or mtime, its sidecar's, or the steps'
versions have changed; the hash then decides. Same hash at a new path is a rename,
recorded without cutting; same hash at the same path is a touch. The steps' version
covers the code of every step that decides text or labels, so a change to it marks
every file (DR-F7). Deletions come from comparing the full set of doc_ids on disk with
the manifest's, which no timestamp ever shows; a corpus folder that is not there is
refused rather than read as every document deleted (DR-F11). `build` reads and cuts
every file and keeps whatever the fresh cut gives, whatever the hashes say: it is the
run that verifies. Every run records, per file, what changed.
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
from needtoknow.ingest import pipeline
from needtoknow.ingest.units import Unit
from needtoknow.settings import data_dir

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE files (
    doc_id TEXT PRIMARY KEY, path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
    sidecar_size INTEGER NOT NULL, sidecar_mtime_ns INTEGER NOT NULL, content_hash TEXT NOT NULL,
    format TEXT NOT NULL, encoding TEXT NOT NULL, language TEXT NOT NULL, versions TEXT NOT NULL, run_id INTEGER NOT NULL
);
CREATE TABLE units (
    unit_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES files(doc_id) ON DELETE CASCADE,
    section_key TEXT NOT NULL, layer TEXT NOT NULL, labels TEXT NOT NULL, path TEXT NOT NULL,
    text TEXT NOT NULL, start INTEGER NOT NULL, "end" INTEGER NOT NULL, language TEXT NOT NULL, run_id INTEGER NOT NULL
);
CREATE INDEX units_by_doc ON units(doc_id);
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, started TEXT NOT NULL, finished TEXT,
    files INTEGER, files_read INTEGER, files_changed INTEGER, files_removed INTEGER, units_added INTEGER,
    units_removed INTEGER, units_total INTEGER
);
CREATE TABLE changes (
    run_id INTEGER NOT NULL, doc_id TEXT NOT NULL, kind TEXT NOT NULL, units_added INTEGER NOT NULL,
    units_removed INTEGER NOT NULL
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


def content_hash(text: str, meta: dict) -> str:
    """What a file's units are cut from: its decoded text and its sidecar's default audience."""
    return hashlib.sha256(f"{meta.get('default_audience', '')}\x1f{text}".encode("utf-8")).hexdigest()


def unique_ids(found: list[Unit]) -> list[Unit]:
    """Two identical units in one section share an id; the second and later get the occurrence appended."""
    seen: dict[str, int] = {}
    out = []
    for u in found:
        n = seen.get(u.unit_id, 0)
        seen[u.unit_id] = n + 1
        out.append(u if n == 0 else replace(u, unit_id=f"{u.unit_id}#{n}"))
    return out


def file_row(db: sqlite3.Connection, run_id: int, rel: str, file: Path, sidecar: Path, meta: dict, digest: str) -> None:
    stat, side = file.stat(), sidecar.stat()
    db.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET path = excluded.path, "
        "size = excluded.size, mtime_ns = excluded.mtime_ns, sidecar_size = excluded.sidecar_size, "
        "sidecar_mtime_ns = excluded.sidecar_mtime_ns, content_hash = excluded.content_hash, format = excluded.format, "
        "encoding = excluded.encoding, language = excluded.language, versions = excluded.versions, run_id = excluded.run_id",
        (meta["doc_id"], rel, stat.st_size, stat.st_mtime_ns, side.st_size, side.st_mtime_ns, digest, meta["format"],
         meta["encoding"], meta["language"], pipeline.VERSIONS, run_id))


def replace_units(db: sqlite3.Connection, run_id: int, doc_id: str, found: list[Unit]) -> tuple[int, int]:
    """A file's units brought up to date; returns the units added and removed."""
    found = unique_ids(found)
    old = {row[0] for row in db.execute("SELECT unit_id FROM units WHERE doc_id = ?", (doc_id,))}
    new = {u.unit_id: u for u in found}
    removed = old - set(new)
    added = [u for uid, u in new.items() if uid not in old]
    db.executemany("DELETE FROM units WHERE unit_id = ?", [(uid,) for uid in removed])
    db.executemany("INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   [(u.unit_id, u.doc_id, u.section_key, u.layer, json.dumps(u.labels), json.dumps(u.path, ensure_ascii=False),
                     u.text, u.start, u.end, u.language, run_id) for u in added])
    return len(added), len(removed)


def scan(full: Path | None = None, path: Path | None = None, force: bool = False) -> dict:
    """Bring the manifest up to date with the corpus, reading only what may have changed unless forced."""
    full = full or data_dir() / "corpus" / "full"
    if not full.is_dir():
        raise ManifestError(f"{full} is not there, so every document would read as deleted: nothing scanned. A swap "
                            f"killed between its renames leaves it so, and the next ntk corpus assemble settles it")
    kind = "build" if force else "scan"
    with opened(path) as db:
        run_id = db.execute("INSERT INTO runs (kind, started) VALUES (?, ?)", (kind, now())).lastrowid
        summary = {"run": run_id, "kind": kind, "files": 0, "files_read": 0, "files_changed": 0, "files_removed": 0,
                   "units_added": 0, "units_removed": 0, "changes": {}}
        known = {row[0]: row[1:] for row in db.execute(
            "SELECT doc_id, path, size, mtime_ns, sidecar_size, sidecar_mtime_ns, content_hash, versions FROM files")}
        unknown: list[str] = []
        found = corpus_files(full)

        def note(doc_id: str, change: str, added: int = 0, removed: int = 0) -> None:
            db.execute("INSERT INTO changes VALUES (?, ?, ?, ?, ?)", (run_id, doc_id, change, added, removed))
            summary["changes"][change] = summary["changes"].get(change, 0) + 1

        for doc_id, (file, sidecar, meta) in sorted(found.items()):
            summary["files"] += 1
            rel = file.relative_to(full).as_posix()
            row = known.get(doc_id)
            stat, side = file.stat(), sidecar.stat()
            same_stats = row is not None and row[1:5] == (stat.st_size, stat.st_mtime_ns, side.st_size, side.st_mtime_ns)
            if not force and same_stats and row[6] == pipeline.VERSIONS:
                if row[0] != rel:
                    db.execute("UPDATE files SET path = ?, run_id = ? WHERE doc_id = ?", (rel, run_id, doc_id))
                    note(doc_id, "renamed")
                continue
            text, cut = pipeline.document_units(file, meta, unknown)
            summary["files_read"] += 1
            digest = content_hash(text, meta)
            if not force and row is not None and row[5] == digest and row[6] == pipeline.VERSIONS:
                file_row(db, run_id, rel, file, sidecar, meta, digest)
                note(doc_id, "renamed" if row[0] != rel else "touched")
                continue
            file_row(db, run_id, rel, file, sidecar, meta, digest)
            added, removed = replace_units(db, run_id, doc_id, cut)
            summary["units_added"] += added
            summary["units_removed"] += removed
            if added or removed or row is None:
                summary["files_changed"] += 1
                note(doc_id, "added" if row is None else "changed", added, removed)
            elif row[0] != rel:
                note(doc_id, "renamed")
            elif not force:
                note(doc_id, "touched")
        for doc_id in sorted(set(known) - set(found)):
            removed = db.execute("SELECT count(*) FROM units WHERE doc_id = ?", (doc_id,)).fetchone()[0]
            db.execute("DELETE FROM files WHERE doc_id = ?", (doc_id,))
            summary["files_removed"] += 1
            summary["units_removed"] += removed
            note(doc_id, "removed", 0, removed)
        summary["units_total"] = db.execute("SELECT count(*) FROM units").fetchone()[0]
        summary["unknown_audiences"] = sorted(set(unknown))
        db.execute("UPDATE runs SET finished = ?, files = ?, files_read = ?, files_changed = ?, files_removed = ?, "
                   "units_added = ?, units_removed = ?, units_total = ? WHERE run_id = ?",
                   (now(), summary["files"], summary["files_read"], summary["files_changed"], summary["files_removed"],
                    summary["units_added"], summary["units_removed"], summary["units_total"], run_id))
    return summary


def build(full: Path | None = None, path: Path | None = None) -> dict:
    """Every file read and cut, whatever its timestamps say."""
    return scan(full, path, force=True)
