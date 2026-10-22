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

import gc
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from needtoknow.corpus import exclusive
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
CREATE INDEX units_by_doc_run ON units(doc_id, run_id);
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
# What the change feed and a document's view show a role, counted under each label as the units are written: a unit
# counts once under every label it holds, in its scan's row of `added` and its segment's row of `segments`. A role's
# poll and its view read the rows of the labels it reads and no other, so their work holds nothing of the text it may
# not read, not a scan's time, not a segment's size (AC-F18, AC-F19). Made in one transaction that counts the units
# already there; the triggers keep them in step from then on, a cascade's deletes included, and refuse an update that
# would move a unit out of its rows. The manifest keeps a digest of this text, and a manifest whose digest is not the
# code's has its counts made again, so a change here reaches every manifest made before it (MT-F29).
SUMMARIES = """
CREATE TABLE added (
    label TEXT NOT NULL, run_id INTEGER NOT NULL, doc_id TEXT NOT NULL, labels TEXT NOT NULL, units INTEGER NOT NULL,
    PRIMARY KEY (label, run_id, doc_id, labels)
) WITHOUT ROWID;
CREATE TABLE segments (
    doc_id TEXT NOT NULL, label TEXT NOT NULL, section_key TEXT NOT NULL, labels TEXT NOT NULL, units INTEGER NOT NULL,
    PRIMARY KEY (doc_id, label, section_key, labels)
) WITHOUT ROWID;
INSERT INTO added SELECT j.value, u.run_id, u.doc_id, u.labels, COUNT(*) FROM units u, json_each(u.labels) j
    GROUP BY j.value, u.run_id, u.doc_id, u.labels;
INSERT INTO segments SELECT u.doc_id, j.value, u.section_key, u.labels, COUNT(*) FROM units u, json_each(u.labels) j
    GROUP BY u.doc_id, j.value, u.section_key, u.labels;
CREATE TRIGGER unit_written AFTER INSERT ON units BEGIN
    INSERT INTO added SELECT value, NEW.run_id, NEW.doc_id, NEW.labels, 1 FROM json_each(NEW.labels) WHERE true
        ON CONFLICT (label, run_id, doc_id, labels) DO UPDATE SET units = units + 1;
    INSERT INTO segments SELECT NEW.doc_id, value, NEW.section_key, NEW.labels, 1 FROM json_each(NEW.labels) WHERE true
        ON CONFLICT (doc_id, label, section_key, labels) DO UPDATE SET units = units + 1;
END;
CREATE TRIGGER unit_removed AFTER DELETE ON units BEGIN
    UPDATE added SET units = units - 1 WHERE label IN (SELECT value FROM json_each(OLD.labels))
        AND run_id = OLD.run_id AND doc_id = OLD.doc_id AND labels = OLD.labels;
    DELETE FROM added WHERE label IN (SELECT value FROM json_each(OLD.labels))
        AND run_id = OLD.run_id AND doc_id = OLD.doc_id AND labels = OLD.labels AND units = 0;
    UPDATE segments SET units = units - 1 WHERE doc_id = OLD.doc_id AND label IN (SELECT value FROM json_each(OLD.labels))
        AND section_key = OLD.section_key AND labels = OLD.labels;
    DELETE FROM segments WHERE doc_id = OLD.doc_id AND label IN (SELECT value FROM json_each(OLD.labels))
        AND section_key = OLD.section_key AND labels = OLD.labels AND units = 0;
END;
CREATE TRIGGER unit_kept BEFORE UPDATE OF unit_id, doc_id, section_key, labels, run_id ON units BEGIN
    SELECT RAISE(ABORT, 'a unit keeps its id, document, section, labels and scan: a change is a new unit');
END;
"""
COUNTS = hashlib.sha256(SUMMARIES.encode("utf-8")).hexdigest()[:16]
# The counts as any earlier text made them, taken away before they are made again.
UNMADE = """
DROP TRIGGER IF EXISTS unit_kept;
DROP TRIGGER IF EXISTS unit_removed;
DROP TRIGGER IF EXISTS unit_written;
DROP TABLE IF EXISTS added;
DROP TABLE IF EXISTS segments;
"""


class ManifestError(RuntimeError):
    """A manifest from another schema version, or one that cannot be opened."""


@contextmanager
def collector_paused():
    """Python's cyclic collector off while a scan cuts and writes: a cut makes millions of objects and no cycles, and
    every full collection meanwhile walks every object the process holds. In a watcher holding the encoder, the
    largest file's cut took 3.09 s with it on and 2.16 s with it off (tick 73), and a change waits for it.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


SCAN_WAIT = 900.0  # seconds a scan waits for another to finish: a full build takes 90 to 175


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
        # A hit's segment, for the expansion: found by its own index rather than by reading its whole document through
        # units_by_doc, 2,034 units at the median and 17,369 in the largest (PF-F5). Made once, in a manifest without it.
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'units_by_segment'").fetchone():
            db.execute("CREATE INDEX units_by_segment ON units(doc_id, section_key, labels, start)")
        # A document's units by the run that added them, for the change feed: read through units_by_doc, every poll
        # read every unit of each document a scan changed, the largest's thousands for its one changed unit (PF-F11).
        # It takes units_by_doc's place, which its first column serves, so a scan maintains no more indexes than before.
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'units_by_doc_run'").fetchone():
            db.execute("CREATE INDEX IF NOT EXISTS units_by_doc_run ON units(doc_id, run_id)")
            db.execute("DROP INDEX IF EXISTS units_by_doc")
        # Each plain text file's parts as its last cut found them, so that the next scan cuts only the parts a change
        # touched (T7.6). Made once, in a manifest without it; a file with no parts here is cut whole.
        db.execute('CREATE TABLE IF NOT EXISTS parts (doc_id TEXT NOT NULL REFERENCES files(doc_id) ON DELETE CASCADE, '
                   'ordinal INTEGER NOT NULL, start INTEGER NOT NULL, "end" INTEGER NOT NULL, digest TEXT NOT NULL, '
                   'PRIMARY KEY (doc_id, ordinal)) WITHOUT ROWID')
        made = db.execute("SELECT digest FROM made WHERE name = 'counts'").fetchone() if db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'made'").fetchone() else None
        if made != (COUNTS,):
            db.executescript(f"BEGIN IMMEDIATE;\n{UNMADE}{SUMMARIES}"
                             "CREATE TABLE IF NOT EXISTS made (name TEXT PRIMARY KEY, digest TEXT NOT NULL);\n"
                             f"INSERT OR REPLACE INTO made VALUES ('counts', '{COUNTS}');\nCOMMIT;\n")
        yield db
        db.commit()
    finally:
        db.close()


@contextmanager
def read(path: Path | None = None):
    """The manifest for reads that depend on one another, every statement against one snapshot: a scan committed while
    they run is seen by none of them. Two statements in turn saw a scan between them, and a segment found by the first
    had no units by the second (DR-F41)."""
    with opened(path) as db:
        db.execute("BEGIN")
        yield db


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


def file_row(db: sqlite3.Connection, run_id: int, rel: str, stat, side, meta: dict, digest: str) -> None:
    """A file's row, with the stats of the file and its sidecar taken before either was read (OP-F6)."""
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
    # Offsets are not in the id: a paragraph inserted above a unit moves it and keeps its id, so a surviving unit's
    # offsets are written again, or the expansion and the citations would read where it used to be (T3.7).
    db.executemany('UPDATE units SET start = ?, "end" = ? WHERE unit_id = ? AND (start != ? OR "end" != ?)',
                   [(u.start, u.end, u.unit_id, u.start, u.end) for uid, u in new.items() if uid in old])
    db.executemany("INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   [(u.unit_id, u.doc_id, u.section_key, u.layer, json.dumps(u.labels), json.dumps(u.path, ensure_ascii=False),
                     u.text, u.start, u.end, u.language, run_id) for u in added])
    return len(added), len(removed)


def part_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def parts_of(text: str, meta: dict) -> list[tuple[int, int, str]]:
    """A plain text file's head and keyword sections, each as start, end and digest; the head's digest holds the
    sidecar's default audience too, which every part's labels can come from. Empty for a file without keyword
    sections, which is always cut whole."""
    starts = pipeline.parts(text, meta["format"])
    if not starts:
        return []
    edges = [0, *starts, len(text)]
    found = [(a, b, part_digest(text[a:b])) for a, b in zip(edges, edges[1:])]
    found[0] = (0, starts[0], part_digest(f"{meta.get('default_audience', '')}\x1f{text[:starts[0]]}"))
    return found


def keep_parts(db: sqlite3.Connection, doc_id: str, found: list[tuple[int, int, str]]) -> None:
    db.execute("DELETE FROM parts WHERE doc_id = ?", (doc_id,))
    db.executemany("INSERT INTO parts VALUES (?, ?, ?, ?, ?)", [(doc_id, k, *part) for k, part in enumerate(found)])


def cut_parts(db: sqlite3.Connection, run_id: int, doc_id: str, text: str, meta: dict,
              found: list[tuple[int, int, str]], unknown: list[str]) -> tuple[int, int] | None:
    """A changed plain text file brought up to date by cutting only the parts a change touched (T7.6): the units added
    and removed, or None, the manifest untouched, when the file is to be cut whole.

    The parts whose digests are the last cut's, counted from the front and from the back, keep their units, those after
    the change moved by its length in one statement; each part between is cut after the file's head, which gives it
    the units the whole cut gives it, and its units are compared with those the manifest held there. Whole instead: no
    parts now or at the last cut; the head changed, which every part's path and labels follow; nothing between; or a
    part between sharing a section with a part kept, since the ids of their identical units depend on each other.
    """
    kept = db.execute('SELECT start, "end", digest FROM parts WHERE doc_id = ? ORDER BY ordinal', (doc_id,)).fetchall()
    if not found or not kept or found[0][2] != kept[0][2]:
        return None
    front = 0
    while front < min(len(kept), len(found)) and kept[front][2] == found[front][2]:
        front += 1
    back = 0
    while back < min(len(kept), len(found)) - front and kept[-1 - back][2] == found[-1 - back][2]:
        back += 1
    old_lo = kept[front][0] if front < len(kept) else kept[-1][1]
    old_hi = kept[len(kept) - back][0] if back else kept[-1][1]
    new_hi = found[len(found) - back][0] if back else found[-1][1]
    if front == len(kept) == len(found):
        return None
    head = text[:found[0][1]]
    cut = []
    for start, end, _ in found[front:len(found) - back]:
        shift = start - len(head)
        cut += [replace(u, start=u.start + shift, end=u.end + shift)
                for u in pipeline.text_units(head + text[start:end], meta, unknown) if u.start >= len(head)]
    cut = unique_ids(cut)
    outside = {key for (key,) in db.execute(
        "SELECT DISTINCT section_key FROM units WHERE doc_id = ? AND (start < ? OR start >= ?)", (doc_id, old_lo, old_hi))}
    if any(u.section_key in outside for u in cut):
        return None
    old = {uid: (start, end) for uid, start, end in db.execute(
        'SELECT unit_id, start, "end" FROM units WHERE doc_id = ? AND start >= ? AND start < ?', (doc_id, old_lo, old_hi))}
    new = {u.unit_id: u for u in cut}
    removed = set(old) - set(new)
    db.executemany("DELETE FROM units WHERE unit_id = ?", [(uid,) for uid in removed])
    if new_hi != old_hi:  # the parts after the change, moved by its length, every unit in one statement
        db.execute('UPDATE units SET start = start + ?, "end" = "end" + ? WHERE doc_id = ? AND start >= ?',
                   (new_hi - old_hi, new_hi - old_hi, doc_id, old_hi))
    db.executemany('UPDATE units SET start = ?, "end" = ? WHERE unit_id = ?',
                   [(u.start, u.end, uid) for uid, u in new.items() if uid in old and old[uid] != (u.start, u.end)])
    added = [u for uid, u in new.items() if uid not in old]
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
    began = time.perf_counter()
    # One scan at a time, the next waiting its turn rather than meeting SQLite's lock and dying after 5 s (HS-F9).
    with exclusive((path or manifest_path()).parent, wait=SCAN_WAIT), collector_paused(), opened(path) as db:
        run_id = db.execute("INSERT INTO runs (kind, started) VALUES (?, ?)", (kind, now())).lastrowid
        summary = {"run": run_id, "kind": kind, "files": 0, "files_read": 0, "files_changed": 0, "files_removed": 0,
                   "units_added": 0, "units_removed": 0, "changes": {}, "changed_docs": []}
        known = {row[0]: row[1:] for row in db.execute(
            "SELECT doc_id, path, size, mtime_ns, sidecar_size, sidecar_mtime_ns, content_hash, versions FROM files")}
        unknown: list[str] = []
        found = corpus_files(full)

        def note(doc_id: str, change: str, added: int = 0, removed: int = 0) -> None:
            db.execute("INSERT INTO changes VALUES (?, ?, ?, ?, ?)", (run_id, doc_id, change, added, removed))
            summary["changes"][change] = summary["changes"].get(change, 0) + 1
            if added or removed:  # a document whose units the index must follow
                summary["changed_docs"].append(doc_id)

        for doc_id, (file, sidecar, meta) in sorted(found.items()):
            summary["files"] += 1
            rel = file.relative_to(full).as_posix()
            row = known.get(doc_id)
            # The stats the row records, taken once and before anything is read: a write landing after them moves
            # them, so the next scan reads the file again. Taken after the cut, they named a version the cut never
            # read, and the write was skipped by every scan after (OP-F6).
            stat, side = file.stat(), sidecar.stat()
            same_stats = row is not None and row[1:5] == (stat.st_size, stat.st_mtime_ns, side.st_size, side.st_mtime_ns)
            if not force and same_stats and row[6] == pipeline.VERSIONS:
                if row[0] != rel:
                    db.execute("UPDATE files SET path = ?, run_id = ? WHERE doc_id = ?", (rel, run_id, doc_id))
                    note(doc_id, "renamed")
                continue
            # The sidecar read again after its stat, since the listing read it before: an audience changed between the
            # two was cut under the older one and recorded under the newer stat (OP-F6). One rewritten as another
            # document meanwhile is left, its stats unrecorded, to the next scan's listing.
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("doc_id") != doc_id or file.name != f"{sidecar.name[:-len('.meta.json')]}.{meta.get('format')}":
                continue
            text = pipeline.load(file, meta)
            summary["files_read"] += 1
            digest = content_hash(text, meta)
            if not force and row is not None and row[5] == digest and row[6] == pipeline.VERSIONS:
                file_row(db, run_id, rel, stat, side, meta, digest)
                note(doc_id, "renamed" if row[0] != rel else "touched")
                continue
            # A file cut before at these versions is cut only where its parts changed; a build, a new file and any
            # other are cut whole, and either way the parts are kept for the next scan (T7.6).
            split = parts_of(text, meta)
            counted = None if force or row is None or row[6] != pipeline.VERSIONS else cut_parts(
                db, run_id, doc_id, text, meta, split, unknown)
            file_row(db, run_id, rel, stat, side, meta, digest)
            added, removed = counted or replace_units(db, run_id, doc_id, pipeline.text_units(text, meta, unknown))
            keep_parts(db, doc_id, split)
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
    summary["seconds"] = time.perf_counter() - began  # committed: what a watcher's change waits for (MA-F6)
    return summary


def build(full: Path | None = None, path: Path | None = None) -> dict:
    """Every file read and cut, whatever its timestamps say."""
    return scan(full, path, force=True)


def changed_since(run_id: int | None, path: Path | None = None) -> tuple[int, list[str]]:
    """The manifest's last run, and the documents whose units the runs after `run_id` added or removed, whichever
    process ran them: what a watcher syncs by, since a scan by hand or a build beside it moves the manifest and the
    watcher's own next scan then finds nothing (OP-F7). None asks for the last run alone. Read only."""
    path = path or manifest_path()
    if not path.is_file():
        return run_id or 0, []
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        last = db.execute("SELECT COALESCE(MAX(run_id), 0) FROM runs").fetchone()[0]
        if run_id is None or last <= run_id:
            return last, []
        return last, [doc_id for (doc_id,) in db.execute(
            "SELECT DISTINCT doc_id FROM changes WHERE run_id > ? AND run_id <= ? AND (units_added > 0 OR units_removed > 0) "
            "ORDER BY doc_id", (run_id, last))]
    finally:
        db.close()
