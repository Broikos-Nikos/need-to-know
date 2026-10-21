"""The manifest (T2.7): a build records every file and unit, a second build over the same corpus changes nothing,
an edit changes only its own units, and a deleted file takes its units with it."""

import json
import sqlite3

import pytest

from needtoknow.cli import main
from needtoknow.corpus import assemble
from needtoknow.corpus.plan import header
from needtoknow.ingest.manifest import ManifestError, build, manifest_path, opened
from needtoknow.ingest.units import units

PROSE = {"en": "The team checks every case against the procedure and records the outcome before it moves on. " * 3,
         "el": "Η ομάδα ελέγχει κάθε υπόθεση με βάση τη διαδικασία και καταγράφει το αποτέλεσμα πριν προχωρήσει. " * 3}
SHAPES = [("sales", "en", "txt"), ("logistics", "el", "txt"), ("returns", "en", "md"), ("legal", "el", "md"),
          ("finance", "en", "html"), ("hr", "el", "html")]


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    docs = []
    for n, (owner, lang, fmt) in enumerate(SHAPES, 1):
        doc = {"id": f"{owner[:3].upper()}_PRO_{n:03d}", "type": "PRO", "type_name": "Procedure", "owner": owner,
               "language": lang, "format": fmt, "target_mb": 0.02, "version": "1.2",
               "sections": [{"label": owner, "topic_en": "Scope", "topic_el": "Πεδίο", "facts": [], "canary": None}]}
        docs.append(doc)
        text = "\n".join(header(doc, 1, doc["sections"][0]) + ["", PROSE[lang], "", PROSE[lang].replace("records", "notes"), ""])
        (base / "narrative" / f"{doc['id']}.txt").write_bytes(text.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps(docs, ensure_ascii=False).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    assert assemble.assemble()["written"] == len(docs)
    return base, docs


def unit_rows(db):
    return dict(db.execute("SELECT unit_id, run_id FROM units").fetchall())


def test_a_second_build_over_the_same_corpus_changes_nothing(corpus):
    first = build()
    assert first["files"] == 6 and first["files_changed"] == 6 and first["units_added"] == first["units_total"] > 0
    with opened() as db:
        before = unit_rows(db)
    second = build()
    assert (second["files_changed"], second["units_added"], second["units_removed"]) == (0, 0, 0)
    assert second["units_total"] == first["units_total"]
    with opened() as db:
        assert unit_rows(db) == before  # same ids, and none rewritten by the second run
        assert [r[0] for r in db.execute("SELECT kind FROM runs ORDER BY run_id")] == ["build", "build"]


def test_an_edit_changes_only_its_own_units(corpus):
    base, docs = corpus
    build()
    target = base / "full" / f"{docs[0]['id']}.txt"
    target.write_bytes(target.read_bytes().replace(b"notes the outcome", b"logs the outcome", 1))
    summary = build()
    assert summary["files_changed"] == 1 and summary["units_added"] == summary["units_removed"] == 1
    with opened() as db:
        changed = {doc for (doc,) in db.execute("SELECT doc_id FROM units WHERE run_id = ?", (summary["run"],))}
    assert changed == {docs[0]["id"]}


def test_a_deleted_file_takes_its_units_with_it(corpus):
    base, docs = corpus
    first = build()
    with opened() as db:
        gone = db.execute("SELECT count(*) FROM units WHERE doc_id = ?", (docs[5]["id"],)).fetchone()[0]
    for path in (base / "full").glob(f"{docs[5]['id']}.*"):
        path.unlink()
    summary = build()
    assert summary["files_removed"] == 1 and summary["units_removed"] == gone > 0
    assert summary["units_total"] == first["units_total"] - gone


def test_the_counts_the_feed_and_a_view_read_follow_every_unit_written_and_removed(corpus):
    # AC-F18, AC-F19: the change feed and a document's view read counts kept under each label as the units are written,
    # so neither reads a unit of a label its role does not read. After a build, an edit and a deleted file, and in a
    # manifest made before the counts, they are what the units themselves give; and no update moves a unit out of them.
    base, docs = corpus

    def kept_and_recounted() -> tuple[list, list]:
        with opened() as db:
            kept = [sorted(db.execute(f"SELECT * FROM {table}")) for table in ("added", "segments")]
            recounted = [sorted(db.execute(f"SELECT {columns}, COUNT(*) FROM units u, json_each(u.labels) j GROUP BY "
                                           f"{columns}")) for columns in ("j.value, u.run_id, u.doc_id, u.labels",
                                                                          "u.doc_id, j.value, u.section_key, u.labels")]
        return kept, recounted
    build()
    target = base / "full" / f"{docs[0]['id']}.txt"
    target.write_bytes(target.read_bytes().replace(b"notes the outcome", b"logs the outcome", 1))
    for path in (base / "full").glob(f"{docs[5]['id']}.*"):
        path.unlink()
    assert build()["units_removed"] > 1
    kept, recounted = kept_and_recounted()
    assert kept == recounted and all(kept), kept
    with opened() as db:  # a manifest as it was before the counts
        db.executescript("DROP TRIGGER unit_kept; DROP TRIGGER unit_written; DROP TRIGGER unit_removed; "
                         "DROP TABLE added; DROP TABLE segments; DROP TABLE made;")
    kept, recounted = kept_and_recounted()
    assert kept == recounted and all(kept), kept
    with opened() as db, pytest.raises(sqlite3.IntegrityError, match="a change is a new unit"):
        db.execute("UPDATE units SET labels = '[\"general\"]' WHERE rowid = 1")


def test_a_change_to_the_counts_reaches_a_manifest_made_before_it(corpus, monkeypatch):
    # MT-F29: the counts were made once, keyed by a trigger's name, every statement IF NOT EXISTS, so a fix to a trigger
    # in the code never reached a manifest made before it, and nothing said so. The manifest keeps the digest of the
    # text that made its counts, and one that is not the code's is made again, recounted from the units.
    from needtoknow.ingest import manifest
    build()
    mended = manifest.SUMMARIES.replace("UPDATE added SET units = units - 1", "UPDATE added SET units = MAX(units - 1, 0)")
    assert mended != manifest.SUMMARIES
    monkeypatch.setattr(manifest, "SUMMARIES", mended)
    monkeypatch.setattr(manifest, "COUNTS", manifest.hashlib.sha256(mended.encode("utf-8")).hexdigest()[:16])
    with opened() as db:
        trigger = db.execute("SELECT sql FROM sqlite_master WHERE name = 'unit_removed'").fetchone()[0]
        made = db.execute("SELECT digest FROM made WHERE name = 'counts'").fetchone()[0]
        kept = sorted(db.execute("SELECT * FROM added"))
        recounted = sorted(db.execute("SELECT j.value, u.run_id, u.doc_id, u.labels, COUNT(*) FROM units u, "
                                      "json_each(u.labels) j GROUP BY j.value, u.run_id, u.doc_id, u.labels"))
    assert "MAX(units - 1, 0)" in trigger and made == manifest.COUNTS and kept == recounted and kept, (trigger, made)


def test_identical_units_in_one_section_keep_distinct_ids(corpus, tmp_path):
    from needtoknow.ingest.manifest import unique_ids
    from needtoknow.ingest.units import make_unit
    twice = [make_unit("DOC", "prose", "Same text here.", (("sales",), ("P",)), 0, 1, "en")] * 2
    ids = [u.unit_id for u in unique_ids(twice)]
    assert len(set(ids)) == 2 and ids[1] == ids[0] + "#1"


def test_units_are_found_by_document_and_run_through_one_index_in_old_manifests_too(corpus):
    # PF-F11: the change feed read every unit of a changed document to count the few its run added; units are now
    # indexed by document and run, an index that takes units_by_doc's place in a manifest made before it.
    def indexes() -> set[str]:
        with opened() as db:
            return {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND "
                                                   "tbl_name = 'units' AND name NOT LIKE 'sqlite_%'")}
    assert indexes() == {"units_by_doc_run", "units_by_segment"}
    with opened() as db:  # a manifest as it was before the index
        db.execute("DROP INDEX units_by_doc_run")
        db.execute("CREATE INDEX units_by_doc ON units(doc_id)")
    assert indexes() == {"units_by_doc_run", "units_by_segment"}
    with opened() as db:
        plan = " ".join(str(row) for row in db.execute(
            "EXPLAIN QUERY PLAN SELECT labels, COUNT(*) FROM units WHERE doc_id = ? AND run_id = ? GROUP BY labels",
            ("DOC", 1)))
    assert "units_by_doc_run (doc_id=? AND run_id=?)" in plan, plan


def test_a_manifest_of_another_schema_version_is_refused(corpus):
    build()
    db = sqlite3.connect(manifest_path())
    db.execute("PRAGMA user_version = 99")
    db.close()
    with pytest.raises(ManifestError, match="schema version 99"):
        build()


def test_the_ingest_command_prints_its_run(corpus, capsys):
    assert main(["ingest", "build"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("build run 1: files 6, read 6, changed 6, removed 0; units added ") and "manifest.db" in out
    head, _, seconds = out.rstrip().rpartition("; in ")
    assert head.endswith("changes by kind: added 6") and seconds.endswith(" s") and float(seconds[:-2]) >= 0


def test_a_scan_cuts_with_the_collector_paused_and_turns_it_back_on(corpus, monkeypatch):
    # T3.16: in a watcher holding the encoder, every full collection during a cut walks every object it holds: the
    # largest file's cut took 3.09 s with the collector on and 2.16 s with it off.
    import gc

    from needtoknow.ingest import pipeline
    seen = []
    real = pipeline.document_units
    monkeypatch.setattr(pipeline, "document_units", lambda *a, **k: seen.append(gc.isenabled()) or real(*a, **k))
    assert gc.isenabled()
    build()
    assert seen and not any(seen) and gc.isenabled()


def test_a_scan_waits_for_the_one_holding_the_manifest_rather_than_dying(corpus):
    # HS-F9: two scans met in SQLite's lock and the second died after its 5 s; now they take turns on the lock.
    import threading
    import time

    from needtoknow.corpus import exclusive
    held, release = threading.Event(), threading.Event()

    def hold():
        with exclusive(manifest_path().parent):
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(10)
    threading.Timer(3.0, release.set).start()  # three times what this corpus's build takes, so a wait shows
    began = time.perf_counter()
    summary = build()
    waited = time.perf_counter() - began
    holder.join()
    assert waited >= 2.9 and summary["files"] == 6
