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


def test_identical_units_in_one_section_keep_distinct_ids(corpus, tmp_path):
    from needtoknow.ingest.manifest import unique_ids
    from needtoknow.ingest.units import make_unit
    twice = [make_unit("DOC", "prose", "Same text here.", (("sales",), ("P",)), 0, 1, "en")] * 2
    ids = [u.unit_id for u in unique_ids(twice)]
    assert len(set(ids)) == 2 and ids[1] == ids[0] + "#1"


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
    # T3.16: in a watcher holding the encoder, full collections during a cut walk 368,000 objects: the largest file's
    # cut took 3.09 s with the collector on and 2.16 s with it off.
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
