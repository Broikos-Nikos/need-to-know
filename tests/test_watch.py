"""The watcher (T2.9): one scan per change once it settles, none while nothing moves, one for a burst of writes,
and an edit visible in the manifest within a few intervals."""

import json
import os
import threading
import time

import pytest

from needtoknow.corpus import assemble
from needtoknow.corpus.plan import header
from needtoknow.ingest.manifest import build, opened
from needtoknow.ingest.watch import fingerprint, watch

PROSE = "The team checks every case against the procedure and records the outcome before it moves on. " * 3


def make_corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    doc = {"id": "RET_PRO_001", "type": "PRO", "type_name": "Procedure", "owner": "returns", "language": "en", "format": "txt",
           "target_mb": 0.01, "version": "1.0",
           "sections": [{"label": "returns", "topic_en": "Scope", "topic_el": "Πεδίο", "facts": [], "canary": None}]}
    text = "\n".join(header(doc, 1, doc["sections"][0]) + ["", PROSE, "", PROSE.replace("records", "notes"), ""])
    (base / "narrative" / "RET_PRO_001.txt").write_bytes(text.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps([doc]).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    assemble.assemble()
    build()
    return base / "full"


class Script:
    """A fake clock: each sleep runs the next step of the script, then the watcher looks."""

    def __init__(self, steps):
        self.steps, self.scans = list(steps), []

    def sleep(self, _interval):
        if self.steps:
            self.steps.pop(0)()

    def stop(self):
        return not self.steps

    def scan(self, full):
        self.scans.append(len(self.steps))
        return {}


def edit(full, old, new):
    def do():
        path = full / "RET_PRO_001.txt"
        path.write_bytes(path.read_bytes().replace(old, new, 1))
    return do


def test_no_scan_while_nothing_moves_and_one_per_settled_change(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    idle = Script([lambda: None] * 5)
    assert watch(full, catch_up=False, stop=idle.stop, sleep=idle.sleep, scan_fn=idle.scan) == 0
    once = Script([edit(full, b"notes", b"logs"), lambda: None, lambda: None, lambda: None])
    assert watch(full, catch_up=False, stop=once.stop, sleep=once.sleep, scan_fn=once.scan) == 1


def test_a_burst_of_writes_is_scanned_once_after_it_settles(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    # Each write changes the size: two writes in one clock tick with one size would look like one (see watch.py).
    burst = Script([edit(full, b"notes", b"logs"), edit(full, b"logs", b"files"), edit(full, b"files", b"record"),
                    lambda: None, lambda: None, lambda: None])
    assert watch(full, catch_up=False, stop=burst.stop, sleep=burst.sleep, scan_fn=burst.scan) == 1
    assert burst.scans == [2]  # after the third write and one quiet look, not during the burst


def test_the_fingerprint_sees_every_file_and_an_absent_folder_is_empty(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_bytes(b"1")
    assert {name for name, _size, _mtime in fingerprint(tmp_path)} == {str((tmp_path / "a" / "x.txt").relative_to(tmp_path))}
    assert fingerprint(tmp_path / "missing") == frozenset()


def test_an_edit_reaches_the_manifest_within_a_few_intervals(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    done = threading.Event()
    thread = threading.Thread(target=watch, kwargs={"full": full, "interval": 0.2, "stop": done.is_set}, daemon=True)
    thread.start()
    try:
        began = time.perf_counter()
        edit(full, b"notes the outcome", b"files the outcome")()
        while time.perf_counter() - began < 10:
            with opened() as db:
                if db.execute("SELECT count(*) FROM units WHERE text LIKE '%files the outcome%'").fetchone()[0]:
                    break
            time.sleep(0.05)
        elapsed = time.perf_counter() - began
    finally:
        done.set()
        thread.join(5)
    assert elapsed < 3, elapsed


def test_the_watcher_reports_a_refused_scan_and_keeps_watching(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    reports = []
    gone = Script([lambda: full.rename(full.with_name("full.retired")), lambda: None,
                   lambda: full.with_name("full.retired").rename(full), lambda: None, lambda: None])
    runs = watch(full, catch_up=False, stop=gone.stop, sleep=gone.sleep, report=reports.append)
    assert runs == 2 and "is not there" in reports[0]["error"] and reports[1]["files"] == 1


def later(path):
    """A touch the fingerprint sees whatever the clock: the file's mtime a second past its own. A touch at the clock's
    time moved nothing when the edit before it fell in the same tick, so the test failed by the machine (DR-F35)."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_a_scan_that_changed_a_unit_is_followed_by_a_sync_and_a_touch_is_not(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    synced = []
    steps = Script([edit(full, b"notes", b"logs"), lambda: None, lambda: None,
                    lambda: later(full / "RET_PRO_001.txt"), lambda: None, lambda: None])
    reports = []
    watch(full, catch_up=False, stop=steps.stop, sleep=steps.sleep, report=reports.append,
          sync_fn=lambda docs: synced.append(docs) or {"upserted": 1})
    assert synced == [["RET_PRO_001"]] and len([r for r in reports if "kind" in r]) == 2


def test_a_failed_sync_is_tried_again_every_interval_until_one_succeeds(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    tries = []

    def flaky(docs):
        tries.append(docs)
        if len(tries) < 3:
            raise ConnectionError("the server is down")
        return {"upserted": 1}

    steps = Script([edit(full, b"notes", b"logs")] + [lambda: None] * 5)
    reports = []
    watch(full, catch_up=False, stop=steps.stop, sleep=steps.sleep, report=reports.append, sync_fn=flaky)
    errors = [r["error"] for r in reports if "error" in r]
    assert tries == [["RET_PRO_001"]] * 3 and len(errors) == 2 and "ConnectionError: the server is down" in errors[0]
    assert reports[-1] == {"upserted": 1}


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
def test_a_narrowed_audience_stops_reaching_its_old_readers_through_the_watcher(monkeypatch, tmp_path):
    # AC-F6: the narrowing reached the manifest and the index went on serving the section.
    import numpy as np
    from qdrant_client import QdrantClient
    from test_embed_cache import Gpu

    from needtoknow.embed import model
    from needtoknow.index.collection import ensure
    from needtoknow.index.search import search
    from needtoknow.index.sync import sync
    from needtoknow.ingest.units import passage

    full = make_corpus(monkeypatch, tmp_path)
    monkeypatch.setattr(model, "tokenize", lambda texts, cut=False: [[0] + [len(w) + 3 for w in t.split()] + [2] for t in texts])
    monkeypatch.setattr(model, "states", Gpu().states)
    client = QdrantClient(location=":memory:")
    with opened() as db:
        passages = [passage(json.loads(p), t) for p, t in db.execute("SELECT path, text FROM units")]
    ensure(client, lambda: iter(passages))
    sync(client)
    question = np.ones(1024, dtype=np.float32) / 32

    def legal_sees_the_section():
        hits = search(client, "outcome", "legal", encode=lambda texts: np.vstack([question] * len(texts)), rerank="none")
        with opened() as db:
            texts = [db.execute("SELECT text FROM units WHERE unit_id = ?", (h["unit_id"],)).fetchone() for h in hits]
        return any(t is None or "records the outcome" in t[0] for t in texts)

    seen = []
    steps = Script([edit(full, b"Audience: Returns", b"Audience: Returns; Legal"), lambda: None,
                    lambda: seen.append(legal_sees_the_section()),
                    edit(full, b"Audience: Returns; Legal", b"Audience: Returns"), lambda: None,
                    lambda: seen.append(legal_sees_the_section())])
    watch(full, stop=steps.stop, sleep=steps.sleep, sync_fn=lambda docs: sync(client, doc_ids=docs))
    assert seen == [True, False]


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
def test_a_watcher_syncs_every_document_first_since_a_dead_one_may_have_left_one_unsynced(monkeypatch, tmp_path):
    # OP-F4: a narrowing scanned by a watcher that died before its sync stayed served after a restart.
    import numpy as np
    from qdrant_client import QdrantClient
    from test_embed_cache import Gpu

    from needtoknow.embed import model
    from needtoknow.index.collection import COLLECTION, ensure
    from needtoknow.index.search import search
    from needtoknow.index.sync import sync
    from needtoknow.ingest.manifest import scan
    from needtoknow.ingest.units import passage

    full = make_corpus(monkeypatch, tmp_path)
    monkeypatch.setattr(model, "tokenize", lambda texts, cut=False: [[0] + [len(w) + 3 for w in t.split()] + [2] for t in texts])
    monkeypatch.setattr(model, "states", Gpu().states)
    client = QdrantClient(location=":memory:")
    edit(full, b"Audience: Returns", b"Audience: Returns; Legal")()
    scan(full)
    with opened() as db:
        passages = [passage(json.loads(p), t) for p, t in db.execute("SELECT path, text FROM units")]
    ensure(client, lambda: iter(passages))
    sync(client)
    edit(full, b"Audience: Returns; Legal", b"Audience: Returns")()
    scan(full)  # the dead watcher's last act: the manifest narrowed, the index not
    question = np.ones(1024, dtype=np.float32) / 32

    def section_served_to_legal() -> bool:
        """Whether legal gets a unit of the narrowed section, or a point the manifest no longer holds."""
        found = search(client, "outcome", "legal", encode=lambda texts: np.vstack([question] * len(texts)), rerank="none")
        with opened() as db:
            texts = [db.execute("SELECT text FROM units WHERE unit_id = ?", (h["unit_id"],)).fetchone() for h in found]
        return any(t is None or "the outcome" in t[0] for t in texts)

    def stale_points() -> int:
        """Points the manifest no longer holds, the narrowed section's old ones among them."""
        records, _ = client.scroll(COLLECTION, limit=10_000, with_payload=["unit_id"])
        with opened() as db:
            held = {u for (u,) in db.execute("SELECT unit_id FROM units")}
        return sum(1 for record in records if record.payload["unit_id"] not in held)

    # DR-F15: the search keeps to the manifest, so the narrowing holds from the scan on, before any sync.
    assert stale_points() > 0 and not section_served_to_legal()
    steps = Script([lambda: None])
    watch(full, stop=steps.stop, sleep=steps.sleep, sync_fn=lambda docs: sync(client, doc_ids=docs))
    assert stale_points() == 0 and not section_served_to_legal()  # OP-F4: the catch up's sync deletes them


def test_a_watcher_scans_first_what_changed_while_none_ran(monkeypatch, tmp_path):
    # OP-F4: an edit made while no watcher ran was never scanned, until some other file changed.
    full = make_corpus(monkeypatch, tmp_path)
    edit(full, b"notes the outcome", b"files the outcome")()
    steps = Script([lambda: None])
    assert watch(full, stop=steps.stop, sleep=steps.sleep) == 1
    with opened() as db:
        assert db.execute("SELECT count(*) FROM units WHERE text LIKE '%files the outcome%'").fetchone()[0] == 1


def test_the_lines_the_watcher_prints_are_the_lines_the_change_benches_time(capsys, monkeypatch):
    # MA-F6: bench/watched.py times each scan and sync from the line ntk ingest watch prints when it ends.
    import sys

    from needtoknow.cli import report_run
    from needtoknow.settings import repo_root
    monkeypatch.syspath_prepend(str(repo_root() / "bench"))
    import watched
    report_run({"kind": "scan", "run": 7, "files": 160, "files_read": 1, "files_changed": 1, "files_removed": 0,
                "units_added": 1, "units_removed": 1, "units_total": 501031, "changes": {"changed": 1},
                "unknown_audiences": [], "seconds": 2.9}, sys.stdout)
    report_run({"deleted": 1, "embedded": 0, "upserted": 1, "points_after": 501031, "units": 17369, "seconds": 0.5,
                "parts": {"read": 0.35, "check": 0.1}}, sys.stdout)
    scan_line, sync_line = capsys.readouterr().out.splitlines()
    assert watched.SCAN.match(scan_line).groups() == ("1", "2.90")
    assert watched.SYNC.match(sync_line).groups() == ("0.50", "read 0.35, check 0.10")


def test_the_watcher_loads_the_encoder_before_it_watches_when_it_syncs_the_index(monkeypatch, tmp_path):
    # T3.16: the first edit after a watcher started paid the encoder's load, 10 to 25 s, before its embedding.
    import argparse
    import io

    from needtoknow import cli
    from needtoknow.embed import model
    from needtoknow.ingest import watch as watching
    calls = []
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(model, "load", lambda *a, **k: calls.append("load"))
    monkeypatch.setattr(watching, "watch", lambda *a, **k: calls.append("watch") or 0)
    cli.cmd_ingest(argparse.Namespace(action="watch", interval=0.5, manifest_only=False), io.StringIO())
    assert calls == ["load", "watch"]
    calls.clear()
    cli.cmd_ingest(argparse.Namespace(action="watch", interval=0.5, manifest_only=True), io.StringIO())
    assert calls == ["watch"]


def test_the_watcher_looks_every_half_second_unless_told_otherwise():
    # N53: a change waits two intervals before its scan, and a look at 320 files costs 9 ms.
    from needtoknow.cli import build_parser
    assert build_parser().parse_args(["ingest", "watch"]).interval == 0.5


def test_a_scan_that_fails_is_said_and_tried_again_and_the_watcher_lives(tmp_path):
    # HS-F9: a scan that met the manifest locked by a build ended the watcher, and with it every later change.
    import sqlite3
    (tmp_path / "a.txt").write_text("x")
    calls, reports, checks = [], [], {"n": 0}

    def scan_fn(full):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return {}

    def stop():
        checks["n"] += 1
        return checks["n"] > 3

    watch(tmp_path, interval=0, stop=stop, report=reports.append, sleep=lambda _seconds: None, scan_fn=scan_fn)
    assert len(calls) == 2
    assert reports[0]["error"] == "manifest not scanned, trying again: OperationalError: database is locked"
