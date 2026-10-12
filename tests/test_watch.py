"""The watcher (T2.9): one scan per change once it settles, none while nothing moves, one for a burst of writes,
and an edit visible in the manifest within a few intervals."""

import json
import threading
import time

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
    assert watch(full, stop=idle.stop, sleep=idle.sleep, scan_fn=idle.scan) == 0
    once = Script([edit(full, b"notes", b"logs"), lambda: None, lambda: None, lambda: None])
    assert watch(full, stop=once.stop, sleep=once.sleep, scan_fn=once.scan) == 1


def test_a_burst_of_writes_is_scanned_once_after_it_settles(monkeypatch, tmp_path):
    full = make_corpus(monkeypatch, tmp_path)
    # Each write changes the size: two writes in one clock tick with one size would look like one (see watch.py).
    burst = Script([edit(full, b"notes", b"logs"), edit(full, b"logs", b"files"), edit(full, b"files", b"record"),
                    lambda: None, lambda: None, lambda: None])
    assert watch(full, stop=burst.stop, sleep=burst.sleep, scan_fn=burst.scan) == 1
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
