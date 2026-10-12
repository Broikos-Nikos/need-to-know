"""How long a one sentence edit takes to reach the manifest through the watcher, and its undoing (T2.9).

    uv run python bench/freshness.py

Needs an assembled corpus and a manifest (`ntk ingest build`). It brings the manifest up
to date, starts the watcher in a thread, changes one planted fact's value in a plain
text file of the corpus, and times how long until a unit holds the new sentence. It
then writes the file's original bytes back and times how long until the old sentence
is back, so the corpus is left as it was found.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from needtoknow.corpus.mutate import corpus_files, encoded, read, replace_value
from needtoknow.ingest.manifest import opened, scan
from needtoknow.ingest.watch import watch
from needtoknow.settings import data_dir


def holds(doc_id: str, sentence: str) -> bool:
    """Whether a unit of the document holds the sentence; one document's units, by the manifest's index."""
    with opened() as db:
        return db.execute("SELECT count(*) FROM units WHERE doc_id = ? AND instr(text, ?) > 0", (doc_id, sentence)).fetchone()[0] > 0


def until(condition, limit: float) -> float:
    began = time.perf_counter()
    while time.perf_counter() - began < limit:
        if condition():
            return time.perf_counter() - began
        time.sleep(0.05)
    raise TimeoutError(f"not seen within {limit} s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Time an edit through the watcher into the manifest.")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    base = data_dir() / "corpus"
    full = base / "full"
    files = corpus_files(full)
    facts = json.loads((base / "plan" / "facts.json").read_text(encoding="utf-8"))
    doc_id, fact = next((d, f) for f in facts if f["values"][0].isdigit() for d in f["placements"]
                        if d in files and files[d][2]["format"] == "txt")
    file, _sidecar, meta = files[doc_id]
    old = fact[meta["language"]]
    new = replace_value(old, fact["values"][0], str(int(fact["values"][0]) + 1))
    scan(full)
    original = file.read_bytes()
    stop = threading.Event()
    watcher = threading.Thread(target=watch, kwargs={"full": full, "interval": args.interval, "stop": stop.is_set}, daemon=True)
    watcher.start()
    try:
        time.sleep(args.interval)  # let the watcher take its first fingerprint
        file.write_bytes(encoded(meta, read(file, meta).replace(old, new, 1)))
        edit = until(lambda: holds(doc_id, new), 30)
        file.write_bytes(original)
        undo = until(lambda: holds(doc_id, old) and not holds(doc_id, new), 30)
    finally:
        stop.set()
        watcher.join(10)
        if file.read_bytes() != original:
            file.write_bytes(original)
    print(f"{doc_id}: the edited sentence reached the manifest in {edit:.1f} s, and its undoing in {undo:.1f} s "
          f"(watch interval {args.interval:g} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
