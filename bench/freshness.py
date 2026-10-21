"""How long a one sentence edit takes to reach a query, through the watcher, the scan, the embedding and the sync, and
its undoing: the change bench of the freshness objective (T2.9, MA-F6).

    uv run python bench/freshness.py [--interval SECONDS] [--runs 5]

Needs the corpus, the manifest, the vector cache, the embed extra and the index on the
running server. It changes one planted fact's value in the largest plain text file that
plants a number, which changes one paragraph's unit, and times until a role that may read
it finds the new sentence by asking it; then it writes the file's original bytes back and
times until the old sentence is found again and the new one is not. The watcher runs in a
thread, started only once its catch up is done, with the encoder already loaded by the
first question, as a running watcher's is; every scan and sync between an edit and the
query that saw it is printed with its seconds, and the units those syncs embedded again
out of the collection's points, PLAN section 1's number 5. It edits and undoes `--runs`
times, five by default, and prints each change's median, largest and count (RS-F13). The
corpus is left as it was found. It exits 1 when any run takes longer than the objective,
5 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from machine import Load
from watched import OBJECTIVE, Watched, past, spread, until

from needtoknow.corpus.mutate import corpus_files, encoded, read, replace_value
from needtoknow.index import server
from needtoknow.index.search import search
from needtoknow.ingest.manifest import opened, scan
from needtoknow.policy import load_policy, readable
from needtoknow.settings import data_dir


def found(client, policy: dict, role: str, sentence: str) -> bool:
    """Whether the role, asking the sentence, gets a unit that holds it."""
    hits = search(client, sentence, role, policy, limit=20, rerank="none")
    with opened() as db:
        texts = [db.execute("SELECT text FROM units WHERE unit_id = ?", (h["unit_id"],)).fetchone() for h in hits]
    return any(row is not None and sentence in " ".join(row[0].split()) for row in texts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Time an edit through the watcher into queries.")
    parser.add_argument("--interval", type=float, help="the watcher's interval; by default its own")
    parser.add_argument("--runs", type=int, default=5, help="how many times to edit and undo; the largest decides")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    base = data_dir() / "corpus"
    full = base / "full"
    files = corpus_files(full)
    facts = json.loads((base / "plan" / "facts.json").read_text(encoding="utf-8"))
    # The largest file it can edit: a whole file's re-embedding would show there first (PLAN section 1, number 5).
    _, doc_id, fact = max(((files[d][0].stat().st_size, d, f) for f in facts if f["values"][0].isdigit()
                           for d in f["placements"] if d in files and files[d][2]["format"] == "txt"),
                          key=lambda c: (c[0], c[1]))
    file, _sidecar, meta = files[doc_id]
    old = fact[meta["language"]]
    new = replace_value(old, fact["values"][0], str(int(fact["values"][0]) + 1))
    scan(full)
    with opened() as db:
        labels = next(json.loads(row[0]) for row in db.execute(
            "SELECT labels FROM units WHERE doc_id = ? AND instr(text, ?) > 0", (doc_id, old)))
    policy = load_policy()
    role = next(r for r in policy["roles"] if r != "admin" and set(labels) & readable(policy, r))
    client = server.client()
    assert found(client, policy, role, old) and not found(client, policy, role, new), "the fact is not as planted"
    original = file.read_bytes()
    edits, undos, embedded = [], [], {"edit": [], "undoing": []}
    load = Load()  # the other programs' load, read before anything of the bench runs (MA-F20)
    print(load.opening(), flush=True)
    watched = Watched(args.interval)
    print(f"{doc_id}, {len(original) / 1e6:.1f} MB, as {role}; the watcher's catch up, a scan and a sync of every "
          f"document: {watched.start():.1f} s", flush=True)
    try:
        for run in range(1, args.runs + 1):
            began = time.perf_counter()
            file.write_bytes(encoded(meta, read(file, meta).replace(old, new, 1)))
            edits.append(until(lambda: found(client, policy, role, new), 60))
            print(f"  run {run}, the edit: {watched.timeline(began, edits[-1])}", flush=True)
            embedded["edit"].append(watched.embedded(began, edits[-1]))
            began = time.perf_counter()
            file.write_bytes(original)
            undos.append(until(lambda: found(client, policy, role, old) and not found(client, policy, role, new), 60))
            print(f"  run {run}, the undoing: {watched.timeline(began, undos[-1])}", flush=True)
            embedded["undoing"].append(watched.embedded(began, undos[-1]))
    finally:
        watched.stop()
        if file.read_bytes() != original:
            file.write_bytes(original)
    print(load.closing(), flush=True)
    print(f"{doc_id}: the edited sentence reached a query in {spread(edits)}; its undoing in {spread(undos)} "
          f"(watch interval {watched.interval:g} s; objective {OBJECTIVE:g} s)")
    # A passage the vector cache holds is not embedded again, so the first edit is the one that measures the change;
    # the undoings and the later edits bring back passages already embedded.
    points = next((p for _, p in reversed(embedded["edit"] + embedded["undoing"]) if p), None)
    print(f"{doc_id}: units embedded again, each edit {[n for n, _ in embedded['edit']]} and each undoing "
          f"{[n for n, _ in embedded['undoing']]}, out of {points} points in the collection")
    if past(edits + undos):
        print(f"past the {OBJECTIVE:g} s objective", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
