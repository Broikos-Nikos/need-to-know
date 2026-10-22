"""How long an audience change takes to reach queries through the watcher, the scan and the sync (T3.13, AC-F6).

    uv run python bench/revocation.py [--interval SECONDS] [--doc DOC_ID] [--runs 10]

Needs the corpus, the manifest, the vector cache, the embed extra and the index on the
running server (`ntk qdrant start`, `ntk index sync`). It picks the first prose section
of a plain text document whose audience is one department that MIXES extends, the
largest such document unless `--doc` names another, the slowest to scan while a scan cut
the whole file, as it does still for a change inside the part of the file that holds most
of it (T7.6); it starts the
watcher with the sync, waits for its catch up, and widens the section's audience to a
second label, as the simulator's relabel does; it times until a role of that second label
finds one of the section's units. Then it puts the file back, which narrows the audience
again, and times until that role finds none: the revocation. Every scan and sync between
an edit and the query that saw it is printed with its seconds (MA-F6). It widens and
narrows `--runs` times, ten by default, the objective holding over ten with the largest
deciding (T7.6), and prints each change's median, largest and
count, since one run is one draw of the watcher's poll, the scan and the machine's load
(RS-F13). The corpus is left as it was found, and the next sync after it leaves the index
as it was. It exits 1 when any run takes longer than the objective, 5 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from machine import Load
from watched import OBJECTIVE, Watched, past, spread, until

from needtoknow.corpus.mutate import corpus_files, encoded, read
from needtoknow.corpus.plan import LABEL_EN, MIXES
from needtoknow.index import server
from needtoknow.index.search import search
from needtoknow.ingest.manifest import opened, scan
from needtoknow.policy import load_policy
from needtoknow.settings import data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Time an audience change through the watcher into queries.")
    parser.add_argument("--interval", type=float, help="the watcher's interval; by default its own")
    parser.add_argument("--doc", help="the document to relabel; by default the largest that qualifies, "
                        "the slowest to scan")
    parser.add_argument("--runs", type=int, default=10, help="how many times to widen and narrow; the largest decides")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    full = data_dir() / "corpus" / "full"
    files = corpus_files(full)
    policy = load_policy()
    candidates = []
    for doc_id, (file, _sidecar, meta) in sorted(files.items()):
        label = meta["owner"]
        line = f"Audience: {LABEL_EN.get(label, '')}\n"
        if meta["format"] == "txt" and meta["language"] == "en" and label in MIXES and line in read(file, meta):
            candidates.append((doc_id, file, meta, label, MIXES[label], line))
    with opened() as db:
        size = dict(db.execute("SELECT doc_id, COUNT(*) FROM units GROUP BY doc_id"))
    pick = (next(c for c in candidates if c[0] == args.doc) if args.doc
            else max(candidates, key=lambda c: size.get(c[0], 0)))  # the largest, the slowest to scan
    doc_id, file, meta, label, added, line = pick
    reader = next(role for role, entry in policy["roles"].items() if added in entry["reads"] and label not in entry["reads"])
    scan(full)
    text = read(file, meta)
    heading = text[:text.index(line)].rstrip("\n").rsplit("\n", 1)[-1]  # the section the audience line opens
    with opened() as db:
        section = [t for t, path in db.execute("SELECT text, path FROM units WHERE doc_id = ? AND layer = 'prose'",
                                               (doc_id,)) if json.loads(path)[-1:] == [heading]][0]
    question, opening = " ".join(section.split()[:40]), " ".join(section.split()[:8])
    asker = server.client()

    def sees() -> bool:
        """Whether the reader gets the section back: one of its units, or a point the manifest no longer holds."""
        hits = search(asker, question, reader, policy, limit=20, rerank="none")  # whether, not in what order
        with opened() as db:
            rows = [db.execute("SELECT doc_id, text FROM units WHERE unit_id = ?", (h["unit_id"],)).fetchone() for h in hits]
        return any(row is None or (row[0] == doc_id and opening in " ".join(row[1].split())) for row in rows)

    assert not sees(), f"{reader} already sees {doc_id}'s section"
    original = file.read_bytes()
    grants, revokes = [], []
    cpu = Load()  # the other programs' load, read before anything of the bench runs (MA-F20)
    print(cpu.opening(), flush=True)
    watched = Watched(args.interval)
    print(f"{doc_id}, as {reader}; the watcher's catch up, a scan and a sync of every document: {watched.start():.1f} s",
          flush=True)
    try:
        widened = line.replace("\n", f"; {LABEL_EN[added]}\n")
        for run in range(1, args.runs + 1):
            began = time.perf_counter()
            file.write_bytes(encoded(meta, read(file, meta).replace(line, widened, 1)))
            grants.append(until(sees, 60))
            print(f"  run {run}, the widening: {watched.timeline(began, grants[-1])}", flush=True)
            began = time.perf_counter()
            file.write_bytes(original)
            revokes.append(until(lambda: not sees(), 60))
            print(f"  run {run}, the narrowing: {watched.timeline(began, revokes[-1])}", flush=True)
    finally:
        watched.stop()
        if file.read_bytes() != original:
            file.write_bytes(original)
    print(cpu.closing(), flush=True)
    print(f"{doc_id}: {label} widened to {label} and {added}, then narrowed back; as {reader}: the section found after "
          f"the edit, {spread(grants)}; gone after the narrowing, {spread(revokes)} (watch interval "
          f"{watched.interval:g} s; objective {OBJECTIVE:g} s)")
    if past(grants + revokes):
        print(f"past the {OBJECTIVE:g} s objective", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
