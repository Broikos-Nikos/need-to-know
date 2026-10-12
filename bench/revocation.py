"""How long an audience change takes to reach queries through the watcher, the scan and the sync (T3.13, AC-F6).

    uv run python bench/revocation.py [--interval SECONDS] [--doc DOC_ID]

Needs the corpus, the manifest, the vector cache, the embed extra and the index on the
running server (`ntk qdrant start`, `ntk index sync`). It picks the first prose section
of a plain text document whose audience is one department that MIXES extends, the
largest such document unless `--doc` names another, since a change costs a scan of its
whole file; it starts the
watcher with the sync, waits for its catch up, and widens the section's audience to a
second label, as the simulator's relabel does; it times until a role of that second label
finds one of the section's units. Then it puts the file back, which narrows the audience
again, and times until that role finds none: the revocation. Every scan and sync between
an edit and the query that saw it is printed with its seconds (MA-F6). The corpus is left
as it was found, and the next sync after it leaves the index as it was. It exits 1 when
either change takes longer than the objective, 5 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from watched import OBJECTIVE, Watched, until

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
    watched = Watched(args.interval)
    print(f"{doc_id}, as {reader}; the watcher's catch up, a scan and a sync of every document: {watched.start():.1f} s",
          flush=True)
    try:
        widened = line.replace("\n", f"; {LABEL_EN[added]}\n")
        began = time.perf_counter()
        file.write_bytes(encoded(meta, read(file, meta).replace(line, widened, 1)))
        grant = until(sees, 60)
        print(f"  the widening: {watched.timeline(began, grant)}", flush=True)
        began = time.perf_counter()
        file.write_bytes(original)
        revoke = until(lambda: not sees(), 60)
        print(f"  the narrowing: {watched.timeline(began, revoke)}", flush=True)
    finally:
        watched.stop()
        if file.read_bytes() != original:
            file.write_bytes(original)
    print(f"{doc_id}: {label} widened to {label} and {added}, then narrowed back; as {reader}: the section was found "
          f"{grant:.1f} s after the edit, and gone {revoke:.1f} s after the narrowing (watch interval {watched.interval:g} s; "
          f"objective {OBJECTIVE:g} s)")
    if max(grant, revoke) > OBJECTIVE:
        print(f"past the {OBJECTIVE:g} s objective", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
