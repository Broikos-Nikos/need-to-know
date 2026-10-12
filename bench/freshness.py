"""How long a one sentence edit takes to reach a query, through the watcher, the scan, the embedding and the sync, and
its undoing: the change bench of the freshness objective (T2.9, MA-F6).

    uv run python bench/freshness.py [--interval SECONDS]

Needs the corpus, the manifest, the vector cache, the embed extra and the index on the
running server. It changes one planted fact's value in a plain text file, which changes
one paragraph's unit, and times until a role that may read it finds the new sentence by
asking it; then it writes the file's original bytes back and times until the old
sentence is found again and the new one is not. The watcher runs in a thread, started
only once its catch up is done, with the encoder already loaded by the first question,
as a running watcher's is; every scan and sync between an edit and the query that saw
it is printed with its seconds. The corpus is left as it was found. It exits 1 when
either change takes longer than the objective, 5 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from watched import OBJECTIVE, Watched, until

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
    with opened() as db:
        labels = next(json.loads(row[0]) for row in db.execute(
            "SELECT labels FROM units WHERE doc_id = ? AND instr(text, ?) > 0", (doc_id, old)))
    policy = load_policy()
    role = next(r for r in policy["roles"] if r != "admin" and set(labels) & readable(policy, r))
    client = server.client()
    assert found(client, policy, role, old) and not found(client, policy, role, new), "the fact is not as planted"
    original = file.read_bytes()
    watched = Watched(args.interval)
    print(f"{doc_id}, as {role}; the watcher's catch up, a scan and a sync of every document: {watched.start():.1f} s",
          flush=True)
    try:
        began = time.perf_counter()
        file.write_bytes(encoded(meta, read(file, meta).replace(old, new, 1)))
        edit = until(lambda: found(client, policy, role, new), 60)
        print(f"  the edit: {watched.timeline(began, edit)}", flush=True)
        began = time.perf_counter()
        file.write_bytes(original)
        undo = until(lambda: found(client, policy, role, old) and not found(client, policy, role, new), 60)
        print(f"  the undoing: {watched.timeline(began, undo)}", flush=True)
    finally:
        watched.stop()
        if file.read_bytes() != original:
            file.write_bytes(original)
    print(f"{doc_id}: the edited sentence reached a query in {edit:.1f} s, and its undoing in {undo:.1f} s "
          f"(watch interval {watched.interval:g} s; objective {OBJECTIVE:g} s)")
    if max(edit, undo) > OBJECTIVE:
        print(f"past the {OBJECTIVE:g} s objective", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
