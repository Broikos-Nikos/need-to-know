"""Parse every file of the assembled corpus and check that every block resolves back to its span (T2.1).

    uv run python bench/parse_corpus.py

Reads NTK_DATA_DIR/corpus/full by its sidecars, each file in its declared encoding.
Prints the files and megabytes read, the blocks by kind, the header rows, the blocks
whose span does not give their text back, and the seconds spent parsing alone.
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from needtoknow.corpus.mutate import corpus_files, read
from needtoknow.ingest.parse import parse, resolve
from needtoknow.settings import data_dir


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    full = data_dir() / "corpus" / "full"
    found = corpus_files(full)
    if not found:
        print(f"no assembled corpus in {full}: run ntk corpus assemble first", file=sys.stderr)
        return 1
    kinds, headers, unresolved, size, seconds = Counter(), 0, [], 0, 0.0
    for doc_id, (file, _sidecar, meta) in sorted(found.items()):
        text = read(file, meta)
        size += file.stat().st_size
        began = time.perf_counter()
        blocks = parse(text, meta["format"])
        seconds += time.perf_counter() - began
        kinds.update(b.kind for b in blocks)
        headers += sum(b.header for b in blocks)
        unresolved += [f"{doc_id} {b.kind} at {b.start}" for b in blocks if resolve(text, meta["format"], b) != b.text]
    print(f"files {len(found)}, {size / 2**20:.1f} MB; blocks {sum(kinds.values())}: "
          + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) + f"; header rows {headers}")
    print(f"blocks that do not resolve to their span: {len(unresolved)}" + (f", such as {unresolved[:3]}" if unresolved else ""))
    print(f"parsing alone: {seconds:.1f} s, {size / 2**20 / seconds:.1f} MB/s")
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
