"""Ingestion step by step over the assembled corpus (PF-F1, PF-F3): where the seconds go, and how much memory
cutting one file takes.

    uv run python bench/ingest_steps.py

Reads NTK_DATA_DIR/corpus/full by its sidecars and runs every file through the steps
`pipeline.document_units` runs, decode, parse, structure, segments and units, in one
process on one thread, timing each apart: wall seconds, and the process's CPU seconds,
which other work on the machine slows less. Prints each step's seconds and share, the
elements and units made, and a digest of every unit's id, labels, path, text and
offsets, which a change meant only to make the steps faster must leave as it was. Then
cuts the largest file once more under tracemalloc and prints its peak traced memory
against its size.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import tracemalloc
from collections import Counter

from needtoknow.corpus.mutate import corpus_files
from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import parse
from needtoknow.ingest.pipeline import document_units
from needtoknow.ingest.segments import segments
from needtoknow.ingest.structure import structure
from needtoknow.ingest.units import units
from needtoknow.settings import data_dir

STEPS = ("decode", "parse", "structure", "segments", "units")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    full = data_dir() / "corpus" / "full"
    found = corpus_files(full)
    if not found:
        print(f"no assembled corpus in {full}: run ntk corpus assemble first", file=sys.stderr)
        return 1
    wall, cpu = dict.fromkeys(STEPS, 0.0), dict.fromkeys(STEPS, 0.0)
    kinds, made, size, digest = Counter(), 0, 0, hashlib.sha256()
    for _doc_id, (file, _sidecar, meta) in sorted(found.items()):
        size += file.stat().st_size
        marks = [(time.perf_counter(), time.process_time())]
        text = load(file, meta)
        marks.append((time.perf_counter(), time.process_time()))
        blocks = parse(text, meta["format"])
        marks.append((time.perf_counter(), time.process_time()))
        elements = structure(blocks, meta["format"])
        marks.append((time.perf_counter(), time.process_time()))
        segmented = segments(elements, meta, [])
        marks.append((time.perf_counter(), time.process_time()))
        cut = units(segmented, meta["doc_id"], meta["language"])
        marks.append((time.perf_counter(), time.process_time()))
        for step, before, after in zip(STEPS, marks, marks[1:]):
            wall[step] += after[0] - before[0]
            cpu[step] += after[1] - before[1]
        kinds.update(e.kind for e in segmented)
        made += len(cut)
        for unit in cut:
            digest.update(json.dumps([unit.unit_id, unit.labels, unit.path, unit.text, unit.start, unit.end],
                                     ensure_ascii=False).encode("utf-8"))
        del text, blocks, elements, segmented, cut
    total_wall, total_cpu = sum(wall.values()), sum(cpu.values())
    print(f"files {len(found)}, {size / 2**20:.1f} MB; elements {sum(kinds.values())}: "
          + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) + f"; units {made}")
    print(f"all steps: wall {total_wall:.1f} s, cpu {total_cpu:.1f} s")
    for step in STEPS:
        print(f"  {step:<10} wall {wall[step]:6.1f} s ({wall[step] / total_wall:4.0%}), "
              f"cpu {cpu[step]:6.1f} s ({cpu[step] / total_cpu:4.0%})")
    print(f"units digest {digest.hexdigest()[:16]}")

    doc_id, (file, _sidecar, meta) = max(found.items(), key=lambda item: item[1][0].stat().st_size)
    tracemalloc.start()
    document_units(file, meta)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    mb = file.stat().st_size / 2**20
    print(f"largest file {doc_id}, {mb:.2f} MB: peak traced memory {peak / 2**20:.0f} MB, {peak / 2**20 / mb:.0f} times "
          f"its size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
