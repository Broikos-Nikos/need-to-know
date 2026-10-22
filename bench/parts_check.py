"""Whether every plain text file of the corpus cuts the same by its parts as whole (T7.6): the file's head, then each
keyword section cut after the head, as a scan cuts the parts a change touched, against the units of the whole cut,
each unit's id, labels, path, text and offsets.

    uv run python bench/parts_check.py

Reads NTK_DATA_DIR/corpus/full by its sidecars and writes nothing. Prints how many plain text files have parts, how
many cut the same both ways, the largest part beside its file, since a change inside a file's largest part still
cuts all of that part, and the seconds of both ways. Exits 1 on any file that cuts otherwise.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace

from needtoknow.corpus.mutate import corpus_files
from needtoknow.ingest import pipeline
from needtoknow.ingest.decode import load
from needtoknow.settings import data_dir


def rows(units) -> list[tuple]:
    return [(u.unit_id, u.labels, u.path, u.text, u.start, u.end) for u in units]


def by_parts(text: str, meta: dict, starts: list[int]) -> list:
    head = text[:starts[0]]
    out = [u for u in pipeline.text_units(head, meta, []) if u.start < len(head)]
    for a, b in zip(starts, [*starts[1:], len(text)]):
        out += [replace(u, start=u.start + a - len(head), end=u.end + a - len(head))
                for u in pipeline.text_units(head + text[a:b], meta, []) if u.start >= len(head)]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    full = data_dir() / "corpus" / "full"
    if not full.is_dir():
        print("no corpus under NTK_DATA_DIR: ntk corpus assemble makes it")
        return 1
    plain = same = 0
    differ, largest = [], []
    seconds = {"whole": 0.0, "parts": 0.0}
    for doc_id, (file, _sidecar, meta) in sorted(corpus_files(full).items()):
        if meta["format"] != "txt":
            continue
        plain += 1
        text = load(file, meta)
        starts = pipeline.parts(text, meta["format"])
        if not starts:
            continue
        began = time.perf_counter()
        whole = pipeline.text_units(text, meta, [])
        seconds["whole"] += time.perf_counter() - began
        began = time.perf_counter()
        parts = by_parts(text, meta, starts)
        seconds["parts"] += time.perf_counter() - began
        largest.append((max(b - a for a, b in zip(starts, [*starts[1:], len(text)])), len(text), len(starts), doc_id))
        if rows(parts) == rows(whole):
            same += 1
        else:
            differ.append(doc_id)
    split = same + len(differ)
    print(f"plain text files {plain}, with keyword sections {split}: cut by their parts the same as whole {same}, "
          f"otherwise {len(differ)}{': ' + ', '.join(differ) if differ else ''}")
    for size, length, count, doc_id in sorted(largest, reverse=True)[:3]:
        print(f"  {doc_id}: {count} keyword sections; its largest part {size:,} of its {length:,} characters")
    print(f"cut whole in {seconds['whole']:.1f} s, by parts in {seconds['parts']:.1f} s")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
