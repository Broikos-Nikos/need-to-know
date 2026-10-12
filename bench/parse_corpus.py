"""Decode and parse every file of the assembled corpus, and check that every block resolves back to its span
(T2.1, T2.2).

    uv run python bench/parse_corpus.py

Reads NTK_DATA_DIR/corpus/full by its sidecars, each file decoded as its sidecar
declares. Prints the files and megabytes read, the blocks by kind, the header rows,
the blocks whose span does not give their text back, the files decoding refused or
changed, and the seconds spent decoding and parsing.
"""

from __future__ import annotations

import html
import re
import sys
import time
from collections import Counter

from needtoknow.corpus.mutate import corpus_files, read
from needtoknow.ingest.decode import DecodeError, load
from needtoknow.ingest.parse import parse, resolve
from needtoknow.ingest.structure import structure
from needtoknow.settings import data_dir

SECTION_LINE = re.compile(r"(?m)^(?:## )?((?:SECTION|ΕΝΟΤΗΤΑ) \d+  .+)$")
SECTION_TAG = re.compile(r"<h2>([^<]+)</h2>")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    full = data_dir() / "corpus" / "full"
    found = corpus_files(full)
    if not found:
        print(f"no assembled corpus in {full}: run ntk corpus assemble first", file=sys.stderr)
        return 1
    kinds, headers, unresolved, size, seconds, rejected, normalised = Counter(), 0, [], 0, 0.0, [], 0
    levels, mismatched, title_only = Counter(), 0, 0
    for doc_id, (file, _sidecar, meta) in sorted(found.items()):
        size += file.stat().st_size
        began = time.perf_counter()
        try:
            text = load(file, meta)
        except DecodeError as error:
            rejected.append(str(error))
            continue
        blocks = parse(text, meta["format"])
        elements = structure(blocks, meta["format"])
        seconds += time.perf_counter() - began
        normalised += text != read(file, meta)
        kinds.update(b.kind for b in blocks)
        headers += sum(b.header for b in blocks)
        unresolved += [f"{doc_id} {b.kind} at {b.start}" for b in blocks if resolve(text, meta["format"], b) != b.text]
        # T2.3: the section headings found, by markup or by wording, against the section lines in the raw text.
        written = SECTION_TAG.findall(text) if meta["format"] == "html" else SECTION_LINE.findall(text)
        found_sections = [e.text for e in elements if e.kind == "heading" and e.level == 2]
        mismatched += found_sections != [html.unescape(s) for s in written]
        levels.update(f"{meta['format']} level {e.level}" for e in elements if e.kind == "heading")
        title_only += sum(e.kind != "heading" and len(e.path) < 2 for e in elements)
    print(f"files {len(found)}, {size / 2**20:.1f} MB; blocks {sum(kinds.values())}: "
          + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) + f"; header rows {headers}")
    print(f"blocks that do not resolve to their span: {len(unresolved)}" + (f", such as {unresolved[:3]}" if unresolved else ""))
    print(f"files refused by decoding: {len(rejected)}" + (f", such as {rejected[:2]}" if rejected else "")
          + f"; files that line ends or NFC changed: {normalised}")
    print("headings: " + ", ".join(f"{k} {n}" for k, n in sorted(levels.items()))
          + f"; files whose section headings differ from their section lines: {mismatched}"
          + f"; paragraphs and rows under the title alone: {title_only}")
    print(f"decoding, parsing and structure: {seconds:.1f} s, {size / 2**20 / seconds:.1f} MB/s")
    return 0 if not unresolved and not rejected and not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
