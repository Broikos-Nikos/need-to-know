"""The build's deterministic parts, each as one digest, for a rebuild to compare with (T7.7): the plan and the
assembled corpus, each file's path and bytes in order of its path, and the manifest's units, each with its document,
section, layer, labels, heading path, text, offsets and language, in order of its id.

    uv run python bench/corpus_digest.py

It reads the corpus and the manifest under `NTK_DATA_DIR`, and prints a line for each; a rebuild from this repository
prints the same three lines. The vectors are left out: the GPU need not give an embedding the same last bits twice,
and the index's layout is the engine's. The scan that wrote each unit is left out too: it numbers the build's history,
not its content.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

from needtoknow.ingest.manifest import manifest_path
from needtoknow.settings import data_dir

UNITS = 'SELECT unit_id, doc_id, section_key, layer, labels, path, text, start, "end", language FROM units ORDER BY unit_id'


def files(folder: Path) -> tuple[int, int, str]:
    """How many files, how many bytes, and one digest of each one's path and bytes, in order of path."""
    digest, count, size = hashlib.sha256(), 0, 0
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "writer.lock"):
        data = path.read_bytes()
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0" + len(data).to_bytes(8, "big") + data)
        count, size = count + 1, size + len(data)
    return count, size, digest.hexdigest()


def units(path: Path) -> tuple[int, str]:
    digest, count = hashlib.sha256(), 0
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        for row in db.execute(UNITS):
            digest.update(repr(row).encode("utf-8") + b"\n")
            count += 1
    return count, digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = data_dir() / "corpus"
    for name in ("plan", "full"):
        if not (corpus / name).is_dir():
            print(f"no {name} under NTK_DATA_DIR/corpus: ntk corpus {'plan' if name == 'plan' else 'assemble'} makes it")
            return 1
        count, size, digest = files(corpus / name)
        print(f"corpus/{name}: {count:,} files, {size:,} bytes, sha256 {digest}")
    if manifest_path().is_file():
        count, digest = units(manifest_path())
        print(f"manifest: {count:,} units, sha256 {digest}")
    else:
        print("no manifest: ntk ingest build makes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
