"""One file through every ingestion step: decode, parse, headings, access segments, units."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from needtoknow.corpus.plan import LABEL_EL, LABEL_EN
from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import PARSER_VERSION, parse
from needtoknow.ingest.segments import segments
from needtoknow.ingest.sentences import SPLITTER_VERSION
from needtoknow.ingest.structure import STRUCTURE_VERSION, structure
from needtoknow.ingest.units import CHUNKER_VERSION, Unit, units
from needtoknow.policy import UNLABELLED

# The files of every step that decides a unit's text, labels or path, this one included. Their bytes, with the
# label names the segments step reads, are part of the version a scan compares, so any change to them, a fix to
# the access rules above all, re-cuts every file at the next scan (DR-F7): a version bumped by hand is one that
# somebody forgets, and decode and segments had none.
STEP_FILES = ("decode.py", "parse.py", "structure.py", "segments.py", "sentences.py", "units.py", "pipeline.py")


def code_digest(root: Path | None = None, names: tuple = (LABEL_EN, LABEL_EL, UNLABELLED)) -> str:
    data = hashlib.sha256()
    for name in STEP_FILES:
        data.update((root or Path(__file__).parent).joinpath(name).read_bytes())
    data.update(json.dumps(names, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return data.hexdigest()[:16]


# A change to any part makes a file's units stale (T2.8).
VERSIONS = (f"parser {PARSER_VERSION}, structure {STRUCTURE_VERSION}, splitter {SPLITTER_VERSION}, "
            f"chunker {CHUNKER_VERSION}, steps {code_digest()}")


def document_units(file: Path, meta: dict, unknown: list[str] | None = None) -> tuple[str, list[Unit]]:
    """The decoded text and the units of one corpus file, as its sidecar describes it."""
    text = load(file, meta)
    elements = segments(structure(parse(text, meta["format"]), meta["format"]), meta, unknown)
    return text, units(elements, meta["doc_id"], meta["language"])
