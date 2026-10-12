"""One file through every ingestion step: decode, parse, headings, access segments, units."""

from __future__ import annotations

from pathlib import Path

from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import PARSER_VERSION, parse
from needtoknow.ingest.segments import segments
from needtoknow.ingest.sentences import SPLITTER_VERSION
from needtoknow.ingest.structure import STRUCTURE_VERSION, structure
from needtoknow.ingest.units import CHUNKER_VERSION, Unit, units

# Every step's version in one string: a change to any of them makes a file's units stale (T2.8).
VERSIONS = f"parser {PARSER_VERSION}, structure {STRUCTURE_VERSION}, splitter {SPLITTER_VERSION}, chunker {CHUNKER_VERSION}"


def document_units(file: Path, meta: dict, unknown: list[str] | None = None) -> tuple[str, list[Unit]]:
    """The decoded text and the units of one corpus file, as its sidecar describes it."""
    text = load(file, meta)
    elements = segments(structure(parse(text, meta["format"]), meta["format"]), meta, unknown)
    return text, units(elements, meta["doc_id"], meta["language"])
