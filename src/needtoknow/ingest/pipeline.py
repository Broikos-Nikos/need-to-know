"""One file through every ingestion step: decode, parse, headings, access segments, units.

A plain text file is also known by its parts: its head, the title and what comes before
its first keyword section, then each keyword section, from a heading such as SECTION 3
or ΕΝΟΤΗΤΑ 3 at the head of a run of lines to the next. Such a heading puts back the
title's path and the title's audience, and no run of lines crosses it, so a part cut
alone after the file's head gives the units the whole cut gives it, which lets a scan
cut only the parts a change touched (T7.6).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from needtoknow.corpus.plan import LABEL_EL, LABEL_EN
from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import PARSER_VERSION, parse
from needtoknow.ingest.segments import segments
from needtoknow.ingest.sentences import SPLITTER_VERSION
from needtoknow.ingest.structure import KEYWORD, STRUCTURE_VERSION, heading_level, structure
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


def text_units(text: str, meta: dict, unknown: list[str] | None = None) -> list[Unit]:
    """The units of a file's decoded text, as its sidecar describes it."""
    elements = segments(structure(parse(text, meta["format"]), meta["format"]), meta, unknown)
    return units(elements, meta["doc_id"], meta["language"])


def document_units(file: Path, meta: dict, unknown: list[str] | None = None) -> tuple[str, list[Unit]]:
    """The decoded text and the units of one corpus file, as its sidecar describes it."""
    text = load(file, meta)
    return text, text_units(text, meta, unknown)


# A line that may open a keyword section, found at the regular expression engine's speed; each is then held to the
# heading rules the structure step reads it by.
KEYWORD_LINE = re.compile(r"^[ \t]*" + KEYWORD.pattern, re.M)
INK = re.compile(r"\S")


def parts(text: str, fmt: str) -> list[int]:
    """Where a plain text file's keyword sections begin, each at a line that the structure step makes a heading of
    level 2 by its keyword, after a blank line and after the file's first run, its title's. Empty in markdown and html,
    whose sections this does not split."""
    if fmt != "txt":
        return []
    first = INK.search(text)
    if first is None:
        return []
    title = text.rfind("\n", 0, first.start()) + 1
    found = []
    for match in KEYWORD_LINE.finditer(text):
        start = match.start()
        if start <= title:
            continue
        end = text.find("\n", start)
        line = text[start:end if end >= 0 else len(text)]
        if heading_level(line) != 2 or not KEYWORD.match(line.strip()):
            continue
        before = text.rfind("\n", 0, start - 1) + 1
        if not INK.search(text, before, start - 1):
            found.append(start)
    return found
