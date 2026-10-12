"""Headings and heading paths (T2.3): every paragraph and table row learns the headings it sits under.

Markdown and html mark their headings, and their levels are taken as marked. Plain
text marks them only in its wording, so a line of a paragraph is a heading when it
is one of:

* a keyword with a number: SECTION 3, ANNEX B, ΕΝΟΤΗΤΑ 3, ΠΑΡΑΡΤΗΜΑ Β, level 2;
* a numbered line, "2" or "2.1" then a title that starts with a capital, as in
  "2.1 Who may approve": level 2 for one number, one more for each further number;
* a short line in capitals, level 2;

and in every case it is short and does not end like a sentence. The first line of a
plain text file is its title, level 1, as markdown's and html's first heading is.
A heading found inside a paragraph splits it, each part keeping its own offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from needtoknow.ingest.parse import Block

STRUCTURE_VERSION = 1
KEYWORD = re.compile(r"(?:SECTION|CHAPTER|PART|ANNEX|APPENDIX|ARTICLE|ΕΝΟΤΗΤΑ|ΚΕΦΑΛΑΙΟ|ΜΕΡΟΣ|ΠΑΡΑΡΤΗΜΑ|ΑΡΘΡΟ)\s+[\dA-ZΑ-Ω]{1,3}\b")
NUMBERED = re.compile(r"(\d{1,2}(?:\.\d{1,2})*)\.?\s+\S")


@dataclass(frozen=True)
class Element:
    kind: str                       # "heading", "paragraph" or "row"
    text: str
    start: int
    end: int
    level: int = 0
    path: tuple[str, ...] = ()      # the headings above it, outermost first; a heading's ends with itself
    cells: tuple[str, ...] = ()
    header: bool = False
    audience: str | None = None


def heading_level(line: str) -> int:
    """The level of a plain text line that reads as a heading, or 0."""
    line = line.strip()
    words = line.split()
    if not words or len(words) > 14 or line[-1] in ".;:,?!" or "|" in line:
        return 0
    if KEYWORD.match(line):
        return 2
    numbered = NUMBERED.match(line)
    if numbered and 1 < len(words) <= 10 and words[1][:1].isupper():
        return 1 + numbered.group(1).count(".") + 1
    letters = [c for c in line if c.isalpha()]
    if len(words) <= 8 and len(letters) >= 4 and all(c.isupper() for c in letters):
        return 2
    return 0


def split_txt(blocks: list[Block]) -> list[Element]:
    """Plain text blocks with their heading lines lifted out, the first line of the file as its title."""
    out: list[Element] = []
    for index, block in enumerate(blocks):
        if block.kind != "paragraph":
            out.append(Element(block.kind, block.text, block.start, block.end, cells=block.cells, header=block.header))
            continue
        pending: list[tuple[int, int]] = []

        def flush() -> None:
            if pending:
                start, end = pending[0][0], pending[-1][1]
                out.append(Element("paragraph", block.text[start - block.start:end - block.start], start, end))
                pending.clear()

        at = block.start
        for number, line in enumerate(block.text.split("\n")):
            level = 1 if index == 0 and number == 0 else heading_level(line)
            if level:
                flush()
                out.append(Element("heading", line, at, at + len(line), level=level))
            else:
                pending.append((at, at + len(line)))
            at += len(line) + 1
        flush()
    return out


def structure(blocks: list[Block], fmt: str) -> list[Element]:
    """The blocks as elements, each with its heading path."""
    if fmt == "txt":
        elements = split_txt(blocks)
    else:
        elements = [Element(b.kind, b.text, b.start, b.end, level=b.level, cells=b.cells, header=b.header, audience=b.audience)
                    for b in blocks]
    stack: list[tuple[int, str]] = []
    out = []
    for e in elements:
        if e.kind == "heading":
            while stack and stack[-1][0] >= e.level:
                stack.pop()
            stack.append((e.level, e.text))
        out.append(Element(e.kind, e.text, e.start, e.end, e.level, tuple(t for _, t in stack), e.cells, e.header, e.audience))
    return out
