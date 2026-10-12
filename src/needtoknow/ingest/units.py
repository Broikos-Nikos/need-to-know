"""Units, the pieces that get embedded and searched (T2.6).

A unit never crosses a segment: it is cut from a run of elements that share their
labels and their heading path. Prose: one paragraph is one unit; a paragraph over
MAX_TOKENS is split at sentence ends into pieces packed up to MAX_TOKENS, and a piece
under MIN_TOKENS joins the next piece of its run, or the one before when it is last
(KB1-F10). Tables: data rows are grouped under their header row, repeated at the head
of every group; a group closes at a row whose text hashes to a boundary, on average
every ROWS_PER_UNIT rows, so inserting a row reshapes one or two groups rather than all
that follow; and no group passes MAX_ROWS rows or MAX_TOKENS estimated tokens.
Headings, audience and classification lines are never units: the heading path, already
cut to what the unit's readers may read (AC-F4), and the labels travel with the unit
instead.

Tokens are estimated from characters, never counted by the model's tokenizer, so a
unit's boundaries are the same with or without the embed extra. The estimate is close
for prose and low for tables: `bench/embed_throughput.py` finds prose units over it by
at most 3 tokens, while table text, dense with codes, digits and dates, runs to 1.80
characters a token in Greek and 2.08 in English, so a table unit passes its estimate by
up to 208 tokens and the longest passage is 486. Nothing is cut for it: the embedder
refuses a passage over its max_length rather than truncating it (T3.1), and the bench
prints the longest passage against that limit.

Ids are content addressed (architecture section 3): a section key from the document
and its heading path, and a unit id from the document, the section key, the text, the
labels and CHUNKER_VERSION. A replay gives the same ids, and a change elsewhere leaves
a unit's id alone.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from needtoknow.ingest.sentences import sentence_spans
from needtoknow.ingest.structure import Element

CHUNKER_VERSION = 1
MAX_TOKENS = 256
MIN_TOKENS = 30
ROWS_PER_UNIT = 12
MAX_ROWS = 24
CHARS_PER_TOKEN = {"en": 4.0, "el": 3.2}
SEP = "\x1f"


def estimate_tokens(text: str, language: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN.get(language, 3.2))


def digest(*parts: str, size: int = 32) -> str:
    return hashlib.sha256(SEP.join(parts).encode("utf-8")).hexdigest()[:size]


@dataclass(frozen=True, slots=True)
class Unit:
    unit_id: str
    doc_id: str
    section_key: str
    layer: str                  # "prose" or "table"
    text: str
    labels: tuple[str, ...]
    path: tuple[str, ...]
    start: int                  # offsets of the elements it came from, in the decoded text
    end: int
    language: str


def passage(path, text: str) -> str:
    """What a unit is searched as, by both encoders: its heading path on a line above its text."""
    return " > ".join(path) + "\n" + text if path else text


def section_key(doc_id: str, path: tuple[str, ...]) -> str:
    return digest(doc_id, *path, size=16)


def make_unit(doc_id: str, layer: str, text: str, run: tuple[tuple[str, ...], tuple[str, ...]], start: int, end: int,
              language: str) -> Unit:
    labels, path = run
    key = section_key(doc_id, path)
    unit_id = digest(doc_id, key, digest(text, size=64), ",".join(labels), str(CHUNKER_VERSION))
    return Unit(unit_id, doc_id, key, layer, text, labels, path, start, end, language)


def prose_pieces(e: Element, language: str) -> list[tuple[str, int, int]]:
    """A paragraph as one piece, or as sentences packed up to MAX_TOKENS when it is longer, each piece with its own
    offsets (DR-F12): exact in txt and markdown, whose element text is its span; in html the element's start plus the
    piece's place in the unescaped text, which orders the pieces though it does not resolve to them."""
    if estimate_tokens(e.text, language) <= MAX_TOKENS:
        return [(e.text, e.start, e.end)]
    pieces: list[tuple[str, int, int]] = []
    current: list[tuple[int, int]] = []
    for s, t in sentence_spans(e.text):
        if current and estimate_tokens(e.text[current[0][0]:t], language) > MAX_TOKENS:
            pieces.append((e.text[current[0][0]:current[-1][1]], e.start + current[0][0], e.start + current[-1][1]))
            current = []
        current.append((s, t))
    if current:
        pieces.append((e.text[current[0][0]:current[-1][1]], e.start + current[0][0], e.start + current[-1][1]))
    return pieces


def merge_short(pieces: list[tuple[str, int, int]], language: str) -> list[tuple[str, int, int]]:
    """A piece under MIN_TOKENS joins the next piece of its run, or the one before when it is last."""
    out: list[tuple[str, int, int]] = []
    carry: tuple[str, int, int] | None = None
    for text, start, end in pieces:
        if carry is not None:
            text, start = carry[0] + "\n\n" + text, carry[1]
            carry = None
        if estimate_tokens(text, language) < MIN_TOKENS:
            carry = (text, start, end)
            continue
        out.append((text, start, end))
    if carry is not None:
        if out:
            text, start, _ = out.pop()
            out.append((text + "\n\n" + carry[0], start, carry[2]))
        else:
            out.append(carry)
    return out


def is_boundary(row_text: str) -> bool:
    return int(digest(row_text, size=8), 16) % ROWS_PER_UNIT == 0


def table_groups(header: Element | None, rows: list[Element], language: str) -> list[tuple[str, int, int]]:
    """Data rows grouped under their header, closing at content defined boundaries and the size limits."""
    head = header.text if header else ""
    groups: list[tuple[str, int, int]] = []
    current: list[Element] = []

    def close() -> None:
        if current:
            text = "\n".join(([head] if head else []) + [r.text for r in current])
            groups.append((text, current[0].start, current[-1].end))
            current.clear()

    for row in rows:
        candidate = "\n".join(([head] if head else []) + [r.text for r in current] + [row.text])
        if current and (len(current) >= MAX_ROWS or estimate_tokens(candidate, language) > MAX_TOKENS):
            close()
        current.append(row)
        if is_boundary(row.text):
            close()
    close()
    return groups


def units(elements: list[Element], doc_id: str, language: str) -> list[Unit]:
    """The document's units, from its segmented elements."""
    out: list[Unit] = []
    run_key = None
    prose: list[tuple[str, int, int]] = []
    header: Element | None = None
    rows: list[Element] = []

    def flush_prose() -> None:
        for text, start, end in merge_short(prose, language):
            if text.strip():
                out.append(make_unit(doc_id, "prose", text, run_key, start, end, language))
        prose.clear()

    def flush_table() -> None:
        nonlocal header
        for text, start, end in table_groups(header, rows, language):
            out.append(make_unit(doc_id, "table", text, run_key, start, end, language))
        rows.clear()
        header = None

    for e in elements:
        key = (e.labels, e.path)
        if e.kind in ("heading", "audience") or key != run_key:
            flush_prose()
            flush_table()
            run_key = key
        if e.kind == "paragraph":
            flush_table()
            prose.extend(prose_pieces(e, language))
        elif e.kind == "row":
            if e.header:
                flush_prose()
                flush_table()
                header = e
            else:
                rows.append(e)
    flush_prose()
    flush_table()
    return out
