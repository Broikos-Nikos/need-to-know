"""Sentences inside a paragraph, with their offsets (T2.5).

A sentence ends at ".", "?" or "!", and at ";" after a Greek letter, which is the
Greek question mark once the text is in NFC (KB1-F14); English keeps its ";" inside a
sentence. The end needs a space after it and, after any closing quotes or brackets, a
capital or a digit to start the next sentence. It is no end after an abbreviation
from the lists below, after a single capital such as an initial, or inside a number
such as 2.3, which has no space after its dot.

Rules rather than Punkt (N47): Punkt's trained models are a download, and untrained
it is a heuristic of this kind anyway; these rules are explicit, and every planted
fact in both languages comes out of them as one whole sentence.
"""

from __future__ import annotations

import re

SPLITTER_VERSION = 1
ABBREVIATIONS = {
    # English, lower case, without the final dot.
    "e.g", "i.e", "etc", "mr", "mrs", "ms", "dr", "no", "nos", "vs", "approx", "fig", "st", "dept", "inc", "ltd", "jan",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "p", "pp", "art", "para", "ref",
    # Greek, lower case, without the final dot.
    "κ", "π.χ", "παρ", "κλπ", "δηλ", "βλ", "αρ", "τηλ", "σελ", "ν", "άρθ", "σημ", "περ", "υπ", "κ.λπ", "κ.ά", "αριθ",
}
END = re.compile(r"[.?!;\u037e]")  # U+037E, the Greek question mark, should NFC not have run
GREEK = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")  # Greek and Greek with accents
CLOSERS = "\"'»”’)]"
STARTS = re.compile(r"[\"'«“‘(\[]*[A-ZΑ-ΩΆΈΉΊΌΎΏ0-9]")


def ends_here(text: str, at: int) -> bool:
    """Whether the mark at `at` ends a sentence."""
    mark = text[at]
    if mark == ";" and not (at > 0 and GREEK.match(text[at - 1])):  # U+037E itself is always Greek
        return False
    after = at + 1
    while after < len(text) and text[after] in CLOSERS:
        after += 1
    if after < len(text) and not text[after].isspace():
        return False  # 2.3, e.g, a.m: no space, no end
    rest = text[after:].lstrip()
    if rest and not STARTS.match(rest):
        return False
    if mark == ".":
        word = re.search(r"(\S+)$", text[:at])
        stem = word.group(1).lstrip("\"'«“‘([") if word else ""
        if stem.casefold() in ABBREVIATIONS or (len(stem) == 1 and stem.isupper()):
            return False
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Spans of the sentences in a paragraph, each without the space around it."""
    spans, start = [], 0
    for mark in END.finditer(text):
        at = mark.start()
        if ends_here(text, at):
            end = at + 1
            while end < len(text) and text[end] in CLOSERS:
                end += 1
            spans.append((start, end))
            start = end
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out
