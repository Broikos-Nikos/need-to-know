"""Access segments (T2.4): every element learns who may read it, from its section's audience line.

A section's audience line, "Audience: Returns; Legal" or "Κοινό: Επιστροφές", names
the labels of everything under its heading, down to the next heading at its level or
above; a subsection without its own line inherits its parent's. Text above the first
audience line, the title and the document line, takes the sidecar's default audience.
A name the labels do not know, or a section with no audience while the sidecar names
no default, leaves the text unlabelled, which only a role that lists "unlabelled"
reads (default deny). The audience and classification lines themselves become
elements of kind "audience": they describe access and are not text to retrieve.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from needtoknow.corpus.plan import LABEL_EL, LABEL_EN
from needtoknow.ingest.structure import Element
from needtoknow.policy import UNLABELLED

# A label by its English or Greek name, as audience lines write it, or by its id, as sidecars do.
NAMES = {**{label: label for label in LABEL_EN},
         **{name.casefold(): label for label, name in LABEL_EN.items()},
         **{name.casefold(): label for label, name in LABEL_EL.items()}}
AUDIENCE = re.compile(r"^(?:Audience|Κοινό)\s*:\s*(.+)$")
CLASSIFICATION = re.compile(r"^(?:Classification|Διαβάθμιση)\s*:")
ROWS_PER_UNIT = 12


def labels_named(names: str, unknown: list[str]) -> tuple[str, ...]:
    """The labels a line of audience names grants, in order; unlabelled when any name is unknown."""
    labels = []
    for name in (n.strip() for n in names.split(";")):
        label = NAMES.get(name.casefold())
        if label is None:
            unknown.append(name)
            return (UNLABELLED,)
        if label not in labels:
            labels.append(label)
    return tuple(labels) or (UNLABELLED,)


def is_access_line(text: str) -> bool:
    return all(AUDIENCE.match(line) or CLASSIFICATION.match(line) for line in text.split("\n") if line.strip())


def segments(elements: list[Element], meta: dict, unknown: list[str] | None = None) -> list[Element]:
    """The elements with their labels, the audience lines marked."""
    unknown = [] if unknown is None else unknown
    default = labels_named(meta["default_audience"], unknown) if meta.get("default_audience") else (UNLABELLED,)
    stack: list[tuple[int, tuple[str, ...], int]] = []  # (heading level, labels, its index in out) per open section
    current, out = default, []
    for e in elements:
        if e.kind == "heading":
            while stack and stack[-1][0] >= e.level:
                stack.pop()
            current = stack[-1][1] if stack else default
            stack.append((e.level, current, len(out)))
            out.append(replace(e, labels=current))
            continue
        if e.kind == "paragraph" and is_access_line(e.text):
            named = next((AUDIENCE.match(line) for line in e.text.split("\n") if AUDIENCE.match(line)), None)
            if named:
                current = labels_named(named.group(1), unknown)
                if stack:
                    level, _, at = stack[-1]
                    stack[-1] = (level, current, at)
                    out[at] = replace(out[at], labels=current)  # the heading belongs to its section's readers
            out.append(replace(e, kind="audience", labels=current))
            continue
        out.append(replace(e, labels=current))
    return out


@dataclass
class Coverage:
    """Units by label, a unit being what T2.6 will cut: a paragraph, or ROWS_PER_UNIT rows of one table."""

    first: Counter = field(default_factory=Counter)      # units by their first label, which sums to all units
    readable: Counter = field(default_factory=Counter)   # units each label can read, which may sum above them
    unknown: list[str] = field(default_factory=list)

    def add(self, elements: list[Element]) -> None:
        rows: list[Element] = []

        def close() -> None:
            if rows:
                self.count(rows[0].labels, math.ceil(len(rows) / ROWS_PER_UNIT))
                rows.clear()

        for e in elements:
            if e.kind == "row" and not e.header and (not rows or rows[0].labels == e.labels):
                rows.append(e)
                continue
            close()
            if e.kind == "row" and not e.header:
                rows.append(e)
            elif e.kind == "paragraph":
                self.count(e.labels, 1)
        close()

    def count(self, labels: tuple[str, ...], units: int) -> None:
        self.first[labels[0]] += units
        for label in labels:
            self.readable[label] += units


def corpus_coverage(full) -> Coverage:
    """The coverage of every file in an assembled corpus, each decoded, parsed and segmented as ingestion will."""
    from needtoknow.corpus.mutate import corpus_files
    from needtoknow.ingest.decode import load
    from needtoknow.ingest.parse import parse
    from needtoknow.ingest.structure import structure

    coverage = Coverage()
    for _doc_id, (file, _sidecar, meta) in sorted(corpus_files(full).items()):
        elements = structure(parse(load(file, meta), meta["format"]), meta["format"])
        coverage.add(segments(elements, meta, coverage.unknown))
    return coverage
