"""Access segments (T2.4, T2.15): every element learns who may read it, and no text can widen that.

A section's audience line, "Audience: Returns; Legal" or "Κοινό: Επιστροφές", in any
case, with or without accents, after a colon or a dash, names the labels of
everything under its heading, down to the next heading at its level or above; a
subsection without its own line inherits its parent's, and text above the first
audience line takes the sidecar's default audience. In html a section's
data-audience declares the same, and when both are given the section takes what both
allow. Only the line right under a heading sets its section's audience: a later one,
once text has appeared, can only narrow it (AC-F1). A name the labels do not know, a
line that names nothing, and a narrowing that leaves nothing all make the text
unlabelled, which only a role that lists "unlabelled" reads (default deny).

The audience and classification lines become elements of kind "audience": they
describe access and are not text to retrieve. Every element's heading path keeps only
the headings that each of its readers may read (AC-F4).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace

from needtoknow.corpus.plan import LABEL_EL, LABEL_EN
from needtoknow.ingest.structure import Element
from needtoknow.policy import UNLABELLED


def fold(text: str) -> str:
    """Lower case without accents, so "ΚΟΙΝΟ", "Κοινό" and "κοινο" read alike."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)).casefold()


# A label by its English or Greek name, as audience lines write it, or by its id, as sidecars do; keys folded.
NAMES = {**{fold(label): label for label in LABEL_EN},
         **{fold(name): label for label, name in LABEL_EN.items()},
         **{fold(name): label for label, name in LABEL_EL.items()}}
# Matched against a folded line: a colon, a hyphen or an en or em dash after the word (AC-F2).
AUDIENCE = re.compile(r"^\s*(?:audience|κοινο)\s*[:\-\u2013\u2014]\s*(.*?)\s*$")
CLASSIFICATION = re.compile(r"^\s*(?:classification|διαβαθμιση)\s*[:\-\u2013\u2014]")
SEPARATOR = re.compile(r"[:\-\u2013\u2014]")


def labels_named(names: str, unknown: list[str]) -> tuple[str, ...]:
    """The labels a line of audience names grants, in order; unlabelled when any name is unknown or none is given."""
    labels = []
    for name in (n.strip() for n in names.split(";")):
        label = NAMES.get(fold(name))
        if label is None:
            unknown.append(name)
            return (UNLABELLED,)
        if label not in labels:
            labels.append(label)
    return tuple(labels) or (UNLABELLED,)


def audience_names(text: str) -> str | None:
    """The names an access paragraph's audience line gives, as written, or None when it has none."""
    for line in text.split("\n"):
        if AUDIENCE.match(fold(line)):
            return SEPARATOR.split(line, maxsplit=1)[1].strip()  # the word before it holds no separator
    return None


def is_access_line(text: str) -> bool:
    return all(AUDIENCE.match(fold(line)) or CLASSIFICATION.match(fold(line)) for line in text.split("\n") if line.strip())


def narrow(labels: tuple[str, ...], allowed: tuple[str, ...]) -> tuple[str, ...]:
    """What both allow, in the first's order; unlabelled when nothing is left."""
    return tuple(label for label in labels if label in allowed and label != UNLABELLED) or (UNLABELLED,)


def segments(elements: list[Element], meta: dict, unknown: list[str] | None = None,
             notes: Counter | None = None) -> list[Element]:
    """The elements with their labels and safe heading paths, the audience lines marked."""
    unknown = [] if unknown is None else unknown
    notes = Counter() if notes is None else notes
    default = labels_named(meta["default_audience"], unknown) if meta.get("default_audience") else (UNLABELLED,)
    stack: list[dict] = []  # the open sections: level, labels, index of the heading in out, settled, declared
    current, out = default, []
    for e in elements:
        if e.kind == "heading":
            while stack and stack[-1]["level"] >= e.level:
                stack.pop()
            current = stack[-1]["labels"] if stack else default
            declared = e.audience is not None
            if declared:  # html: the section's data-audience
                current = labels_named(fold(e.audience), unknown)
            stack.append({"level": e.level, "labels": current, "at": len(out), "settled": False, "declared": declared})
            out.append(replace(e, labels=current))
            continue
        if e.kind == "paragraph" and is_access_line(e.text):
            names = audience_names(e.text)
            if names is not None:
                named = labels_named(names, unknown)
                top = stack[-1] if stack else None
                if top is not None and not top["settled"]:
                    if top["declared"] and set(named) != set(top["labels"]):
                        notes["html data-audience and audience line disagree"] += 1
                    current = narrow(named, top["labels"]) if top["declared"] else named
                else:
                    notes["audience line inside a section, read as narrowing only"] += 1
                    current = narrow(current, named)
                if top is not None:
                    top["labels"], top["settled"] = current, True
                    out[top["at"]] = replace(out[top["at"]], labels=current)  # the heading belongs to its section's readers
            out.append(replace(e, kind="audience", labels=current))
            continue
        if stack:
            stack[-1]["settled"] = True  # text has appeared: from now on an audience line can only narrow
        out.append(replace(e, labels=current))
    return with_safe_paths(out)


def with_safe_paths(elements: list[Element]) -> list[Element]:
    """Each element's heading path, keeping a heading only when every label that reads the element reads it (AC-F4)."""
    open_headings: list[tuple[int, str, frozenset[str]]] = []
    out = []
    for e in elements:
        if e.kind == "heading":
            while open_headings and open_headings[-1][0] >= e.level:
                open_headings.pop()
            open_headings.append((e.level, e.text, frozenset(e.labels)))
        readers = set(e.labels)
        out.append(replace(e, path=tuple(text for _, text, labels in open_headings if readers <= labels)))
    return out


@dataclass
class Coverage:
    """Units by label, as the units step cuts them (T2.6), and units by layer."""

    first: Counter = field(default_factory=Counter)      # units by their first label, which sums to all units
    readable: Counter = field(default_factory=Counter)   # units each label can read, which may sum above them
    layers: Counter = field(default_factory=Counter)     # prose and table units (KB1-F8)
    unknown: list[str] = field(default_factory=list)
    notes: Counter = field(default_factory=Counter)  # audience lines read as narrowing, html disagreements

    def add(self, units) -> None:
        for unit in units:
            self.first[unit.labels[0]] += 1
            self.layers[unit.layer] += 1
            for label in unit.labels:
                self.readable[label] += 1


def corpus_coverage(full) -> Coverage:
    """The coverage of every file in an assembled corpus, each decoded, parsed, segmented and cut as ingestion will."""
    from needtoknow.corpus.mutate import corpus_files
    from needtoknow.ingest.decode import load
    from needtoknow.ingest.parse import parse
    from needtoknow.ingest.structure import structure
    from needtoknow.ingest.units import units

    coverage = Coverage()
    for doc_id, (file, _sidecar, meta) in sorted(corpus_files(full).items()):
        elements = structure(parse(load(file, meta), meta["format"]), meta["format"])
        coverage.add(units(segments(elements, meta, coverage.unknown, coverage.notes), doc_id, meta["language"]))
    return coverage
