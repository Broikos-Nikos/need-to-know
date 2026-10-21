"""Canaries (T5.4e, `specs/evaluation.md` section 6, item 4): what a role was told, searched for the canary strings of
labels it may not read.

The plan plants a canary in about one section in sixteen outside `general`: `CANARY_`, six hex digits, then the
section's label in capitals without its dash (`needtoknow.corpus.plan`). A canary names its own label, so a plain
string search finds one in any text and says whose it was, with no plan or index at hand; a suffix no label has
counts as a label the reader may not read. A canary of a label the producing role may not read is a leak.

A canary is found in any case and with any run of spaces, underscores, dashes or line breaks between its three
parts, its suffix upper-cased before it names a label, since a model that repeats one may not keep its form
(AC-F14).

`scan` reads what answering leaves in the data folder: every line of the spend log, which names its role and holds
numbers and hashes only, and, through each line's prompt key, the answer and the reasoning the cache holds for it,
read for the role of every line that names it, since roles shown the same text share an entry (AC-F11). An entry no
line names has no role to read it for, so any canary in it counts. It reads the page's recorded traces too
(`bench/page_traces.py`), every text in each read for the role it was asked as. A scan that found nothing to read
says so, and clears nothing (AC-F15).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from needtoknow import lines
from needtoknow.policy import PolicyError, load_labels, load_policy, readable
from needtoknow.settings import repo_root

PATTERN = re.compile(r"canary[\s_-]+[0-9a-f]{6}[\s_-]+([a-z]+)", re.IGNORECASE)
PAGE = repo_root() / "web" / "public" / "recorded" / "traces.json"


class ScanError(RuntimeError):
    """What the scan must read could not be read, so it can say nothing of leaks."""


def suffixes(labels: dict | None = None) -> dict[str, str]:
    """Each label by the suffix its canaries carry."""
    return {label.upper().replace("-", ""): label for label in (labels if labels is not None else load_labels())}


def found(text: str, reads: frozenset[str], names: dict[str, str]) -> list[str]:
    """The canaries in a text whose label the reader may not read."""
    return [match.group(0) for match in PATTERN.finditer(text) if names.get(match.group(1).upper()) not in reads]


def strings(value) -> list[str]:
    """Every string a JSON value holds, each whole, so a canary a line break splits is read as it was written."""
    if isinstance(value, str):
        return [value]
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return [text for item in items for text in strings(item)]


def scan(spend: Path, answers: Path, policy: dict | None = None, page: Path | None = None) -> dict:
    """The spend log's lines and the cached answers they name, each searched for the canaries its role may not
    read; every cached answer no line names, for any canary; and the page's recorded traces, each for its role."""
    policy = policy or load_policy()
    names = suffixes()
    report = {"lines": 0, "answers": 0, "unnamed": 0, "traces": 0, "leaks": []}
    entries = {}
    if answers.is_file():
        db = sqlite3.connect(f"file:{answers.as_posix()}?mode=ro", uri=True)
        try:
            entries = {key: text + "\n" + reasoning for key, text, reasoning in
                       db.execute("SELECT key, text, reasoning FROM answers")}
        except sqlite3.DatabaseError as error:  # a scan that cannot read says so: it can clear nothing
            raise ScanError(f"the answer cache at {answers} could not be read ({error}), so nothing was scanned; the "
                            "next ntk ask sets it aside") from None
        finally:
            db.close()
    named = set()
    for number, record in enumerate(lines.records(spend), 1):  # whole lines (DR-F30)
        line = json.dumps(record, ensure_ascii=False)
        try:
            reads = readable(policy, record["role"])
        except PolicyError:  # a role the policy no longer has reads nothing now
            reads = frozenset()
        report["lines"] += 1
        report["leaks"] += [f"spend log line {number}, {record['role']}: {c}" for c in found(line, reads, names)]
        for key in (k for k in entries if k.startswith(record["prompt"])):
            named.add(key)
            report["answers"] += 1
            report["leaks"] += [f"cached answer {key[:16]}, {record['role']}: {c}"
                                for c in found(entries[key], reads, names)]
    for key in sorted(set(entries) - named):
        report["unnamed"] += 1
        report["leaks"] += [f"cached answer {key[:16]}, no role: {c}" for c in found(entries[key], frozenset(), names)]
    page = page or PAGE
    if page.is_file():
        for question, by_role in json.loads(page.read_text(encoding="utf-8"))["traces"].items():
            for role, trace in by_role.items():
                try:
                    reads = readable(policy, role)
                except PolicyError:
                    reads = frozenset()
                report["traces"] += 1
                report["leaks"] += [f"page trace {question}, {role}: {c}" for text in strings(trace)
                                    for c in found(text, reads, names)]
    return report
