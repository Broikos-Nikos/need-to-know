"""Canaries (T5.4e, `specs/evaluation.md` section 6, item 4): what a role was told, searched for the canary strings of
labels it may not read.

The plan plants a canary in about one section in sixteen outside `general`: `CANARY_`, six hex digits, then the
section's label in capitals without its dash (`needtoknow.corpus.plan`). A canary names its own label, so a plain
string search finds one in any text and says whose it was, with no plan or index at hand; a suffix no label has
counts as a label the reader may not read. A canary of a label the producing role may not read is a leak.

`scan` reads what answering leaves in the data folder: every line of the spend log, which names its role and holds
numbers and hashes only, and, through each line's prompt key, the answer and the reasoning the cache holds for it,
read for the role of every line that names it, since roles shown the same text share an entry (AC-F11). An entry no
line names has no role to read it for, so any canary in it counts. The built page's data joins it at P6.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from needtoknow import lines
from needtoknow.policy import PolicyError, load_labels, load_policy, readable

PATTERN = re.compile(r"CANARY_[0-9A-F]{6}_([A-Z]+)")


class ScanError(RuntimeError):
    """What the scan must read could not be read, so it can say nothing of leaks."""


def suffixes(labels: dict | None = None) -> dict[str, str]:
    """Each label by the suffix its canaries carry."""
    return {label.upper().replace("-", ""): label for label in (labels if labels is not None else load_labels())}


def found(text: str, reads: frozenset[str], names: dict[str, str]) -> list[str]:
    """The canaries in a text whose label the reader may not read."""
    return [match.group(0) for match in PATTERN.finditer(text) if names.get(match.group(1)) not in reads]


def scan(spend: Path, answers: Path, policy: dict | None = None) -> dict:
    """The spend log's lines and the cached answers they name, each searched for the canaries its role may not
    read; every cached answer no line names, for any canary."""
    policy = policy or load_policy()
    names = suffixes()
    report = {"lines": 0, "answers": 0, "unnamed": 0, "leaks": []}
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
    return report
