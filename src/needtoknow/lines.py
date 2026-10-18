"""Append only line files, one JSON object a line (DR-F30): the spend log, the change journal and an evaluation's run.

A write stopped part way leaves a last line without its newline, perhaps inside a Greek
letter. Every reader here reads whole lines, as bytes, so such a line is left out as a write
not yet made, where the readers used to raise; every writer cuts it off before it appends,
so the next record starts a line of its own instead of gluing to it; a file with more than
one writer, the spend log, is written under its own lock, so no mend cuts a line another
process is writing at that instant (DR-F33). A whole line that is not a JSON object, or that
a reader's check refuses, is named by its file and line.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path


class LineError(ValueError):
    """A whole line of a line file that is not the record it should be."""


def mend(path: Path) -> bool:
    """Cut off a last line that a stopped write left without its newline, so the next record starts a line of its own;
    whether there was one. Only the last byte is read unless there was."""
    if not path.is_file():
        return False
    with path.open("r+b") as handle:
        size = handle.seek(0, 2)
        if size == 0:
            return False
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return False
        handle.seek(0)
        handle.truncate(handle.read().rfind(b"\n") + 1)
    return True


def append(path: Path, record: dict, shared: bool = False) -> None:
    """One record as a line, after a line a stopped write cut is cut off; a file `shared` by more than one writer
    under its own lock, waited on (DR-F33)."""
    from needtoknow.corpus import exclusive

    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(path.parent, 10.0, name=path.name + ".lock") if shared else nullcontext():
        mend(path)
        with path.open("ab") as handle:
            handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def records(path: Path, check=None) -> list[dict]:
    """The file's whole lines, each a JSON object, none when there is no file; a last line without its newline is a
    write in progress or stopped, and is left out. `check`, given a record, says what is wrong with it, or None."""
    if not path.is_file():
        return []
    found = []
    for number, line in enumerate(path.read_bytes().split(b"\n")[:-1], 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise LineError(f"{path.name} line {number} is not JSON: {error}") from None
        if not isinstance(record, dict):
            raise LineError(f"{path.name} line {number} is JSON but not a record: a record is an object")
        problem = check(record) if check else None
        if problem:
            raise LineError(f"{path.name} line {number}: {problem}")
        found.append(record)
    return found
