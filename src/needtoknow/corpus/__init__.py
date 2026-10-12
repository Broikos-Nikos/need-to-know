"""The corpus: the fact registry and catalogue, the writers' assignments, assembly and a light check."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from needtoknow.settings import repo_root


class CorpusBusy(RuntimeError):
    """Another process is writing the corpus."""


def json_bytes(data) -> bytes:
    """JSON with LF line ends on every platform: Path.write_text would write CRLF on Windows."""
    return (json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def dump_json(path: Path, data) -> None:
    path.write_bytes(json_bytes(data))


def narrative_dir(base: Path) -> Path:
    """The writers' narrative: the data directory's own copy where there is one, as tests and trials make, and
    otherwise the copy committed in the repository's narrative folder, so a fresh clone can build the corpus (N45)."""
    local = base / "narrative"
    return local if local.is_dir() else repo_root() / "narrative"


@contextmanager
def exclusive(base: Path):
    """One writer of the corpus at a time, assembly or the change simulator (OP-F3).

    The lock is the operating system's, on one byte of `writer.lock`, so it ends with
    the process that holds it, and a crash never leaves a stale lock behind.
    """
    handle = open(base / "writer.lock", "a+b")
    try:
        try:
            _lock(handle)
        except OSError:
            raise CorpusBusy(f"another process is writing the corpus: {base / 'writer.lock'} is held") from None
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


if os.name == "nt":
    import msvcrt

    def _lock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_UN)
