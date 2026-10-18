"""The corpus: the fact registry and catalogue, the writers' assignments, assembly and a light check."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from needtoknow.settings import repo_root


class CorpusBusy(RuntimeError):
    """Another process is writing the corpus."""


class NoPlan(RuntimeError):
    """The corpus folder holds no plan, which `ntk corpus plan` makes."""


def read_plan(base: Path, name: str):
    """A file of the plan in the corpus folder `base`, read as JSON; refused in one line when no plan was made there,
    in the words the evaluation's commands use, where it was a traceback (HS-F22)."""
    path = base / "plan" / name
    if not path.is_file():
        raise NoPlan(f"no plan in {path.parent}: run ntk corpus plan first")
    return json.loads(path.read_text(encoding="utf-8"))


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
def exclusive(base: Path, wait: float = 0.0, name: str = "writer.lock"):
    """One writer at a time of what lives under `base`: the corpus, assembly or the change simulator (OP-F3), the
    index, every sync (OP-F5), or the manifest, every scan (HS-F9).

    The lock is the operating system's, on one byte of `writer.lock`, so it ends with
    the process that holds it, and a crash never leaves a stale lock behind. Without
    `wait` a held lock is refused at once; with it, the lock is tried every tenth of a
    second for that many seconds, as a watcher's scan waits out a build. `name` gives one
    file its own lock, as a line file's writers take (DR-F33).
    """
    handle = open(base / name, "a+b")
    try:
        deadline = time.monotonic() + wait
        while True:
            try:
                _lock(handle)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    waited = f" after {wait:g} s" if wait else ""
                    raise CorpusBusy(f"another process is writing here: {base / name} is held{waited}") from None
                time.sleep(0.1)
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
