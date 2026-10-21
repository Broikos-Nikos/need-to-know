"""What a timing ran under, printed beside it (MA-F20, MA-F22): the commit and whether the tree held changes, and how
busy the processor was, so a figure that moves can be told from code that moved. Imported by the benches that time,
not run.

The load is the share of the processor's time that was not idle, every program's counted, from the operating system's
own counters: `GetSystemTimes` on Windows, `/proc/stat` elsewhere, the standard library alone. It is read twice: over
a few seconds before the runs, when nothing of the bench runs yet, which is the other programs' load; and over the
runs, the bench's own work included.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUIET = 3.0  # seconds the load is read for before the runs


def commit() -> str:
    """The commit and whether the tree holds changes, as a record's head says them."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return f"at {git('rev-parse', '--short', 'HEAD')}{', the tree holding changes' if git('status', '--porcelain') else ''}"


def counters() -> tuple[int, int] | None:
    """The machine's idle processor time and its whole processor time so far, or None where neither source is."""
    if sys.platform == "win32":
        idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return idle.value, kernel.value + user.value  # the kernel's time counts the idle time
    try:
        fields = [int(field) for field in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    except (OSError, ValueError):
        return None
    return fields[3] + fields[4], sum(fields)  # idle and waiting on the disk, of all


def busy(began: tuple[int, int] | None, ended: tuple[int, int] | None) -> float | None:
    """The percent of the processor's time between two readings that was not idle."""
    if began is None or ended is None or ended[1] <= began[1]:
        return None
    return 100 * (1 - (ended[0] - began[0]) / (ended[1] - began[1]))


class Load:
    """The processor's load before the runs, read at once over QUIET seconds, and over the runs, read at the end."""

    def __init__(self):
        before = counters()
        time.sleep(QUIET)
        self.began = counters()
        self.before = busy(before, self.began)

    def opening(self) -> str:
        """The commit and the other programs' load, the line a timing bench prints first."""
        load = "unknown" if self.before is None else f"{self.before:.0f} percent busy"
        return f"{commit()}; the processor {load} in the {QUIET:g} s before the runs"

    def closing(self) -> str:
        """The load over the runs, the line a timing bench prints last."""
        during = busy(self.began, counters())
        load = "unknown" if during is None else f"{during:.0f} percent busy"
        return f"the processor {load} over the runs, the bench's own work included"
