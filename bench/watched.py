"""The watcher as the change benches run it (MA-F6): `ntk ingest watch`, the command a deployment runs, in a process of
its own and syncing the index; every scan and sync timed from the line it prints, and a change timed only once the
catch up is done, so a timing begins at the edit and holds nothing else.

Not a thread of the bench: the bench asks its questions every 50 ms meanwhile, and in one interpreter the two take
turns, so the largest document's scan took 4.0 s there against 2.0 s alone (tick 73).

Imported by `bench/freshness.py` and `bench/revocation.py`; not a bench of its own.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

OBJECTIVE = 5.0  # seconds from an edit to a query that sees it: specs/architecture.md, sections 5 and 15
SCAN = re.compile(r"^(?:scan|build) run [0-9]+: files [0-9]+, read ([0-9]+).*; in ([0-9.]+) s$")
SYNC = re.compile(r"^index: .*; in ([0-9.]+) s [(](.*)[)]$")
EVERY = re.compile(r"^watching .* every ([0-9.]+) s;")


class Watched:
    def __init__(self, interval: float | None = None) -> None:
        self.interval = interval  # None: the command's own default, read back from its first line
        self.events: list[dict] = []  # kind, began and seconds, from the moment its line arrived
        self.lines: list[str] = []
        self.caught_up = threading.Event()
        self.process: subprocess.Popen | None = None

    def start(self, limit: float = 600.0) -> float:
        """Start `ntk ingest watch` and return the seconds until its catch up, a scan and a sync of every document."""
        began = time.perf_counter()
        launcher = Path(sys.executable).with_name("ntk.exe" if sys.platform == "win32" else "ntk")
        flag = [] if self.interval is None else ["--interval", f"{self.interval:g}"]
        self.process = subprocess.Popen([str(launcher), "ingest", "watch", *flag], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        threading.Thread(target=self.read, daemon=True).start()
        if not self.caught_up.wait(limit):
            raise TimeoutError(f"the watcher's catch up took over {limit} s: {self.lines[-3:]}")
        return time.perf_counter() - began

    def read(self) -> None:
        for line in self.process.stdout:
            arrived, line = time.perf_counter(), line.rstrip()
            self.lines.append(line)
            every = EVERY.match(line)
            if every:
                self.interval = float(every.group(1))
            scanned, synced = SCAN.match(line), SYNC.match(line)
            if scanned:
                seconds = float(scanned.group(2))
                self.events.append({"kind": "scan", "began": arrived - seconds, "seconds": seconds,
                                    "detail": f"{scanned.group(1)} read"})
            elif synced:
                seconds = float(synced.group(1))
                self.events.append({"kind": "sync", "began": arrived - seconds, "seconds": seconds,
                                    "detail": synced.group(2)})
                self.caught_up.set()

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait(30)

    def timeline(self, edit: float, seen: float) -> str:
        """What the watcher did between an edit and the query that saw it, in seconds from the edit. The query can see
        a change before the sync has done its closing check, so the sync's line is waited for."""
        until(lambda: any(e["kind"] == "sync" and e["began"] >= edit for e in self.events), 30)
        parts = [f"{e['kind']} at {e['began'] - edit:.2f} s took {e['seconds']:.2f} s ({e['detail']})"
                 for e in self.events if edit <= e["began"] <= edit + seen]
        return "; ".join(parts) + f"; seen at {seen:.2f} s"


def spread(seconds: list[float]) -> str:
    """Repeated timings as the change benches report them: one run is one draw of the watcher's poll, the scan and the
    machine's load, so a figure quoted alone moves on the next run (RS-F13)."""
    return f"median {statistics.median(seconds):.1f} s, largest {max(seconds):.1f} s, over {len(seconds)} runs"


def past(seconds: list[float]) -> bool:
    """Whether any run took longer than the objective: the largest decides, never the median."""
    return max(seconds) > OBJECTIVE


def until(condition, limit: float) -> float:
    began = time.perf_counter()
    while time.perf_counter() - began < limit:
        if condition():
            return time.perf_counter() - began
        time.sleep(0.05)
    raise TimeoutError(f"not seen within {limit} s")
