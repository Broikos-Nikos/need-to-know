"""The full build, timed (T3.15): the assembled corpus to a searchable index, phase by phase, with the memory and
disk each phase needs.

    uv run python bench/full_build.py [--fresh]

Runs each phase as its own process, as a user would: `ntk ingest build`, `ntk embed`,
`ntk index create`, `ntk index sync`; then waits for Qdrant to finish indexing what the
sync wrote. Qdrant is started afresh before the index phases, so its peaks are this
build's. For every phase it prints the wall seconds; the CPU seconds and the peak
working set of the process that did the work, read from Windows through the process
handle once it has exited (Qdrant's for the indexing); the share of the machine's
threads busy meanwhile, this build included, since other work slows a timing; and the
peak size of the data directory, sampled every second. Then the manifest's bytes by
column. With `--fresh` it first deletes the manifest, the vector cache, the lexical
statistics, the frequency table and the collection, so every phase starts from nothing;
without it, each phase does only what is left to do. Writes NTK_DATA_DIR/bench/full_build.json.

Windows only, as the pinned Qdrant binary is.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import machine

from needtoknow.index import server
from needtoknow.index.collection import COLLECTION
from needtoknow.settings import data_dir

PHASES = [("ingest", ["ingest", "build"]), ("embed", ["embed"]), ("index create", ["index", "create"]),
          ("index sync", ["index", "sync"])]
GB = 2**30


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def filetime(t: wintypes.FILETIME) -> float:
    return ((t.dwHighDateTime << 32) | t.dwLowDateTime) / 1e7


def process_peaks(handle: int) -> tuple[float, float]:
    """A process's peak working set in bytes and its CPU seconds, user and kernel, by its handle."""
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(MemoryCounters)
    psapi, kernel32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    times = [wintypes.FILETIME() for _ in range(4)]
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    if not kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize, filetime(times[2]) + filetime(times[3])


def pid_peaks(pid: int) -> tuple[float, float]:
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # query limited information, read memory
    if not handle:
        raise ctypes.WinError()
    try:
        return process_peaks(handle)
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


def pid_memory(pid: int) -> tuple[float, float]:
    """A running process's working set and its private bytes, which leave out the pages of files it maps."""
    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
    if not handle:
        raise ctypes.WinError()
    try:
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(MemoryCounters)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.WorkingSetSize, counters.PagefileUsage
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


def serving(questions: int, config: dict) -> int:
    """Qdrant restarted on the built index: its memory once loaded, then after questions asked as admin, whose
    filter passes every unit, through the retrieval API, the exact path included."""
    import random

    from needtoknow.index import search as retrieval
    from needtoknow.ingest.manifest import opened

    if server.running(config):
        server.stop(config)
    pid, _ready, _ = server.start(config)
    client = server.client(config)
    while client.get_collection(COLLECTION).status.value != "green":
        time.sleep(1.0)
    loaded = pid_memory(pid)
    with opened() as db:
        ids = [u for (u,) in db.execute("SELECT unit_id FROM units ORDER BY unit_id")]
        picked = random.Random(71).sample(ids, questions + 1)
        asked = [" ".join(db.execute("SELECT text FROM units WHERE unit_id = ?", (u,)).fetchone()[0].split()[:30])
                 for u in picked]
    retrieval.search(client, asked[0], "admin", rerank="none")  # the encoder loads
    began = time.perf_counter()
    for question in asked[1:]:
        retrieval.search(client, question, "admin", rerank="none")
    each = (time.perf_counter() - began) / questions
    after = pid_memory(pid)
    print(f"qdrant serving, restarted on the built index: working set {loaded[0] / GB:.2f} GB, private "
          f"{loaded[1] / GB:.2f} GB; after {questions} questions as admin, {after[0] / GB:.2f} GB and "
          f"{after[1] / GB:.2f} GB; {each * 1000:.0f} ms a question, encoding and both searches", flush=True)
    return 0


def size_of(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:  # a file removed while the walk passed it
                pass
    return total


class DiskPeak:
    """The data directory's largest size while a phase runs, sampled every second."""

    def __init__(self, folder: Path) -> None:
        self.folder, self.peak, self.done = folder, size_of(folder), threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def watch(self) -> None:
        while not self.done.wait(1.0):
            self.peak = max(self.peak, size_of(self.folder))

    def stop(self) -> int:
        self.done.set()
        self.thread.join()
        return max(self.peak, size_of(self.folder))


def phase(name: str, run, data: Path, pid: int | None = None) -> dict:
    """Time one phase. run() returns the peaks the phase's own process read of itself as it ended; or, given a pid,
    the phase is that process's work meanwhile: its CPU seconds then, and its peak so far."""
    began_load = machine.counters()
    cpu0 = pid_peaks(pid)[1] if pid else 0.0
    disk = DiskPeak(data)
    began = time.perf_counter()
    own = run()
    seconds = time.perf_counter() - began
    peak_disk = disk.stop()
    share = machine.busy(began_load, machine.counters())
    memory, cpu = pid_peaks(pid) if pid else (own["peak_working_set"], own["cpu_seconds"])
    cpu -= cpu0
    row = {"phase": name, "seconds": seconds, "cpu_seconds": cpu, "peak_working_set": memory,
           "machine_busy": None if share is None else share / 100, "peak_data_dir": peak_disk}
    print(f"{name}: {seconds:.0f} s; cpu {cpu:.0f} s; peak working set {memory / GB:.2f} GB; machine busy "
          f"{row['machine_busy']:.0%}; data directory at most {peak_disk / GB:.2f} GB", flush=True)
    return row


def ntk(args: list[str], data: Path) -> dict:
    """Run one ntk command in a process of its own, its output passed through, and return the peaks that process read
    of itself as it ended. Not through a handle from here: in a virtual environment on Windows, python.exe and
    ntk.exe are launchers, and the interpreter doing the work is their child."""
    report = data / "bench" / "full_build.phase.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    code = subprocess.call([sys.executable, __file__, "--run", json.dumps(args), "--report", str(report)],
                           stdout=sys.stdout, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"ntk {' '.join(args)} failed with exit {code}")
    return json.loads(report.read_bytes())


def run_and_report(args: list[str], report: Path) -> int:
    """The child's side: one ntk command, then this process's own peak working set and CPU seconds."""
    from needtoknow.cli import main

    code = main(args)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    memory, cpu = process_peaks(kernel32.GetCurrentProcess())
    report.write_bytes(json.dumps({"peak_working_set": memory, "cpu_seconds": cpu}).encode("utf-8"))
    return code


def manifest_columns(path: Path) -> dict:
    import sqlite3

    db = sqlite3.connect(path)
    try:
        pages = db.execute("PRAGMA page_count").fetchone()[0] * db.execute("PRAGMA page_size").fetchone()[0]
        columns = ["unit_id", "doc_id", "section_key", "layer", "labels", "path", "text", "language"]
        sums = db.execute("SELECT COUNT(*), " + ", ".join(f"SUM(LENGTH(CAST({c} AS BLOB)))" for c in columns)
                          + " FROM units").fetchone()
    finally:
        db.close()
    return {"bytes": pages, "units": sums[0], **dict(zip(columns, sums[1:]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fresh", action="store_true",
                        help="delete the manifest, the vector cache, the lexical statistics and the collection first")
    parser.add_argument("--serving", type=int, metavar="QUESTIONS",
                        help="instead of a build: restart Qdrant on the built index and measure its memory, loaded "
                             "and after this many questions")
    parser.add_argument("--run", help=argparse.SUPPRESS)      # the child's side of one phase
    parser.add_argument("--report", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if os.name != "nt":
        print("Windows only, as the pinned Qdrant binary is", file=sys.stderr)
        return 1
    if args.run:
        return run_and_report(json.loads(args.run), Path(args.report))
    data, config = data_dir(), server.load_config()
    if args.serving:
        return serving(args.serving, config)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.fresh:
        if not server.running(config):
            server.start(config)
        client = server.client(config)
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.close()
        server.stop(config)
        trash = data / "qdrant" / "storage" / ".deleted"
        if trash.is_dir():  # Qdrant moves a deleted collection here and clears it later; once stopped it cannot
            shutil.rmtree(trash)
        gone = [p for p in (data / "manifest.db", *(data / "embed").glob("*.db*"), *(data / "index").glob("*"))
                if p.is_file()] + list(data.glob("manifest.db-*"))
        for path in gone:
            path.unlink()
        print(f"fresh: the collection {COLLECTION} and {len(gone)} files deleted: "
              + ", ".join(sorted({p.name for p in gone})), flush=True)
    elif server.running(config):
        server.stop(config)
    print(f"started {started}; the data directory holds {size_of(data) / GB:.2f} GB", flush=True)
    rows = []
    for name, command in PHASES[:2]:
        rows.append(phase(name, lambda command=command: ntk(command, data), data))
        if name == "ingest":
            columns = manifest_columns(data / "manifest.db")
            print("manifest: " + f"{columns['bytes'] / 2**20:.0f} MB in pages, {columns['units']} units; by column, MB: "
                  + ", ".join(f"{c} {columns[c] / 2**20:.1f}" for c in columns if c not in ("bytes", "units")), flush=True)
    pid, ready, _ = server.start(config)
    print(f"qdrant started afresh as pid {pid}, ready in {ready:.1f} s", flush=True)
    for name, command in PHASES[2:]:
        rows.append(phase(name, lambda command=command: ntk(command, data), data))
    client = server.client(config)

    def indexed() -> None:
        last, steady = None, 0
        while True:
            info = client.get_collection(COLLECTION)
            count = info.indexed_vectors_count or 0
            if info.status.value == "green":
                if count >= info.points_count:
                    return
                steady = steady + 1 if count == last else 0
                if steady >= 15:  # green and unchanged for 30 s: the rest sits in segments too small to index
                    return
            last = count
            time.sleep(2.0)

    rows.append(phase("qdrant indexing", indexed, data, pid=pid))
    memory, cpu = pid_peaks(pid)
    info = client.get_collection(COLLECTION)
    storage = size_of(data / "qdrant")
    total = sum(r["seconds"] for r in rows)
    print(f"qdrant over the build: cpu {cpu:.0f} s, peak working set {memory / GB:.2f} GB; {info.points_count} points, "
          f"{info.indexed_vectors_count} vectors indexed, {info.segments_count} segments; storage {storage / GB:.2f} GB",
          flush=True)
    print(f"full build: {total:.0f} s ({total / 3600:.2f} h); data directory at most "
          f"{max(r['peak_data_dir'] for r in rows) / GB:.2f} GB, now {size_of(data) / GB:.2f} GB", flush=True)
    out = data / "bench" / "full_build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps({"started": started, "fresh": args.fresh, "phases": rows, "manifest": columns,
                                 "qdrant": {"cpu_seconds": cpu, "peak_working_set": memory, "storage": storage,
                                            "points": info.points_count}}) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
