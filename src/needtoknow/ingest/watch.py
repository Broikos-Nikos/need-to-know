"""The watcher (T2.9): changes to the corpus reach the manifest within seconds, through the same scan.

It polls a fingerprint of the corpus folder, every file's path, size and mtime, which
costs one directory walk and no reading. When the fingerprint moves it waits until it
holds still for one interval, so a file written in several steps, or a swap of the
whole folder, is scanned once and whole (debounced). Then it runs the scan of T2.8,
which decides what changed. Polling rather than filesystem events (N48): no new
dependency, the same behaviour on every platform, and a freshness of two intervals
plus one scan.

When the scan changed a unit, the watcher syncs the index too, for the documents it
changed (AC-F6): a narrowed audience that reached only the manifest would go on being
served from the index. The sync deletes before it upserts, so a revocation lands even
when a later step fails, and a sync that fails is tried again every interval, with the
documents of every scan since, until one succeeds.

A write that keeps a file's size inside the same clock tick as the version the manifest
recorded shows in neither the fingerprint nor the scan's shortcut. The quiet interval
makes that impossible for a change the watcher has seen, since the recorded version is
then at least an interval old; `ntk ingest build` verifies every file by its hash.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from needtoknow.ingest.manifest import ManifestError, scan


def fingerprint(full: Path) -> frozenset[tuple[str, int, int]]:
    """Every file under the corpus folder by path, size and mtime; empty while the folder is gone."""
    out = []
    for root, _dirs, files in os.walk(full):
        for name in files:
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except FileNotFoundError:  # moved away between the listing and the stat
                continue
            out.append((os.path.relpath(path, full), stat.st_size, stat.st_mtime_ns))
    return frozenset(out)


def watch(full: Path, interval: float = 1.0, stop: Callable[[], bool] = lambda: False,
          report: Callable[[dict], None] = lambda summary: None, sleep: Callable[[float], None] = time.sleep,
          scan_fn: Callable[[Path], dict] = scan, sync_fn: Callable[[list[str]], dict] | None = None) -> int:
    """Scan after every change once it has settled, then sync the documents it changed, until `stop`; the scans run."""
    last, pending, unsynced, runs = fingerprint(full), False, set(), 0
    while not stop():
        sleep(interval)
        current = fingerprint(full)
        if current != last:
            last, pending = current, True  # still moving: wait for one quiet interval
            continue
        if pending:
            try:
                summary = scan_fn(full)
                report(summary)
                unsynced.update(summary.get("changed_docs", []))
            except ManifestError as error:  # the folder gone, say: said, and the watch goes on (DR-F11)
                report({"error": str(error)})
            runs += 1
            pending = False
        if unsynced and sync_fn is not None:
            try:
                report(sync_fn(sorted(unsynced)))
                unsynced = set()
            except Exception as error:  # the server down, say: said, and tried again next interval (AC-F6)
                report({"error": f"index not synced, trying again: {type(error).__name__}: {error}"})
    return runs
