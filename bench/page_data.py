"""The page's recorded data (T6.2e): the change benches' recorded runs and the build's figures, read from the records
the README's claims are held to, `docs/claims/`, so the page replays what those commands printed, with the command and
the time each record names.

    uv run python bench/page_data.py           writes web/public/recorded/changes.json and scale.json
    uv run python bench/page_data.py --check   exits 1 when either differs from what the records give

Nothing here runs a bench or reads the index: `bench/claims.py record` makes a record, and this reads it. A line of a
record that none of its forms reads is an error, so a bench that prints something new cannot drop out of the page
unseen.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs" / "claims"
OUT = REPO / "web" / "public" / "recorded"
# The change benches, each with its record and the two changes it makes in turn.
BENCHES = {"freshness": ("python-bench-freshness.py.txt", ("edit", "undoing")),
           "revocation": ("python-bench-revocation.py.txt", ("widening", "narrowing"))}
STATS = "ntk-stats.txt"

# What the bench ran under, its first line and its last (bench/machine.py; MA-F20, MA-F22).
MACHINE = re.compile(r"at (?P<commit>[0-9a-f]+)(?P<dirty>, (?:the tree holding changes|with changes to)[^;]*)?; the processor "
                     r"(?:(?P<before>\d+) percent busy|unknown) in the [\d.]+ s before the runs")
DURING = re.compile(r"the processor (?:(?P<during>\d+) percent busy|unknown) over the runs, the bench's own work included")
OPENING = re.compile(r"(?P<doc>[A-Z]+_[A-Z]+_\d+), (?:(?P<size>[\d.]+ MB), )?as (?P<reader>[a-z-]+); the watcher's catch "
                     r"up, a scan and a sync of every document: (?P<catch_up>[\d.]+) s")
RUN = re.compile(r"run (?P<run>\d+), the (?P<change>[a-z]+): (?P<events>.*?);? ?seen at (?P<seen>[\d.]+) s")
EVENT = re.compile(r"(?P<kind>scan|sync) at (?P<at>[\d.]+) s took (?P<took>[\d.]+) s \((?P<detail>[^)]*)\)")
PART = re.compile(r"([a-z]+) ([\d.]+)")
SETTINGS = re.compile(r"\(watch interval (?P<interval>[\d.]+) s; objective (?P<objective>[\d.]+) s\)")
SCALE = re.compile(r"(?P<documents>[\d,]+) documents, (?P<bytes>[\d,]+) bytes; (?P<units>[\d,]+) units \((?P<layers>[^)]*)\); "
                   r"(?P<vectors>[\d,]+) vectors in (?P<collection>\w+); on disk, the index (?P<index>[\d,]+) bytes, the "
                   r"manifest (?P<manifest>[\d,]+), the vector cache (?P<cache>[\d,]+)")


class RecordError(ValueError):
    """A record this reader cannot read whole."""


def split(name: str) -> tuple[dict, list[str]]:
    """A record's head, its command and when it was recorded, and the lines of its output."""
    text = (RECORDS / name).read_bytes().decode("utf-8")
    head, sep, body = text.partition("\n---\n")
    if not sep:
        raise RecordError(f"{name}: no line of three dashes after its head")
    fields = dict(line.split(" ", 1) for line in head.splitlines() if line.strip())
    if "command:" not in fields or "recorded" not in fields:
        raise RecordError(f"{name}: its head names no command or no time")
    return ({"command": fields["command:"], "recorded": f"recorded {fields['recorded']}"},
            [line.rstrip() for line in body.splitlines() if line.strip()])


def change(bench: str) -> dict:
    """One change bench's record: its document, reader and catch up, each run's scans and syncs in seconds from the
    change, when a query saw it, and the lines it closed with."""
    name, kinds = BENCHES[bench]
    head, lines = split(name)
    machine = MACHINE.fullmatch(lines[0].strip())
    if not machine:
        raise RecordError(f"{name}: its first line says no commit and no load: {lines[0][:80]!r}")
    opened = OPENING.fullmatch(lines[1].strip()) if len(lines) > 1 else None
    if not opened:
        raise RecordError(f"{name}: its second line is no watcher's opening: {lines[1][:80] if len(lines) > 1 else ''!r}")
    runs, said, during = [], [], None
    for line in lines[2:]:
        stripped = line.strip()
        ran = RUN.fullmatch(stripped)
        closing = DURING.fullmatch(stripped)
        if ran:
            events = [part for part in ran["events"].split("; ") if part]
            read = [EVENT.fullmatch(part) for part in events]
            if not all(read) or ran["change"] not in kinds:
                raise RecordError(f"{name}: a run line it cannot read: {stripped[:80]!r}")
            runs.append({"run": int(ran["run"]), "change": ran["change"], "seen": float(ran["seen"]),
                         "events": [{"kind": e["kind"], "at": float(e["at"]), "took": float(e["took"]),
                                     "parts": {k: float(v) for k, v in PART.findall(e["detail"])} if e["kind"] == "sync"
                                     else {}, "detail": e["detail"]} for e in read]})
        elif closing:
            during = int(closing["during"]) if closing["during"] else None
        elif stripped.startswith(f"{opened['doc']}: ") or stripped.startswith("past the "):
            said.append(stripped)
        else:
            raise RecordError(f"{name}: a line none of its forms reads: {stripped[:80]!r}")
    settings = next((SETTINGS.search(line) for line in said if SETTINGS.search(line)), None)
    if settings is None or not runs:
        raise RecordError(f"{name}: no runs, or no closing line with the watch interval and the objective")
    before = int(machine["before"]) if machine["before"] else None
    return {"bench": bench, **head, "commit": machine["commit"], "clean": not machine["dirty"], "before": before,
            "during": during, "doc_id": opened["doc"], "size": opened["size"], "reader": opened["reader"],
            "catch_up": float(opened["catch_up"]), "interval": float(settings["interval"]),
            "objective": float(settings["objective"]), "kinds": list(kinds), "runs": runs, "said": said}


def scale() -> dict:
    """The build's figures as `ntk stats` printed them, in the shape `GET /stats` answers with."""
    head, lines = split(STATS)
    found = SCALE.fullmatch(lines[0]) if len(lines) == 1 else None
    if not found:
        raise RecordError(f"{STATS}: not the one line ntk stats prints")
    number = lambda key: int(found[key].replace(",", ""))  # noqa: E731
    return {**head, "documents": number("documents"), "bytes": number("bytes"), "units": number("units"),
            "layers": {k: int(v.replace(",", "")) for k, v in re.findall(r"([a-z]+) ([\d,]+)", found["layers"])},
            "vectors": number("vectors"), "collection": found["collection"],
            "disk": {"index": number("index"), "manifest": number("manifest"), "vector_cache": number("cache")}}


def built() -> dict[str, bytes]:
    """Each data file the page reads, as this writes it."""
    out = {"changes.json": {"changes": [change(bench) for bench in BENCHES]}, "scale.json": scale()}
    return {name: (json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8") for name, data in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the page's data differs from the records")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        files = built()
    except RecordError as error:
        print(f"page_data: {error}")
        return 1
    stale = [name for name, data in files.items() if not (OUT / name).is_file() or (OUT / name).read_bytes() != data]
    if args.check:
        for name in stale:
            print(f"page_data: web/public/recorded/{name} differs from what the records give: run bench/page_data.py")
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (OUT / name).write_bytes(data)
    changes = json.loads(files["changes.json"])["changes"]
    print(f"web/public/recorded: {', '.join(f'{c['bench']} {len(c['runs'])} runs' for c in changes)}; the build's figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
