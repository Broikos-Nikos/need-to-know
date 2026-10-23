"""Is retrieval repeatable on one build (RS-F29)? Every dev item searched as its role, as `ask` searches, several times
in one process, and the excerpts the prompt would hold compared run by run.

    uv run python bench/repeatability.py [--runs 3] [--split dev]

Needs the manifest, the index on the running server, the `embed` extra and the GPU; asks the protocol first. The
release's parity rule (`eval/protocol.toml` [release]) needs every dev item's retrieval equal to the last dev run of
the same configuration, so an item shown different excerpts in two runs of one build is one that rule would fail on
noise. It prints how many items were shown the same excerpts in every run, names the others, and, for them, whether
the fused list the reranker read was the same set of units each time.

    uv run python bench/repeatability.py --against eval/runs/baseline/dev.jsonl

With `--against`, a recorded run of the split, it is the parity check itself, asked for parity: each item's excerpts
are compared with those the run recorded, in every repeat, and it names the items shown others, those whose repeats
there disagreed, and those the run did not ask.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from machine import commit

from needtoknow.eval.golden import golden_path, load_items
from needtoknow.eval.protocol import check_use
from needtoknow.index import server
from needtoknow.index.expand import excerpts
from needtoknow.index.search import search
from needtoknow.policy import load_policy


def written(shown: list[dict]) -> str:
    """Excerpts as they are compared: each one's document and units, in the order shown."""
    return json.dumps([(e["doc_id"], e["units"]) for e in shown])


def recorded(path: Path) -> dict[str, set[str]]:
    """Each item's excerpts in every repeat of a recorded run."""
    found = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        found[row["id"]].add(written(row["excerpts"]))
    return found


def parity(now: dict[str, set[str]], run: dict[str, set[str]]) -> tuple[list[str], list[str], list[str]]:
    """Against a recorded run: the items shown other excerpts than it showed in every repeat, the items whose repeats
    there disagreed, and the items it did not ask."""
    other = [item for item, shown in now.items() if len(run.get(item, ())) == 1 and shown != run[item]]
    return other, [item for item in now if len(run.get(item, ())) > 1], [item for item in now if not run.get(item)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--against", type=Path, help="a recorded run of the split, whose excerpts each item's must equal")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    check_use(args.split, "parity" if args.against else "development")
    print(commit(), flush=True)  # the code the run used (MA-F29)
    items = load_items(golden_path(args.split))
    run = recorded(args.against) if args.against else {}
    client, policy = server.client(), load_policy()
    search(client, items[0]["question"], items[0]["role"], policy)  # the models load before the clock
    began = time.perf_counter()
    moved, now = [], {}
    for item in items:
        shown, candidates = set(), set()
        for _ in range(args.runs):
            hits = search(client, item["question"], item["role"], policy)
            shown.add(written(excerpts(hits)))
            candidates.add(json.dumps(sorted(h["unit_id"] for h in hits)))
        now[item["id"]] = shown
        if len(shown) > 1:
            moved.append(f"{item['id']} ({len(shown)} excerpt sets, {len(candidates)} candidate sets)")
    print(f"{args.split}: {len(items) - len(moved)} of {len(items)} items shown the same excerpts in all {args.runs} "
          f"runs" + (f"; moved: {', '.join(moved)}" if moved else ""))
    other = []
    if args.against:
        other, unsettled, unasked = parity(now, run)
        asked = len(now) - len(unasked)
        print(f"against {args.against.as_posix()}: {asked - len(other) - len(unsettled)} of the {asked} items it asked "
              f"shown the excerpts it recorded in every repeat" + (f"; {len(other)} shown others: {', '.join(other)}"
              if other else "") + (f"; {len(unsettled)} whose repeats there disagreed: {', '.join(unsettled)}"
              if unsettled else "") + (f"; {len(unasked)} not in it: {', '.join(unasked)}" if unasked else ""))
    print(f"{len(items) * args.runs} searches in {time.perf_counter() - began:.0f} s")
    return 1 if moved or other else 0


if __name__ == "__main__":
    raise SystemExit(main())
