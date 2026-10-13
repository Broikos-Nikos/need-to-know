"""Is retrieval repeatable on one build (RS-F29)? Every dev item searched as its role, as `ask` searches, several times
in one process, and the excerpts the prompt would hold compared run by run.

    uv run python bench/repeatability.py [--runs 3] [--split dev]

Needs the manifest, the index on the running server, the `embed` extra and the GPU; asks the protocol first. The
release's parity rule (`eval/protocol.toml` [release]) needs every dev item's retrieval equal to the last dev run of
the same configuration, so an item shown different excerpts in two runs of one build is one that rule would fail on
noise. It prints how many items were shown the same excerpts in every run, names the others, and, for them, whether
the fused list the reranker read was the same set of units each time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from needtoknow.eval.golden import golden_path, load_items
from needtoknow.eval.protocol import check_use
from needtoknow.index import server
from needtoknow.index.expand import excerpts
from needtoknow.index.search import search
from needtoknow.policy import load_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--split", choices=("dev",), default="dev")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    check_use(args.split, "development")
    items = load_items(golden_path(args.split))
    client, policy = server.client(), load_policy()
    search(client, items[0]["question"], items[0]["role"], policy)  # the models load before the clock
    began = time.perf_counter()
    moved = []
    for item in items:
        shown, candidates = set(), set()
        for _ in range(args.runs):
            hits = search(client, item["question"], item["role"], policy)
            shown.add(json.dumps([(e["doc_id"], e["units"]) for e in excerpts(hits)]))
            candidates.add(json.dumps(sorted(h["unit_id"] for h in hits)))
        if len(shown) > 1:
            moved.append(f"{item['id']} ({len(shown)} excerpt sets, {len(candidates)} candidate sets)")
    print(f"{args.split}: {len(items) - len(moved)} of {len(items)} items shown the same excerpts in all {args.runs} "
          f"runs" + (f"; moved: {', '.join(moved)}" if moved else ""))
    print(f"{len(items) * args.runs} searches in {time.perf_counter() - began:.0f} s")
    return 1 if moved else 0


if __name__ == "__main__":
    raise SystemExit(main())
