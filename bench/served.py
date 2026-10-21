"""The served path (T6.1, PF-F7): dev probes asked through the local server, each as its selective reader.

    uv run python bench/served.py [--questions 20]

The server's app as `ntk serve` builds it, over the running index, in this process: the
encoder and the reranker loaded first, as `ntk serve` loads them before it listens; then
a session for each role and the first dev probes through `POST /search`, a probe whose
gold unit the manifest lacks set apart and named (DR-F25). Prints the models' load, the
first search's seconds and the median and 95th percentile of the rest by nearest rank,
what the trace's own queries took, and the fusion check: the engine's fused scores
against reciprocal rank fusion of the two searches as the trace shows them, so the trace
can be read as what the engine fused (N64).

Asks the protocol first: dev, for development. Needs the manifest, the index on the
running server, the `embed` extra and the GPU.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

from fastapi.testclient import TestClient
from machine import Load

from needtoknow import api
from needtoknow.answer.pipeline import warm
from needtoknow.eval.probes import load, lost
from needtoknow.eval.protocol import check_use
from needtoknow.index import server
from needtoknow.index.search import RRF_K, in_manifest


def nearest(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(share * len(ordered)) - 1]


def gaps(body: dict) -> list[float]:
    """For each fused unit, how far its score lies from reciprocal rank fusion of its ranks in the two searches as the
    trace shows them."""
    ranks = [{hit["unit_id"]: hit["rank"] for hit in body[side]} for side in ("dense", "sparse")]
    return [abs(hit["score"] - sum(1 / (RRF_K + side[hit["unit_id"]]) for side in ranks if hit["unit_id"] in side))
            for hit in body["fused"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=20, help="the first probes of dev to ask")
    args = parser.parse_args(argv)
    check_use("dev", "development")
    probes = load("dev")
    missing = lost(probes, in_manifest(sorted({unit for probe in probes for unit in probe["gold"]})))
    probes = [probe for probe in probes if probe["id"] not in missing][:args.questions]
    client = server.client()
    load = Load()  # the other programs' load, read before the models load (MA-F20)
    print(load.opening(), flush=True)
    loaded = warm()
    http = TestClient(api.create(client), base_url=f"http://{api.HOST}")
    tokens = {}
    walls, traced, checked = [], [], []
    for probe in probes:
        role = probe["selective"]
        if role not in tokens:
            tokens[role] = http.post("/session", json={"role": role}).json()["token"]
        began = time.perf_counter()
        response = http.post("/search", json={"question": probe["question"]},
                             headers={"Authorization": f"Bearer {tokens[role]}"})
        walls.append(time.perf_counter() - began)
        if response.status_code != 200:
            print(f"{probe['id']}: {response.status_code} {response.text}", flush=True)
            return 1
        body = response.json()
        traced.append(body["seconds"]["trace"])
        checked += gaps(body)
    if missing:
        print(f"{len(missing)} probes with a gold unit the manifest lacks, left out: {', '.join(missing)}; "
              "ntk eval probes --check says why", flush=True)
    rest = walls[1:] or walls
    print(f"the models loaded in {loaded:.1f} s; {len(walls)} dev probes searched through the server as their "
          f"selective readers, {len(tokens)} sessions: the first in {walls[0]:.2f} s, then a median of "
          f"{statistics.median(rest):.2f} s and a 95th percentile of {nearest(rest, 0.95):.2f} s; the trace's own "
          f"queries a median of {statistics.median(traced):.3f} s", flush=True)
    print(f"the fused scores against reciprocal rank fusion of the two searches as traced: {len(checked)} fused units, "
          f"the largest gap {max(checked):.1e}", flush=True)
    print(load.closing(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
