"""The permission suite at retrieval level (T3.18, `specs/evaluation.md` section 6, items 1 to 3): no role gets a
unit it may not read, from either search or their fusion, and the filter hides nothing a role may read.

    uv run python bench/permission_suite.py [--per-label 20] [--seed 75]

Needs the manifest, the vector cache, the embed extra and the index on the running
server; writes nothing. Probes: for each label, up to `per-label` units, half prose and
half tables where it has both, drawn by the seed; a probe asks the first 30 words of its
unit. Cases: every role as `config/policy.toml` defines it; a user holding two roles,
sales and hr, whose filter is the union; a user with only `general`; and sales with a
membership removed from the policy in memory, `returns`, asked again with no index
change. Every probe is asked as every case three ways, each 100 candidates deep, as the
retrieval API returns them before any reranking: the dense search alone, the sparse
search alone, and their fusion, which is what the reranker reads. A unit the case may
not read is foreign wherever it appears, and the suite exits 1 on any. For the cases
that may read a probe's unit, its availability: how often the unit itself comes back in
the fused top 10 and top 100, so the filter is not passing by hiding everything.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict

from needtoknow.embed import model
from needtoknow.index import search as retrieval
from needtoknow.index import server
from needtoknow.ingest.manifest import opened
from needtoknow.policy import load_policy, readable

MODES = {"dense": ("dense",), "sparse": ("sparse",), "fused": ("dense", "sparse")}
DEPTH = 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=75)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    began = time.perf_counter()
    with opened() as db:
        labels_of = {u: frozenset(json.loads(l)) for u, l in db.execute("SELECT unit_id, labels FROM units")}
        pools = defaultdict(lambda: {"prose": [], "table": []})
        for unit, labels, layer in db.execute("SELECT unit_id, labels, layer FROM units ORDER BY unit_id"):
            for label in json.loads(labels):
                pools[label][layer].append(unit)
        rng = random.Random(args.seed)
        probes = []
        for label in sorted(pools):
            prose, table = pools[label]["prose"], pools[label]["table"]
            take = min(len(prose), args.per_label // 2)
            picked = rng.sample(prose, take) + rng.sample(table, min(len(table), args.per_label - take))
            probes += [(label, unit) for unit in picked]
        texts = {u: " ".join(db.execute("SELECT text FROM units WHERE unit_id = ?", (u,)).fetchone()[0].split()[:30])
                 for _, u in probes}
    vectors = dict(zip([u for _, u in probes], model.encode([texts[u] for _, u in probes])))
    policy = load_policy()
    views = copy.deepcopy(policy)
    union = readable(policy, "sales") | readable(policy, "hr")
    views["roles"]["sales and hr"] = {"reads": sorted(union)}
    views["roles"]["general alone"] = {"reads": ["general"]}
    removed = copy.deepcopy(policy)
    removed["roles"]["sales"]["reads"] = [r for r in removed["roles"]["sales"]["reads"] if r != "returns"]
    cases = [(role, role, views) for role in policy["roles"]] + [
        ("sales and hr", "sales and hr", views), ("general alone", "general alone", views),
        ("sales, returns removed", "sales", removed)]
    print(f"{len(labels_of)} units; {len(probes)} probes over {len(pools)} labels; {len(cases)} cases, each asked "
          f"{len(MODES)} ways {DEPTH} deep", flush=True)
    client = server.client()
    sets = retrieval.label_sets(client)
    retrieval.label_sets = lambda _client: sets
    foreign_total = 0
    for name, role, case_policy in cases:
        reads = readable(case_policy, role)
        foreign, asked, top10, top100, readers = defaultdict(int), 0, 0, 0, 0
        for label, unit in probes:
            may_read = bool(labels_of[unit] & reads)
            for mode, sides in MODES.items():
                hits = retrieval.search(client, texts[unit], role, case_policy, limit=DEPTH, sides=sides, rerank="none",
                                        encode=lambda _texts, v=vectors[unit]: v[None, :])
                foreign[mode] += sum(1 for h in hits if not labels_of.get(h["unit_id"], frozenset()) & reads)
                if mode == "fused" and may_read:
                    ranked = [h["unit_id"] for h in hits]
                    readers += 1
                    top10 += unit in ranked[:10]
                    top100 += unit in ranked
            asked += 1
        foreign_total += sum(foreign.values())
        availability = (f"its own unit in the fused top 10 for {top10} of {readers}, top 100 for {top100}"
                        if readers else "reads none of the probes")
        print(f"  {name:<24} foreign units: dense {foreign['dense']}, sparse {foreign['sparse']}, fused "
              f"{foreign['fused']}; {availability}", flush=True)
    print(f"{len(cases) * len(probes) * len(MODES)} searches in {time.perf_counter() - began:.0f} s; foreign units in "
          f"all: {foreign_total}", flush=True)
    return 1 if foreign_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
