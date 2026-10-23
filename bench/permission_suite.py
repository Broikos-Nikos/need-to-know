"""The permission suite at retrieval level (T3.18, T5.4e, `specs/evaluation.md` section 6, items 1 to 3): no role gets
a unit it may not read, from either search, their fusion or the excerpts a prompt would hold, and the filter hides
nothing a role may read.

    uv run python bench/permission_suite.py [--per-label 20] [--seed 75] [--prefixes]

Needs the manifest, the vector cache, the embed extra, the GPU and the index on the running server; writes nothing.
Probes: for each label, up to `per-label` of the committed dev probes whose gold units carry it, drawn by the seed,
questions worded apart from the text as a person's would be (RS-F16); a probe with a gold unit the manifest no
longer holds is set apart and named (MT-F16). With `--prefixes`, as at tick 75, the first 30 words of up to
`per-label` units a label instead, half prose and half tables where it has both. Cases: every role as
`config/policy.toml` defines it; a user holding two roles, sales and hr, whose filter is the union; a user with only
`general`; and sales with a membership removed from the policy in memory, `returns`, asked again with no index
change. Every probe is asked as every case three ways, each 100 candidates deep as the retrieval API returns them
before any reranking: the dense search alone, the sparse search alone, and their fusion, which is what the reranker
reads; and once more as `ask` searches, reranked, its excerpts built into the prompt they would make, each excerpt's
units checked and the prompt searched for the canaries of labels the case may not read. A unit the case may not
read is foreign wherever it appears, and the suite exits 1 on any. For the cases that may read a probe's gold
units, its availability: how often one comes back in the fused top 10 and top 100, and in the excerpts.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict

from machine import commit

from needtoknow import canaries
from needtoknow.answer.prompt import build, reader_mark
from needtoknow.embed import model
from needtoknow.index import search as retrieval
from needtoknow.index import server
from needtoknow.index.expand import excerpts
from needtoknow.ingest.manifest import opened
from needtoknow.policy import load_policy, readable

MODES = {"dense": ("dense",), "sparse": ("sparse",), "fused": ("dense", "sparse")}
DEPTH = 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--prefixes", action="store_true", help="units' first 30 words for probes, as at tick 75")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(commit(), flush=True)  # the code the run used, for a record taken from its log (MA-F33)
    began = time.perf_counter()
    rng = random.Random(args.seed)
    with opened() as db:
        labels_of = {u: frozenset(json.loads(l)) for u, l in db.execute("SELECT unit_id, labels FROM units")}
        if args.prefixes:
            pools = defaultdict(lambda: {"prose": [], "table": []})
            for unit, labels, layer in db.execute("SELECT unit_id, labels, layer FROM units ORDER BY unit_id"):
                for label in json.loads(labels):
                    pools[label][layer].append(unit)
            probes = []
            for label in sorted(pools):
                prose, table = pools[label]["prose"], pools[label]["table"]
                take = min(len(prose), args.per_label // 2)
                picked = rng.sample(prose, take) + rng.sample(table, min(len(table), args.per_label - take))
                probes += [(label, " ".join(db.execute("SELECT text FROM units WHERE unit_id = ?", (unit,))
                                            .fetchone()[0].split()[:30]), frozenset({unit})) for unit in picked]
    if not args.prefixes:
        from needtoknow.eval.probes import load as load_probes
        from needtoknow.eval.probes import lost
        from needtoknow.eval.protocol import check_use
        check_use("dev", "development")
        asked = load_probes("dev")
        missing = lost(asked, labels_of)
        if missing:
            print(f"{len(missing)} probes with a gold unit the manifest lacks, left out: {', '.join(missing)}; "
                  "ntk eval probes --check says why", flush=True)
        pools = defaultdict(list)
        for probe in asked:
            if probe["id"] not in missing:
                for label in sorted(frozenset().union(*(labels_of[u] for u in probe["gold"]))):
                    pools[label].append(probe)
        probes = []
        for label in sorted(pools):
            probes += [(label, p["question"], frozenset(p["gold"]))
                       for p in rng.sample(pools[label], min(len(pools[label]), args.per_label))]
    texts = sorted({text for _, text, _ in probes})
    vectors = dict(zip(texts, model.encode(texts)))
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
    kind = "units' first 30 words" if args.prefixes else "dev probes' questions"
    print(f"{len(labels_of)} units; {len(probes)} probes, {kind}, over {len(pools)} labels; {len(cases)} cases, each "
          f"asked {len(MODES)} ways {DEPTH} deep and once as ask asks", flush=True)
    client = server.client()
    sets = retrieval.label_sets(client)
    retrieval.label_sets = lambda _client: sets
    names = canaries.suffixes()
    foreign_total = 0
    for name, role, case_policy in cases:
        reads = readable(case_policy, role)
        foreign = defaultdict(int)
        top10 = top100 = shown_gold = readers = 0
        for _, text, gold in probes:
            may_read = any(labels_of[u] & reads for u in gold)
            for mode, sides in MODES.items():
                hits = retrieval.search(client, text, role, case_policy, limit=DEPTH, sides=sides, rerank="none",
                                        encode=lambda _texts, v=vectors[text]: v[None, :])
                foreign[mode] += sum(1 for h in hits if not labels_of.get(h["unit_id"], frozenset()) & reads)
                if mode == "fused" and may_read:
                    ranked = [h["unit_id"] for h in hits]
                    readers += 1
                    top10 += bool(gold & set(ranked[:10]))
                    top100 += bool(gold & set(ranked))
            # As ask searches: the configured reranker, then the excerpts the prompt would hold.
            shown = excerpts(retrieval.search(client, text, role, case_policy,
                                              encode=lambda _texts, v=vectors[text]: v[None, :]))
            units = [u for excerpt in shown for u in excerpt["units"]]
            foreign["excerpts"] += sum(1 for u in units if not labels_of.get(u, frozenset()) & reads)
            foreign["canaries"] += len(canaries.found(build(text, shown, reader=reader_mark(reads))[1], reads, names))
            shown_gold += may_read and bool(gold & set(units))
        foreign_total += sum(foreign.values())
        availability = (f"a gold unit in the fused top 10 for {top10} of {readers}, top 100 for {top100}, the "
                        f"excerpts for {shown_gold}" if readers else "reads none of the probes")
        print(f"  {name:<24} foreign units: dense {foreign['dense']}, sparse {foreign['sparse']}, fused "
              f"{foreign['fused']}, excerpts {foreign['excerpts']}; canaries in the prompt {foreign['canaries']}; "
              f"{availability}", flush=True)
    searches = len(cases) * len(probes) * (len(MODES) + 1)
    print(f"{searches} searches in {time.perf_counter() - began:.0f} s; foreign units and canaries in all: "
          f"{foreign_total}", flush=True)
    return 1 if foreign_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
