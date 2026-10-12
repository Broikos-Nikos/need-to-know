"""The filter bench (T3.8): does the engine's filtered search find what an exact scan finds, for every filter a role
sends, and two no role sends?

    uv run python bench/filter_bench.py [--per-label 20] [--seed 38] [--efs 64,128,256,512] [--report]

Needs the manifest, the vector cache, the index on the running server, the embed extra
and the GPU; writes its numbers to NTK_DATA_DIR/bench/filter_bench.json and prints them.

Probes: for each label, up to `per-label` prose units and as many table units, drawn by
the seed; a probe asks the first 30 words of its unit, encoded as a question is. They
split in two by the seed, dev and test. For each case of `policy.bench_cases()`, every
role's filter exactly as sent plus the bench only views (N31), a probe is in domain when
the case may read its unit and off topic when not, the case the base saw collapse (M7).

For each probe, case and filter form (`labels`, `label_set`, KB1-F3), at each ef and
once more through the exact path the search takes when no ef is named (N51), the
engine's dense top 10, through the retrieval API, is set against the exact top 10 over
the float32 vectors of the units the case may read, from the vector cache, scanned on
the GPU outside the engine. Overlap@10 is reported raw and tie aware (RS-F11): an exact
neighbour the engine missed counts as a miss only when its score passes the engine's
tenth by more than 0.001, so a swap among near equal scores is not one. On dev probes,
each case and form takes the first of the efs, then the exact path, whose tie aware
mean is 0.97 or more in both classes; test probes report it, with the median seconds of
a search, beside the graph's dev mean at the largest ef. Every unit the engine returns
that the case may not read is counted (it must be 0), and for prose probes the share of
table units in the exact path's top 10, KB3-K3's annex intrusion. `--report` prints the
last run's report again from its saved numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict

import numpy as np

from needtoknow.embed import cache, model
from needtoknow.index import search as retrieval
from needtoknow.index import server
from needtoknow.ingest.manifest import opened
from needtoknow.ingest.units import passage
from needtoknow.policy import bench_cases, load_policy
from needtoknow.settings import data_dir

TOLERANCE = 0.001
TARGET = 0.97


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--efs", default="64,128,256,512")
    parser.add_argument("--report", action="store_true",
                        help="print the last run's report again, from filter_bench.json, without searching")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = data_dir() / "bench" / "filter_bench.json"
    if args.report:
        results = json.loads(out.read_bytes())
        report(results, sorted({r["ef"] for r in results if r["ef"] is not None}) + [None])
        return 0
    import torch

    efs = [int(e) for e in args.efs.split(",")] + [None]  # None: the exact path, the search's default (N51)
    began = time.perf_counter()
    with opened() as db:
        rows = db.execute("SELECT unit_id, labels, path, text, layer FROM units ORDER BY unit_id").fetchall()
    ids = [r[0] for r in rows]
    labels = [frozenset(json.loads(r[1])) for r in rows]
    layer = [r[4] for r in rows]
    keys = [cache.passage_key(passage(json.loads(r[2]), r[3])) for r in rows]
    with cache.opened() as db:
        vectors = np.vstack([cache.read(db, keys[start:start + 20000]) for start in range(0, len(keys), 20000)])
    matrix = torch.from_numpy(vectors).to("cuda")
    del vectors
    print(f"{len(ids)} units and their float32 vectors on the GPU in {time.perf_counter() - began:.0f} s", flush=True)

    rng = random.Random(args.seed)
    by_label = defaultdict(lambda: {"prose": [], "table": []})
    for index, (unit_labels, kind) in enumerate(zip(labels, layer)):
        for label in unit_labels:
            by_label[label][kind].append(index)
    probes = []
    for label in sorted(by_label):
        for kind in ("prose", "table"):
            pool = by_label[label][kind]
            probes += [(i, kind) for i in rng.sample(pool, min(args.per_label, len(pool)))]
    rng.shuffle(probes)
    split = {i: ("dev" if n % 2 == 0 else "test") for n, (i, _) in enumerate(probes)}
    texts = [" ".join(rows[i][3].split()[:30]) for i, _ in probes]
    queries = model.encode(texts)
    print(f"{len(probes)} probes, {sum(1 for v in split.values() if v == 'dev')} dev", flush=True)

    policy = load_policy()
    cases = bench_cases(policy)
    views = {"roles": {**policy["roles"], **{name: {"reads": sorted(reads)} for name, reads, only in cases if only}}}
    sets = retrieval.label_sets(server.client())
    retrieval.label_sets = lambda client: sets  # one facet for the whole bench, not one per question
    client = server.client()
    index_of = {unit: n for n, unit in enumerate(ids)}
    results = []
    for case, reads, bench_only in cases:
        allowed = torch.tensor([bool(unit_labels & reads) for unit_labels in labels], device="cuda")
        for n, ((unit, kind), query) in enumerate(zip(probes, queries)):
            q = torch.from_numpy(query).to("cuda")
            scores = matrix @ q
            exact_scores = torch.where(allowed, scores, torch.tensor(-2.0, device="cuda"))
            top = torch.topk(exact_scores, 10)
            exact = [ids[i] for i in top.indices.tolist()]
            tenth_exact = top.values[-1].item()
            domain = "in domain" if labels[unit] & reads else "off topic"
            for form in retrieval.FORMS:
                for ef in efs:
                    asked = time.perf_counter()
                    hits = retrieval.search(client, texts[n], case, views, form=form, limit=10, ef=ef,
                                            encode=lambda _texts, q=query: q[None, :], sides=("dense",), rerank="none")
                    seconds = time.perf_counter() - asked
                    found = [h["unit_id"] for h in hits]
                    found_scores = [scores[index_of[u]].item() for u in found]
                    floor = min(found_scores) if len(found_scores) == 10 else -2.0  # fewer than 10: every gap a miss
                    missed = [u for u, s in zip(exact, top.values.tolist()) if u not in found]
                    real_misses = [u for u in missed if scores[index_of[u]].item() > floor + TOLERANCE]
                    results.append({
                        "case": case, "bench_only": bench_only, "form": form, "ef": ef, "split": split[unit],
                        "class": domain, "kind": kind, "overlap": 1 - len(missed) / 10,
                        "tie_aware": 1 - len(real_misses) / 10, "returned": len(found),
                        "foreign": sum(1 for u in found if not labels[index_of[u]] & reads),
                        "tables": sum(1 for u in found if layer[index_of[u]] == "table"), "tenth": tenth_exact,
                        "seconds": seconds})
        print(f"  {case}: done, {time.perf_counter() - began:.0f} s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(results) + "\n").encode("utf-8"))
    report(results, efs)
    return 0


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def reaches(dev: list[dict]) -> bool:
    """Whether the dev probes' tie aware mean is TARGET or more in each class they hold."""
    return bool(dev) and all(mean([r for r in dev if r["class"] == c], "tie_aware") >= TARGET
                             for c in ("in domain", "off topic") if any(r["class"] == c for r in dev))


def report(results: list[dict], efs: list[int | None]) -> None:
    foreign = sum(r["foreign"] for r in results)
    print(f"searches {len(results)}; units returned that the case may not read: {foreign}")
    prose = [r for r in results if r["kind"] == "prose" and r["class"] == "in domain" and r["ef"] is None]
    with opened() as db:
        share = db.execute("SELECT AVG(layer = 'table') FROM units").fetchone()[0]
    print(f"annex intrusion, prose probes in domain through the exact path: table units {mean(prose, 'tables') / 10:.1%} "
          f"of the top 10, against {share:.1%} of all units")
    for form in ("labels", "label_set"):
        print(f"form {form}: case, the first of the efs then the exact path to reach {TARGET} on dev; on test, tie "
              f"aware overlap@10 in domain and off topic (raw in brackets), lowest probe, median search; the graph's "
              f"dev mean at ef {efs[-2]}")
        for case in dict.fromkeys(r["case"] for r in results):
            rows = [r for r in results if r["case"] == case and r["form"] == form]
            reached = [ef for ef in efs if reaches([r for r in rows if r["ef"] == ef and r["split"] == "dev"])]
            shown = reached[0] if reached else efs[-1]
            test = [r for r in rows if r["ef"] == shown and r["split"] == "test"]
            cells = []
            for c in ("in domain", "off topic"):
                part = [r for r in test if r["class"] == c]
                cells.append(f"{c} {mean(part, 'tie_aware'):.3f} ({mean(part, 'overlap'):.3f}) n {len(part)} "
                             f"min {min((r['tie_aware'] for r in part), default=float('nan')):.1f}")
            took = sorted(r["seconds"] for r in test)
            graph = [r for r in rows if r["ef"] == efs[-2] and r["split"] == "dev"]
            name = ("exact" if shown is None else f"ef {shown}") if reached else f"none reached {TARGET}, exact"
            flag = " bench only" if rows and rows[0]["bench_only"] else ""
            print(f"  {case:<16}{flag:<11} {name}: " + "; ".join(cells) + f"; median {took[len(took) // 2] * 1000:.0f} "
                  f"ms; graph {mean(graph, 'tie_aware'):.3f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
