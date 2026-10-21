"""Do units a role may not read move the order of its tied units on the index's server (AC-F12)? A collection the
bench names, `ties_bench`, made as the index's own is made, holds filler units of every label and units of sales that
tie in both searches. Units of legal that would tie with them are written into it, deleted again, and then written in
bulk, as a served index meets them, and in each state the role's searches are asked as the served query asks them.

    uv run python bench/ties.py [--tied 150] [--filler 30000] [--calls 3]

Needs the running server (`ntk qdrant start`) and nothing else: the vectors are made here, not encoded, so neither the
GPU nor the corpus is read, and the index's own collection is never asked anything. N50 says why it matters: units of
equal score are put in order by the engine, and the fusion ranks each search's units by that order, so an order that
units the role may not read could move would move the role's fused list too. It prints, for each state, whether the
role's dense, lexical and fused lists are the ones it had alone, then whether a build again of the very same units, in
the very same order, gives the role the order it had in the first; a measurement, it exits 0 either way.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from needtoknow.index import server
from needtoknow.index.collection import DENSE, DIMENSIONS, SPARSE, config, indexes, payload, point_id
from needtoknow.index.search import CANDIDATES, RRF_K, TIES, role_filter
from needtoknow.policy import load_policy, readable

NAME = "ties_bench"
ROLE, HIDDEN = "sales", "legal"
TERMS = [11, 22, 33]  # the tied units' lexical terms, which no filler unit holds
BATCH = 500


def unit(kind: str, n: int, labels: list[str], dense: list[float], sparse: tuple[list[int], list[float]]) -> dict:
    unit_id = f"{kind}-{n:05d}"
    return {"id": point_id(unit_id), "dense": dense, "sparse": sparse,
            "payload": payload(unit_id, f"BENCH_{kind.upper()}", "key", "prose", labels, ["Ties"], "en")}


def filler(count: int, labels: list[str], seed: int, kind: str = "filler"):
    """Units of the given labels in turn, their vectors random and far from the tied units', made a batch at a time."""
    rng = np.random.default_rng(seed)
    for start in range(0, count, BATCH):
        dense = rng.standard_normal((min(BATCH, count - start), DIMENSIONS)).astype(np.float32)
        dense /= np.linalg.norm(dense, axis=1, keepdims=True)
        for n, vector in enumerate(dense, start):
            terms = sorted(rng.choice(np.arange(1000, 100_000), size=20, replace=False).tolist())
            yield unit(kind, n, [labels[n % len(labels)]], vector.tolist(), (terms, rng.random(20).tolist()))


def written(client, units) -> None:
    """The units upserted in batches, then the server left to finish whatever optimization they started."""
    from qdrant_client import models

    units, part = iter(units), []
    while True:
        part = [u for _, u in zip(range(BATCH), units)]
        if not part:
            break
        client.upsert(NAME, wait=True, points=models.Batch(
            ids=[u["id"] for u in part], vectors={DENSE: [u["dense"] for u in part], SPARSE: [
                models.SparseVector(indices=u["sparse"][0], values=u["sparse"][1]) for u in part]},
            payloads=[u["payload"] for u in part]))
    settled(client)


def settled(client, wait: float = 600.0) -> None:
    """Until the server's optimizers have nothing left to do on the collection."""
    from qdrant_client import models

    began = time.monotonic()
    while client.get_collection(NAME).status != models.CollectionStatus.GREEN:
        if time.monotonic() - began > wait:
            raise SystemExit(f"{NAME} still optimizing after {wait:.0f} s")
        time.sleep(0.5)


def made(client, *groups) -> None:
    """The bench's collection made afresh as the index's own is made, its units written group by group, in order."""
    if client.collection_exists(NAME):
        client.delete_collection(NAME)
    vectors, sparse = config()
    client.create_collection(NAME, vectors_config=vectors, sparse_vectors_config=sparse)
    for field, schema in indexes().items():
        client.create_payload_index(NAME, field, field_schema=schema)
    for group in groups:
        written(client, group)


def lists(client, flt, question: list[float]) -> dict:
    """The role's searches as the served query asks them: each search alone in the engine's order, and the fused list
    as the role is shown it, ordered by score and then by unit id and cut at the limit (RS-F29)."""
    from qdrant_client import models

    prefetch = [models.Prefetch(query=question, using=DENSE, filter=flt, limit=CANDIDATES,
                                params=models.SearchParams(exact=True)),
                models.Prefetch(query=models.SparseVector(indices=TERMS, values=[1.0] * len(TERMS)), using=SPARSE,
                                filter=flt, limit=CANDIDATES)]
    apart = client.query_batch_points(NAME, requests=[models.QueryRequest(
        query=p.query, using=p.using, filter=p.filter, params=p.params, limit=p.limit, with_payload=["unit_id"])
        for p in prefetch])
    fused = client.query_points(NAME, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=RRF_K + 1)),
                                query_filter=flt, limit=CANDIDATES + TIES, with_payload=["unit_id"]).points
    shown = sorted(fused, key=lambda point: (-point.score, point.payload["unit_id"]))[:CANDIDATES]
    return {"dense": [p.payload["unit_id"] for p in apart[0].points],
            "lexical": [p.payload["unit_id"] for p in apart[1].points],
            "fused": [(p.payload["unit_id"], round(p.score, 9)) for p in shown]}


def asked(client, flt, question: list[float], calls: int) -> dict:
    """The role's lists, asked `calls` times; they must agree with one another before a state can be compared."""
    seen = [lists(client, flt, question) for _ in range(calls)]
    if any(other != seen[0] for other in seen[1:]):
        moved = [side for side in seen[0] if any(other[side] != seen[0][side] for other in seen[1:])]
        raise SystemExit(f"the role's {', '.join(moved)} lists moved between calls of one state, so no state can be "
                         f"compared with another")
    return seen[0]


def compared(state: str, found: dict, alone: dict) -> bool:
    same = {side: found[side] == alone[side] for side in alone}
    print(f"{state}: " + ", ".join(f"{side} {'the same' if same[side] else 'MOVED'}" for side in alone), flush=True)
    return all(same.values())


def main() -> int:
    from qdrant_client import models

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tied", type=int, default=150, help="units of each label that tie, more than the limit")
    parser.add_argument("--filler", type=int, default=30_000, help="units of every label around them")
    parser.add_argument("--calls", type=int, default=3, help="times each state's searches are asked")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    policy = load_policy()
    labels = sorted(readable(policy, "admin"))
    rng = np.random.default_rng(20261021)
    question = rng.standard_normal(DIMENSIONS).astype(np.float32)
    question = (question / np.linalg.norm(question)).tolist()
    tied = (TERMS, [1.0] * len(TERMS))
    role = [unit(ROLE, n, [ROLE], question, tied) for n in range(args.tied)]
    hidden = [unit(HIDDEN, n, [HIDDEN], question, tied) for n in range(args.tied)]
    client, flt = server.client(), role_filter(policy, ROLE)
    began = time.perf_counter()
    print(f"{NAME}: {args.filler:,} filler units over {len(labels)} labels; {args.tied} units of {ROLE} tied in both "
          f"searches, and {args.tied} of {HIDDEN} that would tie with them; a limit of {CANDIDATES}, {args.calls} calls a "
          f"state", flush=True)
    try:
        made(client, filler(args.filler, labels, 1), role)
        alone = asked(client, flt, question, args.calls)
        tops = {side: len(set(alone[side]) & {u["payload"]["unit_id"] for u in role}) for side in ("dense", "lexical")}
        print(f"alone: {tops['dense']} of the {args.tied} tied units in the dense list and {tops['lexical']} in the "
              f"lexical, every call alike; the fused scores {alone['fused'][-1][1]} to {alone['fused'][0][1]}", flush=True)
        # In one build, as the served index lives: units the role may not read written, deleted, and written in bulk.
        written(client, hidden)
        held = [compared(f"{args.tied} of {HIDDEN} written after", asked(client, flt, question, args.calls), alone)]
        client.delete(NAME, points_selector=models.PointIdsList(points=[u["id"] for u in hidden]), wait=True)
        settled(client)
        held.append(compared(f"those {args.tied} deleted again", asked(client, flt, question, args.calls), alone))
        written(client, filler(args.filler // 2, [HIDDEN], 2, "bulk"))
        held.append(compared(f"{args.filler // 2:,} more of {HIDDEN} written after, the optimizer left to settle",
                             asked(client, flt, question, args.calls), alone))
        # And a build again of the very same units in the very same order, against which no hidden unit can be told.
        made(client, filler(args.filler, labels, 1), role)
        rebuilt = compared("built again, the same units in the same order", asked(client, flt, question, args.calls), alone)
    finally:
        if client.collection_exists(NAME):
            client.delete_collection(NAME)
    print(f"in one build, the role's tied units as it had them alone in {sum(held)} of {len(held)} states; built again "
          f"from the same units, its order {'the same' if rebuilt else 'moved'}; {time.perf_counter() - began:.0f} s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
