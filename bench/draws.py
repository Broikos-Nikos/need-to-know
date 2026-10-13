"""Does retrieval hold on another draw of the corpus (T5.5e2, RS-F6)? Each draw `ntk corpus draw` makes, a data folder
of its own, built as the corpus is built and searched as `ask` searches, on the probes its own plan makes.

    uv run python bench/draws.py [--seeds 20261012,2,3] [--every 4]

For each seed the draw is made unless its folder holds it already, at the same share of the catalogue; then, with
NTK_DATA_DIR on the draw's folder, the ingest builds its manifest, the embedder fills its vector cache, and the
collection is created and synced in Qdrant's local mode, in this process's memory, so nothing reaches the served
index (N62). The base seed's draw is the corpus's own documents, so the draws differ only in their values and annexes.
The draw's plan makes the probes as `ntk eval probes` makes them, the families cut to the facts its documents plant,
so every draw is asked the same questions of the facts they share. Each dev probe with a gold unit is asked as its
selective reader, reranked as configured: the rank of its first gold unit, and whether the excerpts a prompt would
hold carry one. It prints each draw's recall@1, recall@10 and shown share with their Wilson intervals, then the
spread across the draws, and saves the rows to NTK_DATA_DIR/bench/draws.json in the main data folder.

Needs the corpus's plan and narrative, the `embed` extra and the GPU; no LLM call. At one document in four, each draw
is about a third of the corpus's size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from needtoknow.eval.stats import wilson
from needtoknow.settings import data_dir


def rate(hits: int, n: int) -> str:
    """A share with its count and Wilson interval."""
    if not n:
        return "no probes"
    low, high = wilson(hits, n)
    return f"{hits / n:.3f} ({hits} of {n}, Wilson {low:.3f} to {high:.3f})"


def spread(values: list[float]) -> str:
    return f"{min(values):.3f} to {max(values):.3f}"


def made(seed: int, every: int, report) -> dict:
    """The draw's folder made, or found made at the same share of the catalogue; refused when it holds another."""
    from needtoknow.corpus.draw import draw, draw_dir
    folder = draw_dir(seed)
    catalogue = json.loads((data_dir() / "corpus" / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    want = [doc["id"] for doc in catalogue[::every]]
    plan = folder / "corpus" / "plan" / "catalogue.json"
    if plan.is_file():
        have = [doc["id"] for doc in json.loads(plan.read_text(encoding="utf-8"))]
        if have != want:
            raise SystemExit(f"{folder} holds a draw of {len(have)} documents, not one in {every}: remove it to draw "
                             "again")
        report(f"draw {seed}: found, {len(have)} documents")
    else:
        began = time.perf_counter()
        drew = draw(seed, every)
        report(f"draw {seed}: {drew['documents']} documents planting {drew['facts']} facts, {drew['values']} of them "
               f"drawn a new value, in {time.perf_counter() - began:.0f} s")
    return {"folder": folder, "documents": len(want)}


def measure(seed: int, every: int, report) -> dict:
    """One draw built in its own folder and its dev probes asked; its rows."""
    from qdrant_client import QdrantClient

    from needtoknow.embed.cache import fill
    from needtoknow.eval.golden import Context
    from needtoknow.eval.probes import generate
    from needtoknow.eval.split import split_path
    from needtoknow.index.collection import ensure
    from needtoknow.index.expand import excerpts
    from needtoknow.index.search import search
    from needtoknow.index.sync import sync
    from needtoknow.ingest.manifest import build, opened
    from needtoknow.ingest.units import passage

    families = json.loads(split_path().read_text(encoding="utf-8"))
    drawn = made(seed, every, report)
    main = os.environ.get("NTK_DATA_DIR")
    os.environ["NTK_DATA_DIR"] = str(drawn["folder"])
    try:
        began = time.perf_counter()
        units = build()["units_total"]
        report(f"  ingest: {units} units in {time.perf_counter() - began:.0f} s")
        began = time.perf_counter()
        filled = fill()
        report(f"  embed: {filled['embedded']} passages, {filled['tokens']} tokens in {time.perf_counter() - began:.0f} s")
        began = time.perf_counter()
        client = QdrantClient(location=":memory:")
        with opened() as db:
            ensure(client, lambda: (passage(json.loads(p), t) for p, t in db.execute("SELECT path, text FROM units")))
        sync(client)
        report(f"  index, Qdrant's local mode: {time.perf_counter() - began:.0f} s")
        plan = json.loads((drawn["folder"] / "corpus" / "plan" / "facts.json").read_text(encoding="utf-8"))
        held = {fact["id"] for fact in plan}
        cut = {**families, "split": [family | {"facts": [f for f in family["facts"] if f in held]}
                                     for family in families["split"]]}
        dev = generate(Context(split=cut))["dev"]
        # A probe whose planted sentence no unit of the draw holds, or no role reads, is named, never dropped (DR-F23).
        left = [p["id"] for p in dev if not (p["gold"] and p["selective"])]
        if left:
            report(f"  {len(left)} dev probes with no gold unit or no reader, left out: {', '.join(left)}")
        probes = [p for p in dev if p["id"] not in left]
        began = time.perf_counter()
        rows = []
        for probe in probes:
            hits = search(client, probe["question"], probe["selective"])
            gold = set(probe["gold"])
            rank = next((k for k, hit in enumerate(hits, 1) if hit["unit_id"] in gold), None)
            shown = {unit for excerpt in excerpts(hits) for unit in excerpt["units"]}
            rows.append({"id": probe["id"], "fact": probe["fact_id"], "lang": probe["lang"], "rank": rank,
                         "shown": bool(shown & gold)})
        report(f"  {len(rows)} dev probes asked in {time.perf_counter() - began:.0f} s")
        client.close()
    finally:
        if main is None:
            os.environ.pop("NTK_DATA_DIR", None)
        else:
            os.environ["NTK_DATA_DIR"] = main
    return {"seed": seed, "documents": drawn["documents"], "units": units, "facts": len(held), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", default="20261012,2,3")
    parser.add_argument("--every", type=int, default=4)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from needtoknow.eval.protocol import check_use
    check_use("dev", "development")
    report = lambda line: print(line, flush=True)  # noqa: E731
    out = data_dir() / "bench" / "draws.json"
    results = []
    for seed in (int(s) for s in args.seeds.split(",")):
        result = measure(seed, args.every, report)
        rows = result["rows"]
        n = len(rows)
        result["recall@1"] = sum(r["rank"] == 1 for r in rows) / n if n else None
        result["recall@10"] = sum(r["rank"] is not None and r["rank"] <= 10 for r in rows) / n if n else None
        result["shown"] = sum(r["shown"] for r in rows) / n if n else None
        report(f"draw {seed}: {result['documents']} documents, {result['units']} units, {result['facts']} facts; "
               f"{n} dev probes: recall@1 {rate(sum(r['rank'] == 1 for r in rows), n)}; recall@10 "
               f"{rate(sum(r['rank'] is not None and r['rank'] <= 10 for r in rows), n)}; shown "
               f"{rate(sum(r['shown'] for r in rows), n)}")
        results.append(result)
    measured = [r for r in results if r["rows"]]
    if len(measured) > 1:
        report(f"spread over the {len(measured)} draws: recall@1 {spread([r['recall@1'] for r in measured])}; "
               f"recall@10 {spread([r['recall@10'] for r in measured])}; shown {spread([r['shown'] for r in measured])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps({"every": args.every, "draws": results}, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
