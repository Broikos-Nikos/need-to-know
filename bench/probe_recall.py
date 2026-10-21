"""Retrieval recall on the probes (T5.4c, RS-F5, RS-F24): the role's search, judged by the units that hold the fact.

    uv run python bench/probe_recall.py [--split dev] [--selective-only]

For each probe, the search its reader runs, through the role's filter and the configured
reranker, as `answer.pipeline.ask` searches; the rank of its first gold unit among the hits;
recall@1, @5 and @10 and the mean reciprocal rank; and recall at the shown excerpts, the
units `ask` would put in the prompt. Counted once a probe, under its selective reader, the
narrowest view that holds it (RS-F5), and per role beside it, every reader asking, so a
fact every role reads is not counted nine times in the headline. Split by whether the
question is in the language of a gold unit's document or across languages, the path
RS-F24 found failing. Templated questions sit closer to the planted text than a person's,
so these recalls are an upper bound; their worth is in comparisons.

Beside each, what the table units take: their share of the top 10 and of the units in the
shown excerpts, and recall@1 and @10 over the same hits with them set aside, gold units
kept, the nearest this index comes to one without them (RS-F28, KB3-K3).

Then the split's answerable golden items, originals only, each asked as its role, as the
baseline asks them (PLAN section 1, number 3): an item is found at rank k when a unit
holding each of its facts, or its table row, sits among the first k hits, and shown when
the excerpts hold each; by unit, since a row's section is a whole register (RS-F31). The rows
are saved to NTK_DATA_DIR/bench/probe_recall_<split>.json, which the README and N43 are
held to.

Asks the protocol first: dev for development, test for its baseline. Needs the manifest, the
index on the running server, the `embed` extra and the GPU.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict

from machine import named

from needtoknow.eval.probes import load, lost
from needtoknow.eval.protocol import check_use
from needtoknow.ingest.manifest import manifest_path
from needtoknow.settings import data_dir

KS = (1, 5, 10)


def measure(probe: dict, role: str, client, search, excerpts, layer) -> dict:
    hits = search(client, probe["question"], role)
    gold = set(probe["gold"])
    rank = next((k for k, hit in enumerate(hits, 1) if hit["unit_id"] in gold), None)
    shown = [unit for excerpt in excerpts(hits) for unit in excerpt["units"]]
    kinds = layer([hit["unit_id"] for hit in hits] + shown)
    kept = [hit for hit in hits if kinds[hit["unit_id"]] != "table" or hit["unit_id"] in gold]
    aside = next((k for k, hit in enumerate(kept, 1) if hit["unit_id"] in gold), None)
    return {"rank": rank, "shown": bool(set(shown) & gold), "top": min(10, len(hits)),
            "tables": sum(kinds[hit["unit_id"]] == "table" for hit in hits[:10]), "shown_units": len(shown),
            "shown_tables": sum(kinds[unit] == "table" for unit in shown), "aside": aside}


def summarise(rows: list[dict]) -> str:
    n = len(rows)
    recall = {k: sum(r["rank"] is not None and r["rank"] <= k for r in rows) / n for k in KS}
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    shown = sum(r["shown"] for r in rows) / n
    return (f"{n} probes; recall@1 {recall[1]:.3f}, @5 {recall[5]:.3f}, @10 {recall[10]:.3f}; MRR {mrr:.3f}; "
            f"in the shown excerpts {shown:.3f}")


def annex(rows: list[dict]) -> str:
    """What the table units take of the top 10 and of the shown excerpts' units, and the recall without them."""
    n = len(rows)
    top = sum(r["tables"] for r in rows) / sum(r["top"] for r in rows)
    shown = sum(r["shown_tables"] for r in rows) / sum(r["shown_units"] for r in rows)
    aside = {k: sum(r["aside"] is not None and r["aside"] <= k for r in rows) / n for k in (1, 10)}
    return (f"table units {top:.1%} of the top 10 and {shown:.1%} of the shown excerpts' units; set aside, "
            f"recall@1 {aside[1]:.3f} and @10 {aside[10]:.3f}")


def found_at(wanted: list[set], placed: list) -> int | None:
    """The first rank by which every place set wanted holds one of the places hit so far; None when the hits never
    cover them all."""
    for k in range(1, len(placed) + 1):
        if all(places & set(placed[:k]) for places in wanted):
            return k
    return None


def golden(split: str, client, search, excerpts) -> list[dict]:
    """The split's answerable golden items, originals only, each asked as its role: the rank by which the hits hold
    a unit holding every fact it needs, or its table row, and whether the shown excerpts do (PLAN section 1, number
    3). By unit, not section: a row's section is a whole register (RS-F31)."""
    from needtoknow.eval.golden import Context, golden_path, load_items
    from needtoknow.eval.score import ANSWERED, Sources
    sources = Sources(Context())
    rows = []
    for item in load_items(golden_path(split)):
        if item["category"] not in ANSWERED or item.get("twin"):
            continue
        holding = [sources.units_of_row(item["row"])] if item["category"] == "I" else [
            sources.units_of_fact(e["fact_id"]) for e in (item.get("expect") or {}).get("facts") or [] if e.get("fact_id")]
        hits = search(client, item["question"], item["role"])
        shown = {unit for excerpt in excerpts(hits) for unit in excerpt["units"]}
        rows.append({"id": item["id"], "category": item["category"], "facts": len(holding),
                     "rank": found_at(holding, [hit["unit_id"] for hit in hits]) if holding else None,
                     "shown": bool(holding) and all(units & shown for units in holding)})
    return rows


def layers(db: sqlite3.Connection):
    """A lookup of units' layers in the manifest, each unit read once."""
    known: dict[str, str] = {}

    def layer(units: list[str]) -> dict[str, str]:
        wanted = [u for u in dict.fromkeys(units) if u not in known]
        for start in range(0, len(wanted), 900):
            part = wanted[start:start + 900]
            known.update(db.execute(f"SELECT unit_id, layer FROM units WHERE unit_id IN ({','.join('?' * len(part))})",
                                    part))
        return known
    return layer


def reach_of(probes: list[dict], selective: list[dict]) -> dict:
    """The probe arm's reach from a saved run: each family's recall@10 over its probes, asked by their selective
    reader, and the best p the registered test could give a candidate that made every family perfect and none worse
    (RS-F30). A saved row names its probe, or, in a run from before rows did, stands at its probe's place."""
    from needtoknow.eval.stats import reach
    by_id = {probe["id"]: probe for probe in probes}
    families: dict[str, list[float]] = defaultdict(list)
    for probe, row in zip(probes, selective):
        family = by_id[row["id"]]["family"] if "id" in row else probe["family"]
        families[family].append(float(row["rank"] is not None and row["rank"] <= 10))
    means = [sum(hits) / len(hits) for hits in families.values()]
    misses = sum(1 for row in selective if row["rank"] is None or row["rank"] > 10)
    return {"probes": len(selective), "misses": misses, "families": len(means),
            "with_room": sum(mean < 1 for mean in means), "p": reach(means)}


def report(split: str, name: str) -> int:
    """The probe arm's reach, read from the run probe_recall saved for the split under these arguments."""
    path = data_dir() / "bench" / name
    if not path.is_file():
        raise SystemExit(f"no run saved under these arguments, {name}: run the bench with them first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    probes = load(split)
    if len(saved["selective"]) != len(probes) and not all("id" in row for row in saved["selective"]):
        raise SystemExit(f"the saved run has {len(saved['selective'])} rows for {len(probes)} probes and names none: "
                         "run the bench again")
    arm = reach_of(probes, saved["selective"])
    print(f"the probe arm's reach on {split}: {arm['misses']} of {arm['probes']} probes outside the top 10, in "
          f"{arm['with_room']} of {arm['families']} families; every family made perfect and none worse: p "
          f"{arm['p']:.4g} by the registered test")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--selective-only", action="store_true", help="ask each probe as its selective reader only")
    parser.add_argument("--report", action="store_true",
                        help="print the probe arm's reach from the saved run instead of searching (MA-F19)")
    args = parser.parse_args(argv)
    check_use(args.split, "development" if args.split == "dev" else "baseline")
    saved_name = named(f"probe_recall_{args.split}", parser, args, ("selective_only",))
    if args.report:
        return report(args.split, saved_name)
    from needtoknow.index import server
    from needtoknow.index.expand import excerpts
    from needtoknow.index.search import search
    probes = load(args.split)
    db = sqlite3.connect(f"file:{manifest_path().as_posix()}?mode=ro", uri=True)
    language = {}
    for probe in probes:
        for unit in probe["gold"]:
            if unit not in language:
                row = db.execute("SELECT language FROM units WHERE unit_id = ?", (unit,)).fetchone()
                language[unit] = row[0] if row else None
    layer = layers(db)
    client = server.client()
    began = time.perf_counter()
    selective, per_role = [], defaultdict(list)
    missing = lost(probes, {unit for unit, found in language.items() if found is not None})
    for probe in probes:
        if probe["id"] in missing:  # counted apart, never as asked across languages (DR-F25)
            continue
        across = probe["lang"] not in {language[u] for u in probe["gold"]}
        roles = [probe["selective"]] if args.selective_only else probe["readers"]
        for role in roles:
            row = {**measure(probe, role, client, search, excerpts, layer), "across": across, "id": probe["id"]}
            per_role[role].append(row)
            if role == probe["selective"]:
                selective.append(row)
    print(f"{args.split}, each probe once, as its selective reader: {summarise(selective)}", flush=True)
    print(f"  {annex(selective)}", flush=True)
    for name, rows in (("same language", [r for r in selective if not r["across"]]),
                       ("across languages", [r for r in selective if r["across"]])):
        if rows:
            print(f"  {name}: {summarise(rows)}", flush=True)
    if not args.selective_only:
        for role in sorted(per_role):
            print(f"as {role}: {summarise(per_role[role])}", flush=True)
            print(f"  {annex(per_role[role])}", flush=True)
    if missing:
        print(f"{len(missing)} probes with a gold unit the manifest lacks, left out: {', '.join(missing)}; "
              "ntk eval probes --check says why", flush=True)
    items = golden(args.split, client, search, excerpts)
    n = len(items)
    within = {k: sum(r["rank"] is not None and r["rank"] <= k for r in items) for k in KS}
    print(f"{args.split} answerable golden items, originals, each asked as its role: {n} items; a unit holding every "
          f"fact they need within the first 1 for {within[1]}, 5 for {within[5]}, 10 for {within[10]} (recall@10 "
          f"{within[10] / n:.3f}); in the shown excerpts {sum(r['shown'] for r in items)}", flush=True)
    missed = [f"{r['id']} ({r['category']}, rank {r['rank']})" for r in items if r["rank"] is None or r["rank"] > 10]
    if missed:
        print(f"  not within the first 10: {', '.join(missed)}", flush=True)
    searches = sum(len(rows) for rows in per_role.values()) + n
    print(f"{searches} searches in {time.perf_counter() - began:.0f} s", flush=True)
    db.close()
    out = data_dir() / "bench" / saved_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps({"selective": selective, "per_role": per_role, "items": items}) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
