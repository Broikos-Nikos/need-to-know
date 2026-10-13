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
kept, the nearest this index comes to one without them (RS-F28, KB3-K3). The rows are
saved to NTK_DATA_DIR/bench/probe_recall_<split>.json, which the README and N43 are held to.

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--selective-only", action="store_true", help="ask each probe as its selective reader only")
    args = parser.parse_args(argv)
    check_use(args.split, "development" if args.split == "dev" else "baseline")
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
            row = {**measure(probe, role, client, search, excerpts, layer), "across": across}
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
    searches = sum(len(rows) for rows in per_role.values())
    print(f"{searches} searches in {time.perf_counter() - began:.0f} s", flush=True)
    db.close()
    out = data_dir() / "bench" / f"probe_recall_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps({"selective": selective, "per_role": per_role}) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
