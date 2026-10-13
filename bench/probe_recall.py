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

Asks the protocol first: dev for development, test for its baseline. Needs the manifest, the
index on the running server, the `embed` extra and the GPU.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict

from needtoknow.eval.probes import load
from needtoknow.eval.protocol import check_use
from needtoknow.ingest.manifest import manifest_path

KS = (1, 5, 10)


def measure(probe: dict, role: str, client, search, excerpts) -> dict:
    hits = search(client, probe["question"], role)
    gold = set(probe["gold"])
    rank = next((k for k, hit in enumerate(hits, 1) if hit["unit_id"] in gold), None)
    shown = {unit for excerpt in excerpts(hits) for unit in excerpt["units"]}
    return {"rank": rank, "shown": bool(shown & gold)}


def lost(probes: list[dict], language: dict) -> list[str]:
    """The probes with a gold unit the manifest no longer holds, whose language is then unknown (DR-F25)."""
    return [probe["id"] for probe in probes if any(language.get(u) is None for u in probe["gold"])]


def summarise(rows: list[dict]) -> str:
    n = len(rows)
    recall = {k: sum(r["rank"] is not None and r["rank"] <= k for r in rows) / n for k in KS}
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    shown = sum(r["shown"] for r in rows) / n
    return (f"{n} probes; recall@1 {recall[1]:.3f}, @5 {recall[5]:.3f}, @10 {recall[10]:.3f}; MRR {mrr:.3f}; "
            f"in the shown excerpts {shown:.3f}")


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
    db.close()
    client = server.client()
    began = time.perf_counter()
    selective, per_role, missing = [], defaultdict(list), lost(probes, language)
    for probe in probes:
        if probe["id"] in missing:  # counted apart, never as asked across languages (DR-F25)
            continue
        across = probe["lang"] not in {language[u] for u in probe["gold"]}
        roles = [probe["selective"]] if args.selective_only else probe["readers"]
        for role in roles:
            row = {**measure(probe, role, client, search, excerpts), "across": across}
            per_role[role].append(row)
            if role == probe["selective"]:
                selective.append(row)
    print(f"{args.split}, each probe once, as its selective reader: {summarise(selective)}", flush=True)
    for name, rows in (("same language", [r for r in selective if not r["across"]]),
                       ("across languages", [r for r in selective if r["across"]])):
        if rows:
            print(f"  {name}: {summarise(rows)}", flush=True)
    if not args.selective_only:
        for role in sorted(per_role):
            print(f"as {role}: {summarise(per_role[role])}", flush=True)
    if missing:
        print(f"{len(missing)} probes with a gold unit the manifest lacks, left out: {', '.join(missing)}; "
              "ntk eval probes --check says why", flush=True)
    searches = sum(len(rows) for rows in per_role.values())
    print(f"{searches} searches in {time.perf_counter() - began:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
