"""P4's exit, live: ten questions about table rows answered and cited, three refused, and the spend logged.

    uv run python bench/smoke_answers.py [--seed 95] [--refusals 3] [--no-cache]

Needs the manifest, the index on the running server, the `embed` extra and a DeepSeek
key. The questions come from table rows, never from planted facts, so no family of
either evaluation split is read (eval/protocol.toml). Label by label, it draws a table
unit by the seed, one of its rows whose key no other unit holds, and a column whose value
appears once in the unit, and asks for that cell by the row's key and the document's
title, in the unit's language, as the role that may read the label with the fewest labels
besides; labels whose keys repeat across documents give their turn to the others, until
ten. A pass is an answered question whose citations include the excerpt holding the row,
whose answer holds the value, and from which nothing was dropped (RS-F14). Then the first
`--refusals` questions whose rows some role may not read are asked again as that role: a
pass is `not_found`, the row's unit among none of the excerpts shown, nothing cited and
the value nowhere in the answer. Everything goes through `answer.pipeline.ask`, its cache
and its spend log; `--no-cache` asks the provider every question again. The bench prints
each question's verdict, the spend log's totals before and after, and exits 1 on any
failure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from needtoknow.answer import pipeline
from needtoknow.index import server
from needtoknow.ingest.manifest import opened
from needtoknow.policy import load_policy, readable

QUESTIONS = {"en": "In {title}, what is the {column} of {key}?",
             "el": "Στο έγγραφο «{title}», ποιο είναι το πεδίο «{column}» για το {key};"}


def normal(text: str) -> str:
    return " ".join(text.lower().split())


def draw(seed: int) -> list[dict]:
    """One row a label, by the seed: the unit, its row's key, a column, its value and the question."""
    rng = random.Random(seed)
    picked = []
    with opened() as db:
        labels = sorted({label for (row,) in db.execute("SELECT DISTINCT labels FROM units WHERE layer = 'table'")
                         for label in json.loads(row)})
        dry: set[str] = set()  # labels none of whose sampled units held a row with a key of its own
        for label in labels * 4:  # rows from the labels whose keys are their own, until ten
            if len(picked) == 10:
                break
            if label in dry:
                continue
            found = len(picked)
            ids = [u for (u,) in db.execute("SELECT unit_id FROM units WHERE layer = 'table' AND labels = ? "
                                            "ORDER BY unit_id", (json.dumps([label]),))]
            for unit in rng.sample(ids, min(len(ids), 40)):
                doc_id, unit_labels, path, text, lang = db.execute(
                    "SELECT doc_id, labels, path, text, language FROM units WHERE unit_id = ?", (unit,)).fetchone()
                lines = [line.split(" | ") for line in text.split("\n") if " | " in line]
                header, rows = lines[0], [r for r in lines[1:] if len(r) == len(lines[0])]
                if not rows:
                    continue
                row = rng.choice(rows)
                # A key in more than one unit makes the question ambiguous: the registers repeat across documents,
                # and at the first run CC781 sat in 358 units, a date key in 386.
                if db.execute("SELECT COUNT(*) FROM units WHERE instr(text, ?) > 0", (row[0],)).fetchone()[0] != 1:
                    continue
                columns = [k for k in range(1, len(header)) if text.count(row[k]) == 1 and len(row[k]) >= 3]
                if not columns:
                    continue
                k = rng.choice(columns)
                title = json.loads(path)[0]
                picked.append({"unit": unit, "doc_id": doc_id, "labels": json.loads(unit_labels), "key": row[0],
                               "value": row[k], "language": lang,
                               "question": QUESTIONS[lang].format(title=title, column=header[k], key=row[0])})
                break
            if len(picked) == found:
                dry.add(label)
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=95)
    parser.add_argument("--refusals", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true", help="ask the provider every question again")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    policy = load_policy()
    roles = {role: readable(policy, role) for role in policy["roles"] if role != "admin"}
    before = pipeline.spend()
    client = server.client()
    failed = 0
    smoke = draw(args.seed)
    for item in smoke:
        reader = min((r for r, reads in roles.items() if set(item["labels"]) & reads), key=lambda r: len(roles[r]))
        result = pipeline.ask(client, item["question"], reader, policy, use_cache=not args.no_cache)
        holding = {(e["doc_id"], e["section"]) for e in result["excerpts"] if item["unit"] in e["units"]}
        cited = {(c["doc_id"], c["section"]) for c in result["citations"]}
        good = (result["status"] == "answered" and bool(holding & cited) and not result["dropped"]
                and normal(item["value"]) in normal(result["answer"]))
        failed += not good
        print(f"{'ok' if good else 'WRONG'} as {reader}: {item['question']} -> {result['status']}: "
              f"{result['answer'][:80]!r}; holding cited {bool(holding & cited)}; dropped {result['dropped']}; "
              f"{'cache' if result['cached'] else f'{result['cost_usd']:.6f} USD'}", flush=True)
    refused = [item for item in smoke if any(not set(item["labels"]) & reads for reads in roles.values())]
    for item in refused[:args.refusals]:  # a row of general has no outsider: every role reads general
        outsider = min((r for r, reads in roles.items() if not set(item["labels"]) & reads), key=lambda r: len(roles[r]))
        result = pipeline.ask(client, item["question"], outsider, policy, use_cache=not args.no_cache)
        shown = [unit for excerpt in result["excerpts"] for unit in excerpt["units"]]
        good = (result["status"] == "not_found" and item["unit"] not in shown and not result["citations"]
                and normal(item["value"]) not in normal(result["answer"]))
        failed += not good
        print(f"{'ok' if good else 'WRONG'} refused to {outsider}: {item['question']} -> {result['status']}", flush=True)
    after = pipeline.spend()
    print(f"{len(smoke)} answered and {min(args.refusals, len(refused))} refused asked, {failed} failed; spend log "
          f"{before['questions']} to {after['questions']} questions, {after['usd'] - before['usd']:.6f} USD added, "
          f"{after['cached'] - before['cached']} from the cache")
    return 1 if failed or len(smoke) < 10 else 0


if __name__ == "__main__":
    raise SystemExit(main())
