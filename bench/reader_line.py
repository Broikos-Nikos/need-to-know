"""Does the reader line move an answer (KB5-K12)? Dev items asked twice with the line that opens the user message
since tick 134 and twice without it, the answer cache bypassed, every answer scored.

    uv run python bench/reader_line.py [--items 24] [--seed 134]

Needs the manifest, the index on the running server, the `embed` extra, the GPU and a DeepSeek key; asks the
protocol first, dev for development. A sample of dev's originals, drawn by the seed, is searched once each as its
role, as `ask` searches; both prompts are built from the same excerpts and differ only in the line, a mark of the
role's label set (AC-F11). Each is sent twice at the catalogue's settings and scored as the evaluation scores it. It
prints how many items each run gets right, how many change verdict between the two runs of one prompt, the
provider's own spread, and between the prompts, run by run; and what the calls cost. About a hundred calls, a few
cents off peak.
"""

from __future__ import annotations

import argparse
import random
import sys

from needtoknow.answer.pipeline import settings_of
from needtoknow.answer.prompt import build, parse, reader_mark
from needtoknow.eval.golden import Context, golden_path, load_items
from needtoknow.eval.protocol import check_use
from needtoknow.eval.score import Sources, score
from needtoknow.index import server
from needtoknow.index.expand import excerpts
from needtoknow.index.search import search
from needtoknow.policy import load_policy, readable
from needtoknow.providers import deepseek
from needtoknow.providers.catalog import load_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--seed", type=int, default=134)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    check_use("dev", "development")
    originals = [item for item in load_items(golden_path("dev")) if not item.get("twin")]
    items = random.Random(args.seed).sample(originals, min(args.items, len(originals)))
    policy, catalog, client = load_policy(), load_catalog(), server.client()
    settings = settings_of(catalog["deepseek"])
    runs = {name: [] for name in ("with 1", "with 2", "without 1", "without 2")}
    usd = 0.0
    for item in items:
        shown = excerpts(search(client, item["question"], item["role"], policy))
        system, user = build(item["question"], shown, reader=reader_mark(readable(policy, item["role"])))
        bare = user.split("\n\n", 1)[1]  # the same message without its first line
        assert user.startswith("Reader: ") and bare.startswith("Documents:")
        for name, message in (("with 1", user), ("with 2", user), ("without 1", bare), ("without 2", bare)):
            done = deepseek.complete(system, [{"role": "user", "content": message}], config=catalog, **settings)
            usd += done.cost_usd or 0.0
            runs[name].append({"id": item["id"], **parse(done.text, shown, item["question"])})
    ctx = Context()
    sources = Sources(ctx)
    right = {name: {r["id"]: r["correct"] for r in score(items, answers, ctx, sources)[0]} for name, answers in
             runs.items()}

    def moved(a: str, b: str) -> int:
        return sum(right[a][item["id"]] != right[b][item["id"]] for item in items)

    print(f"{len(items)} dev originals, seed {args.seed}; right: " + ", ".join(
        f"{name} {sum(right[name].values())}" for name in runs))
    print(f"verdicts that moved between two runs of one prompt: with the line {moved('with 1', 'with 2')}, without "
          f"it {moved('without 1', 'without 2')}; between the prompts, run by run: {moved('with 1', 'without 1')} "
          f"and {moved('with 2', 'without 2')}")
    print(f"{4 * len(items)} calls, {usd:.6f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
