"""The lexical side channel attack (AC-F5, KB1-F4): can a role tell, from ranks alone, which words are common in
text it may not read?

    uv run python bench/idf_attack.py [--role sales] [--pairs 60] [--seed 60]

Needs the manifest and the index on the running server, with the embed extra. It analyses
every passage as the index does and keeps the words the role sees one to three times. A
test pair joins one of them that 30 or more passages the role may not read also hold with
one that no such passage holds and that the role sees exactly as often; a null pair joins
two of the latter, matched the same way. Each pair is asked as
the role, its two words in random order, through the retrieval API, and the pair counts
when both words' units come back: the share of test pairs whose hidden-frequent word's unit
ranks below the other, against the same share in the null pairs. With weights over the
whole collection the two shares part (tick 60: 94 against 45 percent); with weights over
the role's view (N50) they must not.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict

from needtoknow import lexical
from needtoknow.index import server
from needtoknow.index.search import search
from needtoknow.ingest.manifest import manifest_path
from needtoknow.ingest.units import passage
from needtoknow.policy import load_policy, readable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", default="sales")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--sides", default="dense,sparse", help="the searches to run; sparse alone shows the weights")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    policy = load_policy()
    reads = readable(policy, args.role)
    db = sqlite3.connect(f"file:{manifest_path().as_posix()}?mode=ro", uri=True)
    seen = defaultdict(lambda: [0, 0, None])  # term: passages the role reads, passages it may not, one unit it reads
    for unit_id, labels, path, text in db.execute("SELECT unit_id, labels, path, text FROM units"):
        visible = bool(set(json.loads(labels)) & reads)
        for term in set(lexical.terms(passage(json.loads(path), text))):
            if term.isalpha() and len(term) > 4:
                entry = seen[term]
                entry[0 if visible else 1] += 1
                if visible:
                    entry[2] = unit_id
    heavy = sorted(t for t, (v, h, _) in seen.items() if 1 <= v <= 3 and h >= 30)
    clean = sorted(t for t, (v, h, _) in seen.items() if 1 <= v <= 3 and h == 0)
    print(f"as {args.role}, searching {args.sides}: words seen 1 to 3 times, frequent in hidden text {len(heavy)}, "
          f"absent from it {len(clean)}")
    rng = random.Random(args.seed)
    client = server.client()

    def lower(t: str, u: str) -> int | None:
        question = f"{t} {u}" if rng.random() < 0.5 else f"{u} {t}"
        hits = search(client, question, args.role, policy, limit=100, sides=tuple(args.sides.split(",")))
        rank = {h["unit_id"]: h["rank"] for h in hits}
        rt, ru = rank.get(seen[t][2]), rank.get(seen[u][2])
        return None if rt is None or ru is None or seen[t][2] == seen[u][2] else int(rt > ru)

    def partner(t: str, pool: list[str]) -> str:
        """A word the role sees exactly as often as t, so its own view weighs both alike."""
        return rng.choice([u for u in pool if seen[u][0] == seen[t][0] and u != t])

    test = [lower(t, partner(t, clean)) for t in rng.sample(heavy, min(args.pairs, len(heavy)))]
    null = [lower(t, partner(t, clean)) for t in rng.sample(clean, args.pairs)]
    for name, results in (("test pairs, hidden-frequent first", test), ("null pairs", null)):
        kept = [r for r in results if r is not None]
        print(f"{name}: {len(results)}, both units ranked in {len(kept)}, the first word's unit lower in {sum(kept)} "
              f"({sum(kept) / max(len(kept), 1):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
