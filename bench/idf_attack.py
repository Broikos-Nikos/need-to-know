"""The lexical side channel attack (AC-F5, KB1-F4): can a role tell, from ranks alone, which words are common in
text it may not read?

    uv run python bench/idf_attack.py [--role sales] [--pairs 60] [--seed 60] [--global-idf]
    uv run python bench/idf_attack.py --role sales,hr,returns [--global-idf]

Needs the manifest and the index on the running server, with the embed extra. It analyses
every passage as the index does and keeps the words the role sees one to three times. A
test pair joins one of them that 30 or more passages the role may not read also hold with
one that no such passage holds and that the role sees exactly as often; a null pair joins
two of the latter, matched the same way. Each pair is asked as
the role, its two words in random order, through the retrieval API, and the pair counts
when both words' units come back: the share of test pairs whose hidden-frequent word's unit
ranks below the other, against the same share in the null pairs, in the fused order,
where the weights act. `--global-idf` asks with the weights as they were before N50, when
Qdrant's IDF modifier counted every point whatever the role: the same BM25 IDF over every
label set's frequencies, on today's index (MA-F7), since the collection that had the
modifier is gone. With those weights the two shares part. With the role's view (N50) the
ranks are built from text the role may read and one average length, which
tests/test_noninterference.py holds; a gap that remains is one a matched pair cannot tell
from how such words sit in the role's own text, since a null pair matches how often the
role sees a word, not the units that hold it. Each share is printed with its count and
Wilson interval (RS-F12), several roles are pooled, and the counts are saved for N50.
Tick 131, pooled over sales, hr and returns: 74 of 88 over the whole collection, 53 of 88
over the view, 61 of 126 null pairs; tick 74, as hr on the sparse side alone, 97, 55 and
44 percent.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict

from machine import named

from needtoknow import lexical
from needtoknow.index import server
from needtoknow.index.search import search
from needtoknow.ingest.manifest import manifest_path
from needtoknow.eval.stats import wilson
from needtoknow.ingest.units import passage
from needtoknow.policy import load_policy, readable
from needtoknow.settings import data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", default="sales,hr,returns",
                        help="a role, or several joined by commas, pooled at the end; N50's three by default")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--sides", default="dense,sparse", help="the searches to run; sparse alone shows the weights")
    parser.add_argument("--global-idf", action="store_true",
                        help="weights over the whole collection, as before N50, to show the attack the view stops")
    args = parser.parse_args()
    if args.global_idf:  # every label set, whatever the role reads: what the IDF modifier did before N50
        from needtoknow.index import frequencies
        real = frequencies.weights
        frequencies.weights = lambda db, terms, _view: real(db, terms, [s for (s,) in db.execute("SELECT label_set FROM sets")])
    sys.stdout.reconfigure(encoding="utf-8")
    policy = load_policy()
    client = server.client()
    over = "the whole collection" if args.global_idf else "the role's view"
    counts = {}
    for role in args.role.split(","):
        counts[role] = attack(role, args, policy, client, over)
    if len(counts) > 1:
        for name in NAMES:
            lower, kept = (sum(c[name][i] for c in counts.values()) for i in (0, 1))
            print(f"pooled over {', '.join(counts)}, {name}: both units ranked in {kept}, {share(lower, kept)}")
    out = data_dir() / "bench" / named(f"idf_attack_{'whole' if args.global_idf else 'view'}", parser, args,
                                       ("role", "pairs", "seed", "sides"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps({"sides": args.sides, "pairs": args.pairs, "seed": args.seed, "roles": counts},
                                indent=1) + "\n").encode("utf-8"))
    return 0


NAMES = ("test pairs, hidden-frequent first", "null pairs")


def share(lower: int, kept: int) -> str:
    """The pairs whose first word's unit ranks lower, as a count, a share and its Wilson interval (RS-F12)."""
    if not kept:
        return "no pair to read"
    low, high = wilson(lower, kept)
    return f"the first word's unit lower in {lower} ({lower / kept:.0%}, Wilson {low:.2f} to {high:.2f})"


def attack(role: str, args, policy: dict, client, over: str) -> dict[str, list[int]]:
    """One role's test and null pairs asked, each as [lower, both ranked, asked]."""
    reads = readable(policy, role)
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
    db.close()
    heavy = sorted(t for t, (v, h, _) in seen.items() if 1 <= v <= 3 and h >= 30)
    clean = sorted(t for t, (v, h, _) in seen.items() if 1 <= v <= 3 and h == 0)
    print(f"as {role}, searching {args.sides}, weights over {over}: words seen 1 to 3 times, frequent in hidden text "
          f"{len(heavy)}, absent from it {len(clean)}")
    rng = random.Random(args.seed)

    def lower(t: str, u: str) -> int | None:
        question = f"{t} {u}" if rng.random() < 0.5 else f"{u} {t}"
        # The fused order, where the weights act: since T3.6 the search reranks by default, and the cross encoder,
        # which reads no weight, would decide the order the attack reads (tick 74).
        hits = search(client, question, role, policy, limit=100, sides=tuple(args.sides.split(",")), rerank="none")
        rank = {h["unit_id"]: h["rank"] for h in hits}
        rt, ru = rank.get(seen[t][2]), rank.get(seen[u][2])
        return None if rt is None or ru is None or seen[t][2] == seen[u][2] else int(rt > ru)

    def partner(t: str, pool: list[str]) -> str:
        """A word the role sees exactly as often as t, so its own view weighs both alike."""
        return rng.choice([u for u in pool if seen[u][0] == seen[t][0] and u != t])

    test = [lower(t, partner(t, clean)) for t in rng.sample(heavy, min(args.pairs, len(heavy)))]
    null = [lower(t, partner(t, clean)) for t in rng.sample(clean, args.pairs)]
    found = {}
    for name, results in zip(NAMES, (test, null)):
        kept = [r for r in results if r is not None]
        found[name] = [sum(kept), len(kept), len(results)]
        print(f"{name}: {len(results)}, both units ranked in {len(kept)}, {share(sum(kept), len(kept))}", flush=True)
    return found


if __name__ == "__main__":
    raise SystemExit(main())
