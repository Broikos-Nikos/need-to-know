"""What the rerank switch costs on this machine, and how often it changes the first answer (T3.6).

    uv run python bench/rerank_latency.py [--questions 20] [--role admin]

Needs the index on the running server, the manifest, the embed extra and both pinned
models. It asks planted fact sentences, half of them in each language, as the role, at
each position of the switch, after one warm up per mode, and prints for each mode the
seconds a question takes from text to final order, at the median and the 95th percentile,
and in how many questions the first unit differs from the fused order's first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from needtoknow.index import server
from needtoknow.index.rerank import MODES
from needtoknow.index.search import search
from needtoknow.policy import load_policy
from needtoknow.settings import data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--role", default="admin")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    facts = json.loads((data_dir() / "corpus" / "plan" / "facts.json").read_text(encoding="utf-8"))
    questions = [f[lang] for f in facts[: args.questions // 2] for lang in ("en", "el")]
    client, policy = server.client(), load_policy()
    first = {}
    for mode in MODES:
        search(client, "warm up", args.role, policy, rerank=mode)
        seconds, tops = [], []
        for question in questions:
            began = time.perf_counter()
            hits = search(client, question, args.role, policy, rerank=mode)
            seconds.append(time.perf_counter() - began)
            tops.append(hits[0]["unit_id"] if hits else None)
        first[mode] = tops
        changed = sum(a != b for a, b in zip(tops, first["none"]))
        print(f"{mode:<12} {len(questions)} questions as {args.role}, 100 candidates: median {np.median(seconds):.2f} s, "
              f"p95 {np.percentile(seconds, 95):.2f} s; first unit changed from the fused order in {changed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
