"""The page's recorded traces (T6.2c): chosen dev golden questions asked as every role through the app, as the page
asks them live, trimmed to what the page draws, and written with when, at which commit and with which models they were
asked, for the page to replay where no server answers (house rule 10).

    uv run python bench/page_traces.py         asks each question as each role, writes web/public/recorded/traces/

Each role signs in and asks through `POST /ask`, which asks the provider every time (N68): three questions and nine
roles are 27 calls, run outside the provider's peak hours. A trace keeps every stage's first ten candidates, the page's
rows, with how many each stage held, and the text of the units those rows and the excerpts name; nothing a role may not
read is in its trace, as nothing is in the served one. It needs the index on the running server, the `embed` extra, the
GPU and the provider's key.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# An index of the questions, the roles and the provenance, which the page's pickers read at once, and a file per
# question, fetched when it is first replayed (PF-F13).
OUT = REPO / "web" / "public" / "recorded" / "traces"
QUESTIONS = ("D002", "D074", "D001")  # sales' discount in English and in Greek, and legal's call recordings
SHOWN = 10  # the rows the page draws of each stage
STAGES = ("dense", "sparse", "fused", "reranked")


def trimmed(trace: dict) -> dict:
    """The served trace as the page draws it: each stage's first rows, how many the stage held, and the text of the
    units the rows and the excerpts name."""
    kept = {stage: trace[stage][:SHOWN] for stage in STAGES}
    named = {hit["unit_id"] for hits in kept.values() for hit in hits}
    named |= {unit for excerpt in trace["excerpts"] for unit in excerpt["units"]}
    return {**trace, **kept, "counts": {stage: len(trace[stage]) for stage in STAGES},
            "units": {unit: described for unit, described in trace["units"].items() if unit in named}}


def provenance() -> dict:
    """When, at which commit, from a tree holding changes or not, with which embedding and reranking models."""
    from needtoknow.embed.model import load_config as embed_config
    from needtoknow.index.rerank import load_config as rerank_config

    git = lambda *args: subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"recorded": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": git("rev-parse", "--short", "HEAD"),
            "clean": not git("status", "--porcelain"), "embedder": embed_config()["repo"],
            "reranker": rerank_config()["repo"]}


def record(http, policy: dict, items: dict) -> dict:
    """Each question asked as each role through the app's routes, a sign in per role."""
    traces: dict[str, dict] = {}
    for role in policy["roles"]:
        token = http.post("/session", json={"role": role}).json()["token"]
        for qid, item in items.items():
            response = http.post("/ask", json={"question": item["question"]}, headers={"Authorization": f"Bearer {token}"})
            if response.status_code != 200:
                raise SystemExit(f"page_traces: {qid} as {role}: {response.status_code} {response.text[:200]}")
            traces.setdefault(qid, {})[role] = trimmed(response.json())
            print(f"{qid} as {role}: {response.json()['answer']['status']}", flush=True)
    return traces


def main() -> int:
    from fastapi.testclient import TestClient

    from needtoknow import api
    from needtoknow.eval.golden import golden_path, load_items
    from needtoknow.eval.protocol import check_use
    from needtoknow.index import server
    from needtoknow.policy import load_policy

    sys.stdout.reconfigure(encoding="utf-8")
    check_use("dev", "development")
    every = {item["id"]: item for item in load_items(golden_path("dev"))}
    items = {qid: every[qid] for qid in QUESTIONS}
    policy = load_policy()
    http = TestClient(api.create(server.client()), base_url=f"http://{api.HOST}")
    traces = record(http, policy, items)
    models = sorted({trace["answer"]["model"] for by in traces.values() for trace in by.values() if trace["answer"]["model"]})
    index = {**provenance(), "models": models, "roles": list(policy["roles"]),
             "questions": [{"id": qid, "role": item["role"], "lang": item["lang"], "question": item["question"]}
                           for qid, item in items.items()]}
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):  # a question no longer recorded leaves no file behind
        old.unlink()
    files = {"index.json": index, **{f"{qid}.json": {"question": qid, "traces": by_role} for qid, by_role in traces.items()}}
    for name, data in files.items():
        (OUT / name).write_bytes((json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    sizes = ", ".join(f"{name} {(OUT / name).stat().st_size:,} bytes" for name in files)
    print(f"{OUT.relative_to(REPO).as_posix()}: {len(items)} questions as {len(policy['roles'])} roles at "
          f"{index['commit']}{'' if index['clean'] else ' with changes'}, {', '.join(models)}; {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
