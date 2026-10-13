"""The scorer's floor, its ceiling and its citation null (T5.2c, RS-F7, RS-F15): three runs no model answers.

    uv run python bench/null_runs.py [--split dev] [--no-search]

* null: every item answered `not_found`, the floor a system that refuses everything reaches;
  refusing is right for every E, J and N item, so the overall accuracy starts there.
* oracle: every answerable item answered with its first form and citing exactly the sections
  that plant its facts, every refusal refused: the ceiling, and a check that each form
  matches itself once normalised.
* cite everything: the oracle's answers, each citing every excerpt the asking role's search
  shows, as `answer.pipeline.ask` would show them; dev only. Citation precision falls to the
  share of shown excerpts that hold an expected fact, and citation recall becomes the share
  of facts whose planted section the search showed at all.

Needs the plan, the frequency table and the manifest; cite everything also the index on
the running server and the `embed` extra, and `--no-search` leaves it out. The split is
asked of the protocol first: dev for development, test for its baseline, with no search.
"""

from __future__ import annotations

import argparse
import sys
import time

from needtoknow.eval.golden import Context, golden_path, load_items
from needtoknow.eval.protocol import check_use
from needtoknow.eval.score import REFUSALS, Sources, score


def null(items: list[dict]) -> list[dict]:
    return [{"id": item["id"], "status": "not_found", "answer": "", "citations": []} for item in items]


def oracle(items: list[dict], sources: Sources) -> list[dict]:
    answers = []
    for item in items:
        if item["category"] in REFUSALS:
            answers.append({"id": item["id"], "status": "not_found", "answer": "", "citations": []})
            continue
        facts = item["expect"]["facts"]
        places = sources.of_row(item["row"]) if item["category"] == "I" else \
            set().union(*[sources.of_fact(f["fact_id"]) for f in facts])
        answers.append({"id": item["id"], "status": "answered", "answer": " and ".join(f["forms"][0] for f in facts),
                        "citations": [{"doc_id": doc, "section": section} for doc, section in sorted(places)]})
    return answers


def cite_everything(items: list[dict], answers: list[dict]) -> tuple[list[dict], list[int]]:
    """The oracle's answers citing every excerpt the role's search shows, and how many each was shown."""
    from needtoknow.answer.prompt import section_of
    from needtoknow.index import server
    from needtoknow.index.expand import excerpts
    from needtoknow.index.search import search
    client = server.client()
    by_id = {answer["id"]: answer for answer in answers}
    out, shown_counts = [], []
    for item in items:
        answer = by_id[item["id"]]
        if answer["status"] != "answered":
            out.append(answer)
            continue
        shown = excerpts(search(client, item["question"], item["role"]))
        shown_counts.append(len(shown))
        out.append({**answer, "citations": [{"doc_id": e["doc_id"], "section": section_of(e)} for e in shown]})
    return out, shown_counts


def report(name: str, figures: dict) -> None:
    def rate(r: dict) -> str:
        return f"{r['correct']} of {r['items']}" + ("" if r["rate"] is None else f", {r['rate']:.3f}")

    numbers = {k: figures[k] for k in ("partial", "citation_precision", "citation_recall")}
    shown = "; ".join(f"{k.replace('_', ' ')} {'none' if v is None else f'{v:.3f}'}" for k, v in numbers.items())
    print(f"{name}: answerable {rate(figures['answerable'])}; refusals {rate(figures['refusals'])}; "
          f"leaks {figures['leaks']}; overall {rate(figures['overall'])}; {shown}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--no-search", action="store_true", help="leave out the cite everything run")
    args = parser.parse_args(argv)
    check_use(args.split, "development" if args.split == "dev" else "baseline")
    items = load_items(golden_path(args.split))
    ctx = Context()
    sources = Sources(ctx)
    floor = score(items, null(items), ctx, sources)[1]
    report("null", floor)
    best = oracle(items, sources)
    ceiling = score(items, best, ctx, sources)[1]
    report("oracle", ceiling)
    if args.split == "dev" and not args.no_search:
        began = time.perf_counter()
        everything, counts = cite_everything(items, best)
        results, figures = score(items, everything, ctx, sources)
        report("cite everything", figures)
        unseen = [r["id"] for r in results if r["recall"] is not None and r["recall"] < 1]
        shown = {}
        for r in results:
            if r["recall"] is not None:
                shown.setdefault(r["category"], []).append(r["recall"])
        print("cite everything, recall at the shown excerpts by category (RS-F24): " + "; ".join(
            f"{c} {sum(v) / len(v):.3f} of {len(v)}" for c, v in sorted(shown.items())), flush=True)
        print(f"cite everything: {len(counts)} searches, {sum(counts) / len(counts):.1f} excerpts shown on average, "
              f"{time.perf_counter() - began:.0f} s; {len(unseen)} items with a fact the search never showed"
              + (f": {', '.join(unseen)}" if unseen else ""), flush=True)
    print(f"scorer {ceiling['scorer'][:16]}", flush=True)
    return 0 if ceiling["overall"]["rate"] == 1.0 and ceiling["leaks"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
