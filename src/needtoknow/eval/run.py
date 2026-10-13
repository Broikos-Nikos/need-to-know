"""The evaluation run (T5.5c, `eval/protocol.toml` [baseline]): every item of a split asked as its role, through the
answer layer as `ntk ask` asks it, each repeat a fresh call with the answer cache off, as the protocol registers.

Each answer is written as one JSON line the moment it comes back, so a run cut short resumes where it stopped and
bills nothing twice: an item and repeat already in the file are not asked again. A line holds the item's id, the
repeat, the answer as `parse` reads it, the model that answered, which the provider names and an alias does not
(KB5-K14), what the call cost, the excerpts shown, by document, section and units, and how many of those units are
table rows (KB1-F8). With `closed_book` the question is asked with no excerpts at all, the model's own knowledge
alone (KB1-F7). `ntk eval score` reads the file, a repeat at a time and then over them.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from needtoknow.eval.golden import golden_path, load_items
from needtoknow.eval.protocol import check_use


def table_layers(units: list[str]) -> int:
    """How many of these units are table rows, by the manifest."""
    if not units:
        return 0
    from needtoknow.ingest.manifest import opened
    with opened() as db:
        return db.execute(f"SELECT COUNT(*) FROM units WHERE layer = 'table' AND unit_id IN "
                          f"({','.join('?' * len(units))})", units).fetchone()[0]


def asked(out: Path) -> set[tuple[str, int]]:
    """The items and repeats a file already holds."""
    if not out.is_file():
        return set()
    return {(record["id"], record["repeat"]) for record in
            (json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip())}


def run(split: str, purpose: str, repeats: int, out: Path, client, closed_book: bool = False, ask=None,
        tables=table_layers, report=None, first: int | None = None) -> dict:
    """Ask every item of the split as its role, `repeats` times, appending each answer to `out`; what it did. `first`
    asks only that many items, for a smoke run."""
    check_use(split, purpose)
    if ask is None:
        from needtoknow.answer.pipeline import ask
    items = load_items(golden_path(split))[:first]
    done = asked(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"items": len(items), "repeats": repeats, "asked": 0, "kept": len(done), "usd": 0.0, "models": set()}
    for repeat in range(repeats):
        for item in items:
            if (item["id"], repeat) in done:
                continue
            result = ask(client, item["question"], item["role"], use_cache=False, repeat=repeat,
                         closed_book=closed_book)
            units = [unit for excerpt in result["excerpts"] for unit in excerpt["units"]]
            record = {"id": item["id"], "repeat": repeat, "status": result["status"], "answer": result["answer"],
                      "citations": result["citations"], "dropped": result.get("dropped", []),
                      "error": result.get("error"), "model": result["model"], "usd": result["cost_usd"],
                      "closed_book": closed_book, "excerpts": result["excerpts"], "units": len(units),
                      "tables_shown": tables(units),
                      "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            with open(out, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["asked"] += 1
            summary["usd"] += result["cost_usd"] or 0.0
            summary["models"].add(result["model"])
            if report:
                report(f"{item['id']} repeat {repeat}: {result['status']}")
    summary["models"] = sorted(summary["models"])
    return summary
