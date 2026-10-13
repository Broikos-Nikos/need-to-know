"""The evaluation run (T5.5c, `eval/protocol.toml` [baseline]): every item of a split asked as its role, through the
answer layer as `ntk ask` asks it, each repeat a fresh call with the answer cache off, as the protocol registers.

Each answer is written as one JSON line the moment it comes back, so a run cut short resumes where it stopped and
bills nothing twice: an item and repeat already in the file are not asked again. A line holds the item's id, the
repeat, the answer as `parse` reads it, the model that answered, which the provider names and an alias does not
(KB5-K14), what the call cost, the excerpts shown, by document, section and units, and how many of those units are
table rows (KB1-F8). With `closed_book` the question is asked with no excerpts at all, the model's own knowledge
alone (KB1-F7). `ntk eval score` reads the file, a repeat at a time and then over them.

Before the first call a run writes its provenance beside the answers, `<file>.provenance.json`: the commit and whether
the tree was clean, the protocol's version and fingerprint, the split, purpose, repeats and book, the provider, the
model alias and the settings it is asked with, the manifest's last run and unit count, and the lexical analyser's
version. A run is resumed only at the commit it started at, so one file never holds two versions of the code, and a
run for the baseline or a release starts only on a clean tree, the commit alone being what made it: tick 139's test
and closed book runs began while the report was being written beside them.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
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


class RunError(RuntimeError):
    """A run that would mix two versions of the code in one file, or publish one the commit does not name."""


PUBLISHED = ("baseline", "release")


def commit() -> tuple[str, bool]:
    """The repository's commit, and whether its tree is clean."""
    from needtoknow.settings import repo_root
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True,
                            check=True)
    return head.stdout.strip(), not status.stdout.strip()


def provenance(split: str, purpose: str, repeats: int, closed_book: bool, at=commit) -> dict:
    """What a run's answers were made by, as the run starts."""
    from needtoknow import lexical
    from needtoknow.answer.pipeline import settings_of
    from needtoknow.eval.protocol import fingerprint, load_protocol
    from needtoknow.ingest.manifest import opened
    from needtoknow.providers.catalog import load_catalog
    head, clean = at()
    entry = load_catalog()["deepseek"]
    with opened() as db:
        last = db.execute("SELECT run_id, finished, units_total FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    return {"commit": head, "clean": clean, "protocol_version": load_protocol()["version"],
            "protocol": fingerprint(), "split": split, "purpose": purpose, "repeats": repeats,
            "closed_book": closed_book, "provider": "deepseek", "model": entry["model"], "settings": settings_of(entry),
            "manifest": {"run": last[0], "finished": last[1], "units": last[2]} if last else None,
            "analyser": lexical.frozen()["analyser"],
            "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def asked(out: Path) -> set[tuple[str, int]]:
    """The items and repeats a file already holds."""
    if not out.is_file():
        return set()
    return {(record["id"], record["repeat"]) for record in
            (json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip())}


def run(split: str, purpose: str, repeats: int, out: Path, client, closed_book: bool = False, ask=None,
        tables=table_layers, report=None, first: int | None = None, made_by=None) -> dict:
    """Ask every item of the split as its role, `repeats` times, appending each answer to `out`; what it did. `first`
    asks only that many items, for a smoke run."""
    check_use(split, purpose)
    sidecar = out.with_name(out.name + ".provenance.json")
    now = (made_by or provenance)(split, purpose, repeats, closed_book)
    if purpose in PUBLISHED and not now.get("clean", True):
        raise RunError(f"a {purpose} run names the commit that made it, and the tree holds changes: commit them first")
    if sidecar.is_file():
        then = json.loads(sidecar.read_text(encoding="utf-8"))
        if then["commit"] != now["commit"]:
            raise RunError(f"{out} was started at {then['commit'][:12]} and this is {now['commit'][:12]}: resume it "
                           "there, or start another file")
    else:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes((json.dumps(now, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
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
