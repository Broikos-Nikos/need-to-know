"""The pipeline operator's probes on a scratch copy of 24 documents: a killed build, files deleted behind the
pipeline's back, a crash before a change's journal line, and two assemblies compared.

    uv run python bench/operator_probes.py                     the code as it is
    git worktree add ../ntk-before 074907c
    uv run python bench/operator_probes.py --src ../ntk-before/src

The second form runs the same probes on the code before OP-F1 and OP-F2, which is where
the numbers in commits 5547561 and fbd9532 come from. The 24 documents are the first 16
of the catalogue that carry a fact and the first 8 that carry none. Their text is the
writers' narrative when NTK_DATA_DIR holds it, and otherwise a synthetic one built from
the plan, which gives the same counts; `ntk corpus plan` is all either needs. The real
corpus is only read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

FILLER = {"en": "The team follows this step for every case and records the outcome in the system of record.",
          "el": "Η ομάδα ακολουθεί αυτό το βήμα σε κάθε περίπτωση και καταγράφει το αποτέλεσμα στο σύστημα."}


def configured_corpus() -> Path:
    """The corpus this checkout's settings point at, read before any needtoknow module is imported."""
    spec = importlib.util.spec_from_file_location("ntk_settings", Path(__file__).resolve().parents[1] / "src" / "needtoknow" / "settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings.data_dir() / "corpus"


def main() -> int:
    parser = argparse.ArgumentParser(description="The pipeline operator's probes on a scratch copy of 24 documents.")
    parser.add_argument("--src", type=Path, help="import needtoknow from this src folder, such as an older commit's worktree")
    parser.add_argument("--corpus", type=Path, help="the corpus folder holding plan/ and, if written, narrative/")
    parser.add_argument("--synthetic", action="store_true", help="use a synthetic narrative even where the writers' exists")
    args = parser.parse_args()
    real = args.corpus or configured_corpus()
    if not (real / "plan" / "catalogue.json").is_file():
        print(f"no plan in {real}: run `ntk corpus plan` first", file=sys.stderr)
        return 1
    scratch = Path(tempfile.mkdtemp(prefix="ntk-operator-"))
    os.environ["NTK_DATA_DIR"] = str(scratch)
    if args.src:
        sys.path.insert(0, str(args.src.resolve()))
    from needtoknow.corpus import assemble, check, mutate
    from needtoknow.corpus.plan import header

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        docs = json.loads((real / "plan" / "catalogue.json").read_text(encoding="utf-8"))
        facts = json.loads((real / "plan" / "facts.json").read_text(encoding="utf-8"))
        placed = {d for f in facts for d in f["placements"]}
        picked = [d for d in docs if d["id"] in placed][:16] + [d for d in docs if d["id"] not in placed][:8]
        kept = {d["id"] for d in picked}
        for d in picked:
            d["target_mb"] = 0.2
        for f in facts:
            f["placements"] = [p for p in f["placements"] if p in kept]
        by_id = {f["id"]: f for f in facts}
        base = scratch / "corpus"
        (base / "plan").mkdir(parents=True)
        (base / "narrative").mkdir()
        (base / "plan" / "facts.json").write_bytes(json.dumps([f for f in facts if f["placements"]], ensure_ascii=False).encode("utf-8"))
        (base / "plan" / "catalogue.json").write_bytes(json.dumps(picked, ensure_ascii=False).encode("utf-8"))
        written = (real / "narrative").is_dir() and not args.synthetic
        for d in picked:
            target = base / "narrative" / f"{d['id']}.txt"
            if written:
                shutil.copy(real / "narrative" / f"{d['id']}.txt", target)
                continue
            lines = []
            for n, section in enumerate(d["sections"], 1):
                body = " ".join([FILLER[d["language"]]] + [by_id[fid][d["language"]] for fid in section["facts"]]
                                + [section["canary"] or ""]).strip()
                lines += header(d, n, section) + ["", body, ""]
            target.write_bytes("\n".join(lines).encode("utf-8"))
        full = base / "full"

        def digest() -> dict[str, str]:
            return {p.relative_to(full).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in full.rglob("*") if p.is_file()}

        print(f"code: needtoknow from {Path(assemble.__file__).parents[2]}")
        print(f"documents: {len(picked)}, {sum(d['id'] in placed for d in picked)} carrying a fact placement; "
              f"narrative: {'the writers' if written else 'synthetic'}")

        assemble.assemble()
        first = digest()
        assemble.assemble()
        print(f"P3 two assemblies: identical bytes {first == digest()} in {len(first)} files")

        real_doc, calls = assemble.assemble_doc, []

        def dying(*a, **k):
            calls.append(1)
            if len(calls) > 15:
                raise KeyboardInterrupt("killed")
            return real_doc(*a, **k)

        assemble.assemble_doc = dying
        try:
            assemble.assemble()
        except KeyboardInterrupt:
            pass
        assemble.assemble_doc = real_doc
        present = [p for p in full.rglob("*") if p.is_file() and not p.name.endswith(".meta.json")] if full.is_dir() else []
        problems = check.check()["problems"]
        print(f"P1 a build killed after 15 of {len(picked)} documents: {len(picked) - len(present)} files missing; "
              f"problems naming a missing file {sum('no assembled file' in p for p in problems)}")

        assemble.assemble()
        gone = [d["id"] for d in picked[15:]]
        for doc_id in gone:
            next(p for p in full.glob(f"{doc_id}.*") if not p.name.endswith(".meta.json")).unlink()
        problems = check.check()["problems"]
        named = sum(any(p.startswith(f"{g}: no assembled file") for p in problems) for g in gone)
        print(f"P1c {len(gone)} files deleted by hand, {sum(g in placed for g in gone)} carrying a fact: the check names {named}")

        assemble.assemble()
        before = digest()

        def failing(*a, **k):
            raise OSError("disk full while writing the journal")

        # The journal line made to fail: since fbd9532 mutate.append writes it, and before that a json.dumps
        # inside mutate() did.
        owner, name = (mutate, "append") if hasattr(mutate, "append") else (mutate.json, "dumps")
        original = getattr(owner, name)
        setattr(owner, name, failing)
        try:
            mutate.mutate(count=1, seed=4)
        except OSError:
            pass
        finally:
            setattr(owner, name, original)
        after = digest()
        journal = base / "full.journal.jsonl"
        lines = [line for line in journal.read_text(encoding="utf-8").splitlines() if line] if journal.is_file() else []
        print(f"P2 the journal line made to fail: files changed {sum(before[k] != after.get(k) for k in before)}; journal lines {len(lines)}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
