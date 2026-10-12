"""Every write of nine changes killed in turn: does the next run leave the corpus and its journal as a run
without the crash would have?

    uv run python bench/crash_points.py

It builds three small documents in a temporary folder, runs the change simulator once
to record the state after each change, then kills it at each write in turn (a journal
line cut in half where the write is one) and runs it again with nothing to add. A kill
point counts as equal when the files and the journal match the crash free state after
the same number of changes. Commit fbd9532 quotes this script's output.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

FACT = {"en": "Returned items are inspected within 3 working days of arrival at the warehouse.",
        "el": "Τα επιστρεφόμενα είδη ελέγχονται εντός 3 εργάσιμων ημερών από την άφιξή τους στην αποθήκη."}
PROSE = {"en": "The returns desk checks each parcel against its authorisation and records the result in Dock before anything moves on. " * 3,
         "el": "Το γραφείο επιστροφών ελέγχει κάθε δέμα με βάση την εξουσιοδότηση και καταγράφει το αποτέλεσμα στο Dock πριν προχωρήσει. " * 3}
HEAD = {"en": ["SECTION 1  Inspection", "Audience: Returns", "Classification: Internal"],
        "el": ["ΕΝΟΤΗΤΑ 1  Έλεγχος", "Κοινό: Επιστροφές", "Διαβάθμιση: Εσωτερικό"]}


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="ntk-crash-"))
    os.environ["NTK_DATA_DIR"] = str(scratch)
    from needtoknow.corpus import assemble, mutate

    sys.stdout.reconfigure(encoding="utf-8")
    base = scratch / "corpus"
    try:
        (base / "plan").mkdir(parents=True)
        (base / "narrative").mkdir()
        docs = []
        for n, (lang, fmt) in enumerate([("en", "txt"), ("el", "md"), ("en", "html")], 1):
            doc_id = f"RET_PRO_{n:03d}"
            docs.append({"id": doc_id, "type": "PRO", "type_name": "Procedure", "owner": "returns", "language": lang,
                         "format": fmt, "target_mb": 0.05, "version": "1.0", "sections": []})
            text = "\n".join(HEAD[lang] + ["", PROSE[lang] + FACT[lang], "", PROSE[lang], ""])
            (base / "narrative" / f"{doc_id}.txt").write_bytes(text.encode("utf-8"))
        facts = [{"id": "F1", "family": "F1", "label": "returns", "kind": "single", "values": ["3", "3"], **FACT,
                  "placements": [d["id"] for d in docs]}]
        (base / "plan" / "facts.json").write_bytes(json.dumps(facts, ensure_ascii=False).encode("utf-8"))
        (base / "plan" / "catalogue.json").write_bytes(json.dumps(docs, ensure_ascii=False).encode("utf-8"))
        assemble.assemble()
        pristine = scratch / "pristine"
        shutil.copytree(base / "full", pristine)

        def state():
            changes = [{k: v for k, v in e.items() if k != "finished_after_a_crash"} for e in mutate.entries(base / "full.journal.jsonl")]
            return changes, {p.relative_to(base / "full").as_posix(): p.read_bytes() for p in (base / "full").rglob("*") if p.is_file()}

        expected, writes, during = [state()], [], []
        real = {"append": mutate.append, "replace": os.replace, "write_bytes": Path.write_bytes, "unlink": Path.unlink}

        def run(kill_at=None):
            def counted(name):
                def write(*a, **k):
                    if len(writes) == kill_at:
                        if name == "append":
                            line = (json.dumps(a[1], ensure_ascii=False) + "\n").encode("utf-8")
                            with open(a[0], "ab") as handle:
                                handle.write(line[: len(line) // 2])
                        raise KeyboardInterrupt
                    writes.append(name)
                    if kill_at is None:
                        during.append(len(expected))  # the change being made, counted from 1
                    result = real[name](*a, **k)
                    if kill_at is None and name == "append" and a[1]["state"] == "applied":
                        expected.append(state())
                    return result
                return write

            mutate.append, os.replace = counted("append"), counted("replace")
            Path.write_bytes, Path.unlink = counted("write_bytes"), counted("unlink")
            try:
                mutate.mutate(count=len(mutate.KINDS), seed=3)
            finally:
                mutate.append, os.replace = real["append"], real["replace"]
                Path.write_bytes, Path.unlink = real["write_bytes"], real["unlink"]

        run()
        clean, change_of = list(writes), list(during)
        finished = dropped = equal = 0
        for kill_at in range(len(clean)):
            shutil.rmtree(base / "full")
            shutil.copytree(pristine, base / "full")
            (base / "full.journal.jsonl").unlink(missing_ok=True)
            shutil.rmtree(base / "full.staging", ignore_errors=True)
            writes.clear()
            try:
                run(kill_at)
            except KeyboardInterrupt:
                pass
            mutate.mutate(count=0)
            changes, files = state()
            finished += len(changes) == change_of[kill_at]
            dropped += len(changes) == change_of[kill_at] - 1
            equal += (changes, files) == expected[len(changes)]
        kinds = ", ".join(f"{name} {clean.count(name)}" for name in sorted(set(clean)))
        print(f"writes in {len(mutate.KINDS)} changes: {len(clean)} ({kinds})")
        print(f"kill points {len(clean)}: the change finished by the next run {finished}, dropped {dropped}; "
              f"equal to the crash free state {equal} of {len(clean)}")
        return 0 if equal == len(clean) else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
