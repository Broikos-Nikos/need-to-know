"""Assembly on a three document corpus in a temporary folder: a build or a swap killed at any step leaves a
corpus and a journal that agree, and the check names every file that is missing or unexplained (OP-F1)."""

import json
import os
import shutil
from pathlib import Path

import pytest

from needtoknow.corpus import assemble, check, mutate
from needtoknow.corpus.plan import header

FACT = {"en": "Returned items are inspected within 3 working days of arrival at the warehouse.",
        "el": "Τα επιστρεφόμενα είδη ελέγχονται εντός 3 εργάσιμων ημερών από την άφιξή τους στην αποθήκη."}
PROSE = {"en": "The returns desk checks each parcel against its authorisation and records the result in Dock before anything moves on. " * 3,
         "el": "Το γραφείο επιστροφών ελέγχει κάθε δέμα με βάση την εξουσιοδότηση και καταγράφει το αποτέλεσμα στο Dock πριν προχωρήσει. " * 3}


@pytest.fixture
def base(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    docs = []
    for n, (lang, fmt) in enumerate([("en", "txt"), ("el", "md"), ("en", "html")], 1):
        doc = {"id": f"RET_PRO_{n:03d}", "type": "PRO", "type_name": "Procedure", "owner": "returns", "language": lang,
               "format": fmt, "target_mb": 0.01, "version": "1.0",
               "sections": [{"label": "returns", "topic_en": "Inspection", "topic_el": "Έλεγχος", "facts": ["F1"], "canary": None}]}
        docs.append(doc)
        text = "\n".join(header(doc, 1, doc["sections"][0]) + ["", PROSE[lang] + FACT[lang], ""])
        (base / "narrative" / f"{doc['id']}.txt").write_bytes(text.encode("utf-8"))
    facts = [{"id": "F1", "family": "F1", "label": "returns", "kind": "single", "values": ["3", "3"], **FACT,
              "placements": [d["id"] for d in docs]}]
    (base / "plan" / "catalogue.json").write_bytes(json.dumps(docs, ensure_ascii=False).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(json.dumps(facts, ensure_ascii=False).encode("utf-8"))
    assert assemble.assemble()["written"] == 3
    return base


def state(base: Path):
    """Every byte of the corpus and of its journal: what a kill must never leave disagreeing."""
    journal = base / "full.journal.jsonl"
    files = {p.relative_to(base).as_posix(): p.read_bytes() for p in (base / "full").rglob("*") if p.is_file()}
    return files, journal.read_bytes() if journal.is_file() else None


def leftovers(base: Path) -> list[str]:
    return [n for n in ("full.building", "full.retired", "full.retired.journal.jsonl") if (base / n).exists()]


def test_a_clean_build_passes_and_every_kind_of_journaled_change_still_passes(base):
    assert check.check()["problems"] == [] and check.check()["counts"]["documents in full"] == 3
    entries = {e["kind"]: e for e in mutate.mutate(count=len(mutate.KINDS), seed=3)}
    assert check.check()["problems"] == []
    # The same changes without their journal: each one the inventory can see is named.
    assert len({entries[k]["doc_id"] for k in ("delete_file", "new_file", "value_update")}) == 3
    (base / "full.journal.jsonl").unlink()
    problems = check.check()["problems"]
    assert f"{entries['new_file']['doc_id']}: in full/ but neither in the catalogue nor added by the journal" in problems
    assert f"{entries['delete_file']['doc_id']}: no assembled file in full/" in problems
    assert any(entries["value_update"]["doc_id"] in p and "fact F1 missing from the assembled file" in p for p in problems)


def test_every_missing_file_sidecar_and_stray_is_named(base):
    full = base / "full"
    (full / "RET_PRO_001.txt").unlink()
    (full / "RET_PRO_002.meta.json").unlink()
    (full / "notes.txt").write_bytes(b"left here by hand\n")
    (full / "copy").mkdir()
    for name in ("RET_PRO_003.html", "RET_PRO_003.meta.json"):
        shutil.copy(full / name, full / "copy" / name)
    problems = check.check()["problems"]
    for expected in ["RET_PRO_001.meta.json: a sidecar without its file", "RET_PRO_001: no assembled file in full/",
                     "RET_PRO_002.md: a file no sidecar describes", "RET_PRO_002: no assembled file in full/",
                     "notes.txt: a file no sidecar describes",
                     "copy/RET_PRO_003.meta.json: a second file for RET_PRO_003, the first is RET_PRO_003.meta.json"]:
        assert expected in problems, expected
    assert len(problems) == 6, problems


def test_a_build_killed_halfway_leaves_the_previous_corpus_and_its_journal_untouched(base, monkeypatch):
    mutate.mutate(count=len(mutate.KINDS), seed=3)
    before = state(base)
    real, calls = assemble.assemble_doc, []

    def dying(*args, **kwargs):
        calls.append(args[0]["id"])
        if len(calls) > 2:
            raise KeyboardInterrupt("killed after two documents")
        return real(*args, **kwargs)

    monkeypatch.setattr(assemble, "assemble_doc", dying)
    with pytest.raises(KeyboardInterrupt):
        assemble.assemble()
    assert state(base) == before and check.check()["problems"] == []
    assert leftovers(base) == ["full.building"]
    monkeypatch.setattr(assemble, "assemble_doc", real)
    assemble.settle(assemble.places(base, "full"))
    assert state(base) == before and leftovers(base) == []


@pytest.mark.parametrize("survive", [0, 1, 2, 3])
def test_a_swap_killed_at_any_step_leaves_the_corpus_and_its_journal_agreeing(base, monkeypatch, survive):
    """The swap is three renames and a cleanup: a kill after any number of them, then the next build's settle."""
    fresh = state(base)
    mutate.mutate(count=len(mutate.KINDS), seed=3)
    before = state(base)
    steps = []

    def counted(real):
        def step(*args, **kwargs):
            if len(steps) == survive:
                raise KeyboardInterrupt(f"killed after {survive} steps")
            steps.append(real.__name__)
            return real(*args, **kwargs)
        return step

    with monkeypatch.context() as m:
        m.setattr(Path, "rename", counted(Path.rename))
        m.setattr(shutil, "rmtree", counted(shutil.rmtree))
        with pytest.raises(KeyboardInterrupt):
            assemble.assemble()
    assert steps == ["rename", "rename", "rename", "rmtree"][:survive]
    # A corpus in place always comes with its own journal; between the renames there is none, and the check says so.
    assert check.check()["problems"] == ([] if survive in (0, 3) else ["full/: no assembled corpus"])
    assemble.settle(assemble.places(base, "full"))
    assert state(base) == (fresh if survive == 3 else before) and leftovers(base) == []
    assert check.check()["problems"] == []


def test_a_change_left_pending_is_named_until_the_next_run_finishes_it(base, monkeypatch):
    # OP-F2: the journal line is written, then the simulator dies before the change is made.
    def killed(*args, **kwargs):
        raise KeyboardInterrupt("killed between the journal line and the change")

    with monkeypatch.context() as m:
        m.setattr(os, "replace", killed)
        with pytest.raises(KeyboardInterrupt):
            mutate.mutate(count=1, seed=3)
    assert check.check()["problems"] == ["journal change 1 (value_update) is pending: the next ntk corpus mutate finishes it"]
    assert mutate.mutate(count=0) == [] and check.check()["problems"] == []
