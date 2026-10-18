"""The change simulator on a small corpus built in a temporary folder: each change does what its journal says."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from needtoknow.corpus import CorpusBusy, assemble, exclusive, mutate

FACT_EN = "Returned items are inspected within 3 working days of arrival at the warehouse."
FACT_EL = "Τα επιστρεφόμενα είδη ελέγχονται εντός 3 εργάσιμων ημερών από την άφιξή τους στην αποθήκη."
PROSE_EN = "The returns desk checks each parcel against its authorisation and records the result in Dock before anything moves on. " * 3
PROSE_EL = "Το γραφείο επιστροφών ελέγχει κάθε δέμα με βάση την εξουσιοδότηση και καταγράφει το αποτέλεσμα στο Dock πριν προχωρήσει. " * 3


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    facts = [{"id": "F1", "family": "F1", "label": "returns", "kind": "single", "values": ["3", "3"],
              "en": FACT_EN, "el": FACT_EL, "placements": ["RET_PRO_001", "RET_PRO_002", "RET_PRO_003"]}]
    docs = []
    for n, (lang, fmt) in enumerate([("en", "txt"), ("el", "md"), ("en", "html")], 1):
        doc_id = f"RET_PRO_{n:03d}"
        docs.append({"id": doc_id, "type": "PRO", "type_name": "Procedure", "owner": "returns", "language": lang,
                     "format": fmt, "target_mb": 0.05, "version": "1.0", "sections": []})
        head = (["ΕΝΟΤΗΤΑ 1  Έλεγχος", "Κοινό: Επιστροφές", "Διαβάθμιση: Εσωτερικό"] if lang == "el"
                else ["SECTION 1  Inspection", "Audience: Returns", "Classification: Internal"])
        prose, fact = (PROSE_EL, FACT_EL) if lang == "el" else (PROSE_EN, FACT_EN)
        (base / "narrative" / f"{doc_id}.txt").write_text("\n".join(head + ["", f"{prose}{fact}", "", prose, ""]), encoding="utf-8")
    (base / "plan" / "facts.json").write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    (base / "plan" / "catalogue.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    assert assemble.assemble()["written"] == 3
    return base


def test_every_kind_of_change_is_applied_and_journaled(corpus):
    entries = mutate.mutate(count=len(mutate.KINDS), seed=3)
    journal = mutate.entries(corpus / "full.journal.jsonl")
    assert [e["seq"] for e in journal] == list(range(1, len(mutate.KINDS) + 1))
    assert all(e["state"] == "applied" for e in journal) and not (corpus / "full.staging").exists()
    assert all(e["expected"] for e in entries)
    # A fallback is journaled as what was done, with what was asked beside it.
    assert all(e["kind"] in mutate.KINDS and e["requested"] in mutate.KINDS for e in journal)
    assert all(e["kind"] == e["requested"] or "note" in e for e in journal)


def test_a_value_update_changes_the_value_and_nothing_else(corpus):
    before = {d: f.read_bytes() for d, (f, _, _) in mutate.corpus_files(corpus / "full").items()}
    (entry,) = mutate.mutate(count=1, seed=1)
    assert entry["kind"] == "value_update" and entry["old_value"] == "3" and entry["new_value"] == "4"
    file, _, meta = mutate.corpus_files(corpus / "full")[entry["doc_id"]]
    text = mutate.read(file, meta)
    assert mutate.as_written(entry["sentence"], meta["format"]) in text
    old = FACT_EL if meta["language"] == "el" else FACT_EN
    assert mutate.as_written(old, meta["format"]) not in text
    assert len(file.read_bytes()) == len(before[entry["doc_id"]])  # one digit for one digit


def test_a_value_is_replaced_as_a_whole_token_never_inside_a_code():
    sentence = "Tickets of priority P1 receive a first response within 1 hours."
    assert mutate.replace_value(sentence, "1", "2") == "Tickets of priority P1 receive a first response within 2 hours."
    assert mutate.replace_value("Courier C1 picks up 1 pallet.", "1", "3") == "Courier C1 picks up 3 pallet."


def test_a_relabel_touches_only_the_audience(corpus):
    mutate.mutate(count=2, seed=5)  # value_update, insert_paragraph
    files = mutate.corpus_files(corpus / "full")
    before = {d: mutate.read(f, m) for d, (f, _, m) in files.items()}
    (entry,) = mutate.mutate(count=1, seed=5)
    assert entry["kind"] == "relabel" and entry["to_labels"] == ["returns", "legal"]
    file, _, meta = mutate.corpus_files(corpus / "full")[entry["doc_id"]]
    old_lines, new_lines = before[entry["doc_id"]].splitlines(), mutate.read(file, meta).splitlines()
    changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
    assert len(old_lines) == len(new_lines) and 1 <= len(changed) <= 2
    assert all(("Legal" in b or "Νομικό Τμήμα" in b) for _, b in changed)


def test_a_rename_and_a_re_encode_keep_the_text(corpus):
    mutate.mutate(count=3, seed=7)
    texts = {d: mutate.read(f, m) for d, (f, _, m) in mutate.corpus_files(corpus / "full").items()}
    renamed, encoded = mutate.mutate(count=2, seed=7)
    assert renamed["kind"] == "rename" and renamed["file"].startswith("moved/")
    assert encoded["kind"] == "re_encode"
    files = mutate.corpus_files(corpus / "full")
    for entry in (renamed, encoded):
        file, _, meta = files[entry["doc_id"]]
        assert mutate.read(file, meta) == texts[entry["doc_id"]]


def journal_and_corpus(base: Path):
    """The journal's changes and every byte under full/: what a crash and the next run must leave agreeing."""
    changes = [{k: v for k, v in e.items() if k != "finished_after_a_crash"} for e in mutate.entries(base / "full.journal.jsonl")]
    return changes, {p.relative_to(base / "full").as_posix(): p.read_bytes() for p in (base / "full").rglob("*") if p.is_file()}


def test_a_crash_at_any_write_of_a_change_is_finished_or_dropped_by_the_next_run(corpus, monkeypatch):
    """OP-F2. Every write of nine changes is killed in turn, a journal line cut in half where the write is one.
    The next run must leave the corpus and journal exactly as a run without the crash left them after the
    same number of changes: a change journaled before the kill is finished, any other is dropped."""
    pristine = corpus.parent / "pristine"
    shutil.copytree(corpus / "full", pristine)
    expected, writes = [journal_and_corpus(corpus)], []

    def run(kill_at=None):
        def counted(real, name):
            def write(*args, **kwargs):
                if len(writes) == kill_at:
                    if name == "append":
                        line = (json.dumps(args[1], ensure_ascii=False) + "\n").encode("utf-8")
                        with open(args[0], "ab") as handle:
                            handle.write(line[: len(line) // 2])
                    raise KeyboardInterrupt(f"killed at write {kill_at}, {name}")
                writes.append(name)
                result = real(*args, **kwargs)
                if kill_at is None and name == "append" and args[1]["state"] == "applied":
                    expected.append(journal_and_corpus(corpus))
                return result
            return write

        with monkeypatch.context() as m:
            m.setattr(mutate, "append", counted(mutate.append, "append"))
            m.setattr(os, "replace", counted(os.replace, "replace"))
            m.setattr(Path, "write_bytes", counted(Path.write_bytes, "write_bytes"))
            m.setattr(Path, "unlink", counted(Path.unlink, "unlink"))
            mutate.mutate(count=len(mutate.KINDS), seed=3)

    run()
    clean = list(writes)
    assert len(expected) == len(mutate.KINDS) + 1 and {"append", "replace", "write_bytes", "unlink"} == set(clean)
    for kill_at in range(len(clean)):
        shutil.rmtree(corpus / "full")
        shutil.copytree(pristine, corpus / "full")
        (corpus / "full.journal.jsonl").unlink(missing_ok=True)
        shutil.rmtree(corpus / "full.staging", ignore_errors=True)
        writes.clear()
        with pytest.raises(KeyboardInterrupt):
            run(kill_at)
        assert mutate.mutate(count=0) == []
        changes, files = journal_and_corpus(corpus)
        assert all(c["state"] == "applied" for c in changes), kill_at
        assert (changes, files) == expected[len(changes)], f"killed at write {kill_at}, {clean[kill_at]}"
        assert not (corpus / "full.staging").exists(), kill_at
        # The journal takes the next change cleanly, so nothing of a line cut short was left in it.
        mutate.mutate(count=1, seed=11)
        assert [c["seq"] for c in mutate.entries(corpus / "full.journal.jsonl")] == list(range(1, len(changes) + 2)), kill_at


def test_one_writer_at_a_time(corpus):
    # OP-F3: two simulators, or a simulator and a build, would interleave their writes and their journal.
    with exclusive(corpus):
        with pytest.raises(CorpusBusy):
            mutate.mutate(count=1)
        with pytest.raises(CorpusBusy):
            assemble.assemble()
    assert not (corpus / "full.journal.jsonl").exists()
    assert [e["seq"] for e in mutate.mutate(count=1)] == [1]


HOLD = ("import pathlib, sys, time\nfrom needtoknow.corpus import exclusive\n"
        "with exclusive(pathlib.Path(sys.argv[1])):\n    print('held', flush=True)\n    time.sleep(60)\n")


def test_a_writer_that_dies_leaves_no_lock_behind(corpus):
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(corpus)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(CorpusBusy):
            mutate.mutate(count=1)
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()
    for _ in range(50):
        try:
            assert [e["seq"] for e in mutate.mutate(count=1)] == [1]
            break
        except CorpusBusy:
            time.sleep(0.1)
    else:
        pytest.fail("the lock outlived the process that held it")


def test_the_simulator_leaves_no_file_open(corpus):
    # DR-F6: measured with -X dev, the first simulator left the journal open after counting its lines.
    code = "from needtoknow.corpus.mutate import mutate\nmutate(9, seed=3)\nmutate(9, seed=4)\n"
    run = subprocess.run([sys.executable, "-X", "dev", "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert run.returncode == 0, run.stderr[-2000:]
    assert "ResourceWarning" not in run.stderr, run.stderr[-2000:]


def test_the_journal_is_read_in_whole_lines_even_cut_inside_a_greek_letter(tmp_path):
    # DR-F30: entries() decoded the whole file first, so a line cut inside a Greek letter raised.
    journal = tmp_path / "full.journal.jsonl"
    line = json.dumps({"seq": 1, "state": "applied", "new": "Ελληνικά"}, ensure_ascii=False).encode("utf-8")
    journal.write_bytes(line + b"\n" + line[:line.index("Ελ".encode("utf-8")) + 3])
    assert [entry["seq"] for entry in mutate.entries(journal)] == [1]
