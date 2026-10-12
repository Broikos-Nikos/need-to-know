"""The change simulator on a small corpus built in a temporary folder: each change does what its journal says."""

import json

import pytest

from needtoknow.corpus import assemble, mutate

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
    journal = [json.loads(line) for line in (corpus / "full.journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in journal] == list(range(1, len(mutate.KINDS) + 1))
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
