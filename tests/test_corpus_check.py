"""The light check on a two section corpus in a temporary folder: it passes a faithful writer and names each kind of drift."""

import json

import pytest

from needtoknow.corpus import assemble, check
from needtoknow.corpus.plan import header

FACT = "Returned items are inspected within 3 working days of arrival at the warehouse."
PROSE = "The returns desk checks each parcel against its authorisation and records the result in Dock before it moves on. " * 2


@pytest.fixture
def write(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    doc = {"id": "RET_PRO_001", "language": "en", "owner": "returns", "type": "PRO", "type_name": "Procedure",
           "format": "txt", "target_mb": 0.01, "version": "1.0", "sections": [
        {"label": "returns", "topic_en": "Inspection", "topic_el": "Έλεγχος", "facts": ["F1"], "canary": None},
        {"label": "returns", "topic_en": "Inspection, part 2", "topic_el": "Έλεγχος, μέρος 2", "facts": [], "canary": None}]}
    fact = {"id": "F1", "label": "returns", "en": FACT, "el": FACT, "placements": ["RET_PRO_001"]}
    (base / "plan" / "catalogue.json").write_text(json.dumps([doc]), encoding="utf-8")
    (base / "plan" / "facts.json").write_text(json.dumps([fact]), encoding="utf-8")

    def narrative(first_heading=None, body=f"{PROSE}{FACT}", tamper=None, assembled=True):
        heads = [header(doc, n, s) for n, s in enumerate(doc["sections"], 1)]
        if first_heading:
            heads[1][0] = first_heading
        text = "\n".join(heads[0] + ["", body, ""] + heads[1] + ["", PROSE, ""])
        (base / "narrative" / "RET_PRO_001.txt").write_text(text, encoding="utf-8")
        if assembled:
            assemble.assemble()
        if tamper is not None:
            path = base / "full" / "RET_PRO_001.txt"
            path.write_bytes(tamper(path.read_bytes().decode("utf-8")).encode("utf-8"))
        return check.check()
    return narrative


@pytest.mark.parametrize("text", ["within 3 working days", "εντός 3 εργάσιμων ημερών", "εντός 5 ημερών",
                                  "a fee of 30 €", "up to €250", "within 48 hours", "15%", "για 2 έτη", "4 μήνες"])
def test_every_form_of_a_policy_value_is_seen(text):
    # DR-F1: the first detector missed five of the first seven of these.
    assert check.policy_values(text), text


@pytest.mark.parametrize("text", ["form RF2", "step 3 of the procedure", "courier C1", "priority P1", "since 2025"])
def test_codes_and_steps_are_not_policy_values(text):
    assert check.policy_values(text) == [], text


def test_a_fact_missing_from_the_assembled_file_is_named(write):
    # DR-F3: the narrative is not what gets indexed; the assembled file is.
    assert write(tamper=lambda text: text)["problems"] == []
    result = write(tamper=lambda text: text.replace(FACT, "The warehouse inspects returned items."))
    assert any("fact F1 missing from the assembled file" in p for p in result["problems"])


def test_a_faithful_writer_passes(write):
    result = write()
    assert result["problems"] == [] and result["counts"]["facts found"] == 1


def test_a_corpus_never_assembled_is_named(write):
    # OP-F1: a check that passes with no corpus would pass a swap killed halfway.
    assert write(assembled=False)["problems"] == ["full/: no assembled corpus"]


def test_a_heading_that_drifted_from_the_plan_is_named(write):
    result = write(first_heading="SECTION 2  Inspection")
    assert any("heading" in p and "part 2" in p for p in result["problems"])


def test_a_fact_with_a_changed_value_is_named(write):
    result = write(body=PROSE + FACT.replace("3", "4"))
    assert any("fact F1 missing or altered" in p for p in result["problems"])
