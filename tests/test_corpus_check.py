"""The light check on a two section corpus in a temporary folder: it passes a faithful writer and names each kind of drift."""

import json

import pytest

from needtoknow.corpus import check
from needtoknow.corpus.plan import header

FACT = "Returned items are inspected within 3 working days of arrival at the warehouse."
PROSE = "The returns desk checks each parcel against its authorisation and records the result in Dock before it moves on. " * 2


@pytest.fixture
def write(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    doc = {"id": "RET_PRO_001", "language": "en", "owner": "returns", "sections": [
        {"label": "returns", "topic_en": "Inspection", "topic_el": "Έλεγχος", "facts": ["F1"], "canary": None},
        {"label": "returns", "topic_en": "Inspection, part 2", "topic_el": "Έλεγχος, μέρος 2", "facts": [], "canary": None}]}
    fact = {"id": "F1", "label": "returns", "en": FACT, "el": FACT, "placements": ["RET_PRO_001"]}
    (base / "plan" / "catalogue.json").write_text(json.dumps([doc]), encoding="utf-8")
    (base / "plan" / "facts.json").write_text(json.dumps([fact]), encoding="utf-8")

    def narrative(first_heading=None, body=f"{PROSE}{FACT}"):
        heads = [header(doc, n, s) for n, s in enumerate(doc["sections"], 1)]
        if first_heading:
            heads[1][0] = first_heading
        text = "\n".join(heads[0] + ["", body, ""] + heads[1] + ["", PROSE, ""])
        (base / "narrative" / "RET_PRO_001.txt").write_text(text, encoding="utf-8")
        return check.check()
    return narrative


def test_a_faithful_writer_passes(write):
    result = write()
    assert result["problems"] == [] and result["counts"]["facts found"] == 1


def test_a_heading_that_drifted_from_the_plan_is_named(write):
    result = write(first_heading="SECTION 2  Inspection")
    assert any("heading" in p and "part 2" in p for p in result["problems"])


def test_a_fact_with_a_changed_value_is_named(write):
    result = write(body=PROSE + FACT.replace("3", "4"))
    assert any("fact F1 missing or altered" in p for p in result["problems"])
