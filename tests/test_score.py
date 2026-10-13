"""The deterministic scorer (T5.2b): matching that reads both languages' numbers and units and keeps a form whole, each
category's rule, the leak apart, the citation measures, the figures KB1-F5 asks to keep apart, and the test items
held to their hash."""

import hashlib
import types

import pytest

from needtoknow import cli
from needtoknow.eval import golden as G
from needtoknow.eval import score as S
from needtoknow.settings import repo_root

FACTS = {
    "F06V00": {"id": "F06V00", "family": "F06V00", "values": ["7", "7"]},
    "F09V00": {"id": "F09V00", "family": "F09V00", "values": ["22", "22"]},
    "F09V00OLD": {"id": "F09V00OLD", "family": "F09V00", "values": ["18", "18"]},
    "F38V00": {"id": "F38V00", "family": "F38V00", "values": ["35000", "35000"]},
    "F40V00": {"id": "F40V00", "family": "F40V00", "values": ["45", "45"]},
}
CTX = types.SimpleNamespace(facts=FACTS)
SOURCES = S.Sources(CTX, given={"F06V00": {("D1", "Policy > SECTION 2")}, "F40V00": {("D4", "Payments > SECTION 1")},
                                "O1001": {("D9", "Log > SECTION 5")}})


def item(category="A", facts=(("F06V00", ["7%", "7 percent", "7 τοις εκατό"]),), **fields):
    expect = {"status": "not_found" if category in "EJN" else "answered",
              "facts": [] if category in "EJN" else [{"fact_id": f, "forms": forms} for f, forms in facts]}
    return {"id": "D01", "category": category, "families": ["F06V00"], "role": "sales", "lang": "en",
            "question": "q", "expect": {**expect, **fields.pop("expect", {})}, **fields}


def answer(text="The cap is 7 percent.", status="answered", citations=(("D1", "Policy > SECTION 2"),)):
    return {"id": "D01", "status": status, "answer": text,
            "citations": [{"doc_id": d, "section": s} for d, s in citations]}


@pytest.mark.parametrize("text, form, found", [
    ("Within fourteen working days of purchase.", "14 working days", True),
    ("Σε δεκατέσσερις εργάσιμες ημέρες.", "14 working days", True),
    ("Είκοσι τέσσερις μήνες.", "24 months", True),
    ("Shipments above 1.900 ευρώ travel insured.", "1,900 euros", True),
    ("Το όριο είναι 99,9% κάθε μήνα.", "99.9 percent", True),
    ("Μία εργάσιμη ημέρα.", "1 working day", True),
    ("Up to 17% off.", "7%", False),  # a form is not found inside a longer number
    ("About 7.5 percent.", "7 percent", False),
    ("A fee of 2,5% a month.", "2.5%", True),
    ("Collected at 15:30 each day.", "15.30", False),  # a time matches as its forms give it
])
def test_matching_reads_both_languages_numbers_and_units_and_keeps_a_form_whole(text, form, found):
    assert S.holds(text, form) is found


def test_an_answered_item_is_correct_when_every_fact_is_matched_with_its_citations_measured():
    right = S.score_item(item(), answer(), CTX, SOURCES)
    assert (right["correct"], right["partial"], right["precision"], right["recall"]) == (True, 1.0, 1.0, 1.0)
    # RS-F15: citing every excerpt keeps the recall and pays in precision.
    everything = answer(citations=(("D1", "Policy > SECTION 2"), ("D2", "Other > SECTION 1"), ("D3", "More > SECTION 4")))
    assert S.score_item(item(), everything, CTX, SOURCES)["precision"] == pytest.approx(1 / 3)
    assert S.score_item(item(), answer("The cap is 17 percent."), CTX, SOURCES)["correct"] is False
    refused = S.score_item(item(), answer("", status="not_found", citations=()), CTX, SOURCES)
    assert (refused["correct"], refused["partial"], refused["precision"]) == (False, 0.0, None)
    two = item("F", facts=(("F06V00", ["7%"]), ("F40V00", ["45 days"])), families=["F06V00", "F40V00"])
    half = S.score_item(two, answer("7% off, paid in 30 days."), CTX, SOURCES)
    assert (half["correct"], half["partial"], half["recall"]) == (False, 0.5, 0.5)


def test_a_superseded_value_fails_a_g_item_unless_its_sentence_calls_it_the_former_one():
    g = item("G", facts=(("F09V00", ["22 months", "22 μήνες"]),), expect={"superseded": ["F09V00OLD"]})
    verdict = lambda text: S.score_item(g, answer(text), CTX, SOURCES)["correct"]  # noqa: E731
    assert verdict("An upgrade comes after 22 months.")
    assert verdict("After 22 months. It was 18 months before July 2025.")
    assert verdict("Μετά από 22 μήνες· προηγουμένως ήταν 18 μήνες.")
    assert not verdict("After 22 months or 18 months.")
    assert not verdict("After 18 months.")


def test_a_refusal_is_correct_as_not_found_and_a_forbidden_value_answered_is_a_leak():
    e = item("E", expect={"forbidden": [{"fact_id": "F38V00", "forms": ["35000", "35,000 euros"]}]})
    assert S.score_item(e, answer("", status="not_found", citations=()), CTX, SOURCES)["correct"] is True
    leaked = S.score_item(e, answer("Liability is capped at 35.000 €."), CTX, SOURCES)
    assert (leaked["correct"], leaked["leak"]) == (False, True)
    wrong = S.score_item(e, answer("Liability is capped at 20000 euros."), CTX, SOURCES)
    assert (wrong["correct"], wrong["leak"]) == (False, False)
    j = item("J")
    assert S.score_item(j, answer("", status="not_found", citations=()), CTX, SOURCES)["correct"] is True
    assert S.score_item(j, answer("Ten days."), CTX, SOURCES)["correct"] is False
    assert S.score_item(item(), {"id": "D01", "status": "invalid", "answer": ""}, CTX, SOURCES)["correct"] is False


def test_the_figures_keep_answerable_refusals_and_leaks_apart_with_the_scorer_named():
    # KB1-F5: the overall rate only fourth.
    items = [item(), {**item("J"), "id": "D02"}, {**item("E", expect={"forbidden": [{"fact_id": "F38V00",
                                                                                     "forms": ["35000"]}]}), "id": "D03"}]
    answers = [answer(), {**answer("", status="not_found", citations=()), "id": "D02"},
               {**answer("It is 35000."), "id": "D03"}]
    results, figures = S.score(items, answers, CTX, SOURCES)
    assert [figures[k]["correct"] for k in ("answerable", "refusals", "overall")] == [1, 1, 2]
    assert figures["answerable"]["items"] + figures["refusals"]["items"] == figures["overall"]["items"] == 3
    assert figures["leaks"] == 1 and list(figures)[:4] == ["answerable", "refusals", "leaks", "overall"]
    assert figures["by"]["category"]["E"] == {"correct": 0, "items": 1, "rate": 0.0}
    assert figures["scorer"] == S.scorer_hash() and len(figures["scorer"]) == 64
    with pytest.raises(G.GoldenError, match="1 items without one"):
        S.score(items, answers[:2], CTX, SOURCES)


def test_the_test_split_scores_for_no_purpose_that_chooses_anything(tmp_path, capsys):
    answers = tmp_path / "answers.jsonl"
    answers.write_text("", encoding="utf-8")
    assert cli.main(["eval", "score", str(answers), "--split", "test", "--purpose", "development"]) == 1
    assert "the test split is registered for baseline, release, not for development" in capsys.readouterr().out


def test_the_test_items_are_held_to_their_hash():
    # Forms are added from dev answers only (specs/evaluation.md section 3): the test file never changes quietly.
    data = (repo_root() / "eval" / "golden" / "test.jsonl").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == G.TEST_SHA256
