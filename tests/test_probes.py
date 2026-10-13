"""The retrieval probes (T5.4b): two a fact from the plan's own slots, worded apart from the statements, every one within
the overlap cap, and the committed files the ones the command makes."""

import pytest

from needtoknow.corpus.plan import TEMPLATES
from needtoknow.eval import probes as P
from needtoknow.eval.golden import Context


def test_every_statement_template_has_a_question_in_each_language():
    assert sorted(P.QUESTIONS) == list(range(len(TEMPLATES)))
    greek = lambda text: any(0x0370 <= ord(c) <= 0x03FF for c in text)  # noqa: E731
    assert all(not greek(en) and greek(el) for en, el in P.QUESTIONS.values())


def test_a_probe_is_filled_from_the_slots_the_plan_used():
    world = {"regions": {"en": ["Attica", "Crete"], "el": ["Αττική", "Κρήτη"]},
             "warehouses": [{"en": "the Sindos warehouse", "el": "η αποθήκη Σίνδου"}]}
    assert P.slots("F05R01", world)[:2] == ("Crete", "Κρήτη")
    assert P.slots("F21V01", world) == ("the Sindos warehouse", "αποθήκη Σίνδου", "C2", "C2")
    assert P.fill(P.QUESTIONS[21][0], "the Sindos warehouse", "C2") == \
        "What time does courier C2 make its final collection at our Sindos site?"


def test_a_probe_over_the_cap_or_without_gold_is_a_problem():
    ctx = type("Ctx", (), {"protocol": {"items": {"max_shared_rare_terms": 1}}})()
    rows = {"dev": [{"id": "DP001", "fact_id": "F1", "shared_rare": 2, "gold": ["u"], "readers": ["sales"]},
                    {"id": "DP002", "fact_id": "F1", "shared_rare": 0, "gold": [], "readers": []},
                    {"id": "DP003", "fact_id": "F2", "shared_rare": 1, "gold": ["v"], "readers": ["hr"]}]}
    found = P.problems(rows, ctx)
    assert len(found) == 2 and "DP001: shares 2" in found[0] and "DP002: no gold unit" in found[1]


def test_the_committed_probes_are_the_ones_the_command_makes():
    # Needs the plan, the frequency table and the manifest, which a build makes; a bare clone skips it.
    from needtoknow.eval.split import SplitError, load_plan
    from needtoknow.index.frequencies import table_path
    from needtoknow.ingest.manifest import manifest_path
    try:
        load_plan()
    except SplitError:
        pytest.skip("no plan: ntk corpus plan makes it")
    if not table_path().is_file() or not manifest_path().is_file():
        pytest.skip("no frequency table or manifest: ntk ingest build and ntk index sync make them")
    ctx = Context()
    made = P.generate(ctx)
    assert P.problems(made, ctx) == []
    assert {split: P.load(split) for split in P.SPLITS} == made


def test_a_fact_no_unit_holds_is_named_by_the_check_not_a_traceback(monkeypatch):
    # DR-F23: min() of no readers stopped the generator before problems could name the fact.
    from needtoknow.eval.split import SplitError, load_plan
    from needtoknow.index.frequencies import table_path
    from needtoknow.ingest.manifest import manifest_path
    try:
        load_plan()
    except SplitError:
        pytest.skip("no plan: ntk corpus plan makes it")
    if not table_path().is_file() or not manifest_path().is_file():
        pytest.skip("no frequency table or manifest: ntk ingest build and ntk index sync make them")
    real = P.gold_units
    monkeypatch.setattr(P, "gold_units", lambda fact, ctx, db: [] if fact["id"] == "F06V00" else real(fact, ctx, db))
    ctx = Context()
    made = P.generate(ctx)
    lost = [p for p in made["dev"] if p["fact_id"] == "F06V00"]
    assert lost and all(p["selective"] is None for p in lost)
    assert any("no gold unit or no reader for F06V00" in p for p in P.problems(made, ctx))
