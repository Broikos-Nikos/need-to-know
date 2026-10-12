"""The corpus plan: the same seed gives the same Company, and every fact is planted where its label belongs."""

import random

from needtoknow.corpus.plan import SEED, make_catalogue, make_facts, world


def plan():
    rng = random.Random(SEED)
    facts = make_facts(rng, world())
    return facts, make_catalogue(rng, facts)


def test_the_same_seed_gives_the_same_facts_and_catalogue():
    first, second = plan(), plan()
    assert [f["en"] for f in first[0]] == [f["en"] for f in second[0]]
    assert [d["id"] for d in first[1]] == [d["id"] for d in second[1]]


def test_every_fact_is_rendered_in_both_languages_with_its_value():
    facts, _ = plan()
    for f in facts:
        assert "{" not in f["en"] + f["el"], f["id"]
        assert f["values"][0] in f["en"] and f["values"][1] in f["el"], f["id"]
        assert any("Ͱ" <= c <= "Ͽ" for c in f["el"]) and not any("Ͱ" <= c <= "Ͽ" for c in f["en"])


def test_every_fact_is_planted_in_a_section_of_its_own_label():
    facts, docs = plan()
    sections = {(d["id"], fid): s["label"] for d in docs for s in d["sections"] for fid in s["facts"]}
    for f in facts:
        assert f["placements"], f["id"]
        for doc_id in f["placements"]:
            assert sections[(doc_id, f["id"])] == f["label"], (doc_id, f["id"])


def test_no_document_repeats_a_heading():
    # DR-F2: topics are drawn with replacement, so a repeat becomes "..., part 2".
    _, docs = plan()
    for doc in docs:
        headings = [s["topic_en"] for s in doc["sections"]]
        assert len(headings) == len(set(headings)), doc["id"]
        greek = [s["topic_el"] for s in doc["sections"]]
        assert len(greek) == len(set(greek)), doc["id"]


def test_a_superseded_value_differs_from_the_one_that_replaced_it():
    facts, _ = plan()
    by_id = {f["id"]: f for f in facts}
    old = [f for f in facts if f["kind"] == "superseded"]
    assert old
    for f in old:
        assert f["values"] != by_id[f["superseded_by"]]["values"], f["id"]
