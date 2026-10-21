"""The corpus plan: the same seed gives the same Company, and every fact is planted where its label belongs."""

import random
import re

from needtoknow.corpus.plan import SEED, lower_first, make_catalogue, make_facts, world
from needtoknow.settings import repo_root

# A value of one before a plural unit, a word between allowed ("1 working days"), in both languages: the units are
# any a value can stand before, whether or not a template writes them today.
PLURALS = ("days", "hours", "minutes", "months", "weeks", "years", "euros", "kilograms", "ημέρες", "ημερών", "ώρες",
           "ωρών", "λεπτά", "λεπτών", "μήνες", "μηνών", "εβδομάδες", "εβδομάδων", "έτη", "ετών", "χρόνια", "χρόνων", "κιλά")
ONE_BEFORE_PLURAL = re.compile(rf"(?<![\d.,])1 (?:\w+ )?(?:{'|'.join(PLURALS)})(?!\w)")


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


def test_a_dated_sentence_goes_on_in_lower_case_unless_it_starts_with_an_acronym():
    # KB3-K7: "From 1 July 2025, A credit check ..." in 8 English and 6 Greek sentences.
    assert lower_first("A credit check is required.") == "a credit check is required."
    assert lower_first("Ο πελάτης δικαιούται.") == "ο πελάτης δικαιούται."
    assert lower_first("SLA breaches are reported.") == "SLA breaches are reported."
    facts, _ = plan()
    frame = re.compile(r"^(?:From \d+ \w+ \d{4}|Από την \d+ \w+ \d{4}), (\w+)")
    firsts = [m.group(1) for f in facts for lang in ("en", "el") if (m := frame.match(f[lang]))]
    assert len(firsts) == 2 * sum(f["kind"] in ("versioned", "superseded") for f in facts) > 0
    assert all(w[0].islower() or (len(w) > 1 and w.isupper()) for w in firsts), firsts


def test_a_superseded_value_differs_from_the_one_that_replaced_it():
    facts, _ = plan()
    by_id = {f["id"]: f for f in facts}
    old = [f for f in facts if f["kind"] == "superseded"]
    assert old
    for f in old:
        assert f["values"] != by_id[f["superseded_by"]]["values"], f["id"]


def test_a_value_of_one_takes_its_unit_in_the_singular():
    # "Call recordings are kept for 1 years" was in legal's own question as the page replays it, and "1 εργάσιμες
    # ημέρες" in five documents' delivery sections: a planted value of one takes the singular, in both languages.
    facts, _ = plan()
    ones = {f["id"]: f for f in facts if f["values"][0] == "1"}
    assert ones and not [f[lang] for f in facts for lang in ("en", "el") if ONE_BEFORE_PLURAL.search(f[lang])]
    assert ones["F35V00"]["en"] == "Call recordings are kept for 1 year and then deleted."
    assert ones["F22R00"]["el"].endswith("διαρκεί 1 εργάσιμη ημέρα.")


def test_no_value_of_one_stands_before_a_plural_unit_in_the_narrative():
    # The narrative is the committed source of the corpus, so the rule holds in a clone that has built nothing.
    texts = {path.name: path.read_text(encoding="utf-8") for path in sorted((repo_root() / "narrative").glob("*.txt"))}
    found = [(name, m.group(0)) for name, text in texts.items() for m in ONE_BEFORE_PLURAL.finditer(text)]
    assert texts and not found, found
