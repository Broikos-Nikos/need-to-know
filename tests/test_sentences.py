"""Sentences (T2.5): the Greek question mark ends a sentence and English ";" does not, abbreviations, initials and
numbers do not, and every planted fact in both languages comes out as one whole sentence (KB1-F14)."""

import random

import pytest

from needtoknow.corpus.plan import SEED, make_facts, world
from needtoknow.ingest.sentences import sentence_spans

GREEK_QUESTION_MARK = chr(0x037E)


def split(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


@pytest.mark.parametrize("text, sentences", [
    ("Τι ώρα είναι; Είναι πέντε.", ["Τι ώρα είναι;", "Είναι πέντε."]),
    ("Τι ώρα είναι" + GREEK_QUESTION_MARK + " Είναι πέντε.", ["Τι ώρα είναι" + GREEK_QUESTION_MARK, "Είναι πέντε."]),
    ("Check the parcel; Then record it.", ["Check the parcel; Then record it."]),
    ("The fee, e.g. for returns, is set. The next step follows.", ["The fee, e.g. for returns, is set.", "The next step follows."]),
    ("Βλ. π.χ. την παρ. 3 του άρθρου. Η επόμενη ενότητα.", ["Βλ. π.χ. την παρ. 3 του άρθρου.", "Η επόμενη ενότητα."]),
    ("Version 2.3 applies from today. 14 days later it ends.", ["Version 2.3 applies from today.", "14 days later it ends."]),
    ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
    ("Signed by J. Smith on Monday. It holds.", ["Signed by J. Smith on Monday.", "It holds."]),
    ("Is it open? Yes! It opens at nine.", ["Is it open?", "Yes!", "It opens at nine."]),
    ("no capital follows. so it goes on.", ["no capital follows. so it goes on."]),
    ("", []), ("   ", []),
])
def test_sentence_ends(text, sentences):
    assert split(text) == sentences


def test_every_planted_fact_is_one_whole_sentence_in_both_languages():
    facts = make_facts(random.Random(SEED), world())
    broken = [(f["id"], lang) for f in facts for lang in ("en", "el") if split(f[lang]) != [f[lang]]]
    assert facts and broken == []


def test_spans_cover_every_word_and_stay_inside_the_text():
    text = "  First one. Second one; still second? Τρίτη; Τέταρτη.  "
    spans = sentence_spans(text)
    assert all(0 <= s < e <= len(text) and text[s:e] == text[s:e].strip() for s, e in spans)
    assert " ".join(text[s:e] for s, e in spans).split() == text.split()
