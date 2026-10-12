"""The lexical analyser and BM25 vectors (T3.2): folding, stopwords, stemming by script, the
values Qdrant gets, and statistics frozen per index (KB1-F15, RS-F3)."""

import hashlib
import importlib.metadata
from pathlib import Path

import pytest

from needtoknow import lexical
from needtoknow.lexical import K1, LexicalError, document, freeze, frozen, measure, query, terms


def test_case_accents_and_final_sigma_leave_a_term_alone():
    assert terms("ΕΠΙΣΤΡΟΦΕΣ") == terms("Επιστροφές") == terms("επιστροφες") == ["επιστροφ"]
    assert terms("Είναι") == []  # a stopword once its accent is gone
    assert terms("café 10ός") == terms("cafe 10ος") == ["cafe", "10οσ"]  # words the Greek stemmer never sees


def test_each_word_is_stemmed_by_its_own_script_not_by_its_unit_language():
    assert terms("Η εφαρμογή Ledger κάνει processing των devices") == \
        ["εφαρμογ", "ledger", "καν", "process", "devic"]
    assert terms("the shop of Παπαδόπουλος") == ["shop", "παπαδοπουλ"]


def test_codes_numbers_and_ordinals_stay_whole():
    assert terms("RMA133420, T4698317 and CC591 on 2024-10-10") == ["rma133420", "t4698317", "cc591", "2024", "10", "10"]
    assert terms("10ος όροφος") == ["10οσ", "οροφ"]  # the Greek stemmer would make the ordinal the number 10


def test_the_stopwords_of_both_languages_are_dropped_as_lucene_lists_them():
    assert terms("the returns are for the shop και οι επιστροφές για το κατάστημα") == \
        ["return", "shop", "επιστροφ", "καταστημ"]
    assert len(lexical.STOPWORDS) == 108


def test_rs_f3_the_question_shares_no_word_and_three_stems_with_its_planted_sentence():
    question = "πόσες ημέρες δέχεστε επιστροφές συσκευών πελάτες"
    planted = "πελάτης μπορεί επιστρέψει συσκευή 30 ημερών"
    assert set(question.split()) & set(planted.split()) == set()
    assert len(terms(question)) == len(terms(planted)) == 6
    assert set(terms(question)) & set(terms(planted)) == {"ημερ", "συσκευ", "πελατ"}


def test_a_passage_value_is_bm25_term_frequency_at_the_frozen_average_length():
    indices, values = document("cat dog dog", avgdl=3.0)
    weight = dict(zip(indices, values))
    assert weight[lexical.term_id("dog")] == pytest.approx(2 * (K1 + 1) / (2 + K1))
    assert weight[lexical.term_id("cat")] == pytest.approx(1.0)
    _, shorter_average = document("cat dog dog", avgdl=1.5)
    assert max(shorter_average) < max(values)
    assert indices == sorted(indices) and all(0 <= i < 2 ** 32 for i in indices)


def test_terms_that_collide_share_one_weight(monkeypatch):
    monkeypatch.setattr(lexical, "term_id", lambda term: 7)
    assert document("cat dog", avgdl=2.0) == ([7], [pytest.approx(2 * (K1 + 1) / (2 + K1))])


def test_a_question_sends_each_of_its_terms_once_at_one():
    indices, values = query("returns, returns and the return window")
    assert len(indices) == 2 and values == [1.0, 1.0]


def test_statistics_are_measured_once_and_refused_when_another_analyser_wrote_them(tmp_path):
    path = tmp_path / "lexical.json"
    with pytest.raises(LexicalError, match="does not exist"):
        frozen(path)
    stats = measure(["cat dog", "cat", "the and of"])
    assert (stats["passages"], stats["avgdl"]) == (3, 1.0)
    freeze(stats, path)
    assert frozen(path) == stats
    path.write_bytes(path.read_bytes().replace(lexical.ANALYSER_VERSION.encode(), b"an older analyser"))
    with pytest.raises(LexicalError, match="rebuild the sparse side"):
        frozen(path)


def test_the_analyser_version_is_its_code_and_its_stemmer_release():
    digest = hashlib.sha256(Path(lexical.__file__).read_bytes()).hexdigest()[:16]
    assert lexical.ANALYSER_VERSION == f"{digest} snowballstemmer {importlib.metadata.version('snowballstemmer')}"
