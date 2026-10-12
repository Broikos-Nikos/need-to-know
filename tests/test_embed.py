"""BGE-M3 on the GPU: the model card's numbers reproduce, vectors are sane, Greek works, order is kept.

Skipped where the `embed` extra, a CUDA device or the cached model is missing, so the
suite still runs on a machine without them.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from needtoknow.embed.model import EmbedError, encode, load  # noqa: E402

CARD_QUERIES = ["What is BGE M3?", "Defination of BM25"]
CARD_PASSAGES = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
]
CARD_SIMILARITY = [[0.6265, 0.3477], [0.3499, 0.678]]  # BAAI/bge-m3 model card, fp16

QUESTION_EL = "Σε πόσες ημέρες μπορεί ο πελάτης να επιστρέψει μια συσκευή;"
QUESTION_EN = "How many days does a customer have to return a device?"
ANSWER_EL = "Ο πελάτης μπορεί να επιστρέψει τη συσκευή εντός 30 ημερών από την αγορά, με την απόδειξη."
UNRELATED_EL = "Η αποθήκη της Θεσσαλονίκης παραλαμβάνει φορτία από τις 08:00 έως τις 16:00."


@pytest.fixture(scope="module", autouse=True)
def model():
    try:
        return load()
    except EmbedError as error:
        pytest.skip(str(error))


def test_the_model_cards_reference_similarities_reproduce():
    similarity = encode(CARD_QUERIES) @ encode(CARD_PASSAGES).T
    assert np.abs(similarity - np.array(CARD_SIMILARITY)).max() < 0.002, similarity


def test_vectors_are_finite_unit_length_rows_even_for_empty_and_overlong_text():
    texts = ["short", "", "λέξη " * 3000, "Mixed γλώσσα text, 30 ημέρες, 15:30"]
    vectors = encode(texts)
    assert vectors.shape == (len(texts), 1024) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)


def test_rows_come_back_in_the_order_given_though_batches_are_sorted_by_length():
    texts = ["a much longer sentence about returns, devices, receipts and the thirty day window", "short", "medium length"]
    batched = encode(texts, batch_size=2)
    alone = np.vstack([encode([text]) for text in texts])
    assert np.sum(batched * alone, axis=1).min() > 0.999


def test_a_greek_question_finds_its_greek_answer_and_so_does_its_english_twin():
    question_el, question_en, answer, unrelated = encode([QUESTION_EL, QUESTION_EN, ANSWER_EL, UNRELATED_EL])
    assert question_el @ answer > question_el @ unrelated + 0.15
    assert question_en @ answer > question_en @ unrelated + 0.15
