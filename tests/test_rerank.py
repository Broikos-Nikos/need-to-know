"""The rerank switch (T3.6): each reranker puts the passage that answers above ones that do not, in either language;
a reranker reorders only the candidates it is given; and the switch refuses a mode it does not know."""

import pytest

from needtoknow.index import rerank

QUESTION = {"en": "How many days does a customer have to return a device?",
            "el": "Σε πόσες ημέρες μπορεί ο πελάτης να επιστρέψει μια συσκευή;"}
ANSWER = "Returns > SECTION 2  Window\nThe customer may return the device within 30 days of purchase, with the receipt."
OTHERS = ["Logistics > SECTION 4  Pallets\nPallets are wrapped and labelled before they leave the warehouse.",
          "Finance > SECTION 1  Ledger\nThe ledger closes on the last working day of each month."]
HITS = [{"unit_id": unit, "rank": rank, "score": 0.03} for rank, unit in enumerate(["pallets", "ledger", "answer"], 1)]


def test_a_reranker_reorders_the_candidates_it_was_given_and_adds_none(monkeypatch):
    monkeypatch.setattr(rerank, "cross_scores", lambda question, passages: [0.1, 0.9, 0.5])
    out = rerank.rerank("q", HITS, ["a", "b", "c"], "cross")
    assert [(h["unit_id"], h["rank"], h["score"]) for h in out] == [("ledger", 1, 0.9), ("answer", 2, 0.5), ("pallets", 3, 0.1)]


def test_none_keeps_the_fused_order_and_an_unknown_mode_is_refused():
    assert rerank.rerank("q", HITS, ["a", "b", "c"], "none") is HITS
    with pytest.raises(ValueError, match="no reranker 'colbert'"):
        rerank.rerank("q", HITS, ["a", "b", "c"], "colbert")


@pytest.mark.parametrize("mode", ["multivector", "cross"])
@pytest.mark.parametrize("language", ["en", "el"])
def test_each_reranker_puts_the_answer_first_in_either_language(mode, language):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    out = rerank.rerank(QUESTION[language], HITS, [*OTHERS, ANSWER], mode)
    assert out[0]["unit_id"] == "answer" and sorted(h["unit_id"] for h in out) == ["answer", "ledger", "pallets"]
    assert [h["rank"] for h in out] == [1, 2, 3]


def test_the_cross_encoder_scores_the_shortest_pairs_together_and_answers_in_the_order_given(monkeypatch):
    # PF-F4: batches of 16 in rank order, each padded to its longest pair; sorted by length a question took 572 ms
    # instead of 843 on 10 questions, the same top 8.
    import torch
    batches = []

    class Inputs(dict):
        def to(self, _device):
            return self

    def tokenizer(pairs, padding=False, truncation=True, max_length=None, return_tensors=None):
        ids = [list(range(len(passage) // 4 + 1)) for _, passage in pairs]
        if return_tensors is None:
            return {"input_ids": ids}
        batches.append([len(i) for i in ids])
        return Inputs(lengths=torch.tensor([float(len(passage)) for _, passage in pairs]))

    class Scorer:
        device = "cpu"

        def __call__(self, lengths):
            return type("Out", (), {"logits": lengths.view(-1, 1)})()

    monkeypatch.setattr(rerank, "cross_encoder", lambda: (tokenizer, Scorer()))
    passages = ["x" * n for n in (900, 40, 400, 8, 2000, 120) * 6]
    assert rerank.cross_scores("q", passages, batch=16) == [float(len(p)) for p in passages]
    assert len(batches) == 3 and all(max(b) <= min(c) for b, c in zip(batches, batches[1:]))
