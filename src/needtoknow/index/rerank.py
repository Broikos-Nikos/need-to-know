"""The second stage (T3.6): the fused candidates reordered by a stronger model, behind one switch.

`multivector` is BGE-M3's late interaction: its ColBERT head over the token states of the
question and of each candidate passage, encoded at query time, and MaxSim, the mean over
the question's tokens of each one's best match in the passage. The base measured it 7 to
8 points of recall@1 above dense alone. `cross` is bge-reranker-v2-m3, which reads the
question and a passage together: 4 points more again, at about a second a question for
100 paragraphs on this card (`03-retrieval-rag/12` section 9). `none` keeps the fused
order. `config/models.toml` pins both models and names the default.

A reranker only reorders the candidates the role's filtered search returned. It adds
none, so it cannot widen what a role sees, and it reads no statistic of the collection,
so nothing a role may not read moves its scores.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from needtoknow.embed import model
from needtoknow.settings import repo_root

MODES = ("none", "multivector", "cross")


def load_config(path: Path | None = None) -> dict:
    with open(path or repo_root() / "config" / "models.toml", "rb") as handle:
        return tomllib.load(handle)["rerank"]


def rerank(question: str, hits: list[dict], passages: list[str], mode: str | None = None) -> list[dict]:
    """The hits in the reranker's order, each with its rank and the reranker's score; none added, none dropped."""
    mode = mode or load_config()["mode"]
    if mode not in MODES:
        raise ValueError(f"no reranker {mode!r}; the switch takes {', '.join(MODES)}")
    if mode == "none" or not hits:
        return hits
    scores = multivector_scores(question, passages) if mode == "multivector" else cross_scores(question, passages)
    order = sorted(range(len(hits)), key=lambda i: -scores[i])
    return [{"unit_id": hits[i]["unit_id"], "rank": rank, "score": float(scores[i])} for rank, i in enumerate(order, 1)]


@lru_cache(maxsize=1)
def colbert_head():
    """BGE-M3's ColBERT projection, from the pinned snapshot in the local cache."""
    import torch
    from huggingface_hub import try_to_load_from_cache

    config = model.load_config()
    _, encoder = model.load()
    path = try_to_load_from_cache(config["repo"], "colbert_linear.pt", revision=config["revision"])
    if not isinstance(path, str):
        raise model.EmbedError(f"colbert_linear.pt of {config['repo']} at {config['revision'][:12]} is not in the local cache")
    head = torch.nn.Linear(config["dimensions"], config["dimensions"])
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return head.to(device=encoder.device, dtype=encoder.dtype).eval()


def token_vectors(texts: list[str]) -> list:
    """Each text's normalised ColBERT vectors, one per token after the first, padding left out."""
    import numpy as np
    import torch

    tokenizer, encoder = model.load()
    head = colbert_head()
    ids = model.tokenize(texts, cut=True)
    out = [None] * len(texts)
    with torch.inference_mode():
        for rows in model.batches([len(x) for x in ids], model.load_config()["batch_tokens"]):
            width = len(ids[rows[-1]])
            tokens = np.full((len(rows), width), tokenizer.pad_token_id, dtype=np.int64)
            mask = np.zeros((len(rows), width), dtype=np.int64)
            for j, i in enumerate(rows):
                tokens[j, :len(ids[i])] = ids[i]
                mask[j, :len(ids[i])] = 1
            states = encoder(input_ids=torch.from_numpy(tokens).to(encoder.device),
                             attention_mask=torch.from_numpy(mask).to(encoder.device)).last_hidden_state
            vectors = torch.nn.functional.normalize(head(states[:, 1:]).float(), dim=-1)
            for j, i in enumerate(rows):
                out[i] = vectors[j, :len(ids[i]) - 1]
    return out


def multivector_scores(question: str, passages: list[str]) -> list[float]:
    """MaxSim of the question against each passage."""
    question_vectors, *passage_vectors = token_vectors([question, *passages])
    return [float((question_vectors @ p.T).max(dim=1).values.mean()) for p in passage_vectors]


@lru_cache(maxsize=1)
def cross_encoder():
    """The pinned cross encoder and its tokenizer, from the local cache, with no byte to or from the network."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model.offline()
    config = load_config()
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["repo"], revision=config["revision"], local_files_only=True)
        scorer = AutoModelForSequenceClassification.from_pretrained(
            config["repo"], revision=config["revision"], local_files_only=True, dtype=torch.float16)
    except OSError as error:
        raise model.EmbedError(f"{config['repo']} at {config['revision'][:12]} is not in the local Hugging Face cache") from error
    return tokenizer, scorer.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def cross_scores(question: str, passages: list[str], batch: int = 16) -> list[float]:
    """The cross encoder's relevance of each passage to the question, in the order given. The pairs are scored
    shortest first, so each batch pads to its own longest pair rather than to the longest the rank order happened to
    put beside it: on 10 questions, 572 ms a question instead of 843, the same top 8 (PF-F4)."""
    import torch

    tokenizer, scorer = cross_encoder()
    limit = load_config()["max_length"]
    pairs = [[question, p] for p in passages]
    lengths = [len(ids) for ids in tokenizer(pairs, truncation=True, max_length=limit)["input_ids"]]
    order = sorted(range(len(pairs)), key=lengths.__getitem__)
    scores = [0.0] * len(pairs)
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            part = order[start:start + batch]
            inputs = tokenizer([pairs[i] for i in part], padding=True, truncation=True, max_length=limit,
                               return_tensors="pt").to(scorer.device)
            for i, score in zip(part, scorer(**inputs).logits.view(-1).float().tolist()):
                scores[i] = score
    return scores
