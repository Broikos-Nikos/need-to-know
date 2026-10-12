"""BGE-M3 dense vectors on the GPU in fp16, from the local Hugging Face cache.

The repository and revision are pinned in config/models.toml, and loading never
touches the network. A vector is BGE-M3's dense recipe: the last hidden state of
the first token, normalised to unit length. `states` returns that state as the model
computed it, in its own dtype, which is what the vector cache keeps (T3.1), and
`normalise` turns it into the float32 row that is searched, the same way for a fresh
state and a cached one.

Tokenising is apart from the model, so a caller can check lengths before the GPU
sees anything. Texts are sorted by token count and cut into batches of at most
`batch_tokens` padded tokens, so short texts go many to a batch and long ones few,
and the rows come back in the order given. `states` refuses an input over
`max_length` rather than cutting it, because a cut passage loses its tail from the
index without a word (`12-infrastructure/08`); `encode`, for questions, cuts. The
allocator is capped below the card's memory, so a batch too big for it fails with
an out of memory error instead of spilling into system memory and crawling, which
the Windows driver otherwise allows (same doc, section 2).

Needs the `embed` extra: `uv sync --extra embed`.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

from needtoknow.settings import repo_root

# The share of the card's memory torch may take; the knowledge base's M1 ran at the same cap.
MEMORY_FRACTION = 0.92


class EmbedError(Exception):
    """The model, its revision, the GPU or an input is not what the configuration asks for."""


def load_config(path: Path | None = None) -> dict:
    with open(path or repo_root() / "config" / "models.toml", "rb") as handle:
        return tomllib.load(handle)["embed"]


@lru_cache(maxsize=1)
def load(device: str = "cuda"):
    """The tokenizer and the model, loaded once per process, with no byte to or from the network."""
    # `local_files_only` is not enough: given a .bin checkpoint, as BGE-M3's pinned revision is, transformers starts a
    # thread that asks the Hub about a safetensors conversion of the model, and can ask for one. Every load did,
    # until tick 61. The first switch is read when the model loads, the second when huggingface_hub is imported.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise EmbedError(f"{error.name} is not installed; run `uv sync --extra embed`") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise EmbedError("no CUDA device is visible to torch")
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    config = load_config()
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["repo"], revision=config["revision"], local_files_only=True)
        model = AutoModel.from_pretrained(config["repo"], revision=config["revision"], local_files_only=True,
                                          dtype=getattr(torch, config["dtype"]))
    except OSError as error:
        raise EmbedError(f"{config['repo']} at {config['revision'][:12]} is not in the local Hugging Face cache") from error
    return tokenizer, model.to(device).eval()


def tokenize(texts: list[str], cut: bool = False) -> list[list[int]]:
    """Token ids with the model's special tokens, one list per text; cut at max_length only when asked."""
    tokenizer, _ = load()
    limit = load_config()["max_length"]
    return tokenizer(list(texts), add_special_tokens=True, truncation=cut, max_length=limit if cut else None)["input_ids"]


def batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices sorted by length and cut so that no batch pads past `budget` tokens; a longer text goes alone."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if out and (len(out[-1]) + 1) * lengths[i] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def states(ids: list[list[int]], budget: int | None = None):
    """The first token's last hidden state for each list of token ids, in the model's dtype, in the order given."""
    import numpy as np
    import torch

    config = load_config()
    tokenizer, model = load()
    longest = max(map(len, ids), default=0)
    if longest > config["max_length"]:
        over = sum(len(x) > config["max_length"] for x in ids)
        raise EmbedError(f"{over} inputs are longer than max_length {config['max_length']}, the longest "
                         f"{longest} tokens: refused rather than cut")
    out = np.empty((len(ids), config["dimensions"]), dtype=np.dtype(config["dtype"]))
    with torch.inference_mode():
        for rows in batches([len(x) for x in ids], budget or config["batch_tokens"]):
            width = len(ids[rows[-1]])
            tokens = np.full((len(rows), width), tokenizer.pad_token_id, dtype=np.int64)
            mask = np.zeros((len(rows), width), dtype=np.int64)
            for j, i in enumerate(rows):
                tokens[j, :len(ids[i])] = ids[i]
                mask[j, :len(ids[i])] = 1
            first = model(input_ids=torch.from_numpy(tokens).to(model.device),
                          attention_mask=torch.from_numpy(mask).to(model.device)).last_hidden_state[:, 0]
            raw = first.cpu().numpy()
            check_batch(normalise(raw))
            out[rows] = raw
    return out


def normalise(rows):
    """Float32 rows of unit length; a row of zeros comes out as NaN, which check_batch refuses."""
    import numpy as np

    rows = np.asarray(rows, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def encode(texts: list[str], budget: int | None = None):
    """Unit length float32 rows, one per text, in the order given; a text over max_length is cut, as a question may be."""
    return normalise(states(tokenize(texts, cut=True), budget))


def check_batch(vectors):
    """Refuse a batch holding a value that is not finite or a row that is not of unit length.

    Both are failures that load and run without raising anything (`12-infrastructure/08`).
    """
    import numpy as np

    if not np.isfinite(vectors).all():
        raise EmbedError("a batch came back with NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise EmbedError(f"a batch came back with norms from {norms.min():.4f} to {norms.max():.4f}, not 1")
    return vectors
