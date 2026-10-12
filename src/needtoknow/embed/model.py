"""BGE-M3 dense vectors on the GPU in fp16, from the local Hugging Face cache.

The repository and revision are pinned in config/models.toml, and loading never
touches the network. A vector is BGE-M3's dense recipe: the last hidden state of
the first token, normalised to unit length, returned as float32. Texts are batched
by length, so padding stays short, and the rows come back in the order given.
Needs the `embed` extra: `uv sync --extra embed`.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from needtoknow.settings import repo_root


class EmbedError(Exception):
    """The model, its revision or the GPU is not what the configuration asks for."""


def load_config(path: Path | None = None) -> dict:
    with open(path or repo_root() / "config" / "models.toml", "rb") as handle:
        return tomllib.load(handle)["embed"]


@lru_cache(maxsize=1)
def load(device: str = "cuda"):
    """The tokenizer and the model, loaded once per process."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise EmbedError(f"{error.name} is not installed; run `uv sync --extra embed`") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise EmbedError("no CUDA device is visible to torch")
    config = load_config()
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["repo"], revision=config["revision"], local_files_only=True)
        model = AutoModel.from_pretrained(config["repo"], revision=config["revision"], local_files_only=True,
                                          dtype=getattr(torch, config["dtype"]))
    except OSError as error:
        raise EmbedError(f"{config['repo']} at {config['revision'][:12]} is not in the local Hugging Face cache") from error
    return tokenizer, model.to(device).eval()


def encode(texts: list[str], batch_size: int = 32, max_length: int | None = None):
    """Unit length float32 rows, one per text, in the order given."""
    import numpy as np
    import torch

    config = load_config()
    tokenizer, model = load()
    vectors = np.empty((len(texts), config["dimensions"]), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                              max_length=max_length or config["max_length"], return_tensors="pt").to(model.device)
            first = model(**batch).last_hidden_state[:, 0].float()
            vectors[rows] = torch.nn.functional.normalize(first, dim=-1).cpu().numpy()
    return vectors
