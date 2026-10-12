"""How long the corpus's passages really are in BGE-M3 tokens, and how fast this machine embeds them (T3.1).

    uv run python bench/embed_throughput.py [--sample 4096] [--seed 50] [--budgets 8192,16384,32768,65536,131072]

Needs a manifest (`ntk ingest build`), the embed extra and the GPU; it writes nothing.

Tokens: every unit's text and passage through BGE-M3's tokenizer, by layer and language:
the mean and longest passage, the fewest characters per token of a text, and how far a
text's real count runs past the estimate the units step cuts by (`estimate_tokens`).

Throughput: a fixed sample of units, after a warm up, embedded three ways: in batches of
32 texts sorted by characters and tokenised batch by batch, as encode() did before T3.1;
tokenised first and cut by padded tokens, at each budget; and each language alone at the
configured budget. Each line gives the seconds from text to state, tokens per second
counting real tokens only, the share of padding, the GPU's mean utilisation as nvidia-smi
samples it, and torch's peak memory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np

from needtoknow.embed import model
from needtoknow.ingest.manifest import opened
from needtoknow.ingest.units import estimate_tokens, passage

STEP = 20000


class Utilisation:
    """The GPU's utilisation from nvidia-smi, sampled every 200 ms while the block runs."""

    def __enter__(self):
        self.samples: list[int] = []
        self.process = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                         "-lms", "200"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read, daemon=True).start()
        return self

    def read(self) -> None:
        for line in self.process.stdout:
            if line.strip().isdigit():
                self.samples.append(int(line))

    def __exit__(self, *exc) -> None:
        self.process.terminate()
        self.process.wait()

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")


def lengths(texts: list[str]) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.int64)
    for start in range(0, len(texts), STEP):
        out[start:start + STEP] = [len(ids) for ids in model.tokenize(texts[start:start + STEP])]
    return out


def token_report(units: list[tuple], passages: list[str], limit: int) -> None:
    texts = [text for _, _, text in units]
    real = lengths(texts) - 2  # the text alone, without the two special tokens
    whole = lengths(passages)
    chars = np.array([len(text) for text in texts])
    estimate = np.array([estimate_tokens(text, language) for language, _, text in units])
    keys = np.array([f"{layer} {language}" for language, layer, _ in units])
    print(f"units {len(units)}; tokens of all passages {int(whole.sum())}; passages over max_length {limit}: "
          f"{int((whole > limit).sum())}")
    for key in sorted(set(keys)):
        at = keys == key
        short = real[at] - estimate[at]
        print(f"  {key}: units {int(at.sum())}; passage tokens mean {whole[at].mean():.1f}, longest {int(whole[at].max())}; "
              f"fewest characters per text token {(chars[at] / np.maximum(real[at], 1)).min():.2f}; "
              f"text over its estimate in {int((short > 0).sum())} units, by up to {max(int(short.max()), 0)} tokens")


def before_t31(texts: list[str]) -> int:
    """encode() as it was before T3.1; the padded positions it ran."""
    import torch

    tokenizer, encoder = model.load()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    padded = 0
    with torch.inference_mode():
        for start in range(0, len(order), 32):
            batch = tokenizer([texts[i] for i in order[start:start + 32]], padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to(encoder.device)
            first = encoder(**batch).last_hidden_state[:, 0].float()
            torch.nn.functional.normalize(first, dim=-1).cpu().numpy()
            padded += batch["input_ids"].numel()
    return padded


def budgeted(texts: list[str], budget: int) -> int:
    ids = model.tokenize(texts)
    model.states(ids, budget)
    sizes = [len(x) for x in ids]
    return sum(len(rows) * sizes[rows[-1]] for rows in model.batches(sizes, budget))


def timed(label: str, run, tokens: int) -> float:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        with Utilisation() as busy:
            began = time.perf_counter()
            padded = run()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - began
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {label}: out of memory")
        return 0.0
    print(f"  {label}: {seconds:.1f} s, {tokens / seconds:,.0f} tokens/s, padding {1 - tokens / padded:.1%}, "
          f"GPU busy {busy.mean():.0f}%, peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB", flush=True)
    return tokens / seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--budgets", default="8192,16384,32768,65536,131072")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config = model.load_config()
    with opened() as db:
        units = db.execute("SELECT language, layer, text, path FROM units ORDER BY unit_id").fetchall()
    passages = [passage(json.loads(path), text) for _, _, text, path in units]
    units = [(language, layer, text) for language, layer, text, _ in units]
    print(f"model {config['repo']}@{config['revision'][:12]} {config['dtype']}, batch_tokens {config['batch_tokens']}")
    token_report(units, passages, config["max_length"])

    rng = np.random.default_rng(args.seed)
    sample = [passages[i] for i in rng.choice(len(passages), size=args.sample, replace=False)]
    tokens = int(lengths(sample).sum())
    model.states(model.tokenize(sample[:256]))  # warm up
    print(f"sample of {args.sample} units, seed {args.seed}, {tokens} tokens:")
    timed("batches of 32 by count (encode before T3.1)", lambda: before_t31(sample), tokens)
    for budget in (int(b) for b in args.budgets.split(",")):
        timed(f"batches of up to {budget} padded tokens", lambda: budgeted(sample, budget), tokens)
    for language in ("en", "el"):
        picked = [passages[i] for i in np.flatnonzero(np.array([u[0] for u in units]) == language)]
        part = [picked[i] for i in rng.choice(len(picked), size=args.sample // 2, replace=False)]
        count = int(lengths(part).sum())
        timed(f"{language} alone, {args.sample // 2} units, at {config['batch_tokens']}",
              lambda: budgeted(part, config["batch_tokens"]), count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
