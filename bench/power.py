"""The smallest effect the registered test sees on the dev units (T5.3b, RS-F18), by simulation.

    uv run python bench/power.py [--trials 2000] [--resamples 2000]

The units are the dev split's, as `needtoknow.eval.stats.unit_of` groups its items: a family,
two families asked together, an own item alone, with as many items as each holds. A
candidate differs from the incumbent on a share q of items, the discordance; among those it
fixes as many more than it breaks as a true effect of d points needs, so q is at least d.
Each trial draws the items' differences, averages them by unit, and runs the registered
test: the two sided sign flip permutation test of the mean unit difference, by seeded
resamples, adopting when p is under alpha and the mean is above zero. Power is the share of
trials adopted; the smallest visible effect, the smallest d adopted in at least 80 percent
of trials, for each q.

No answer has been scored yet, so q is an assumption: `eval/protocol.toml` records the figure
at q = 0.2 and names this bench, and the baseline's measured discordance replaces the
assumption when it exists. numpy comes with qdrant-client.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from needtoknow.eval.golden import golden_path, load_items
from needtoknow.eval.protocol import load_protocol
from needtoknow.eval.stats import SEED, unit_of

DISCORDANCE = (0.10, 0.15, 0.20, 0.30)
EFFECTS = tuple(round(0.01 * k, 2) for k in range(2, 31))
POWER = 0.80


def unit_sizes(items: list[dict]) -> list[int]:
    sizes: dict[str, int] = {}
    for item in items:
        unit = unit_of(item)
        sizes[unit] = sizes.get(unit, 0) + 1
    return [sizes[unit] for unit in sorted(sizes)]


def sign_flip_p(means: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """The registered test's resampled p for each trial's unit means, the observed counted as one resample."""
    observed = np.abs(means.sum(axis=1))
    flipped = np.abs(means @ signs.T)
    extreme = (flipped >= observed[:, None] - 1e-12).sum(axis=1)
    return (1 + extreme) / (1 + signs.shape[0])


def power(sizes: list[int], q: float, effect: float, alpha: float, trials: int, resamples: int,
          rng: np.random.Generator) -> float:
    """The share of simulated experiments the registered test adopts, at discordance q and a true effect."""
    n, units = sum(sizes), len(sizes)
    owner = np.repeat(np.arange(units), sizes)
    membership = np.zeros((n, units))
    membership[np.arange(n), owner] = 1.0 / np.repeat(np.array(sizes, dtype=float), sizes)
    fix = (q + effect) / (2 * q)
    discordant = rng.random((trials, n)) < q
    sign = np.where(rng.random((trials, n)) < fix, 1.0, -1.0)
    means = (discordant * sign) @ membership
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, units))
    p = sign_flip_p(means, signs)
    return float(((p < alpha) & (means.sum(axis=1) > 0)).mean())


def smallest_visible(sizes: list[int], q: float, alpha: float, trials: int, resamples: int, seed: int) -> float | None:
    rng = np.random.default_rng(seed)
    for effect in EFFECTS:
        if effect > q:
            return None
        if power(sizes, q, effect, alpha, trials, resamples, rng) >= POWER:
            return effect
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    protocol = load_protocol()
    alpha = protocol["decision"]["alpha"]
    items = load_items(golden_path("dev"))
    sizes = unit_sizes(items)
    print(f"dev: {len(items)} items in {len(sizes)} units, {min(sizes)} to {max(sizes)} items a unit; alpha {alpha}; "
          f"{args.trials} trials of {args.resamples} resamples each, seed {SEED}", flush=True)
    for q in DISCORDANCE:
        found = smallest_visible(sizes, q, alpha, args.trials, args.resamples, SEED)
        shown = "none within the discordance" if found is None else f"{found:.2f}"
        print(f"discordance {q:.2f}: smallest effect seen in {POWER:.0%} of trials {shown}", flush=True)
    print(f"registered: smallest_visible_effect {protocol['decision']['smallest_visible_effect']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
