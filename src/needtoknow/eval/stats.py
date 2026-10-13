"""The statistics the protocol registers (T5.3, `eval/protocol.toml` [decision], N43): every comparison over units.

Items about one fact family are not independent, so the unit of every test is the family:
an item counts toward the family it asks, an item asking two families toward the pair as
one unit, and a table item, an absent fact or a near miss, which belong to no family, is a
unit of its own. A unit's score is the mean over its items, and over the repeats of each.

* `wilson`: the 95 percent interval of a rate, for every rate a report gives.
* `sign_flip`: the two sided sign flip permutation test of the mean of the units' paired
  differences, exact when fourteen units or fewer differ, else by the seeded resamples.
* `bootstrap`: the percentile bootstrap interval of that mean, over units, seeded.
* `decide`: the registered arms. Up: the mean difference above zero by the test under
  alpha. Cost: cost or p95 seconds down by the registered cut, and the interval's lower
  end above minus the margin. Probe: probe recall up by the test, and accuracy non
  inferior as for cost. Otherwise "too small to tell", with the smallest effect these
  units can show beside it, never a winner.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict

Z95 = 1.959963984540054
SEED = 20261012  # the protocol's registered seed, so a rerun gives the same resamples


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval of a rate: at 40 of 50, 0.670 to 0.888."""
    if n == 0:
        raise ValueError("a rate over no items has no interval")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def unit_of(item: dict) -> str:
    """The unit an item counts toward: its family, its two families as one, or itself when it has none."""
    families = item.get("families") or []
    return "+".join(sorted(families)) if families else item["id"]


def unit_means(items: list[dict], scores: dict[str, list[float]]) -> dict[str, float]:
    """Each unit's mean score: an item's repeats averaged first, then the unit's items."""
    per_unit: dict[str, list[float]] = defaultdict(list)
    for item in items:
        runs = scores[item["id"]]
        per_unit[unit_of(item)].append(sum(runs) / len(runs))
    return {unit: sum(values) / len(values) for unit, values in sorted(per_unit.items())}


def differences(items: list[dict], incumbent: dict[str, list[float]], candidate: dict[str, list[float]]) -> list[float]:
    """The candidate's unit means less the incumbent's, unit by unit, over the same items."""
    before, after = unit_means(items, incumbent), unit_means(items, candidate)
    return [after[unit] - before[unit] for unit in before]


def sign_flip(diffs: list[float], resamples: int = 10000, seed: int = SEED, exact_up_to: int = 14) -> float:
    """Two sided p value of the mean difference under random signs: exact over every sign pattern of the units that
    differ when there are few, else the share of seeded resamples at least as far from zero, counting the observed."""
    if not diffs:
        raise ValueError("no units to test")
    observed = abs(sum(diffs))
    nonzero = [d for d in diffs if d != 0]
    tolerance = 1e-12 * max(1.0, observed)
    if len(nonzero) <= exact_up_to:
        patterns = list(itertools.product((1, -1), repeat=len(nonzero)))
        extreme = sum(abs(sum(s * d for s, d in zip(signs, nonzero))) >= observed - tolerance for signs in patterns)
        return extreme / len(patterns)
    rng = random.Random(seed)
    extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - tolerance
                  for _ in range(resamples))
    return (1 + extreme) / (1 + resamples)


def bootstrap(diffs: list[float], resamples: int = 10000, level: float = 0.95, seed: int = SEED) -> tuple[float, float]:
    """The percentile bootstrap interval of the mean difference, units drawn with replacement."""
    if not diffs:
        raise ValueError("no units to resample")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low = means[int(math.floor((1 - level) / 2 * resamples))]
    high = means[min(resamples - 1, int(math.ceil((1 + level) / 2 * resamples)) - 1)]
    return low, high


def difference(first: list[float], second: list[float], resamples: int = 10000, level: float = 0.95,
               seed: int = SEED) -> tuple[float, float, float]:
    """The mean of the first units less the mean of the second, and its percentile bootstrap interval, each side
    drawn with replacement from its own units: two sets of units that are not paired, such as two splits' families."""
    if not first or not second:
        raise ValueError("no units to resample")
    rng = random.Random(seed)
    observed = sum(first) / len(first) - sum(second) / len(second)
    gaps = sorted(sum(first[rng.randrange(len(first))] for _ in first) / len(first)
                  - sum(second[rng.randrange(len(second))] for _ in second) / len(second) for _ in range(resamples))
    low = gaps[int(math.floor((1 - level) / 2 * resamples))]
    high = gaps[min(resamples - 1, int(math.ceil((1 + level) / 2 * resamples)) - 1)]
    return observed, low, high


def decide(diffs: list[float], alpha: float, margin: float, smallest_visible: float, cost_cut: float | None = None,
           cost_ratio: float | None = None, probe_diffs: list[float] | None = None) -> dict:
    """The registered arms over one comparison. `cost_ratio` is the candidate's cost over the incumbent's, USD a
    question or p95 seconds; `probe_diffs` the dev probes' recall differences by unit."""
    p = sign_flip(diffs)
    low, high = bootstrap(diffs)
    mean = sum(diffs) / len(diffs)
    not_worse = low > -margin
    verdict = "too small to tell"
    if p < alpha and mean > 0:
        verdict = "up"
    elif cost_cut is not None and cost_ratio is not None and cost_ratio <= 1 - cost_cut and not_worse:
        verdict = "cost"
    elif probe_diffs and sign_flip(probe_diffs) < alpha and sum(probe_diffs) > 0 and not_worse:
        verdict = "probe"
    elif p < alpha and mean < 0:
        verdict = "worse"
    return {"verdict": verdict, "mean": mean, "p": p, "interval": (low, high), "units": len(diffs),
            "smallest_visible_effect": smallest_visible}
