"""The provider catalogue in config/providers.toml, and what a call costs.

Cost is computed from the token counts the API reports, split three ways
because DeepSeek prices a cache hit at a fiftieth of a miss, and at the price in
force at the moment of the call, because peak hours cost double.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

from needtoknow.settings import repo_root


def catalog_path() -> Path:
    return repo_root() / "config" / "providers.toml"


def load_catalog(path: Path | None = None) -> dict:
    with open(path or catalog_path(), "rb") as handle:
        return tomllib.load(handle)


def is_peak(when: dt.datetime, peak: dict) -> bool:
    if when.tzinfo is None:
        raise ValueError("a naive datetime cannot be placed in a UTC peak window")
    utc = when.astimezone(dt.timezone.utc)
    if utc.weekday() not in peak["weekdays"]:
        return False
    return any(start <= utc.hour < end for start, end in peak["windows_utc"])


def cost_usd(price: dict, usage: dict, when: dt.datetime, peak: dict) -> float:
    """USD for one call. `usage` holds input_hit, input_miss and output token counts."""
    rates = price["peak"] if is_peak(when, peak) else price["off_peak"]
    return sum(usage.get(kind, 0) * rates[kind] for kind in ("input_hit", "input_miss", "output")) / 1_000_000
