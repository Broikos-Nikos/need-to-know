"""The registered evaluation protocol in eval/protocol.toml: which split may serve what.

The test split reports and never decides. Every command that reads golden items or
probes asks check_use first, so no tuning step can read the test split, and the
protocol itself is held to the hashes recorded in eval/protocol_history.toml.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from needtoknow.settings import repo_root

SPLITS = ("dev", "test")
# Purposes whose results choose something: a prompt, a scorer form, a parameter.
DECIDING = frozenset({"development", "experiment"})


class ProtocolError(Exception):
    """The protocol is invalid, or a split was asked for a purpose it is not registered for."""


def protocol_path() -> Path:
    return repo_root() / "eval" / "protocol.toml"


def history_path() -> Path:
    return repo_root() / "eval" / "protocol_history.toml"


def load_protocol(path: Path | None = None) -> dict:
    with open(path or protocol_path(), "rb") as handle:
        protocol = tomllib.load(handle)
    problems = validate(protocol)
    if problems:
        raise ProtocolError("; ".join(problems))
    return protocol


def validate(protocol: dict) -> list[str]:
    """Every way the protocol could let the test split decide, or leave a rule unset."""
    problems = []
    splits = protocol.get("splits", {})
    for name in SPLITS:
        if not splits.get(name, {}).get("purposes"):
            problems.append(f"splits.{name}.purposes is missing")
    test_uses = set(splits.get("test", {}).get("purposes", []))
    if test_uses & DECIDING:
        problems.append(f"the test split may not serve {', '.join(sorted(test_uses & DECIDING))}, because those results choose something")
    if "release" not in test_uses:
        problems.append("the test split is not run at release")
    if "experiment" not in splits.get("dev", {}).get("purposes", []):
        problems.append("the dev split does not serve experiments")
    decision = protocol.get("decision", {})
    margin = decision.get("non_inferiority_margin")
    if not isinstance(margin, float) or not 0 < margin <= 0.1:
        problems.append("decision.non_inferiority_margin must be a share above 0 and at most 0.1")
    alpha = decision.get("alpha")
    if not isinstance(alpha, float) or not 0 < alpha <= 0.05:
        problems.append("decision.alpha must be a share above 0 and at most 0.05")
    ids = [entry.split(" ", 1)[0] for entry in protocol.get("experiments", {}).get("order", [])]
    if not ids or ids != [f"E{i}" for i in range(1, len(ids) + 1)]:
        problems.append("experiments.order must run E1, E2 and onwards with no gap or repeat")
    return problems


def check_use(split: str, purpose: str, protocol: dict | None = None) -> None:
    """Raise unless the protocol registers this split for this purpose."""
    protocol = protocol if protocol is not None else load_protocol()
    entry = protocol.get("splits", {}).get(split)
    if entry is None:
        raise ProtocolError(f"no split named {split!r}; the protocol has {', '.join(SPLITS)}")
    if purpose not in entry["purposes"]:
        raise ProtocolError(f"the {split} split is registered for {', '.join(entry['purposes'])}, not for {purpose}")


def fingerprint(path: Path | None = None) -> str:
    """sha256 of the protocol's bytes with LF line ends, the same on every checkout."""
    return hashlib.sha256((path or protocol_path()).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
