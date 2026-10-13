"""The registered evaluation protocol in eval/protocol.toml: which split may serve what.

The test split reports and never decides. Every command that scores golden items or
probes, or uses them to choose anything, asks check_use first, so no tuning step can read
the test split; the construction checks of `needtoknow.eval.golden` read items and no
outcome. The protocol itself is held to the hashes recorded in eval/protocol_history.toml.
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
    if protocol.get("baseline", {}).get("on_disagreement") != "report":
        problems.append("baseline.on_disagreement must be 'report': a dev and test difference may change nothing")
    if splits.get("balance", {}).get("when") != "construction":
        problems.append("splits.balance.when must be 'construction': balance is checked before any item is scored")
    decision = protocol.get("decision", {})
    margin = decision.get("non_inferiority_margin")
    if not isinstance(margin, float) or not 0 < margin <= 0.1:
        problems.append("decision.non_inferiority_margin must be a share above 0 and at most 0.1")
    alpha = decision.get("alpha")
    if not isinstance(alpha, float) or not 0 < alpha <= 0.05:
        problems.append("decision.alpha must be a share above 0 and at most 0.05")
    # Version 3 (N43): items about one family cluster, so the family is the unit of the split and of every test.
    if splits.get("unit") != "fact family" or decision.get("unit") != "fact family":
        problems.append("splits.unit and decision.unit must be 'fact family': items about one fact are not independent")
    if "section" not in str(splits.get("family", "")):
        problems.append("splits.family must join facts planted in one section: an excerpt about one shows the "
                        "others' sentences (N56)")
    if splits.get("test", {}).get("items_per_family") != 1:
        problems.append("splits.test.items_per_family must be 1: a test family carries one item, by construction")
    if not splits.get("balance", {}).get("rewrites"):
        problems.append("splits.balance.rewrites must say how a rewrite is logged (KB2-K2)")
    items = protocol.get("items", {})
    rare, shared = items.get("rare_terms"), items.get("max_shared_rare_terms")
    if items.get("overlap_terms") != "needtoknow.lexical.terms":
        problems.append("items.overlap_terms must be needtoknow.lexical.terms, the terms BM25 matches (RS-F3)")
    if not (isinstance(rare, int) and isinstance(shared, int) and 0 <= shared < rare):
        problems.append("items.rare_terms and items.max_shared_rare_terms must cap the shared rare terms (KB3-K4)")
    # Version 5 (N57): every count fixed before any item, so none can be chosen once items exist.
    test_items, dev_items = items.get("test", {}), items.get("dev", {})
    used = sum(test_items.get(code, 0) for code in "ABCDEGH") + 2 * sum(test_items.get(code, 0) for code in "FM")
    if used != splits.get("test_families"):
        problems.append(f"items.test uses {used} families, not the {splits.get('test_families')} test families: "
                        "every test family is asked once, F and M items using two")
    if any(not isinstance(counts.get(code), int) for counts in (test_items, dev_items) for code in "IJN"):
        problems.append("items.test and items.dev must count the I, J and N items, which belong to no family (RS-F17)")
    if not items.get("defaults"):
        problems.append("items.defaults must say which items may ask a deliberate default (RS-F19)")
    if not {"category", "language"} <= set(splits.get("balance", {}).get("item_properties", [])):
        problems.append("splits.balance.item_properties must hold the category and the language (RS-F20)")
    baseline = protocol.get("baseline", {})
    if baseline.get("cache") != "off":
        problems.append("baseline.cache must be 'off': a repeat sends the same prompt and would come back from the "
                        "answer cache (KB5-K13)")
    if not baseline.get("aggregate"):
        problems.append("baseline.aggregate must say how repeats make one score (RS-F2)")
    if not decision.get("several_values"):
        problems.append("decision.several_values must say how an experiment with several values adopts one (RS-F8)")
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
