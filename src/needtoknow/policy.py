"""Who may read what: config/labels.toml and config/policy.toml.

The labels a role reads are the filter its queries send, so the query path and the
filtered search bench both take them from readable() and from nowhere else. The
administrator is no special case here: it reads everything because the policy lists
everything for it, and its queries run the same filter as everyone else's.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from needtoknow.settings import repo_root

# Text with no audience line. Only a role whose policy lists it may read it.
UNLABELLED = "unlabelled"

# Filters no role sends, measured by the filtered search bench and granted to
# nobody: roles without `general`, below the most selective filter a role sends.
BENCH_ONLY = {
    "outside counsel": frozenset({"legal"}),
    "hr alone": frozenset({"hr"}),
}

# The characters of an unknown role that an error quotes (HS-F20).
QUOTED = 40


class PolicyError(Exception):
    """The policy is inconsistent, or a role it does not define was asked for."""


def config_path(name: str) -> Path:
    return repo_root() / "config" / name


def load_labels(path: Path | None = None) -> dict:
    with open(path or config_path("labels.toml"), "rb") as handle:
        return tomllib.load(handle)["labels"]


def load_policy(path: Path | None = None, labels: dict | None = None) -> dict:
    with open(path or config_path("policy.toml"), "rb") as handle:
        policy = tomllib.load(handle)
    problems = validate(policy, labels if labels is not None else load_labels())
    if problems:
        raise PolicyError("; ".join(problems))
    return policy


def validate(policy: dict, labels: dict) -> list[str]:
    """Every way the policy could grant a label nobody defined, or leave text that nobody may read."""
    known = set(labels) | {UNLABELLED}
    roles = policy.get("roles", {})
    problems = [] if roles else ["config/policy.toml defines no roles"]
    for role, entry in roles.items():
        unknown = set(entry.get("reads", [])) - known
        if unknown:
            problems.append(f"role {role} reads {', '.join(sorted(unknown))}, which config/labels.toml does not define")
    unread = known - set().union(*(set(entry.get("reads", [])) for entry in roles.values()))
    if unread:
        problems.append(f"no role reads {', '.join(sorted(unread))}, so that text is visible to nobody")
    total = sum(label["target_share"] for label in labels.values())
    if abs(total - 1.0) > 0.005:
        problems.append(f"label target shares sum to {total:.3f}, not 1")
    return problems


def readable(policy: dict, role: str) -> frozenset[str]:
    """The labels a role's queries send as their filter."""
    entry = policy.get("roles", {}).get(role)
    if entry is None:  # quoted short: a sign in sent a million characters and had them all back (HS-F20)
        named = repr(role) if len(role) <= QUOTED else f"{role[:QUOTED]!r}, cut from {len(role):,} characters,"
        raise PolicyError(f"no role named {named}; config/policy.toml defines {', '.join(policy.get('roles', {}))}")
    return frozenset(entry["reads"])


def bench_cases(policy: dict) -> list[tuple[str, frozenset[str], bool]]:
    """The filtered search bench's cases: every role's filter exactly as sent, then the bench only views, flagged."""
    cases = [(role, readable(policy, role), False) for role in policy["roles"]]
    return cases + [(name, reads, True) for name, reads in BENCH_ONLY.items()]


def planned_share(labels: dict, reads: frozenset[str]) -> float:
    """Target share of units a filter admits; an upper bound, since one section can carry two labels."""
    return sum(labels[label]["target_share"] for label in reads if label in labels)
