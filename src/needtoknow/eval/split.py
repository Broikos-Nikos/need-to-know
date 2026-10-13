"""The split of the fact families into test and dev (T5.1b, N56): drawn under protocol version 4, unchanged since.

A family here is the protocol's, not the registry's: a registry family, its versions and
regional variants, joined to every fact planted in the same section of a document,
transitively, since an excerpt about one shows the others' sentences. The plan's 103
registry families make 81 of them.

The split draws `test_families` of them at random within each stratum, the family's
kind and, within it, its label, by the registered seed: each kind's seats are the largest
remainder of its proportional share, then each label's within the kind, ties broken in
the seed's order. A flat allocation hands the seats left over to the largest fractions,
and the three variant families, each alone in its stratum, were left with none, the test
with no near duplicate item. Then the balance, on properties that carry no outcome: how many
places each of its facts is planted in, on average, the mean size of their documents,
the share of them in Greek, and the salience of its planted sentences, each one's share
of its paragraph's characters in the writers' narrative, which a short paragraph about
one fact makes large (KB1-F9). Placements are counted per fact, not per family: a
variant family plants thirteen regional facts, far more places in all than any other
family, and alone in its stratum it sits where the quotas put it whatever the seed, so
counted per family the placements could not balance. Each property's standardized mean difference between test and dev must stay
under the registered limit; a draw that fails is replaced by the next seed's, and every
draw is recorded. The plan's deliberate defaults are flagged on their facts (KB1-F7). The
result is eval/families.json, committed; `ntk eval split --check` draws it again and
compares.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from needtoknow.eval.protocol import load_protocol
from needtoknow.settings import data_dir, repo_root

PROPERTIES = ("placements", "document_size", "language", "salience")
# Values the plan gives that match a rule or a convention a model may know without the documents, so a closed book
# answer can be right for them (KB1-F7); the closed book run of T5.5 measures which are.
DELIBERATE_DEFAULTS = {
    "F34V00": "the 14 days of withdrawal from a distance contract, EU law",
    "F01V00": "a yearly renewal", "F03V00": "a yearly confirmation",
    "F07V00": "a 12 month minimum term", "F07V01": "a 12 month minimum term", "F07V02": "a 12 month minimum term",
    "F07V03": "a 24 month minimum term", "F07V05": "a 24 month minimum term",
    "F13V00": "a 30 day return window", "F05R00": "shops opening at 09:00",
    "F46V00": "a night maintenance window", "F47V00": "99.9 percent availability",
}


class SplitError(Exception):
    """The plan is missing, a planted sentence is not where the plan put it, or no seed balances the split."""


def split_path() -> Path:
    return repo_root() / "eval" / "families.json"


def load_plan() -> tuple[list[dict], list[dict]]:
    plan = data_dir() / "corpus" / "plan"
    try:
        facts = json.loads((plan / "facts.json").read_text(encoding="utf-8"))
        catalogue = json.loads((plan / "catalogue.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SplitError(f"no plan in {plan}: run ntk corpus plan first") from None
    return facts, catalogue


def groups(facts: list[dict], catalogue: list[dict]) -> list[list[dict]]:
    """The facts joined by registry family and by every section that plants two of them, transitively."""
    parent = {f["id"]: f["id"] for f in facts}

    def root(fid: str) -> str:
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    def join(a: str, b: str) -> None:
        parent[root(a)] = root(b)

    by_registry = defaultdict(list)
    for f in facts:
        by_registry[f["family"]].append(f["id"])
    for ids in by_registry.values():
        for other in ids[1:]:
            join(ids[0], other)
    for doc in catalogue:
        for section in doc["sections"]:
            for other in section["facts"][1:]:
                join(section["facts"][0], other)
    joined = defaultdict(list)
    for f in facts:
        joined[root(f["id"])].append(f)
    return sorted(joined.values(), key=lambda members: min(f["family"] for f in members))


def salience(sentence: str, text: str, nth: int = 0) -> float:
    """The planted sentence's share of its paragraph's characters, at its nth occurrence: a document that plants a fact
    twice is measured at each (DR-F21)."""
    at = -1
    for _ in range(nth + 1):
        at = text.find(sentence, at + 1)
        if at < 0:
            raise SplitError(f"the planted sentence {sentence[:60]!r} is not " + (f"{nth + 1} times " if nth else "")
                             + "in its document's narrative")
    start = text.rfind("\n\n", 0, at)
    end = text.find("\n\n", at)
    paragraph = text[start + 2 if start >= 0 else 0:end if end >= 0 else len(text)].strip()
    return len(sentence) / len(paragraph)


def describe(members: list[dict], docs: dict[str, dict], narrative: Path) -> dict:
    """A family: its id, stratum, registry families, facts and the four balance properties."""
    sizes, greek, shares = [], [], []
    for f in members:
        seen: Counter = Counter()
        for doc_id in f["placements"]:
            doc = docs[doc_id]
            sizes.append(doc["target_mb"])
            greek.append(doc["language"] == "el")
            shares.append(salience(f[doc["language"]], (narrative / f"{doc_id}.txt").read_text(encoding="utf-8"),
                                   seen[doc_id]))
            seen[doc_id] += 1
    labels = Counter(f["label"] for f in members)
    kinds = {f["kind"] for f in members}
    kind = "variant" if "variant" in kinds else "versioned" if kinds & {"versioned", "superseded"} else "single"
    return {"id": min(f["family"] for f in members),
            "label": sorted(labels, key=lambda label: (-labels[label], label))[0], "kind": kind,
            "registry_families": sorted({f["family"] for f in members}), "facts": sorted(f["id"] for f in members),
            "defaults": {f["id"]: DELIBERATE_DEFAULTS[f["id"]] for f in members if f["id"] in DELIBERATE_DEFAULTS},
            "placements": round(statistics.mean(len(f["placements"]) for f in members), 4), "document_size": round(statistics.mean(sizes), 4),
            "language": round(statistics.mean(greek), 4), "salience": round(statistics.mean(shares), 4)}


def quotas(sizes: dict, total: int, rng: random.Random) -> dict:
    """Each stratum's test seats: the floor of its proportional share, the seats left going to the largest fractions,
    ties in the seed's order rather than the strata's names."""
    whole = sum(sizes.values())
    exact = {key: count * total / whole for key, count in sizes.items()}
    out = {key: math.floor(value) for key, value in exact.items()}
    keys = sorted(exact)
    rng.shuffle(keys)
    keys.sort(key=lambda key: out[key] - exact[key])  # stable: equal fractions keep the shuffled order
    for key in keys[:total - sum(out.values())]:
        out[key] += 1
    return out


def draw(families: list[dict], total: int, seed: int) -> set[str]:
    """The test families of one draw: seats by kind first, then by label within each kind, so a kind of three
    families gets its share; within each stratum the families shuffled by the seed and its seats taken."""
    rng = random.Random(seed)
    strata: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for family in families:
        strata[family["kind"]][family["label"]].append(family["id"])
    by_kind = quotas({kind: sum(map(len, labels.values())) for kind, labels in strata.items()}, total, rng)
    test: set[str] = set()
    for kind in sorted(strata):
        labels = strata[kind]
        by_label = quotas({label: len(ids) for label, ids in labels.items()}, by_kind[kind], rng)
        for label in sorted(labels):
            ids = sorted(labels[label])
            rng.shuffle(ids)
            test.update(ids[:by_label[label]])
    return test


def smd(test: list[float], dev: list[float]) -> float:
    """The standardized mean difference, test minus dev, over the pooled standard deviation."""
    pooled = math.sqrt((statistics.variance(test) + statistics.variance(dev)) / 2)
    return 0.0 if pooled == 0 else (statistics.mean(test) - statistics.mean(dev)) / pooled


def split(protocol: dict | None = None, facts: list[dict] | None = None, catalogue: list[dict] | None = None,
          narrative: Path | None = None, attempts: int = 10000) -> dict:
    """The registered split: every draw from the protocol's seed until one balances, and the families it gives."""
    protocol = protocol or load_protocol()
    if facts is None or catalogue is None:
        facts, catalogue = load_plan()
    docs = {doc["id"]: doc for doc in catalogue}
    families = [describe(members, docs, narrative or repo_root() / "narrative") for members in groups(facts, catalogue)]
    settings = protocol["splits"]
    total, limit = settings["test_families"], settings["balance"]["max_standardized_mean_difference"]
    draws = []
    for seed in range(settings["seed"], settings["seed"] + attempts):
        test = draw(families, total, seed)
        gaps = {p: smd([f[p] for f in families if f["id"] in test], [f[p] for f in families if f["id"] not in test])
                for p in PROPERTIES}
        balanced = all(abs(gap) < limit for gap in gaps.values())
        draws.append({"seed": seed, "smd": {p: round(gap, 4) for p, gap in gaps.items()}, "balanced": balanced})
        if balanced:
            break
    else:
        raise SplitError(f"no seed from {settings['seed']} in {attempts} draws balanced the split")
    for family in families:
        family["split"] = "test" if family["id"] in test else "dev"
    return {"protocol_version": protocol["version"], "seed": draws[-1]["seed"], "registry_families":
            len({f["family"] for f in facts}), "families": len(families), "test_families": total,
            "limit": limit, "draws": draws, "split": families}


def write(result: dict, path: Path | None = None) -> Path:
    path = path or split_path()
    path.write_bytes((json.dumps(result, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return path
