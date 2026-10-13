"""The golden items and their construction checks (T5.1c2, `specs/evaluation.md` 2.3, protocol version 5).

One item per line of `eval/golden/test.jsonl` and `eval/golden/dev.jsonl`:

    {"id": "T01", "category": "A", "families": ["F06V00"], "role": "sales", "lang": "en",
     "question": "...",
     "expect": {"status": "answered", "facts": [{"fact_id": "F06V00", "forms": ["7 percent", "7%"]}],
                "superseded": [], "forbidden": []}}

A table item carries `"row": {"doc_id", "key", "column", "value"}` and its expected
`forms`; an E item names the fact it asks under `forbidden`, with its forms, since its
answer must hold none of them; a near miss names the present fact it neighbours in
`"near"`; a dev item may name its paraphrase twin in `"twin"`. `check` reads both files and no outcome, so it decides
nothing and needs no purpose of the protocol: it holds each item to its category's rules,
and each split to the counts and shares version 5 registered.

Per item: a known category, and `not_found` exactly for E, J and N; the question's script
is its language; family items name their families, two for F and M, each in this split;
the role exists; an F item's facts sit under two labels its role reads; every expected fact is in the registry,
in one of the item's families, each of its forms holding its planted value or naming it; an answered item's
role may read a section each fact is planted in, an E item's none; C asks in the other
language than every document its facts sit in; G asks the current value and names the
superseded one, H a regional variant; a deliberate default is asked only by E (RS-F19);
and the overlap cap: of the three rarest terms of each planted sentence by the corpus's
document frequency, at most one in the question, counted by `needtoknow.lexical.terms`,
the terms BM25 matches (RS-F3, KB3-K4). Per split: the test's category counts, every test
family asked once; two or three items for each dev family; the own items' counts; the
categories' shares and the Greek share. Each item has a layer, table for I and prose for
the rest, so every figure can be given by layer (KB1-F8).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from needtoknow import lexical
from needtoknow.answer.prompt import language
from needtoknow.eval.protocol import load_protocol
from needtoknow.eval.split import DELIBERATE_DEFAULTS, load_plan, split_path
from needtoknow.index import frequencies
from needtoknow.policy import load_policy, readable
from needtoknow.settings import repo_root

FAMILY = set("ABCDEFGHM")
OWN = set("IJN")
REFUSED = set("EJN")
TWO_FAMILIES = set("FM")
SPLITS = ("test", "dev")


class GoldenError(Exception):
    """A golden file that cannot be read as items."""


def golden_path(split: str) -> Path:
    return repo_root() / "eval" / "golden" / f"{split}.jsonl"


def load_items(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(json.loads(line))
            except ValueError as error:
                raise GoldenError(f"{path.name} line {number} is not JSON: {error}") from None
    return items


def corpus_df(path: Path | None = None) -> Callable[[str], int]:
    """A term's document frequency over the whole corpus, every label set summed, from the sync's frequency table."""
    db = sqlite3.connect(f"file:{(path or frequencies.table_path()).as_posix()}?mode=ro", uri=True)
    return lambda term: db.execute("SELECT COALESCE(SUM(n), 0) FROM df WHERE term = ?",
                                   (lexical.term_id(term),)).fetchone()[0]


def rare_terms(sentence: str, df: Callable[[str], int], count: int) -> list[str]:
    """The sentence's rarest terms by document frequency, first occurrence breaking ties; an empty term, which the
    Greek stemmer leaves for some words, is no term a question could share."""
    distinct = [term for term in dict.fromkeys(lexical.terms(sentence)) if term]
    return sorted(distinct, key=lambda term: (df(term), distinct.index(term)))[:count]


class Context:
    """What the checks read: the protocol, the split, the registry, the catalogue, the policy and the frequencies."""

    def __init__(self, protocol=None, split=None, facts=None, catalogue=None, policy=None, df=None):
        self.protocol = protocol or load_protocol()
        result = split or json.loads(split_path().read_text(encoding="utf-8"))
        self.families = {f["id"]: f for f in result["split"]}
        if facts is None or catalogue is None:
            facts, catalogue = load_plan()
        self.facts = {f["id"]: f for f in facts}
        self.docs = {d["id"]: d for d in catalogue}
        self.policy = policy or load_policy()
        self.df = df or corpus_df()

    def sections(self, fact_id: str) -> list[tuple[str, dict]]:
        """Each document a fact is planted in, with the section that plants it."""
        fact = self.facts[fact_id]
        return [(doc_id, next(s for s in self.docs[doc_id]["sections"] if fact_id in s["facts"]))
                for doc_id in fact["placements"]]


def layer(item: dict) -> str:
    return "table" if item.get("category") == "I" else "prose"


def check_item(item: dict, split: str, ctx: Context) -> list[str]:
    """Every rule of the item's category that the item breaks, each as one line."""
    name = item.get("id", "(no id)")
    category, expect = item.get("category"), item.get("expect") or {}
    problems: list[str] = []
    say = problems.append
    if category not in FAMILY | OWN:
        return [f"{name}: no category {category!r}"]
    if (expect.get("status") == "not_found") != (category in REFUSED):
        say(f"{name}: a {category} item expects {'not_found' if category in REFUSED else 'answered'}")
    question = item.get("question") or ""
    if item.get("lang") not in ("en", "el") or language(question) != item.get("lang"):
        say(f"{name}: lang {item.get('lang')!r} is not the question's script")
    role = item.get("role")
    if role not in ctx.policy["roles"]:
        return problems + [f"{name}: no role {role!r}"]
    families = item.get("families") or []
    if category in OWN:
        if families:
            say(f"{name}: an {category} item belongs to no family")
    else:
        if len(families) != (2 if category in TWO_FAMILIES else 1):
            say(f"{name}: a {category} item names {2 if category in TWO_FAMILIES else 1} families")
        for family in families:
            if family not in ctx.families:
                say(f"{name}: no family {family}")
            elif ctx.families[family]["split"] != split:
                say(f"{name}: family {family} is in {ctx.families[family]['split']}, not {split}")
    reads = readable(ctx.policy, role)
    expected = (expect.get("forbidden") if category == "E" else expect.get("facts")) or []
    if category == "I" and not all((item.get("row") or {}).get(k) for k in ("doc_id", "key", "column", "value")):
        say(f"{name}: a table item names its row: doc_id, key, column and value")
    for entry in expected:
        fid = entry.get("fact_id")
        if category == "I":
            if not entry.get("forms"):
                say(f"{name}: a table item names the forms of its value")
            continue
        if fid not in ctx.facts:
            say(f"{name}: no fact {fid!r} in the registry")
            continue
        fact = ctx.facts[fid]
        if not any(fid in ctx.families.get(family, {}).get("facts", []) for family in families):
            say(f"{name}: {fid} is in none of the item's families")
        if not any(value in form for value in fact["values"] for form in entry.get("forms") or []):
            say(f"{name}: no form of {fid} holds either of its planted values {fact['values']}")
        placed = ctx.sections(fid)
        if category == "E":
            if any(section["label"] in reads for _, section in placed):
                say(f"{name}: {role} may read a section {fid} is planted in, so it is no E item")
        elif not any(section["label"] in reads for _, section in placed):
            say(f"{name}: {role} may read no section {fid} is planted in")
        if category == "C" and any(ctx.docs[doc_id]["language"] == item.get("lang") for doc_id, _ in placed):
            say(f"{name}: a C item asks in another language than every document of {fid}")
        if fid in DELIBERATE_DEFAULTS and category != "E":
            say(f"{name}: {fid} is a deliberate default, asked only by an E item")
        cap = ctx.protocol["items"]["max_shared_rare_terms"]
        asked = set(lexical.terms(question))
        for doc_id, _ in placed:
            planted = fact[ctx.docs[doc_id]["language"]]
            shared = [t for t in rare_terms(planted, ctx.df, ctx.protocol["items"]["rare_terms"]) if t in asked]
            if len(shared) > cap:
                say(f"{name}: shares {len(shared)} of the rarest terms of {fid}'s planted sentence: {shared}")
    if category == "F" and len({ctx.facts[e["fact_id"]]["label"] for e in expected
                                if e.get("fact_id") in ctx.facts}) < 2:
        say(f"{name}: an F item asks facts under two labels, each of which its role reads")
    if category == "G":
        olds = [fid for fid in expect.get("superseded") or [] if ctx.facts.get(fid, {}).get("kind") == "superseded"]
        if not olds or not all(ctx.facts.get(e.get("fact_id"), {}).get("kind") == "versioned" for e in expected):
            say(f"{name}: a G item expects the current value and names the superseded fact")
    if category == "H" and not all(ctx.facts.get(e.get("fact_id"), {}).get("kind") == "variant" for e in expected):
        say(f"{name}: an H item expects a regional variant")
    if category in FAMILY and not expected:
        say(f"{name}: a family item names the fact it asks" + (" under forbidden" if category == "E" else ""))
    if category in {"J", "N"} and expect.get("facts"):
        say(f"{name}: a {category} item expects no fact")
    if category == "N" and item.get("near") not in ctx.facts:
        say(f"{name}: a near miss names the present fact it neighbours")
    return problems


def check_split(items: list[dict], split: str, ctx: Context, test_shares: dict | None = None) -> list[str]:
    """The split's counts and shares against version 5."""
    problems: list[str] = []
    counts = Counter(item.get("category") for item in items)
    uses = Counter(family for item in items for family in item.get("families") or [])
    own = ctx.protocol["items"][split]
    mine = [fid for fid, family in ctx.families.items() if family["split"] == split]
    if split == "test":
        for code in sorted(FAMILY | OWN):
            if counts[code] != ctx.protocol["items"]["test"].get(code, 0):
                problems.append(f"test: {counts[code]} {code} items, registered {ctx.protocol['items']['test'][code]}")
        for family in mine:
            if uses[family] != 1:
                problems.append(f"test: family {family} asked {uses[family]} times, not once")
    else:
        for code in sorted(OWN):
            if counts[code] != own[code]:
                problems.append(f"dev: {counts[code]} {code} items, registered {own[code]}")
        for family in mine:
            if not 2 <= uses[family] <= 3:
                problems.append(f"dev: family {family} asked {uses[family]} times, not two or three")
        if test_shares and items:
            tolerance = own["share_tolerance"]
            for code, share in test_shares.items():
                if abs(counts[code] / len(items) - share) > tolerance:
                    problems.append(f"dev: {code} is {counts[code] / len(items):.3f} of the items, the test's "
                                    f"{share:.3f}, beyond {tolerance}")
    if items:
        low, high = ctx.protocol["items"]["dev"]["greek_share"]
        greek = sum(item.get("lang") == "el" for item in items) / len(items)
        if not low <= greek <= high:
            problems.append(f"{split}: Greek is {greek:.3f} of the items, outside {low} to {high}")
    return problems


def check(ctx: Context | None = None) -> tuple[dict, list[str]]:
    """Both splits' items, each against its rules, and a summary: items by split, category and layer."""
    ctx = ctx or Context()
    loaded = {split: load_items(golden_path(split)) for split in SPLITS}
    problems: list[str] = []
    ids = Counter(item.get("id") for split in SPLITS for item in loaded[split])
    problems += [f"id {i} used {n} times" for i, n in ids.items() if n > 1]
    for split in SPLITS:
        for item in loaded[split]:
            problems += check_item(item, split, ctx)
    test_counts = Counter(item.get("category") for item in loaded["test"])
    shares = {code: test_counts[code] / len(loaded["test"]) for code in FAMILY | OWN} if loaded["test"] else None
    for split in SPLITS:
        problems += check_split(loaded[split], split, ctx, shares if split == "dev" else None)
    summary = {split: {"items": len(loaded[split]),
                       "categories": dict(sorted(Counter(i.get("category") for i in loaded[split]).items())),
                       "layers": dict(Counter(layer(i) for i in loaded[split]))} for split in SPLITS}
    return summary, problems
