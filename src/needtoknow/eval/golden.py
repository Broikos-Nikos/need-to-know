"""The golden items and their construction checks (T5.1c2, `specs/evaluation.md` 2.3, protocol version 6).

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
and each split to the counts and shares version 6 registered.

Per item: a known category, and `not_found` exactly for E, J and N; the question's script
is its language; family items name their families, two for F and M, each in this split;
the role exists; A asks in English and B in Greek, each in the language of a document its fact sits in; D asks a
fact from a file that also holds a section its role may not read; an F item's facts sit under two labels its role
reads; an M item's two facts share no document; every expected fact is in the registry,
in one of the item's families, each of its forms holding its planted value or naming it; an answered item's
role may read a section each fact is planted in, an E item's none; C asks in the other
language than every document its facts sit in; G asks the current value and names the
superseded one of its own registry family, H a regional variant; E, J and N expect no fact, and a near miss's
neighbour is one its role reads, of a dev family for a dev item; a deliberate default is asked only by E (RS-F19);
a twin is a dev item the same as its original in every field
but its wording, sharing no more of the rarest terms than its original and at most half of its terms, one an item, as many
as registered (N59); an absent fact or near miss names its probes in `eval/golden/probes.json`, and
no prose unit of the corpus matches one (MT-F11); a table item's row is in the manifest, its key in one unit alone, a table of the named
document under a label its role reads, the cell under its column the value, which a form holds (DR-F20);
and the overlap cap: of the three rarest terms of each planted sentence by the corpus's
document frequency, at most one in the question, counted by `needtoknow.lexical.terms`,
the terms BM25 matches (RS-F3, KB3-K4). Per split: the test's category counts, every test
family asked once; two or three items for each dev family, and none for one whose every
fact is a deliberate default under the general label, which no item can ask (N58); the own items' counts; the
categories' shares and the Greek share. Each item has a layer, table for I and prose for
the rest, so every figure can be given by layer (KB1-F8).
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from needtoknow import lexical
from needtoknow.answer.prompt import language
from needtoknow.eval.protocol import load_protocol
from needtoknow.eval.split import DELIBERATE_DEFAULTS, load_plan, quotas, split_path
from needtoknow.index import frequencies
from needtoknow.ingest.manifest import manifest_path
from needtoknow.policy import load_policy, readable
from needtoknow.settings import repo_root

FAMILY = set("ABCDEFGHM")
OWN = set("IJN")
REFUSED = set("EJN")
TWO_FAMILIES = set("FM")
SPLITS = ("test", "dev")
# The test items are written once and never tuned on: no form is added to one after its answers are seen, so the file
# is held to this hash (`specs/evaluation.md` section 3). Changing it means saying why, here and in the history.
TEST_SHA256 = "a3a8ca9549d6ef12a816bc107e84faf458c928b15c4ea4366a0fd9d4b5fea5d3"
# The dev table items' rows (DR-F20, MT-F11). Only these registers key a row in one unit of the corpus: the ticket log,
# the stock list and the return authorisation log; the others reuse a key across documents with other values.
ROW_SEED = 20261013
# The dev items given a paraphrase twin (KB3-K4, N59), drawn by this seed.
TWIN_SEED = 20261014
# What a twin keeps of its original: all but its id, its wording and the twin field itself.
TWIN_KEEPS = ("category", "families", "role", "lang", "expect", "row", "near")
ROW_PAIRS = (("customer-care", "en"), ("customer-care", "el"), ("logistics", "en"), ("logistics", "el"),
             ("returns", "en"), ("returns", "el"), ("logistics", "en"), ("customer-care", "el"))


class GoldenError(Exception):
    """A golden file that cannot be read as items."""


def golden_path(split: str) -> Path:
    return repo_root() / "eval" / "golden" / f"{split}.jsonl"


def probes_path() -> Path:
    """The probes of every absent fact and near miss, by item id: regular expressions none of whose matches the
    corpus prose may hold (MT-F11)."""
    return repo_root() / "eval" / "golden" / "probes.json"


def load_items(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                item = json.loads(line)
            except ValueError as error:
                raise GoldenError(f"{path.name} line {number} is not JSON: {error}") from None
            if not isinstance(item, dict):
                raise GoldenError(f"{path.name} line {number} is JSON but not an item: an item is an object")
            items.append(item)
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
    """What the checks read: the protocol, the split, the registry, the catalogue, the policy, the frequencies and,
    for a table item, the manifest's units."""

    def __init__(self, protocol=None, split=None, facts=None, catalogue=None, policy=None, df=None, units=None):
        self.protocol = protocol or load_protocol()
        if split is None and not split_path().is_file():
            raise GoldenError(f"no {split_path().name}: ntk eval split writes it")
        result = split or json.loads(split_path().read_text(encoding="utf-8"))
        self.families = {f["id"]: f for f in result["split"]}
        if facts is None or catalogue is None:
            facts, catalogue = load_plan()
        self.facts = {f["id"]: f for f in facts}
        self.docs = {d["id"]: d for d in catalogue}
        self.policy = policy or load_policy()
        self.df = df or corpus_df()
        self.units = units  # given in a test; read from the manifest otherwise
        self.held: dict[str, list[dict]] = {}
        self.prose_units: list[dict] | None = None

    def sections(self, fact_id: str) -> list[tuple[str, dict]]:
        """Every section that plants a fact, with its document: a document that plants it twice is read once, and
        each of its sections that holds the fact is given (DR-F21)."""
        fact = self.facts[fact_id]
        return [(doc_id, section) for doc_id in dict.fromkeys(fact["placements"])
                for section in self.docs[doc_id]["sections"] if fact_id in section["facts"]]

    def holding(self, keys: list[str]) -> dict[str, list[dict]]:
        """Every unit whose text holds each key: one read of the manifest for all the keys not yet looked up."""
        wanted = [key for key in dict.fromkeys(keys) if key not in self.held]
        if wanted:
            found = self.units if self.units is not None else manifest_units(wanted)
            for key in wanted:
                self.held[key] = [unit for unit in found if key in unit["text"]]
        return {key: self.held[key] for key in keys}

    def prose(self) -> list[dict]:
        """Every prose unit of the corpus, read once: the 1,641 units an absent fact's probes are run over."""
        if self.prose_units is None:
            self.prose_units = ([unit for unit in self.units if unit["layer"] == "prose"] if self.units is not None
                                else manifest_prose())
        return self.prose_units


def manifest_prose() -> list[dict]:
    """The prose units with document, layer, labels and text, the manifest read only."""
    path = manifest_path()
    if not path.is_file():
        raise GoldenError(f"no manifest at {path}: ntk ingest build makes it, and an absent fact is checked against it")
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return [{"doc_id": doc_id, "layer": "prose", "labels": json.loads(labels), "text": text}
                for doc_id, labels, text in db.execute("SELECT doc_id, labels, text FROM units WHERE layer = 'prose'")]
    finally:
        db.close()


def absent_problems(loaded: dict[str, list[dict]], ctx: Context, probes: dict[str, list[str]]) -> list[str]:
    """Each absent fact and near miss against the corpus prose (MT-F11): it names its probes, each a regular
    expression, and no prose unit holds a match of any; a probe naming no such item is refused too."""
    problems = []
    asked = {item.get("id"): item for split in SPLITS for item in loaded[split] if item.get("category") in {"J", "N"}}
    for name, item in asked.items():
        if not probes.get(name):
            problems.append(f"{name}: an absent fact names the probes its absence rests on, in eval/golden/probes.json")
        for probe in probes.get(name) or []:
            try:
                pattern = re.compile(probe)
            except re.error as error:
                problems.append(f"{name}: the probe {probe!r} is no regular expression: {error}")
                continue
            hits = sorted({unit["doc_id"] for unit in ctx.prose() if pattern.search(unit["text"])})
            if hits:
                problems.append(f"{name}: the corpus prose answers it: {probe!r} matches in {', '.join(hits[:3])}"
                                + (f" and {len(hits) - 3} more" if len(hits) > 3 else ""))
    problems += [f"eval/golden/probes.json names {name}, no absent fact or near miss" for name in probes
                 if name not in asked]
    return problems


def draw_rows(seed: int = ROW_SEED, pairs: tuple = ROW_PAIRS, attempts: int = 50) -> list[dict]:
    """The dev table items' rows as the seed draws them: for each label and language, the table units of that label
    in that language in id order, one drawn; in it a data row and a column other than the key, each drawn; kept when
    the key is in one unit of the whole manifest and its cell is not empty, else the next draw."""
    path = manifest_path()
    if not path.is_file():
        raise GoldenError(f"no manifest at {path}: ntk ingest build makes it")
    rng = random.Random(seed)
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    drawn = []
    try:
        tables: dict[tuple[str, str], list[str]] = {}  # one scan, not one a pair: each costs seconds
        for unit_id, labels, lang in db.execute("SELECT unit_id, labels, language FROM units WHERE layer = 'table' "
                                                    "ORDER BY unit_id"):
            tables.setdefault((labels, lang), []).append(unit_id)
        for label, lang in pairs:
            ids = tables.get((json.dumps([label]), lang), [])
            if not ids:  # a corpus with no such register: named, not an IndexError (DR-F23)
                raise GoldenError(f"no {label} table units in {lang} to draw a row from")
            for _ in range(attempts):
                doc_id, text = db.execute("SELECT doc_id, text FROM units WHERE unit_id = ?",
                                          (rng.choice(ids),)).fetchone()
                lines = text.splitlines()
                header = [c.strip() for c in lines[0].split("|")]
                data = [[c.strip() for c in line.split("|")] for line in lines[1:]]
                data = [cells for cells in data if len(cells) == len(header) and cells[0]]
                if len(header) < 2 or not data:
                    continue
                key, column = rng.choice(data)[0], rng.choice(header[1:])
                value = cell(text, key, column)
                if value and db.execute("SELECT COUNT(*) FROM units WHERE instr(text, ?) > 0", (key,)).fetchone()[0] == 1:
                    drawn.append({"label": label, "language": lang, "doc_id": doc_id, "key": key,
                                  "column": column, "value": value})
                    break
            else:
                raise GoldenError(f"no {label} table row in {lang} had a key of its own in {attempts} draws")
    finally:
        db.close()
    return drawn


def draw_twins(dev: list[dict], test: list[dict], count: int, seed: int = TWIN_SEED) -> list[str]:
    """The dev items given a paraphrase twin (KB3-K4, N59). A twin adds an item to its family, so an item may take one
    only when each of its families holds two, and no family takes two twins; a table item, an absent fact and a near
    miss take none, their counts being registered exactly. Seats go by the test's category shares,
    largest remainder with the seed's ties, among the categories with such an item; the categories with the fewest
    such items draw first, so an F or M item, whose two families any other draw could take, keeps its seat; within
    each its items are taken in the seed's order, skipping one whose family already has a twin. Twins written are
    left out."""
    originals = [item for item in dev if not item.get("twin")]
    uses = Counter(family for item in originals for family in item.get("families") or [])
    eligible = defaultdict(list)
    for item in originals:
        if item["category"] in FAMILY and all(uses[family] == 2 for family in item["families"]):
            eligible[item["category"]].append(item["id"])
    shares = Counter(item["category"] for item in test)
    rng = random.Random(seed)
    seats = quotas({category: shares[category] for category in sorted(eligible) if shares[category]}, count, rng)
    families = {item["id"]: item.get("families") or [] for item in originals}
    taken: set[str] = set()
    drawn: list[str] = []
    for category in sorted(seats, key=lambda category: (len(eligible[category]), category)):
        order = sorted(eligible[category])
        rng.shuffle(order)
        for name in order:
            if seats[category] == 0:
                break
            if not taken & set(families[name]):
                drawn.append(name)
                taken |= set(families[name])
                seats[category] -= 1
    return drawn


def twin_problems(loaded: dict[str, list[dict]], ctx: Context) -> list[str]:
    """Each twin against its original (N59): a dev item, the same in every field but its wording, which shares no more
    of the rarest terms of the facts' planted sentences than the original does, a name the question needs at most,
    and at most half of its terms with the original's question; an item has one twin at most, a twin none, and the
    dev split holds the registered number."""
    problems = []
    problems += [f"{item.get('id')}: a twin is a dev item" for item in loaded["test"] if item.get("twin")]
    dev = {item.get("id"): item for item in loaded["dev"]}
    twins = [item for item in loaded["dev"] if item.get("twin")]
    for twin, n in Counter(item["twin"] for item in twins).items():
        if n > 1:
            problems.append(f"{twin} has {n} twins, not one")
    count = ctx.protocol["items"]["paraphrase_twins"]
    if twins and len(twins) != count:
        problems.append(f"dev: {len(twins)} twins, registered {count}")
    rare_count = ctx.protocol["items"]["rare_terms"]
    for item in twins:
        name, original = item.get("id"), dev.get(item["twin"])
        if original is None or original.get("twin"):
            problems.append(f"{name}: its twin {item['twin']} is a dev item that is no twin itself")
            continue
        differ = [field for field in TWIN_KEEPS if item.get(field) != original.get(field)]
        if differ:
            problems.append(f"{name}: a twin keeps its original's {', '.join(differ)}")
        mine, theirs = set(lexical.terms(item.get("question") or "")), set(lexical.terms(original.get("question") or ""))
        if mine == theirs or len(mine & theirs) > len(mine | theirs) / 2:
            problems.append(f"{name}: shares {len(mine & theirs)} of {len(mine | theirs)} terms with {item['twin']}, "
                            "more than half")
        for entry in (item.get("expect") or {}).get("facts") or (item.get("expect") or {}).get("forbidden") or []:
            fact = ctx.facts.get(entry.get("fact_id"))
            for doc_id, _ in ctx.sections(fact["id"]) if fact else []:
                rare = rare_terms(fact[ctx.docs[doc_id]["language"]], ctx.df, rare_count)
                shared, before = [t for t in rare if t in mine], [t for t in rare if t in theirs]
                if len(shared) > len(before):
                    problems.append(f"{name}: shares {shared} of the rarest terms of {fact['id']}, more than its "
                                    f"original's {before}")
                    break
    return problems


def manifest_units(keys: list[str]) -> list[dict]:
    """The units whose text holds any of the keys, with document, layer, labels and text, the manifest read only."""
    path = manifest_path()
    if not path.is_file():
        raise GoldenError(f"no manifest at {path}: ntk ingest build makes it, and a table item is checked against it")
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        query = "SELECT doc_id, layer, labels, text FROM units WHERE " + " OR ".join(["instr(text, ?) > 0"] * len(keys))
        return [{"doc_id": doc_id, "layer": layer, "labels": json.loads(labels), "text": text}
                for doc_id, layer, labels, text in db.execute(query, keys)]
    finally:
        db.close()


def cell(text: str, key: str, column: str) -> str | None:
    """The cell under the column of the table row whose first cell is the key; the header is the unit's first line."""
    lines = text.splitlines()
    header = [c.strip() for c in lines[0].split("|")] if lines else []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        if cells[0] == key and column in header and len(cells) == len(header):
            return cells[header.index(column)]
    return None


def table_row(name: str, row: dict, role: str, reads: frozenset[str], ctx: Context) -> list[str]:
    """A table item's row as the manifest holds it (DR-F20): one unit holds the key, in the named document, a table
    under a label the role reads, its row's cell under the column the value."""
    holders = ctx.holding([row["key"]])[row["key"]]
    if len(holders) != 1 or holders[0]["doc_id"] != row["doc_id"]:
        return [f"{name}: the key {row['key']} is in {len(holders)} units, not in one unit of {row['doc_id']} alone"]
    unit, problems = holders[0], []
    if unit["layer"] != "table":
        problems.append(f"{name}: the unit holding {row['key']} is no table")
    if not set(unit["labels"]) & reads:
        problems.append(f"{name}: {role} may read no label of the table holding {row['key']}, {unit['labels']}")
    found = cell(unit["text"], row["key"], row["column"])
    if found != row["value"]:
        problems.append(f"{name}: the row of {row['key']} holds {found!r} under {row['column']!r}, not {row['value']!r}")
    return problems


def masked(sentence: str, values: list[str], rare: list[str]) -> str:
    """A planted sentence with its value and every word holding one of its rarest terms blanked."""
    for value in sorted(set(values), key=len, reverse=True):  # whole numbers only: a value of 24 leaves 2024 alone
        sentence = re.sub("(?<![0-9])" + re.escape(value) + "(?![0-9])", "[value]", sentence)
    return " ".join("___" if set(lexical.terms(word)) & set(rare) else word for word in sentence.split())


def draft_view(split: str, ctx: Context) -> list[str]:
    """The split's families as a writer drafts from them (RS-F21): each fact's planted sentence with its value and its
    three rarest terms blanked, so the subject is plain and the words that decide retrieval are not in view; and where
    it sits, each document's language, the section's label and the other labels in the file, for C, D and E."""
    lines = []
    count = ctx.protocol["items"]["rare_terms"]
    for fid, family in sorted(ctx.families.items()):
        if family["split"] != split:
            continue
        lines.append(f"{fid}: {family['kind']}, {family['label']}"
                     + (f"; defaults {', '.join(sorted(family['defaults']))}" if family["defaults"] else ""))
        for fact_id in family["facts"]:
            fact = ctx.facts[fact_id]
            for language_of in sorted({ctx.docs[d]["language"] for d in fact["placements"]}):
                rare = rare_terms(fact[language_of], ctx.df, count)
                lines.append(f"  {fact_id} {fact['kind']} {language_of}: {masked(fact[language_of], fact['values'], rare)}"
                             f"  [rare: {', '.join(rare)}; at most one in a question]")
            where = []
            for doc_id, section in ctx.sections(fact_id):
                others = sorted({s["label"] for s in ctx.docs[doc_id]["sections"]} - {section["label"]})
                where.append(f"{doc_id} {ctx.docs[doc_id]['language']} {section['label']}"
                             + (f" beside {', '.join(others)}" if others else ""))
            lines.append(f"    value {fact['values']}; {'; '.join(where)}")
    return lines


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
    row = item.get("row") or {}
    if category == "I" and not all(row.get(k) for k in ("doc_id", "key", "column", "value")):
        say(f"{name}: a table item names its row: doc_id, key, column and value")
    elif category == "I":
        problems += table_row(name, row, role, reads, ctx)
    for entry in expected:
        fid = entry.get("fact_id")
        if category == "I":
            if not entry.get("forms"):
                say(f"{name}: a table item names the forms of its value")
            elif row.get("value") and not any(row["value"] in form for form in entry["forms"]):
                say(f"{name}: no form holds the row's value {row['value']!r}")
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
        if category in {"A", "B"} and not any(ctx.docs[doc_id]["language"] == item.get("lang") for doc_id, _ in placed):
            say(f"{name}: {'an A' if category == 'A' else 'a B'} item asks in the language of a document {fid} sits in")
        if category == "D" and not any(section["label"] in reads and
                                       any(other["label"] not in reads for other in ctx.docs[doc_id]["sections"])
                                       for doc_id, section in placed):
            say(f"{name}: a D item asks {fid} from a file that also holds a section {role} may not read")
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
    if category in {"A", "B"} and item.get("lang") != {"A": "en", "B": "el"}[category]:
        say(f"{name}: {'an A item asks in English' if category == 'A' else 'a B item asks in Greek'}")
    if category == "M":
        held = [{doc_id for doc_id, _ in ctx.sections(e["fact_id"])} for e in expected if e.get("fact_id") in ctx.facts]
        if len(held) != 2 or held[0] & held[1]:
            say(f"{name}: an M item asks two facts that no one document holds together")
    if category == "F" and len({ctx.facts[e["fact_id"]]["label"] for e in expected
                                if e.get("fact_id") in ctx.facts}) < 2:
        say(f"{name}: an F item asks facts under two labels, each of which its role reads")
    if category == "G":
        olds = [fid for fid in expect.get("superseded") or [] if ctx.facts.get(fid, {}).get("kind") == "superseded"]
        if not olds or not all(ctx.facts.get(e.get("fact_id"), {}).get("kind") == "versioned" for e in expected):
            say(f"{name}: a G item expects the current value and names the superseded fact")
        asked = {ctx.facts[e["fact_id"]]["family"] for e in expected if e.get("fact_id") in ctx.facts}
        if any(ctx.facts[old]["family"] not in asked for old in olds):
            say(f"{name}: a G item's superseded fact is a former value of the fact it asks")
    if category == "H" and not all(ctx.facts.get(e.get("fact_id"), {}).get("kind") == "variant" for e in expected):
        say(f"{name}: an H item expects a regional variant")
    if category in FAMILY and not expected:
        say(f"{name}: a family item names the fact it asks" + (" under forbidden" if category == "E" else ""))
    if category in REFUSED and expect.get("facts"):
        say(f"{name}: {'a J' if category == 'J' else 'an ' + category} item expects no fact"
            + ("; what it must not say is under forbidden" if category == "E" else ""))
    if category == "N" and item.get("near") not in ctx.facts:
        say(f"{name}: a near miss names the present fact it neighbours")
    elif category == "N" and not any(section["label"] in reads for _, section in ctx.sections(item["near"])):
        say(f"{name}: a near miss's neighbour is a fact {role} may read")
    # A near miss is written to draw its neighbour's section: beside a test fact, a dev one would bring that sentence
    # into development, which the split by family keeps out (N56). A test one beside a dev fact shows test nothing new.
    if category == "N" and split == "dev" and item.get("near") in ctx.facts and not any(
            item["near"] in family["facts"] for family in ctx.families.values() if family["split"] == "dev"):
        say(f"{name}: a dev near miss's neighbour is a fact of a dev family")
    return problems


def exempt(family: dict, ctx: Context) -> bool:
    """Every fact a deliberate default under the general label: only a refusal may ask a default, and no role is
    refused general, so no item can ask the family (N58)."""
    return all(fid in family["defaults"] and all(s["label"] == "general" for _, s in ctx.sections(fid))
               for fid in family["facts"])


def check_split(items: list[dict], split: str, ctx: Context, test_shares: dict | None = None) -> list[str]:
    """The split's counts and shares against version 6."""
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
            if exempt(ctx.families[family], ctx):
                if uses[family]:
                    problems.append(f"dev: family {family} is exempt, every fact a general default, and was asked")
                continue
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
    ctx.holding([key for split in SPLITS for item in loaded[split] if item.get("category") == "I"
                 for key in [(item.get("row") or {}).get("key")] if key])  # every table row in one read
    ids = Counter(item.get("id") for split in SPLITS for item in loaded[split])
    problems += [f"id {i} used {n} times" for i, n in ids.items() if n > 1]
    for split in SPLITS:
        for item in loaded[split]:
            problems += check_item(item, split, ctx)
    test_counts = Counter(item.get("category") for item in loaded["test"])
    shares = {code: test_counts[code] / len(loaded["test"]) for code in FAMILY | OWN} if loaded["test"] else None
    for split in SPLITS:
        problems += check_split(loaded[split], split, ctx, shares if split == "dev" else None)
    probes = json.loads(probes_path().read_text(encoding="utf-8")) if probes_path().is_file() else {}
    problems += absent_problems(loaded, ctx, probes)
    problems += twin_problems(loaded, ctx)
    summary = {split: {"items": len(loaded[split]), "greek": sum(i.get("lang") == "el" for i in loaded[split]),
                       "family_items": sum(i.get("category") in FAMILY for i in loaded[split]),
                       "categories": dict(sorted(Counter(i.get("category") for i in loaded[split]).items())),
                       "layers": dict(Counter(layer(i) for i in loaded[split]))} for split in SPLITS}
    summary["dev"]["exempt"] = sorted(fid for fid, family in ctx.families.items()
                                      if family["split"] == "dev" and exempt(family, ctx))  # MA-F14
    return summary, problems
