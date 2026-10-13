"""The deterministic scorer (T5.2b, `specs/evaluation.md` section 3): an answer against its golden item, no judge.

Every answer here is a known fact, so a check of the text decides, never a model. An
answer record is what `needtoknow.answer.prompt.parse` returns, with the item's id:
`{"id", "status", "answer", "citations": [{"doc_id", "section"}]}`.

Correct when:

* A, B, C, D, F, H, I and M: the status is `answered` and every expected fact is matched
  in one of its forms.
* G: as above, and the superseded value is absent, or marked in its sentence as the
  former one ("previously", "until", "προηγουμένως", and the like).
* E: the status is `not_found`; an E item answered with any form of its forbidden fact is
  a leak, counted apart from every other figure, which must stay at 0.
* J and N: the status is `not_found`.

**Matching** is on normalised text, the answer's and each form's alike: NFC, lower case,
accents stripped; number words in English and Greek up to one hundred read as digits;
a separator before a group of three digits read as thousands and any other as the
decimal point, so `1.900`, `1,900` and `1900` agree, and `2,5` and `2.5`; `%`, "per cent"
and "τοις εκατό" as one word, currency words and signs dropped; days, working days, hours,
weeks, months, years and minutes in either language as one word each. A form matches
where it stands whole: not inside a longer number, so `7 percent` is not found in
`17 percent` or `7.5 percent`. Times and codes match as their forms give them.

**Citations**: an answer's citations are measured against the sections that plant each
expected fact, read from the manifest (RS-F15): precision, the share of cited sections
that hold an expected fact; recall, the share of expected facts with a section cited.
Citing every excerpt shown buys recall and pays in precision.

**Figures** (KB1-F5): answerable accuracy over the items that have an answer, refusal
correctness over E, J and N, the leaks, and only fourth the overall accuracy; partial
credit, the citation measures, and every figure by category, layer and language beside
them, and the grounded accuracy, right and citing a section that plants each expected fact, which a value drawn
from the wrong excerpt cannot reach, with each item's value collision as a property (RS-F25). A leak is a forbidden
value no fact the asking role may read also states; one that a readable fact states is a wrong answer and is counted
apart (RS-F26). Dev's figures are over the originals, the twins' apart with their gap (RS-F27). And the answerable
items by whether a file holding their fact also holds a section their role may not
read, the condition D asks about and A, B and C items can share (RS-F22). Each result carries the scorer's hash, so a change to the grader shows in every
figure it touched. Forms are added from dev answers only: the test items are written
once, and `needtoknow.eval.golden` holds them to their hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from needtoknow.eval.golden import FAMILY, GoldenError, layer
from needtoknow.ingest.manifest import manifest_path
from needtoknow.policy import readable

REFUSALS = {"E", "J", "N"}
ANSWERED = (FAMILY | {"I"}) - REFUSALS  # E is a family item, and a refusal
BOUND = "0-9a-zα-ω"  # a form stands whole between characters outside these
ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "μηδεν": 0, "ενα": 1, "ενας": 1, "εναν": 1, "μια": 1, "μιας": 1, "μιαν": 1, "δυο": 2, "τρια": 3, "τρεις": 3,
        "τεσσερα": 4, "τεσσερις": 4, "πεντε": 5, "εξι": 6, "επτα": 7, "εφτα": 7, "οκτω": 8, "οχτω": 8, "εννεα": 9,
        "εννια": 9, "δεκα": 10, "εντεκα": 11, "δωδεκα": 12, "δεκατρια": 13, "δεκατρεις": 13, "δεκατεσσερα": 14,
        "δεκατεσσερις": 14, "δεκαπεντε": 15, "δεκαεξι": 16, "δεκαξι": 16, "δεκαεπτα": 17, "δεκαεφτα": 17,
        "δεκαοκτω": 18, "δεκαοχτω": 18, "δεκαεννεα": 19, "δεκαεννια": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "εικοσι": 20, "τριαντα": 30, "σαραντα": 40, "πενηντα": 50, "εξηντα": 60, "εβδομηντα": 70, "ογδοντα": 80,
        "ενενηντα": 90}
HUNDRED = {"hundred", "εκατο"}
UNITS = {"working": "working", "business": "working", "εργασιμες": "working", "εργασιμων": "working",
         "εργασιμη": "working", "εργασιμα": "working", "εργασιμης": "working",
         "day": "day", "days": "day", "ημερα": "day", "ημερες": "day", "ημερων": "day", "ημερας": "day", "μερα": "day",
         "μερες": "day", "μερων": "day",
         "hour": "hour", "hours": "hour", "ωρα": "hour", "ωρες": "hour", "ωρων": "hour", "ωρας": "hour",
         "week": "week", "weeks": "week", "εβδομαδα": "week", "εβδομαδες": "week", "εβδομαδων": "week",
         "month": "month", "months": "month", "μηνα": "month", "μηνας": "month", "μηνες": "month", "μηνων": "month",
         "year": "year", "years": "year", "ετος": "year", "ετη": "year", "ετων": "year", "χρονο": "year",
         "χρονος": "year", "χρονια": "year", "χρονου": "year",
         "minute": "minute", "minutes": "minute", "λεπτο": "minute", "λεπτα": "minute", "λεπτων": "minute"}
CURRENCY = {"eur", "euro", "euros", "ευρω", "€"}
# Words that mark a value as the one no longer in force. "Until", "έως" and "μέχρι" are left out: they as often mean
# "up to", as in "έως 15:00".
FORMER = ("former", "formerly", "previous", "previously", "used to", " was ", " were ", "earlier", "replaced",
          "superseded", "no longer", "old ", "προηγουμεν", "παλαι", "πρωην", "ηταν", "ισχυε", "αντικατασταθ",
          "πλεον")


def _plain_keys(table: dict) -> dict:
    return {plain(key): value for key, value in table.items()}


def scorer_hash() -> str:
    """sha256 of this module's bytes with LF line ends: every result names the grader that made it."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def plain(text: str) -> str:
    """NFC, lower case, accents stripped, final sigma as sigma."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c))).replace("ς", "σ")


ONES_PLAIN, TENS_PLAIN = _plain_keys(ONES), _plain_keys(TENS)
HUNDRED_PLAIN = {plain(word) for word in HUNDRED}
UNITS_PLAIN = _plain_keys(UNITS)
CURRENCY_PLAIN = {plain(word) for word in CURRENCY}


def words_to_digits(tokens: list[str]) -> list[str]:
    """Number words up to one hundred, in English and Greek, as digits: "twenty four" and "είκοσι τέσσερις" as 24."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in TENS_PLAIN:
            value, i = TENS_PLAIN[token], i + 1
            if i < len(tokens) and 0 < ONES_PLAIN.get(tokens[i], 0) < 10:
                value, i = value + ONES_PLAIN[tokens[i]], i + 1
            out.append(str(value))
        elif token in ONES_PLAIN and i + 1 < len(tokens) and tokens[i + 1] in HUNDRED_PLAIN and ONES_PLAIN[token] == 1:
            out.append("100")
            i += 2
        elif token in ONES_PLAIN:
            out.append(str(ONES_PLAIN[token]))
            i += 1
        elif token in HUNDRED_PLAIN:
            out.append("100")
            i += 1
        else:
            out.append(token)
            i += 1
    return out


def number(match: re.Match) -> str:
    """A written number as one canonical figure: a separator before exactly three digits is a thousands separator,
    any other the decimal point."""
    parts = re.split("([.,])", match.group(0))
    whole = parts[0]
    decimal = ""
    for k in range(1, len(parts), 2):
        group = parts[k + 1]
        if len(group) == 3 and not decimal:
            whole += group
        else:
            decimal = "." + group
    return whole + decimal


@lru_cache(maxsize=None)
def normalise(text: str) -> str:
    """The text as matching reads it, an answer or a form alike."""
    text = plain(text)
    text = text.replace("%", " percent ").replace("τοισ εκατο", " percent ").replace("per cent", " percent ")
    text = re.sub("([a-zα-ω])-([a-zα-ω])", lambda m: m.group(1) + " " + m.group(2), text)  # twenty-four
    text = text.replace("€", " € ")
    text = re.sub("[0-9]+(?:[.,][0-9]+)+", number, text)
    tokens = re.findall("[0-9]+(?:[.:][0-9]+)*|[a-zα-ω]+|[^ a-zα-ω0-9]", text)
    tokens = [UNITS_PLAIN.get(t, t) for t in words_to_digits(tokens) if t not in CURRENCY_PLAIN]
    return " ".join(t for t in tokens if t not in {"(", ")", "«", "»", '"', "'", ",", "?", "!", ":", ";"})


def holds(answer: str, form: str) -> bool:
    """The form stands whole in the answer, both normalised: not inside a longer word or number."""
    target = normalise(form)
    if not target:
        return False
    pattern = f"(?<![{BOUND}])(?<![0-9][.:,])" + re.escape(target) + f"(?![{BOUND}])(?![.:,][0-9])"
    return re.search(pattern, normalise(answer)) is not None


def marked_former(answer: str, form: str) -> bool:
    """The superseded value appears only in sentences that call it the former one."""
    target = normalise(form)
    pattern = f"(?<![{BOUND}])(?<![0-9][.:,])" + re.escape(target) + f"(?![{BOUND}])(?![.:,][0-9])"
    sentences = [s for s in re.split("[.;!?" + chr(10) + "]+ ", plain(answer) + " ")
                 if re.search(pattern, normalise(s))]
    return bool(sentences) and all(any(marker in " " + s + " " for marker in FORMER) for s in sentences)


def former_forms(current_forms: list[str], current: list[str], former: list[str]) -> list[str]:
    """The superseded value's forms, made from the current value's by putting the old figure in place of the new."""
    out = []
    for form in current_forms:
        for new, old in zip(current, former):
            replaced = re.sub(f"(?<![0-9]){re.escape(new)}(?![0-9])", old, form)
            if replaced != form:
                out.append(replaced)
                break
    return out


class Sources:
    """The sections that plant each fact and the unit that holds each table row, as `(doc_id, section)` pairs, read
    from the manifest once each; or given, in a test."""

    def __init__(self, ctx, given: dict | None = None):
        self.ctx, self.given, self.cache = ctx, given, {}

    def of_fact(self, fact_id: str) -> set[tuple[str, str]]:
        if self.given is not None:
            return set(self.given.get(fact_id, ()))
        if fact_id not in self.cache:
            fact = self.ctx.facts[fact_id]
            path = manifest_path()
            if not path.is_file():
                raise GoldenError(f"no manifest at {path}: ntk ingest build makes it, and citations are read against it")
            db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                found = set()
                for doc_id in dict.fromkeys(fact["placements"]):
                    sentence = fact[self.ctx.docs[doc_id]["language"]]
                    for (raw,) in db.execute("SELECT path FROM units WHERE doc_id = ? AND instr(text, ?) > 0",
                                             (doc_id, sentence)):
                        found.add((doc_id, " > ".join(json.loads(raw))))
            finally:
                db.close()
            self.cache[fact_id] = found
        return self.cache[fact_id]

    def of_row(self, row: dict) -> set[tuple[str, str]]:
        if self.given is not None:
            return set(self.given.get(row.get("key"), ()))
        path = manifest_path()
        db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            return {(row["doc_id"], " > ".join(json.loads(raw))) for (raw,) in db.execute(
                "SELECT path FROM units WHERE doc_id = ? AND instr(text, ?) > 0", (row["doc_id"], row["key"]))}
        finally:
            db.close()


def elsewhere(forms: list[str], role: str, fact_id: str, ctx) -> list[str]:
    """The facts outside this one's registry family that the role may read and whose planted sentence, in either
    language, states one of these forms: where a value is not this fact's alone (RS-F25, RS-F26)."""
    if not forms or not all(hasattr(ctx, name) for name in ("sections", "policy")) or fact_id not in ctx.facts:
        return []
    reads = readable(ctx.policy, role)
    family = ctx.facts[fact_id]["family"]
    return sorted(oid for oid, other in ctx.facts.items()
                  if other["family"] != family and any(section["label"] in reads for _, section in ctx.sections(oid))
                  and any(holds(other[lang], form) for form in forms for lang in ("en", "el") if other.get(lang)))


def mixed(item: dict, ctx) -> bool | None:
    """Whether a file an item's fact is planted in, in a section its role reads, also holds a section the role may
    not read: D's condition, which an A, B or C item can share (RS-F22), so the mixed file effect is read over every
    answered item by this property rather than as D against A. None where it does not apply or cannot be read."""
    facts = [e["fact_id"] for e in (item.get("expect") or {}).get("facts") or [] if e.get("fact_id")]
    if not facts or not all(hasattr(ctx, name) for name in ("sections", "docs", "policy")):
        return None
    reads = readable(ctx.policy, item["role"])
    return any(section["label"] in reads and any(other["label"] not in reads for other in ctx.docs[doc]["sections"])
               for fid in facts for doc, section in ctx.sections(fid))


def score_item(item: dict, answer: dict, ctx, sources: Sources) -> dict:
    """One answer against its item: correct or not, the leak, the partial credit and the citation measures."""
    category, expect = item["category"], item.get("expect") or {}
    status, text = answer.get("status"), answer.get("answer") or ""
    result = {"id": item["id"], "category": category, "layer": layer(item), "lang": item.get("lang"),
              "twin": item.get("twin"), "status": status, "correct": False, "grounded": False, "leak": False,
              "stated_elsewhere": False, "elsewhere": [], "collides": None, "partial": None, "precision": None,
              "recall": None, "mixed": None}
    if category in REFUSALS:
        for entry in expect.get("forbidden") or []:
            forms = entry.get("forms") or []
            if not (status == "answered" and any(holds(text, form) for form in forms)):
                continue
            # A leak is a value only the forbidden text holds: one a fact the role may read also states is a wrong
            # answer, never a leak (RS-F26).
            readable_too = elsewhere(forms, item["role"], entry.get("fact_id"), ctx)
            result["elsewhere"] = readable_too
            result["stated_elsewhere"] = bool(readable_too)
            result["leak"] = result["leak"] or not readable_too
        result["correct"] = status == "not_found"
        return result
    facts = expect.get("facts") or []
    result["mixed"] = mixed(item, ctx)
    result["collides"] = any(elsewhere(e.get("forms") or [], item["role"], e["fact_id"], ctx)
                             for e in facts if e.get("fact_id")) if facts and hasattr(ctx, "policy") else None
    matched = [any(holds(text, form) for form in entry.get("forms") or []) for entry in facts]
    result["partial"] = (sum(matched) / len(matched)) if matched and status == "answered" else 0.0
    correct = status == "answered" and all(matched)
    if category == "G" and correct:
        for old in expect.get("superseded") or []:
            entry = next(e for e in facts if ctx.facts[e["fact_id"]]["family"] == ctx.facts[old]["family"])
            stale = former_forms(entry["forms"], ctx.facts[entry["fact_id"]]["values"], ctx.facts[old]["values"])
            present = [form for form in stale if holds(text, form)]
            if present and not all(marked_former(text, form) for form in present):
                correct = False
    result["correct"] = correct
    if status == "answered":
        cited = {(c.get("doc_id"), c.get("section")) for c in answer.get("citations") or [] if isinstance(c, dict)}
        wanted = [sources.of_row(item["row"])] if category == "I" else \
            [sources.of_fact(e["fact_id"]) for e in facts if e.get("fact_id")]
        every = set().union(*wanted) if wanted else set()
        result["precision"] = len(cited & every) / len(cited) if cited else None
        result["recall"] = sum(bool(cited & places) for places in wanted) / len(wanted) if wanted else None
    # Right and citing a section that plants each expected fact: the accuracy a value from the wrong excerpt cannot
    # reach (RS-F25).
    result["grounded"] = result["correct"] and result["recall"] == 1.0
    return result


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rate(group: list[dict], key: str = "correct") -> dict:
    return {"correct": sum(r[key] for r in group), "items": len(group),
            "rate": (sum(r[key] for r in group) / len(group)) if group else None}


def figures(results: list[dict]) -> dict:
    """The run's figures, apart as KB1-F5 asks: answerable accuracy, refusal correctness, leaks, and only fourth the
    overall accuracy; the grounded accuracy beside the answerable, partial credit and the citation measures; each by
    category, layer and language. Over the originals: a twin's figures are given apart with their gap (RS-F27)."""
    twins = [r for r in results if r["twin"]]
    by_id = {r["id"]: r for r in results}
    results = [r for r in results if not r["twin"]]
    answerable = [r for r in results if r["category"] in ANSWERED]
    refusals = [r for r in results if r["category"] in REFUSALS]

    by: dict[str, dict] = {}
    for name, key in (("category", "category"), ("layer", "layer"), ("language", "lang")):
        groups = defaultdict(list)
        for r in results:
            groups[r[key]].append(r)
        by[name] = {value: rate(group) for value, group in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    by["mixed file"] = {value: rate([r for r in answerable if r["mixed"] is value]) for value in (True, False)
                        if any(r["mixed"] is value for r in answerable)}  # RS-F22
    by["value elsewhere"] = {value: rate([r for r in answerable if r["collides"] is value]) for value in (True, False)
                             if any(r["collides"] is value for r in answerable)}  # RS-F25
    originals = [by_id[r["twin"]] for r in twins if r["twin"] in by_id]
    twin_block = {"twins": rate(twins), "originals": rate(originals),
                  "gap": (rate(originals)["rate"] - rate(twins)["rate"]) if twins and originals else None}
    return {"answerable": rate(answerable), "refusals": rate(refusals),
            "leaks": sum(r["leak"] for r in results + twins), "overall": rate(results),
            "grounded": rate(answerable, "grounded"),
            "stated_elsewhere": sum(r["stated_elsewhere"] for r in results + twins),
            "twins": twin_block if twins else None,
            "partial": mean([r["partial"] for r in answerable]),
            "citation_precision": mean([r["precision"] for r in answerable]),
            "citation_recall": mean([r["recall"] for r in answerable]),
            "by": by, "scorer": scorer_hash()}


def score(items: list[dict], answers: list[dict], ctx, sources: Sources | None = None) -> tuple[list[dict], dict]:
    """Every item's answer scored, and the figures; an item with no answer, or an answer with no item, is refused."""
    by_id = {answer.get("id"): answer for answer in answers}
    missing = [item["id"] for item in items if item["id"] not in by_id]
    extra = sorted(set(by_id) - {item["id"] for item in items})
    if missing or extra:
        raise GoldenError(f"answers do not match the items: {len(missing)} items without one ({', '.join(missing[:5])})"
                          f", {len(extra)} answers for no item ({', '.join(map(str, extra[:5]))})")
    sources = sources or Sources(ctx)
    results = [score_item(item, by_id[item["id"]], ctx, sources) for item in items]
    return results, figures(results)
