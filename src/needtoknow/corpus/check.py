"""A light check of the corpus: every planted fact where it was registered and nowhere it was not, headers
that parse, canaries in place, the language each file claims, and prose that invents no policy value.

Light on purpose: it reads the writers' narrative, where every risk to consistency
lives, and counts the assembled files and their sizes. The full verifier (T1.8)
can come later.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from needtoknow.corpus.assemble import parse_sections
from needtoknow.corpus.plan import LABEL_EL, LABEL_EN
from needtoknow.settings import data_dir

LABELS = {**{v: k for k, v in LABEL_EN.items()}, **{v: k for k, v in LABEL_EL.items()}}
# A number followed by a policy unit: a value a writer may have invented.
POLICY_VALUE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|days?|hours?|minutes?|months?|weeks?|years?|euros?|EUR|percent|kilograms?|"
                          r"ημέρ\w*|ώρ\w*|λεπτ\w*|μήν\w*|μην\w*|εβδομάδ\w*|έτ\w*|ευρώ|τοις εκατό|κιλ\w*)", re.IGNORECASE)


def squash(text: str) -> str:
    return " ".join(text.split())


def check() -> dict:
    base = data_dir() / "corpus"
    docs = json.loads((base / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    facts = {f["id"]: f for f in json.loads((base / "plan" / "facts.json").read_text(encoding="utf-8"))}
    sentences = {lang: {squash(f[lang]): fid for fid, f in facts.items()} for lang in ("en", "el")}
    report = Counter()
    problems: list[str] = []
    invented: list[str] = []
    for doc in docs:
        path = base / "narrative" / f"{doc['id']}.txt"
        if not path.is_file():
            problems.append(f"{doc['id']}: no narrative")
            continue
        lang = doc["language"]
        found = parse_sections(path.read_text(encoding="utf-8"))
        report["documents"] += 1
        if len(found) != len(doc["sections"]):
            problems.append(f"{doc['id']}: {len(found)} sections written, {len(doc['sections'])} planned")
        for n, ((head, body), planned) in enumerate(zip(found, doc["sections"]), 1):
            report["sections"] += 1
            audience = head[1].split(":", 1)[1].strip() if len(head) > 1 and ":" in head[1] else ""
            if LABELS.get(audience) != planned["label"]:
                problems.append(f"{doc['id']} section {n}: audience {audience!r}, planned {planned['label']}")
            flat = squash(body)
            words = len(flat.split())
            report["words"] += words
            report["short sections"] += words < 80
            for fid in planned["facts"]:
                report["facts planned"] += 1
                if squash(facts[fid][lang]) in flat:
                    report["facts found"] += 1
                else:
                    problems.append(f"{doc['id']} section {n}: fact {fid} missing or altered")
            for sentence, fid in sentences[lang].items():
                if sentence in flat and fid not in planned["facts"]:
                    report["facts out of place"] += 1
                    problems.append(f"{doc['id']} section {n}: fact {fid} appears where it was not planted")
            if planned["canary"]:
                report["canaries planned"] += 1
                report["canaries found"] += planned["canary"] in body
            letters = [c for c in body if c.isalpha()]
            greek = sum("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿" for c in letters) / max(1, len(letters))
            if (lang == "el") != (greek > 0.5):
                problems.append(f"{doc['id']} section {n}: Greek share {greek:.0%} in a {lang} document")
            rest = flat
            for fid in planned["facts"]:
                rest = rest.replace(squash(facts[fid][lang]), " ")
            for match in POLICY_VALUE.finditer(rest):
                report["values outside the facts"] += 1
                invented.append(f"{doc['id']} section {n}: {match.group(0)}")
    full = base / "full"
    files = [p for p in full.glob("*") if not p.name.endswith(".meta.json")] if full.is_dir() else []
    report["files assembled"] = len(files)
    report["megabytes"] = round(sum(p.stat().st_size for p in files) / 2**20, 1)
    return {"counts": dict(report), "problems": problems, "invented_examples": invented[:15]}
