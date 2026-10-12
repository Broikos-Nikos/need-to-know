"""The tables of the corpus (KB3-K3, KB3-K5): named columns in every format, each department's own register, one
glossary per document, revisions in order, nothing dated after the last day, and label shares that meet the plan."""

import json
import random
import re
from collections import Counter
from datetime import date

import pytest

from needtoknow.corpus import assemble, mutate
from needtoknow.corpus.assemble import LAST_DAY, TITLES, budgets, register_rows, rows_per_unit, small_tables, units_of
from needtoknow.corpus.plan import DEPARTMENTS, LABEL_EN, SEED, header, make_catalogue, make_facts, world
from needtoknow.policy import load_labels

PROSE = {"en": "The team checks every case against the procedure and records the outcome before it moves on. " * 3,
         "el": "Η ομάδα ελέγχει κάθε υπόθεση με βάση τη διαδικασία και καταγράφει το αποτέλεσμα πριν προχωρήσει. " * 3}


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    """Three documents of three departments, one per format."""
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    docs = []
    for n, (owner, lang, fmt) in enumerate([("logistics", "en", "txt"), ("sales", "el", "md"), ("legal", "en", "html")], 1):
        doc = {"id": f"{owner[:3].upper()}_PRO_{n:03d}", "type": "PRO", "type_name": "Procedure", "owner": owner, "language": lang,
               "format": fmt, "target_mb": 0.05, "version": "2.3",
               "sections": [{"label": owner, "topic_en": "Scope", "topic_el": "Πεδίο", "facts": [], "canary": None}]}
        docs.append(doc)
        text = "\n".join(header(doc, 1, doc["sections"][0]) + ["", PROSE[lang], ""])
        (base / "narrative" / f"{doc['id']}.txt").write_bytes(text.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps(docs, ensure_ascii=False).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    report = assemble.assemble()
    assert report["written"] == 3
    return base, docs, report


def test_the_shares_of_units_meet_the_registered_shares_on_the_plan():
    rng = random.Random(SEED)
    w = world()
    docs = make_catalogue(rng, make_facts(rng, w))
    shares = {label: entry["target_share"] for label, entry in load_labels().items()}
    units = Counter(s["label"] for d in docs for s in d["sections"])
    for d in docs:
        for kind, audience, rows in small_tables(d, w):
            units[audience] += units_of(kind, d["language"], rows)
    by_id = {d["id"]: d for d in docs}
    for doc_id, rows in budgets(docs, shares, w).items():
        d = by_id[doc_id]
        # The units the budgeted rows will make, at the rate the units step cuts each table in its language.
        units[d["owner"]] += rows["register"] / rows_per_unit(d["owner"], d["language"])
        units["general"] += rows["directory"] / rows_per_unit("directory", d["language"])
    total = sum(units.values())
    for label, share in shares.items():
        assert abs(units[label] / total - share) < 0.005, (label, units[label] / total, share)


def test_every_table_names_its_columns_in_every_format(corpus):
    base, docs, _ = corpus
    for doc in docs:
        text = (base / "full" / f"{doc['id']}.{doc['format']}").read_bytes().decode("utf-8")
        kinds = [doc["owner"], "directory", "forms", "glossary", "revisions"]
        columns = [TITLES[k][3 if doc["language"] == "el" else 2] for k in kinds]
        for names in columns:
            if doc["format"] == "txt":
                assert f"\n{' | '.join(names)}\n" in text, names
            elif doc["format"] == "md":
                assert f"\n| {' | '.join(names)} |\n|" in text, names
            else:
                assert "<tr>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>" in text, names
        assert "| c1 |" not in text


def test_each_department_has_its_own_register_and_the_directory_and_glossary_are_for_everyone(corpus):
    base, docs, _ = corpus
    text = (base / "full" / f"{docs[0]['id']}.txt").read_bytes().decode("utf-8")
    audience = {m.group(1): m.group(2) for m in re.finditer(r"SECTION \d+  Annex [A-Z]: (.+)\nAudience: (.+)\n", text)}
    assert audience == {"Stock list": "Logistics", "Shop directory": "All staff", "Form definitions": "Logistics",
                        "Glossary": "All staff", "Revision history": "Logistics"}
    w = world()
    shapes = {re.sub(r"\d", "9", register_rows(owner, "en", random.Random(1), w, 1)[0].split(" | ")[0]) for owner in DEPARTMENTS}
    assert len(shapes) == len(DEPARTMENTS), shapes


def test_one_glossary_per_document_with_each_term_once(corpus):
    base, docs, _ = corpus
    for doc in docs:
        (glossary,) = [rows for kind, _a, rows in small_tables(doc, world()) if kind == "glossary"]
        terms = [row.split(" | ")[0] for row in glossary]
        assert 3 <= len(terms) == len(set(terms))
    text = (base / "full" / f"{docs[0]['id']}.txt").read_bytes().decode("utf-8")
    assert text.count("Annex") and text.count(": Glossary\n") == 1


def test_revisions_run_in_order_up_to_the_version_and_no_event_is_after_the_last_day():
    w = world()
    doc = {"id": "LOG_MAN_001", "owner": "logistics", "language": "en", "version": "3.4", "sections": []}
    (revisions,) = [rows for kind, _a, rows in small_tables(doc, w) if kind == "revisions"]
    versions = [row.split(" | ")[0] for row in revisions]
    dates = [date.fromisoformat(row.split(" | ")[1]) for row in revisions]
    assert versions[0] == "1.0" and versions[-1] == "3.4" and len(versions) == 25
    assert dates == sorted(dates) and len(set(dates)) == len(dates) and dates[-1] <= LAST_DAY
    for owner in DEPARTMENTS:
        for lang in ("en", "el"):
            for row in register_rows(owner, lang, random.Random(owner), w, 50):
                cells = row.split(" | ")
                when = [date.fromisoformat(c) for c in cells if re.fullmatch(r"\d{4}-\d\d-\d\d", c)]
                done = when[:1] if owner == "legal" else when  # a contract's renewal may lie ahead
                assert all(d <= LAST_DAY for d in done), (owner, row)
                assert len(cells) == len(TITLES[owner][2]), row


def test_a_directory_refresh_changes_extensions_alone_in_every_format(corpus):
    base, docs, _ = corpus
    files = mutate.corpus_files(base / "full")
    for doc in docs:
        entry, effects = mutate.plan("annex_refresh", doc["id"], files[doc["id"]], [], random.Random(7), base / "full", 1)
        assert entry["kind"] == "annex_refresh" and entry["rows_changed"] >= 1, doc["format"]
        file, _, meta = files[doc["id"]]
        before, after = mutate.read(file, meta), effects[0][2].decode("utf-8")
        assert len(before) == len(after) and before != after
        changed = [(a, b) for a, b in zip(re.split(r"(\d{5})", before), re.split(r"(\d{5})", after)) if a != b]
        assert changed and all(re.fullmatch(r"\d{5}", a) and re.fullmatch(r"\d{5}", b) for a, b in changed), changed[:3]


def test_the_report_counts_every_unit_it_wrote(corpus):
    _, docs, report = corpus
    prose = len(docs)
    assert sum(report["units"].values()) > prose and report["units"]["general"] > 0
    assert set(report["units"]) <= set(LABEL_EN)
