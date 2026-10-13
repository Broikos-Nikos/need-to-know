"""Another draw of the corpus (T5.5e, RS-F6): the base seed gives the corpus's own documents, and another seed the same
documents and prose with new values written where the plan planted them and new annexes, in a data folder of its own
whose plan agrees with itself (DR-F29)."""

import json
import sqlite3

import pytest

from needtoknow.corpus import draw as D
from needtoknow.corpus.assemble import assemble_doc, budgets, legacy_id
from needtoknow.corpus.plan import SEED, build, world
from needtoknow.eval.golden import Context
from needtoknow.eval.probes import gold_units
from needtoknow.policy import load_labels
from needtoknow.settings import data_dir, repo_root


@pytest.fixture
def plan(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    build()
    base = data_dir() / "corpus"
    return (json.loads((base / "plan" / "catalogue.json").read_text(encoding="utf-8")),
            json.loads((base / "plan" / "facts.json").read_text(encoding="utf-8")))


def test_another_seed_draws_new_values_for_single_facts_and_variants_and_keeps_a_versioned_pair(plan):
    _, facts = plan
    assert D.redraw(facts, SEED) == facts
    drawn = D.redraw(facts, 7)
    moved = [g for f, g in zip(facts, drawn) if f["values"] != g["values"]]
    assert moved and all(g["kind"] in ("single", "variant") for g in moved)
    assert all(f == g for f, g in zip(facts, drawn) if f["kind"] in ("versioned", "superseded"))
    for f, g in zip(facts, drawn):
        if f["values"] != g["values"]:
            assert g["values"][0] in g["en"] and g["values"][1] in g["el"], g["id"]


def test_the_base_seed_draws_the_corpus_own_documents_and_another_seed_new_values_and_annexes(plan):
    catalogue, facts = plan
    base, other = D.draw(SEED, 80), D.draw(7, 80)
    assert base["documents"] == other["documents"] == len(catalogue[::80]) and base["values"] == 0
    shares = {label: entry["target_share"] for label, entry in load_labels().items()}
    budget = budgets(catalogue, shares, world())
    drawn = {g["id"]: g for g in json.loads((D.draw_dir(7) / "corpus" / "plan" / "facts.json").read_text(
        encoding="utf-8"))}
    for doc in catalogue[::80]:
        name = f"{doc['id']}.{doc['format']}"
        size, _ = assemble_doc(doc, repo_root() / "narrative" / f"{doc['id']}.txt", world(), data_dir(),
                               legacy_id(catalogue), budget[doc["id"]])
        assert (D.draw_dir(SEED) / "corpus" / "full" / name).read_bytes() == (data_dir() / name).read_bytes()
        # The annexes are drawn again, not only the sentences: the table rows themselves differ (RS-F6).
        rows = [[line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if " | " in line]
                for path in (D.draw_dir(7) / "corpus" / "full" / name, data_dir() / name)]
        shared = set(rows[0]) & set(rows[1])
        assert rows[0] and rows[1] and len(shared) < len(rows[1]) / 2, (name, len(shared), len(rows[1]))
        lang = doc["language"]
        text = (D.draw_dir(7) / "corpus" / "narrative" / f"{doc['id']}.txt").read_text(encoding="utf-8")
        for f in facts:
            if doc["id"] in f["placements"]:
                assert drawn[f["id"]][lang] in text, f["id"]
                if drawn[f["id"]][lang] != f[lang]:
                    assert f[lang] not in text, f["id"]


def test_a_draw_counts_and_plans_only_what_its_documents_hold(plan):
    # DR-F29: the line counted new values over every fact of the plan beside sentences in its own documents, its plan
    # kept facts placed only in documents it lacked, on which the probe generator stopped, and --every took 0 and
    # negatives.
    catalogue, facts = plan
    report = D.draw(7, 40)
    folder = D.draw_dir(7) / "corpus" / "plan"
    docs = json.loads((folder / "catalogue.json").read_text(encoding="utf-8"))
    drawn = json.loads((folder / "facts.json").read_text(encoding="utf-8"))
    held = {doc["id"] for doc in catalogue[::40]}
    assert {doc["id"] for doc in docs} == held
    assert {g["id"] for g in drawn} == {f["id"] for f in facts if held & set(f["placements"])}
    assert all(g["placements"] and set(g["placements"]) <= held for g in drawn)
    plan_values = {f["id"]: f["values"] for f in facts}
    assert report["facts"] == len(drawn) < len(facts)
    assert report["values"] == sum(g["values"] != plan_values[g["id"]] for g in drawn)
    ctx = Context(facts=drawn, catalogue=docs, df=lambda term: 0)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE units (unit_id TEXT, doc_id TEXT, labels TEXT, text TEXT)")
    assert all(gold_units(g, ctx, db) == [] for g in drawn)  # no unit yet, and no document it cannot find
    for every in (0, -8):
        with pytest.raises(D.DrawError, match="at least 1"):
            D.draw(7, every)
