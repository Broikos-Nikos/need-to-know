"""Another draw of the corpus (T5.5e, RS-F6): the base seed gives the corpus's own documents, and another seed the same
documents and prose with new values written where the plan planted them and new annexes."""

import json

import pytest

from needtoknow.corpus import draw as D
from needtoknow.corpus.assemble import assemble_doc, budgets, legacy_id
from needtoknow.corpus.plan import SEED, build, world
from needtoknow.policy import load_labels
from needtoknow.settings import data_dir, repo_root


@pytest.fixture
def plan(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    build()
    base = data_dir() / "corpus" / "plan"
    return (json.loads((base / "catalogue.json").read_text(encoding="utf-8")),
            json.loads((base / "facts.json").read_text(encoding="utf-8")))


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
    corpus = data_dir() / "corpus"
    shares = {label: entry["target_share"] for label, entry in load_labels().items()}
    budget = budgets(catalogue, shares, world())
    drawn = {g["id"]: g for g in json.loads((corpus / "draw-7" / "plan" / "facts.json").read_text(encoding="utf-8"))}
    for doc in catalogue[::80]:
        name = f"{doc['id']}.{doc['format']}"
        size, _ = assemble_doc(doc, repo_root() / "narrative" / f"{doc['id']}.txt", world(), data_dir(),
                               legacy_id(catalogue), budget[doc["id"]])
        assert (corpus / f"draw-{SEED}" / name).read_bytes() == (data_dir() / name).read_bytes()
        # The annexes are drawn again, not only the sentences: the table rows themselves differ (RS-F6).
        rows = [[line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if " | " in line]
                for path in (corpus / "draw-7" / name, data_dir() / name)]
        shared = set(rows[0]) & set(rows[1])
        assert rows[0] and rows[1] and len(shared) < len(rows[1]) / 2, (name, len(shared), len(rows[1]))
        lang = doc["language"]
        text = (corpus / "draw-7" / "narrative" / f"{doc['id']}.txt").read_text(encoding="utf-8")
        for f in facts:
            if doc["id"] in f["placements"]:
                assert drawn[f["id"]][lang] in text, f["id"]
                if drawn[f["id"]][lang] != f[lang]:
                    assert f[lang] not in text, f["id"]
