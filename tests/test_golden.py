"""The golden items' construction checks (T5.1c2): each category's rules, the overlap cap in the analyser's terms, the
defaults rule, and the split's counts and shares, on a synthetic registry, split and policy."""

import copy
import json

import pytest

from needtoknow.eval import golden as G
from needtoknow.eval.protocol import load_protocol

POLICY = {"roles": {"sales": {"reads": ["general", "sales"]}, "legal": {"reads": ["general", "legal"]},
                    "admin": {"reads": ["general", "sales", "legal"]}}}
FACTS = [
    {"id": "F06V00", "family": "F06V00", "label": "sales", "kind": "single", "placements": ["D1"], "values": ["7", "7"],
     "en": "In shops the maximum discount is 7 percent.", "el": "Στα καταστήματα η μέγιστη έκπτωση είναι 7 τοις εκατό."},
    {"id": "F38V00", "family": "F38V00", "label": "legal", "kind": "single", "placements": ["D2"],
     "values": ["35000", "35000"], "en": "Liability is capped at 35000 euros.",
     "el": "Η ευθύνη περιορίζεται στα 35000 ευρώ."},
    {"id": "F09V00", "family": "F09V00", "label": "sales", "kind": "versioned", "placements": ["D1"],
     "values": ["22", "22"], "en": "From 1 July 2025, an upgrade comes after 22 months.",
     "el": "Από 1 Ιουλίου 2025, η αναβάθμιση έρχεται μετά από 22 μήνες."},
    {"id": "F09V00OLD", "family": "F09V00", "label": "sales", "kind": "superseded", "placements": ["D1"],
     "values": ["18", "18"], "en": "From 1 March 2024, an upgrade comes after 18 months.",
     "el": "Από 1 Μαρτίου 2024, η αναβάθμιση έρχεται μετά από 18 μήνες."},
    {"id": "F34V00", "family": "F34V00", "label": "legal", "kind": "single", "placements": ["D2"], "values": ["14", "14"],
     "en": "Distance contracts may be cancelled within 14 days.", "el": "Οι συμβάσεις από απόσταση ακυρώνονται σε 14 ημέρες."},
]
CATALOGUE = [{"id": "D1", "language": "en", "sections": [{"label": "sales", "facts": ["F06V00", "F09V00", "F09V00OLD"]}]},
             {"id": "D2", "language": "el", "sections": [{"label": "legal", "facts": ["F38V00", "F34V00"]}]}]
SPLIT = {"split": [{"id": "F06V00", "split": "test", "kind": "single", "label": "sales", "defaults": {},
                    "facts": ["F06V00"]},
                   {"id": "F38V00", "split": "test", "kind": "single", "label": "legal",
                    "defaults": {"F34V00": "the 14 days of withdrawal"}, "facts": ["F38V00", "F34V00"]},
                   {"id": "F09V00", "split": "dev", "kind": "versioned", "label": "sales", "defaults": {},
                    "facts": ["F09V00", "F09V00OLD"]}]}
RARE = {"maximum": 1, "percent": 2, "discount": 3, "shop": 9}  # every other term common


@pytest.fixture
def ctx():
    df = lambda term: RARE.get(term, 1000)  # noqa: E731
    return G.Context(protocol=load_protocol(), split=SPLIT, facts=FACTS, catalogue=CATALOGUE, policy=POLICY, df=df)


def item(**fields):
    base = {"id": "T01", "category": "A", "families": ["F06V00"], "role": "sales", "lang": "en",
            "question": "How far may a sales advisor cut a price without approval?",
            "expect": {"status": "answered", "facts": [{"fact_id": "F06V00", "forms": ["7", "7 percent"]}]}}
    return {**base, **fields}


def test_a_well_formed_item_keeps_every_rule(ctx):
    assert G.check_item(item(), "test", ctx) == []
    # A form holds the planted value in its context; no bare number need stand alone among them.
    in_context = item(expect={"status": "answered", "facts": [{"fact_id": "F06V00", "forms": ["7 percent"]}]})
    assert G.check_item(in_context, "test", ctx) == []
    assert "" not in G.rare_terms("Ο ρόλος εγκρίνει έως 100 ευρώ.", lambda term: 0 if term == "" else 5, 3)


@pytest.mark.parametrize("fields, words", [
    ({"expect": {"status": "not_found", "facts": [{"fact_id": "F06V00", "forms": ["7"]}]}}, "expects answered"),
    ({"lang": "el"}, "not the question's script"),
    ({"families": ["F09V00"], "expect": {"status": "answered", "facts": [{"fact_id": "F09V00", "forms": ["22"]}],
                                         "superseded": []}}, "is in dev, not test"),
    ({"role": "legal"}, "may read no section F06V00 is planted in"),
    ({"expect": {"status": "answered", "facts": [{"fact_id": "F06V00", "forms": ["seven"]}]}}, "holds either of its planted"),
    ({"question": "What is the maximum in percent?"}, "shares 2 of the rarest terms"),
    ({"category": "F"}, "asks facts under two labels"),
    ({"category": "C"}, "another language than every document"),
    ({"lang": "el", "question": "Πόση είναι η έκπτωση;"}, "an A item asks in English"),
    ({"category": "B", "lang": "el", "question": "Πόση είναι η έκπτωση;"}, "a B item asks in the language of a document"),
    ({"category": "D"}, "a D item asks F06V00 from a file that also holds a section sales may not read"),
    ({"category": "M", "families": ["F06V00", "F38V00"], "role": "admin",
      "expect": {"status": "answered", "facts": [{"fact_id": "F06V00", "forms": ["7"]}, {"fact_id": "F09V00", "forms": ["22"]}]}},
     "two facts that no one document holds together"),
])
def test_each_category_rule_names_what_it_breaks(ctx, fields, words):
    problems = G.check_item(item(**fields), "test", ctx)
    assert any(words in problem for problem in problems), problems


def test_a_no_access_item_asks_through_forbidden_where_its_role_cannot_read(ctx):
    e = item(id="T02", category="E", families=["F38V00"], role="sales",
             expect={"status": "not_found", "facts": [], "forbidden": [{"fact_id": "F38V00", "forms": ["35000"]}]})
    assert G.check_item(e, "test", ctx) == []
    readable = {**e, "role": "legal"}
    assert any("so it is no E item" in p for p in G.check_item(readable, "test", ctx))


def test_a_deliberate_default_is_asked_only_by_a_no_access_item(ctx):
    # RS-F19: F34V00 is the 14 days of withdrawal a model may know without the documents.
    answered = item(id="T03", families=["F38V00"], role="legal", lang="el", question="Σε πόσες ημέρες ακυρώνεται;",
                    expect={"status": "answered", "facts": [{"fact_id": "F34V00", "forms": ["14"]}]})
    assert any("deliberate default" in p for p in G.check_item(answered, "test", ctx))
    refused = {**answered, "category": "E", "role": "sales",
               "expect": {"status": "not_found", "forbidden": [{"fact_id": "F34V00", "forms": ["14"]}]}}
    assert not any("deliberate default" in p for p in G.check_item(refused, "test", ctx))


def test_a_superseded_item_names_both_values_and_own_unit_items_keep_theirs(ctx):
    g = item(id="D01", category="G", families=["F09V00"],
             expect={"status": "answered", "facts": [{"fact_id": "F09V00", "forms": ["22"]}], "superseded": []})
    assert any("names the superseded fact" in p for p in G.check_item(g, "dev", ctx))
    g["expect"]["superseded"] = ["F09V00OLD"]
    assert G.check_item(g, "dev", ctx) == []
    table = item(id="D02", category="I", families=[], expect={"status": "answered", "facts": [{"forms": ["E08"]}]})
    assert any("names its row" in p for p in G.check_item(table, "dev", ctx))
    absent = item(id="D03", category="J", families=[], expect={"status": "not_found", "facts": [{"fact_id": "F06V00"}]})
    assert any("expects no fact" in p for p in G.check_item(absent, "dev", ctx))


def test_a_superseded_fact_an_e_items_facts_and_a_near_misss_neighbour_are_held_to_their_category(ctx):
    # DR-F22: each was left to the writer.
    ctx.facts["F06V00OLD"] = {**ctx.facts["F06V00"], "id": "F06V00OLD", "kind": "superseded", "values": ["5", "5"]}
    ctx.docs["D1"] = copy.deepcopy(ctx.docs["D1"])  # the module's catalogue stays as every other test reads it
    ctx.docs["D1"]["sections"][0]["facts"].append("F06V00OLD")
    g = item(id="D01", category="G", families=["F09V00"],
             expect={"status": "answered", "facts": [{"fact_id": "F09V00", "forms": ["22"]}], "superseded": ["F09V00OLD"]})
    assert G.check_item(g, "dev", ctx) == []
    g["expect"]["superseded"] = ["F06V00OLD"]  # superseded, but another fact's former value
    assert any("superseded fact is a former value of the fact it asks" in p for p in G.check_item(g, "dev", ctx))
    e = item(id="T02", category="E", families=["F38V00"], role="sales",
             expect={"status": "not_found", "facts": [{"fact_id": "F38V00", "forms": ["35000"]}],
                     "forbidden": [{"fact_id": "F38V00", "forms": ["35000"]}]})
    assert any("an E item expects no fact" in p for p in G.check_item(e, "test", ctx))
    near = item(id="D02", category="N", families=[], role="sales", near="F09V00",
                expect={"status": "not_found", "facts": []})
    assert G.check_item(near, "dev", ctx) == []
    assert any("neighbour is a fact legal may read" in p for p in G.check_item({**near, "role": "legal"}, "dev", ctx))
    # Beside a test fact a dev near miss would draw that fact's section into development; a test one may sit anywhere.
    assert any("a dev near miss's neighbour is a fact of a dev family" in p
               for p in G.check_item({**near, "near": "F06V00"}, "dev", ctx))
    assert G.check_item({**near, "id": "T03", "near": "F06V00"}, "test", ctx) == []


def test_the_split_must_hold_the_registered_counts_and_ask_each_test_family_once(ctx):
    protocol = copy.deepcopy(ctx.protocol)
    protocol["items"]["test"] = {code: 0 for code in "ABCDEFGHMIJN"} | {"A": 2}
    ctx.protocol = protocol
    twice = [item(id="T01"), item(id="T02")]
    problems = G.check_split(twice, "test", ctx)
    assert any("family F06V00 asked 2 times" in p for p in problems)
    assert any("family F38V00 asked 0 times" in p for p in problems)
    assert any("Greek is 0.000 of the items" in p for p in problems)
    dev = [item(id="D01", families=["F09V00"])]
    assert any("family F09V00 asked 1 times, not two or three" in p for p in G.check_split(dev, "dev", ctx))


def test_a_dev_family_of_general_defaults_carries_no_item_and_may_not_be_asked(ctx):
    # N58: only a refusal may ask a default and no role is refused general, so no item can ask such a family.
    ctx.facts["F91V00"] = {"id": "F91V00", "family": "F91V00", "label": "general", "kind": "single",
                           "placements": ["D3"], "values": ["1", "1"], "en": "Badges are renewed once a year.",
                           "el": "Οι κάρτες ανανεώνονται μία φορά τον χρόνο."}
    ctx.docs["D3"] = {"id": "D3", "language": "en", "sections": [{"label": "general", "facts": ["F91V00"]}]}
    ctx.families["F91V00"] = {"id": "F91V00", "split": "dev", "kind": "single", "label": "general",
                              "defaults": {"F91V00": "a yearly renewal"}, "facts": ["F91V00"]}
    dev = [item(id="D01", families=["F09V00"]), item(id="D02", families=["F09V00"])]
    assert not any("F91V00" in p for p in G.check_split(dev, "dev", ctx))
    asked = dev + [item(id="D03", families=["F91V00"])]
    assert any("family F91V00 is exempt" in p for p in G.check_split(asked, "dev", ctx))
    ctx.docs["D3"]["sections"][0]["label"] = "sales"  # a default some role is refused is asked by a refusal
    assert any("family F91V00 asked 0 times" in p for p in G.check_split(dev, "dev", ctx))
    ctx.docs["D3"]["sections"][0]["label"] = "general"
    ctx.families["F91V00"]["defaults"] = {}  # a general fact no model knows is asked like any other
    assert any("family F91V00 asked 0 times" in p for p in G.check_split(dev, "dev", ctx))


def test_a_table_items_row_is_looked_up_in_the_manifest(ctx):
    # DR-F20: the row was taken on the writer's word.
    head = "Order | Shop | Status\n"
    ctx.units = [{"doc_id": "D1", "layer": "table", "labels": ["sales"], "text": head + "O1001 | S1 | shipped\nO1002 | S2 | held"},
                 {"doc_id": "D2", "layer": "table", "labels": ["legal"], "text": head + "O2001 | S1 | held"},
                 {"doc_id": "D2", "layer": "table", "labels": ["legal"], "text": head + "O2001 | S9 | held"}]
    table = item(id="D01", category="I", families=[], question="What is the status of order O1001?",
                 expect={"status": "answered", "facts": [{"forms": ["shipped"]}]},
                 row={"doc_id": "D1", "key": "O1001", "column": "Status", "value": "shipped"})
    assert G.check_item(table, "dev", ctx) == []
    for row, words in [({"value": "held"}, "holds 'shipped' under 'Status', not 'held'"),
                       ({"key": "O2001", "doc_id": "D2", "value": "held"}, "is in 2 units"),
                       ({"key": "O9999"}, "is in 0 units"),
                       ({"doc_id": "D2"}, "not in one unit of D2 alone")]:
        problems = G.check_item({**table, "row": {**table["row"], **row}}, "dev", ctx)
        assert any(words in p for p in problems), (row, problems)
    assert any("legal may read no label of the table" in p for p in G.check_item({**table, "role": "legal"}, "dev", ctx))
    other = {**table, "expect": {"status": "answered", "facts": [{"forms": ["dispatched"]}]}}
    assert any("no form holds the row's value" in p for p in G.check_item(other, "dev", ctx))
    ctx.units[0]["layer"] = "prose"
    assert any("is no table" in p for p in G.check_item(table, "dev", ctx))


def test_an_absent_fact_names_its_probes_and_the_corpus_prose_holds_none(ctx):
    # MT-F11: the absence was checked once, by a script outside the repository.
    ctx.units = [{"doc_id": "D1", "layer": "prose", "labels": ["sales"], "text": "Students get no special terms."},
                 {"doc_id": "D2", "layer": "table", "labels": ["legal"], "text": "Key | Value\nK1 | bereavement"}]
    absent = item(id="D05", category="J", families=[], question="How many days of bereavement leave are there?",
                  expect={"status": "not_found", "facts": []})
    loaded = {"test": [], "dev": [absent]}
    assert G.absent_problems(loaded, ctx, {"D05": ["(?i)bereavement"]}) == []  # a table's cell answers no question
    assert any("names the probes its absence rests on" in p for p in G.absent_problems(loaded, ctx, {}))
    assert any("the corpus prose answers it: '(?i)student' matches in D1" in p
               for p in G.absent_problems(loaded, ctx, {"D05": ["(?i)student"]}))
    assert any("is no regular expression" in p for p in G.absent_problems(loaded, ctx, {"D05": ["(unclosed"]}))
    assert any("names D99, no absent fact or near miss" in p
               for p in G.absent_problems(loaded, ctx, {"D05": ["(?i)bereavement"], "D99": ["x"]}))


def test_a_twin_is_its_original_reworded_away_from_the_planted_sentence(ctx):
    # N59 and KB3-K4: twenty dev items get a twin, the gap between the two to be reported.
    ctx.protocol = copy.deepcopy(ctx.protocol)
    ctx.protocol["items"]["paraphrase_twins"] = 1
    original = item(id="D01", category="A", families=["F06V00"])
    twin = {**original, "id": "D02", "question": "What reduction can someone selling in a store agree to alone?",
            "twin": "D01"}
    problems = lambda *dev, test=(): G.twin_problems({"dev": list(dev), "test": list(test)}, ctx)  # noqa: E731
    assert problems(original, twin) == []
    assert any("keeps its original's role" in p for p in problems(original, {**twin, "role": "admin"}))
    assert any("more than half" in p for p in problems(original, {**twin, "question": original["question"]}))
    assert any("of the rarest terms of F06V00, more than its original's []" in p
               for p in problems(original, {**twin, "question": "What is the maximum shop discount?"}))
    assert any("D01 has 2 twins" in p for p in problems(original, twin, {**twin, "id": "D03"}))
    assert any("is a dev item that is no twin itself" in p for p in problems(original, twin, {**twin, "id": "D03",
                                                                                               "twin": "D02"}))
    assert any("a twin is a dev item" in p for p in problems(original, twin, test=[{**twin, "id": "T05"}]))
    ctx.protocol["items"]["paraphrase_twins"] = 20
    assert any("dev: 1 twins, registered 20" in p for p in problems(original, twin))


def test_the_dev_twins_pair_the_items_the_seed_draws():
    # N59: the items given a twin are drawn by a committed seed, in the test's category shares; this needs no build.
    dev, test = G.load_items(G.golden_path("dev")), G.load_items(G.golden_path("test"))
    assert [twin["twin"] for twin in dev if twin.get("twin")] == G.draw_twins(dev, test, 20)


def test_the_seed_questions_are_ten_from_public_documents_with_the_corpus_beside_each():
    # KB1-F11: ten questions written from the domain's public documents anchor a set a model's prose surrounds.
    from needtoknow.settings import repo_root
    text = (repo_root() / "eval" / "golden" / "seeds.md").read_text(encoding="utf-8")
    rows = [line.split("|")[1:-1] for line in text.splitlines()
            if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert [int(cells[0]) for cells in rows] == list(range(1, 11))
    assert all(len(cells) == 5 and all(cell.strip() for cell in cells) for cells in rows)
    assert sum(G.language(cells[1]) == "el" for cells in rows) == 5  # as many in Greek as in English
    assert all("Directive" in cells[2] or "procedure" in cells[2] for cells in rows)  # each names its document


def test_the_dev_table_items_are_the_rows_the_seed_draws():
    # MT-F11: the draw ran from a script outside the repository; ntk eval rows repeats it.
    from needtoknow.ingest.manifest import manifest_path
    if not manifest_path().is_file():
        pytest.skip("no manifest: ntk ingest build makes it")
    rows = [item["row"] for item in G.load_items(G.golden_path("dev")) if item["category"] == "I"]
    assert [{k: row[k] for k in ("doc_id", "key", "column", "value")} for row in G.draw_rows()] == rows


def test_a_fact_planted_in_two_sections_of_one_document_is_seen_in_both(ctx):
    # DR-F21: only the first section a document plants a fact in was read.
    ctx.docs["D3"] = {"id": "D3", "language": "en", "sections": [{"label": "legal", "facts": ["F92V00"]},
                                                                  {"label": "sales", "facts": ["F92V00"]}]}
    ctx.facts["F92V00"] = {"id": "F92V00", "family": "F92V00", "label": "legal", "kind": "single",
                           "placements": ["D3", "D3"], "values": ["9", "9"], "en": "Notice is given in 9 days.",
                           "el": "Η ειδοποίηση δίνεται σε 9 ημέρες."}
    ctx.families["F92V00"] = {"id": "F92V00", "split": "test", "kind": "single", "label": "legal", "defaults": {},
                              "facts": ["F92V00"]}
    assert [section["label"] for _, section in ctx.sections("F92V00")] == ["legal", "sales"]
    refused = item(id="T02", category="E", families=["F92V00"], role="sales", question="How much notice is given?",
                   expect={"status": "not_found", "facts": [], "forbidden": [{"fact_id": "F92V00", "forms": ["9 days"]}]})
    assert any("sales may read a section F92V00 is planted in" in p for p in G.check_item(refused, "test", ctx))


def test_the_command_reads_the_golden_files_and_names_a_bad_line(tmp_path, monkeypatch, ctx):
    (tmp_path / "test.jsonl").write_text(json.dumps(item()) + "\n{not json\n", encoding="utf-8")
    monkeypatch.setattr(G, "golden_path", lambda split: tmp_path / f"{split}.jsonl")
    with pytest.raises(G.GoldenError, match="test.jsonl line 2 is not JSON"):
        G.check(ctx)


def test_the_committed_items_keep_their_rules():
    # Needs the plan, the frequency table and the manifest, which a build makes; a bare clone skips it.
    from needtoknow.eval.split import SplitError, load_plan
    from needtoknow.index.frequencies import table_path
    from needtoknow.ingest.manifest import manifest_path
    try:
        load_plan()
    except SplitError:
        pytest.skip("no plan: ntk corpus plan makes it")
    if not table_path().is_file():
        pytest.skip("no frequency table: ntk index sync makes it")
    if not manifest_path().is_file():
        pytest.skip("no manifest: ntk ingest build makes it")
    summary, problems = G.check()
    assert summary["test"]["items"] == 35
    assert problems == []  # every item, every count and every share, in both splits
    # MA-F14: the numbers eval/golden/README.md gives, as ntk eval check prints them.
    assert (summary["test"]["greek"], summary["test"]["family_items"], summary["dev"]["exempt"]) ==         (16, 29, ["F01V00", "F03V00"])


def test_a_line_that_is_json_but_no_item_and_a_missing_split_are_refused_by_name(tmp_path, monkeypatch, ctx):
    # HS-F16: both reached the command as a traceback.
    (tmp_path / "test.jsonl").write_text(json.dumps(item()) + "\n[1, 2]\n", encoding="utf-8")
    monkeypatch.setattr(G, "golden_path", lambda split: tmp_path / f"{split}.jsonl")
    with pytest.raises(G.GoldenError, match="test.jsonl line 2 is JSON but not an item"):
        G.check(ctx)
    monkeypatch.setattr(G, "split_path", lambda: tmp_path / "families.json")
    with pytest.raises(G.GoldenError, match="no families.json: ntk eval split writes it"):
        G.Context(protocol=ctx.protocol, facts=FACTS, catalogue=CATALOGUE, policy=POLICY, df=ctx.df)


def test_the_writers_view_blanks_the_value_and_the_rare_words_and_names_the_stems(ctx):
    # RS-F21: items are drafted without the words that decide retrieval in view.
    assert G.masked("From 1 March 2024, an upgrade comes after 24 months.", ["24"], ["march", "upgrad"]) == \
        "From 1 ___ 2024, an ___ comes after [value] months."  # a value of 24 leaves the year whole
    view = "\n".join(G.draft_view("test", ctx))
    assert "In shops the ___ ___ is [value] ___" in view and "[rare: maximum, percent, discount;" in view
    assert "maximum discount" not in view and "F09V00" not in view  # a dev family stays out of the test's view


def test_the_units_holding_a_key_are_read_once_a_manifest_state_and_again_after_a_scan(monkeypatch, tmp_path):
    # PF-F9: each key's units were found by reading every unit's text, at every check and draw: seconds a key, the
    # suite's slowest test. They are kept beside the manifest under its last run, and read again only after it moves.
    import sqlite3

    from needtoknow.ingest.manifest import manifest_path, opened
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    with opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC', 'p', 1, 1, 1, 1, 'h', 'txt', 'utf-8', 'en', '[]', 1)")
        db.execute("INSERT INTO runs (run_id, kind, started, finished) VALUES (1, 'build', 't', 't1')")
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'k', 'table', '[\"sales\"]', '[]', ?, 0, 1, 'en', 1)",
                       [("u1", "K-1 | a"), ("u2", "K-1 | b | K-2")])
    reader = sqlite3.connect(f"file:{manifest_path().as_posix()}?mode=ro", uri=True)
    scans = []
    reader.set_trace_callback(lambda sql: scans.append(sql) if "instr(text" in sql else None)
    first = G.Holding(reader)
    assert [row[1] for row in first("K-1")] == ["u1", "u2"] and [row[1] for row in first("K-2")] == ["u2"]
    first.save()
    assert len(G.Holding(reader)("K-1")) == 2 and len(scans) == 2  # the second check reads no text
    with opened() as db:
        db.execute("INSERT INTO runs (run_id, kind, started, finished) VALUES (2, 'scan', 't', 't2')")
    assert len(G.Holding(reader)("K-1")) == 2 and len(scans) == 3  # a scan since: read again
    reader.close()


def test_the_units_holding_a_key_are_read_again_when_the_query_that_found_them_changes(monkeypatch, tmp_path):
    # MT-F24: the kept rows were keyed by the manifest and its last run alone, so a changed query was served the old
    # query's rows, and a changed column list failed as an unpacking error far from its cause.
    import sqlite3

    from needtoknow.ingest.manifest import manifest_path, opened
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    with opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC', 'p', 1, 1, 1, 1, 'h', 'txt', 'utf-8', 'en', '[]', 1)")
        db.execute("INSERT INTO runs (run_id, kind, started, finished) VALUES (1, 'build', 't', 't1')")
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'k', 'table', '[\"sales\"]', '[]', ?, 0, 1, 'en', 1)",
                       [("u1", "K-1 | a"), ("u2", "K-1 | b")])
    reader = sqlite3.connect(f"file:{manifest_path().as_posix()}?mode=ro", uri=True)
    kept = G.Holding(reader)
    assert [row[1] for row in kept("K-1")] == ["u1", "u2"]
    kept.save()
    monkeypatch.setattr(G.Holding, "QUERY", G.Holding.QUERY.replace("ORDER BY rowid", "ORDER BY rowid DESC"))
    assert [row[1] for row in G.Holding(reader)("K-1")] == ["u2", "u1"]  # the new query's rows, not the kept ones
    reader.close()
