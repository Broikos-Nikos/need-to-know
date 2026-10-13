"""The benches run anywhere and still print what the history quotes them for (MA-F1)."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from needtoknow.corpus.plan import build

BENCH = Path(__file__).resolve().parents[1] / "bench"


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(BENCH / script), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def test_a_change_bench_reports_the_spread_of_its_runs_and_the_largest_decides():
    # RS-F13: N53 quoted one run's 4.6 s; three runs later took 4.0 to 4.5 s.
    spec = importlib.util.spec_from_file_location("watched", BENCH / "watched.py")
    watched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watched)
    assert watched.spread([4.0, 4.1, 5.5]) == "median 4.1 s, largest 5.5 s, over 3 runs"
    assert watched.past([4.0, 5.2, 4.1]) and not watched.past([4.9, 4.9])


def test_every_kill_point_of_the_simulator_ends_equal_to_the_crash_free_state():
    result = run("crash_points.py")
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
    total = int(re.search(r"kill points (\d+):", result.stdout).group(1))
    assert total > 0 and f"equal to the crash free state {total} of {total}" in result.stdout, result.stdout


def test_the_operator_probes_run_on_a_plan_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    build()
    result = run("operator_probes.py", "--corpus", str(tmp_path / "corpus"), "--synthetic")
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
    for line in ["P3 two assemblies: identical bytes True in 48 files",
                 "P1 a build killed after 15 of 24 documents: 0 files missing; problems naming a missing file 0",
                 "P1c 9 files deleted by hand, 1 carrying a fact: the check names 9",
                 "P2 the journal line made to fail: files changed 0; journal lines 0"]:
        assert line in result.stdout, result.stdout


def test_the_null_refuses_everything_and_the_oracle_answers_every_item_with_its_own_form():
    # RS-F7: the floor and the ceiling every figure is read against, run through the scorer itself.
    import types

    from needtoknow.eval import score as S
    spec = importlib.util.spec_from_file_location("null_runs", BENCH / "null_runs.py")
    runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runs)
    ctx = types.SimpleNamespace(facts={"F06V00": {"id": "F06V00", "family": "F06V00", "values": ["7", "7"]}})
    sources = S.Sources(ctx, given={"F06V00": {("D1", "Policy > SECTION 2")}})
    items = [{"id": "D01", "category": "A", "lang": "en", "expect": {"facts": [{"fact_id": "F06V00", "forms": ["7%"]}]}},
             {"id": "D02", "category": "J", "lang": "el", "expect": {"facts": []}}]
    floor = S.score(items, runs.null(items), ctx, sources)[1]
    assert (floor["answerable"]["correct"], floor["refusals"]["correct"], floor["overall"]["rate"]) == (0, 1, 0.5)
    best = runs.oracle(items, sources)
    assert best[0]["citations"] == [{"doc_id": "D1", "section": "Policy > SECTION 2"}]
    ceiling = S.score(items, best, ctx, sources)[1]
    assert (ceiling["overall"]["rate"], ceiling["citation_precision"], ceiling["citation_recall"]) == (1.0, 1.0, 1.0)


def test_the_protocols_smallest_visible_effect_is_the_figure_the_power_bench_prints():
    # RS-F18, N60: 0.12 at the assumed discordance of 0.2, on the dev units as they are.
    from needtoknow.eval.golden import golden_path, load_items
    from needtoknow.eval.protocol import load_protocol
    from needtoknow.eval.stats import SEED
    spec = importlib.util.spec_from_file_location("power", BENCH / "power.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    protocol = load_protocol()
    sizes = bench.unit_sizes(load_items(golden_path("dev")))
    assert "0.2" in protocol["decision"]["smallest_visible_assumes"]
    figure = bench.smallest_visible(sizes, 0.2, protocol["decision"]["alpha"], 2000, 2000, SEED)
    assert figure == protocol["decision"]["smallest_visible_effect"]


def test_the_injection_bench_judges_english_by_its_words_and_not_by_its_letters_alone():
    # KB5-K15: any answer in unaccented Latin letters passed as English.
    spec = importlib.util.spec_from_file_location("injection", BENCH / "injection.py")
    injection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(injection)
    assert injection.in_language("Within 30 days of purchase, with the receipt.", "en")
    assert injection.in_language("30 days.", "en")  # too short to judge by words: the letters alone
    assert not injection.in_language("El cliente puede devolver el dispositivo dentro de 30 dias.", "en")
    assert not injection.in_language("Asiakas voi palauttaa laitteen 30 päivän kuluessa.", "en")
    assert injection.in_language("Μέσα σε 30 ημέρες.", "el")


def test_the_golden_readme_quotes_the_floor_the_null_run_scores():
    # Tick 127: RS-F27 moved dev's figures to the originals and the README kept 35 of 153 for a tick; the floor it
    # quotes is now held to the scorer's. Needs the plan and the manifest; a bare clone skips it.
    from needtoknow.eval.golden import Context, golden_path, load_items
    from needtoknow.eval.score import Sources, score
    from needtoknow.eval.split import SplitError, load_plan
    from needtoknow.ingest.manifest import manifest_path
    from needtoknow.settings import repo_root
    try:
        load_plan()
    except SplitError:
        pytest.skip("no plan: ntk corpus plan makes it")
    if not manifest_path().is_file():
        pytest.skip("no manifest: ntk ingest build makes it")
    spec = importlib.util.spec_from_file_location("null_runs", BENCH / "null_runs.py")
    runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runs)
    ctx = Context()
    text = " ".join((repo_root() / "eval" / "golden" / "README.md").read_text(encoding="utf-8").split())
    for split, phrase in (("dev", "of dev's {items} originals, {rate:.3f}"), ("test", "of test's {items}, {rate:.3f}")):
        items = load_items(golden_path(split))
        overall = score(items, runs.null(items), ctx, Sources(ctx))[1]["overall"]
        assert f"{overall['correct']} " + phrase.format(items=overall["items"], rate=overall["rate"]) in text, split



def test_the_probe_recall_bench_counts_ranks_and_the_shown_excerpts_as_it_says():
    # RS-F5: recall and MRR over probes, a probe with no gold unit among the hits counted as a miss.
    spec = importlib.util.spec_from_file_location("probe_recall", BENCH / "probe_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    rows = [{"rank": 1, "shown": True}, {"rank": 3, "shown": True}, {"rank": None, "shown": False},
            {"rank": 12, "shown": False}]
    assert recall.summarise(rows) == ("4 probes; recall@1 0.250, @5 0.500, @10 0.500; MRR 0.354; "
                                      "in the shown excerpts 0.500")
    hits = [{"unit_id": "u9"}, {"unit_id": "u2"}, {"unit_id": "u1"}]
    layer = {"u9": "table", "u2": "prose", "u1": "prose", "u8": "table"}
    found = recall.measure({"question": "q", "gold": ["u1", "u5"]}, "sales", None, lambda c, q, r: hits,
                           lambda h: [{"units": ["u9", "u8", "u1"]}], lambda units: layer)
    assert found == {"rank": 3, "shown": True, "top": 3, "tables": 1, "shown_units": 3, "shown_tables": 2, "aside": 2}
    # Among the hits but cut from the excerpts the prompt shows: found, not shown.
    cut = recall.measure({"question": "q", "gold": ["u1"]}, "sales", None, lambda c, q, r: hits,
                         lambda h: [{"units": ["u9", "u2"]}], lambda units: layer)
    assert (cut["rank"], cut["shown"]) == (3, False)


def test_the_recall_bench_says_what_the_table_units_take_and_what_recall_is_without_them():
    # RS-F28: N43 reverses if the annexes take a real share of a prose question's top 10, which the filter bench
    # measured on the dense side alone; this bench measures it through the whole search, as ask searches.
    spec = importlib.util.spec_from_file_location("probe_recall", BENCH / "probe_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    hits = [{"unit_id": u} for u in ("t1", "t2", "p1", "g1", "p2", "t3", "p3", "p4", "p5", "p6", "t4")]
    layer = {u["unit_id"]: "table" if u["unit_id"].startswith("t") else "prose" for u in hits}
    row = recall.measure({"question": "q", "gold": ["g1"]}, "sales", None, lambda c, q, r: hits,
                         lambda h: [{"units": ["t1", "t2"]}, {"units": ["g1"]}], lambda units: layer)
    assert (row["rank"], row["aside"], row["tables"], row["top"]) == (4, 2, 3, 10)
    missed = {"rank": None, "shown": False, "top": 10, "tables": 0, "shown_units": 1, "shown_tables": 0, "aside": None}
    assert recall.annex([row, missed]) == ("table units 15.0% of the top 10 and 50.0% of the shown excerpts' units; "
                                           "set aside, recall@1 0.000 and @10 0.500")
    # A table unit that is gold stays in the hits set aside.
    gold_table = recall.measure({"question": "q", "gold": ["t2"]}, "sales", None, lambda c, q, r: hits,
                                lambda h: [{"units": ["t2"]}], lambda units: layer)
    assert (gold_table["rank"], gold_table["aside"]) == (2, 1)


def test_n51_and_n43_quote_the_filter_benchs_saved_runs():
    # Tick 129, RS-F16: the decisions quote the filter bench on units' first 30 words and on the dev probes'
    # questions; each figure from its saved numbers is held to them. A machine without both runs skips.
    import json

    from needtoknow.settings import data_dir, repo_root
    runs = []
    for name in ("filter_bench.json", "filter_bench_questions.json"):
        path = data_dir() / "bench" / name
        if not path.is_file():
            pytest.skip(f"no {name} here: bench/filter_bench.py makes it")
        runs.append(json.loads(path.read_bytes()))
    text = " ".join((repo_root() / "docs" / "DECISIONS.md").read_text(encoding="utf-8").split())

    def section(number: str) -> str:
        start = text.index(f"## {number}. ")
        return text[start:text.index(" ## ", start)]

    def mean(rows: list[dict], key: str = "tie_aware") -> float:
        return sum(r[key] for r in rows) / len(rows)

    def figures(results: list[dict]) -> dict:
        rows = [r for r in results if r["form"] == "labels"]
        cases = {r["case"]: r for r in rows}

        def first(case: str) -> int | None:
            for ef in (64, 128, 256, 512):
                at = [r for r in rows if r["case"] == case and r["split"] == "dev" and r["ef"] == ef]
                if all(mean([r for r in at if r["class"] == c]) >= 0.97 for c in {r["class"] for r in at}):
                    return ef
            return None

        def median(case: str) -> str:
            took = sorted(r["seconds"] for r in rows if r["case"] == case and r["split"] == "test" and r["ef"] is None)
            return f"{took[len(took) // 2] * 1000:.0f}"

        roles = [c for c, r in cases.items() if not r["bench_only"]]
        graph = {c: mean([r for r in rows if r["case"] == c and r["ef"] == 512 and r["split"] == "dev"]) for c in cases}
        medians = sorted(int(median(c)) for c in roles if c != "admin")
        busy = sorted(int(f"{cases[c]['busy']:.0%}"[:-1]) for c in roles)
        prose = [r for r in results if r["kind"] in ("prose", "question") and r["class"] == "in domain"
                 and r["ef"] is None]
        assert all(r["tie_aware"] == 1.0 for r in results if r["ef"] is None)
        assert sum(r["foreign"] for r in results) == 0
        return {"first": {c: first(c) for c in cases}, "graph": graph, "roles": roles, "searches": len(results),
                "range": f"{min(graph[c] for c in roles):.3f} to {max(graph[c] for c in roles):.3f}",
                "median": f"{medians[0]} to {medians[-1]} ms a search and {median('admin')} for admin",
                "busy": f"{busy[0]} to {busy[-1]} percent busy",
                "annex": f"{100 * mean(prose, 'tables') / 10:.1f} percent"}

    units, questions = (figures(results) for results in runs)
    n51, n43 = section("N51"), section("N43")
    assert len(questions["roles"]) == 9 and not any(questions["first"][c] for c in questions["roles"])
    for phrase in (f"ran from {units['range']}", f"median of {units['median']}", f"machine {units['busy']}",
                   f"ran from {questions['range']} over the nine roles",
                   f"admin's fell from {units['graph']['admin']:.3f} to {questions['graph']['admin']:.3f}",
                   f"hr alone still reached it at ef {questions['first']['hr alone']}",
                   f"which reached it at ef {units['first']['outside counsel']} on unit prefixes, ended at "
                   f"{questions['graph']['outside counsel']:.3f}",
                   f"in {questions['searches']:,} searches", f"median of {questions['median']}",
                   f"the machine {questions['busy']}"):
        assert phrase in n51, phrase
    assert questions["first"]["outside counsel"] is None
    for phrase in (f"first 30 words: {units['annex']}", f"questions: {questions['annex']}"):
        assert phrase in n43, phrase


def test_the_readme_and_n43_quote_the_recall_benchs_saved_run():
    # Tick 130, RS-F28: the README's recall on the dev probes and N43's table share through the whole search are
    # held to the recall bench's saved run. A machine without it skips.
    import json

    from needtoknow.settings import data_dir, repo_root
    path = data_dir() / "bench" / "probe_recall_dev.json"
    if not path.is_file():
        pytest.skip("no probe_recall_dev.json here: bench/probe_recall.py makes it")
    run = json.loads(path.read_bytes())
    selective, admin = run["selective"], run["per_role"]["admin"]

    def at(rows: list[dict], k: int, key: str = "rank") -> str:
        return f"{sum(r[key] is not None and r[key] <= k for r in rows) / len(rows):.3f}"

    def share(rows: list[dict], part: str, whole: str) -> str:
        return f"{100 * sum(r[part] for r in rows) / sum(r[whole] for r in rows):.1f}"

    same = [r for r in selective if not r["across"]]
    across = [r for r in selective if r["across"]]
    readme = " ".join((repo_root() / "README.md").read_text(encoding="utf-8").split())
    for phrase in (f"top 10 for {at(selective, 10)} of them: {at(same, 10)} when the question",
                   f"and {at(across, 10)} across languages", f"{at(across, 1)} against {at(same, 1)}.",
                   f"take {share(selective, 'tables', 'top')} percent of that top 10 and "
                   f"{share(selective, 'shown_tables', 'shown_units')} percent of the excerpts' units shown"):
        assert phrase in readme, phrase
    text = " ".join((repo_root() / "docs" / "DECISIONS.md").read_text(encoding="utf-8").split())
    start = text.index("## N43. ")
    n43 = text[start:text.index(" ## ", start)]
    assert all(at(rows, 1) == at(rows, 1, "aside") for rows in run["per_role"].values())
    for phrase in (f"{share(selective, 'tables', 'top')} percent of the top 10 and "
                   f"{share(selective, 'shown_tables', 'shown_units')} of the shown excerpts' units",
                   f"{share(admin, 'tables', 'top')} and {share(admin, 'shown_tables', 'shown_units')} for admin",
                   "recall@1 is the same for every role",
                   f"recall@10 moves from {at(selective, 10)} to {at(selective, 10, 'aside')}, and from "
                   f"{at(admin, 10)} to {at(admin, 10, 'aside')} for admin"):
        assert phrase in n43, phrase


def test_n50_quotes_the_attacks_saved_runs():
    # Tick 131, RS-F12: N50 gave the shares over the whole collection and the null and left out the view's; all
    # three, pooled, with their counts and Wilson intervals, are held to the attack's saved runs. A machine without
    # both skips.
    import json

    from needtoknow.eval.stats import wilson
    from needtoknow.settings import data_dir, repo_root
    runs = {}
    for mode in ("whole", "view"):
        path = data_dir() / "bench" / f"idf_attack_{mode}.json"
        if not path.is_file():
            pytest.skip(f"no idf_attack_{mode}.json here: bench/idf_attack.py makes it")
        runs[mode] = json.loads(path.read_bytes())
    text = " ".join((repo_root() / "docs" / "DECISIONS.md").read_text(encoding="utf-8").split())
    start = text.index("## N50. ")
    n50 = text[start:text.index(" ## ", start)]
    quoted = []
    for mode in ("whole", "view"):
        assert sorted(runs[mode]["roles"]) == ["hr", "returns", "sales"], mode
        for name in ("test pairs, hidden-frequent first", "null pairs"):
            lower, kept = (sum(role[name][i] for role in runs[mode]["roles"].values()) for i in (0, 1))
            low, high = wilson(lower, kept)
            quoted.append(f"{lower} of {kept}")
            assert re.search(rf"{lower} of {kept}[^()]*\(Wilson {low:.2f} to {high:.2f}\)", n50), (mode, name)
    # In the order the decision reads: the whole collection, its null, the view, its null.
    assert [n50.index(q) for q in quoted] == sorted(n50.index(q) for q in quoted), quoted


def test_the_row_draw_names_a_register_the_corpus_lacks(monkeypatch, tmp_path):
    # DR-F23's class: rng.choice of no table units was an IndexError.
    import sqlite3

    from needtoknow.eval import golden as G
    db = tmp_path / "manifest.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE units (unit_id TEXT, doc_id TEXT, labels TEXT, language TEXT, layer TEXT, text TEXT)")
    monkeypatch.setattr(G, "manifest_path", lambda: db)
    with pytest.raises(G.GoldenError, match="no customer-care table units in en"):
        G.draw_rows()


def test_a_probe_whose_gold_unit_is_gone_is_set_apart():
    # DR-F25: its unknown language counted it as asked across languages in the recall bench; MT-F16: in the filter
    # bench it counted as off topic for every case.
    from needtoknow.eval.probes import lost
    probes = [{"id": "DP001", "gold": ["u1"]}, {"id": "DP002", "gold": ["u1", "u9"]}]
    assert lost(probes, {"u1": 0}) == ["DP002"] and lost(probes, {"u1", "u9"}) == []


def test_every_bench_that_asks_the_probes_sets_lost_ones_apart():
    # MT-F16: the filter bench's question mode, written beside DR-F25's fix, kept the old shape.
    asking = [path.name for path in sorted(BENCH.glob("*.py"))
              if "needtoknow.eval.probes import" in path.read_text(encoding="utf-8")]
    assert {"filter_bench.py", "permission_suite.py", "probe_recall.py"} <= set(asking), asking
    for name in asking:
        assert "lost(" in (BENCH / name).read_text(encoding="utf-8"), name
