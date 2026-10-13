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
    found = recall.measure({"question": "q", "gold": ["u1", "u5"]}, "sales", None, lambda c, q, r: hits,
                           lambda h: [{"units": ["u9", "u1"]}])
    assert found == {"rank": 3, "shown": True}
    # Among the hits but cut from the excerpts the prompt shows: found, not shown.
    cut = recall.measure({"question": "q", "gold": ["u1"]}, "sales", None, lambda c, q, r: hits,
                         lambda h: [{"units": ["u9", "u2"]}])
    assert cut == {"rank": 3, "shown": False}


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


def test_the_recall_bench_sets_apart_a_probe_whose_gold_unit_is_gone():
    # DR-F25: its unknown language counted it as asked across languages.
    spec = importlib.util.spec_from_file_location("probe_recall", BENCH / "probe_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    probes = [{"id": "DP001", "gold": ["u1"]}, {"id": "DP002", "gold": ["u1", "u9"]}]
    assert recall.lost(probes, {"u1": "en", "u9": None}) == ["DP002"]
