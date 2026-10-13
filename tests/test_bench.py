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
