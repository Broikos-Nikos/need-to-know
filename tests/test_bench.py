"""The benches run anywhere and still print what the history quotes them for (MA-F1)."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

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
