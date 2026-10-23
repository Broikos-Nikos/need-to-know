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


def test_every_bench_script_has_a_section_in_the_bench_readme_or_is_named_no_bench():
    # MT-F14, MT-F19: probe_recall.py, which the README quotes, had no section, and nothing asked a bench for one.
    text = (BENCH / "README.md").read_text(encoding="utf-8")
    helpers = set(re.findall(r"`(\w+\.py)` is no bench", text))
    sections = set(re.findall(r"^## (\S+\.py)$", text, re.MULTILINE))
    scripts = {path.name for path in BENCH.glob("*.py")}
    assert helpers == {"watched.py", "machine.py"} and not scripts - helpers - sections, sorted(scripts - helpers - sections)
    assert not sections - scripts, sorted(sections - scripts)  # no section for a script that is gone


def test_no_function_rebinds_a_name_its_module_imports():
    # Tick 189: served.py's main bound `load` to the processor's reader, a name the module imports to load the probes,
    # so its first line raised UnboundLocalError; nothing ran the bench's main to see it. A function's own name for a
    # thing its module imports under that name is the shape.
    import ast
    clashes = []
    for path in sorted([*BENCH.glob("*.py"), *(BENCH.parent / "src").rglob("*.py")]):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {(alias.asname or alias.name).split(".")[0] for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            bound = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            bound |= {arg.arg for arg in function.args.args + function.args.kwonlyargs}
            clashes += [f"{path.name}: {function.name} binds {name}" for name in sorted(bound & imported)]
    assert not clashes, clashes


def test_a_saved_run_is_named_by_every_argument_given_away_from_its_default():
    # MT-F13: the filter bench saved one file per mode whatever its --per-label, --seed or --efs, the recall bench
    # whatever --selective-only, the attack whatever --role, so any other run replaced the one a decision quotes.
    import argparse
    import sys
    sys.path.insert(0, str(BENCH))
    import machine
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--efs", default="64,128")
    parser.add_argument("--selective-only", action="store_true")
    parser.add_argument("--report", action="store_true")
    keys = ("seed", "efs", "selective_only")
    assert machine.named("filter_bench", parser, parser.parse_args([]), keys) == "filter_bench.json"
    assert machine.named("filter_bench", parser, parser.parse_args(["--report"]), keys) == "filter_bench.json"
    assert machine.named("filter_bench", parser, parser.parse_args(["--seed", "7", "--efs", "64,256"]),
                         keys) == "filter_bench_seed-7_efs-64+256.json"
    assert machine.named("probe_recall_dev", parser, parser.parse_args(["--selective-only"]),
                         keys) == "probe_recall_dev_selective-only.json"


def test_every_control_breaks_text_its_files_hold_once_and_names_a_test_there_is():
    # MA-F16: the controls ran from the session's scratchpad, so their count in a commit message could not be run
    # again; now each is data, and the suite holds each change to text its file holds once, in order, and each control
    # to a test that exists, so a control goes stale here, not silently. bench/controls.py runs them.
    import tomllib
    controls = tomllib.loads((BENCH / "controls.toml").read_text(encoding="utf-8"))["control"]
    assert len({control["name"] for control in controls}) == len(controls) > 0
    for control in controls:
        texts = {}
        for change in control["change"]:
            path = BENCH.parent / change["file"]
            text = texts.setdefault(path, path.read_bytes().decode("utf-8"))
            assert text.count(change["old"]) == 1, (control["name"], change["file"], text.count(change["old"]))
            texts[path] = text.replace(change["old"], change["new"])
        test_file, _, test_name = control["test"].partition("::")
        source = (BENCH.parent / test_file).read_text(encoding="utf-8")
        assert not test_name or f"def {test_name}(" in source, (control["name"], control["test"])


def test_a_control_whose_test_was_skipped_or_never_collected_did_not_run(monkeypatch, tmp_path):
    # MT-F25: the runner called a control passed whenever its test's pytest exited 0, and pytest exits 0 on a skip, so
    # a control on a test that needs the GPU, the plan or a built index read as a gate that cannot fail where they are
    # absent. A skip, and a test pytest never collected, is a control that did not run, with its reason.
    import types
    spec = importlib.util.spec_from_file_location("controls", BENCH / "controls.py")
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    target = tmp_path / "gate.txt"
    target.write_bytes(b"guarded\n")
    control = {"name": "a guard taken away", "test": "tests/test_x.py::test_x",
               "change": [{"file": str(target), "old": "guarded", "new": "open"}]}
    said = {"skip": (0, "SKIPPED [1] tests/test_x.py:3: no CUDA device is visible\n1 skipped in 0.01s\n"),
            "none": (5, "no tests ran in 0.01s\n"), "pass": (0, "1 passed in 0.01s\n"), "fail": (1, "E   assert 1 == 2\n")}
    outcomes = {}
    for case, (code, stdout) in said.items():
        monkeypatch.setattr(controls, "subprocess", types.SimpleNamespace(
            run=lambda *args, code=code, stdout=stdout, **kwargs: types.SimpleNamespace(returncode=code, stdout=stdout)))
        outcomes[case] = controls.run(control)
        assert target.read_bytes() == b"guarded\n"  # restored byte for byte, whatever the outcome
    assert outcomes == {"skip": ("skipped", "no CUDA device is visible"), "none": ("skipped", "no test collected"),
                        "pass": ("passed", ""), "fail": ("failed", "assert 1 == 2")}, outcomes


def test_the_draws_bench_reuses_a_draw_only_at_the_same_share_and_states_each_share_with_its_count(monkeypatch,
                                                                                                   tmp_path):
    # RS-F6: a draw folder left from another share of the catalogue would be built and probed as if it were this one.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    build()
    spec = importlib.util.spec_from_file_location("draws", BENCH / "draws.py")
    draws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draws)
    said = []
    first = draws.made(7, 80, said.append)
    assert draws.made(7, 80, said.append) == first and said[-1] == f"draw 7: found, {first['documents']} documents"
    with pytest.raises(SystemExit, match="not one in 40"):
        draws.made(7, 40, said.append)
    assert draws.rate(9, 10).startswith("0.900 (9 of 10, Wilson ") and draws.rate(0, 0) == "no probes"
    assert draws.spread([0.5, 0.9, 0.7]) == "0.500 to 0.900"


def test_the_baseline_ledger_finds_each_answers_seconds_once_and_reads_p95_by_nearest_rank():
    # PLAN section 1, number 7: the seconds of a run that kept none come from the spend log line ask wrote for each
    # answer, matched by role, repeat, status, cost and time, each line taken once; a run's own seconds come first.
    spec = importlib.util.spec_from_file_location("baseline", BENCH / "baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    answers = [{"id": "D1", "repeat": 0, "status": "answered", "usd": 0.0002, "at": "2026-10-13T13:00:03+00:00"},
               {"id": "D1", "repeat": 1, "status": "answered", "usd": 0.0002, "at": "2026-10-13T13:00:09+00:00"},
               {"id": "D2", "repeat": 0, "status": "not_found", "usd": 0.0001, "at": "2026-10-13T13:00:20+00:00",
                "seconds": {"total": 9.0}}]
    spend = [{"role": "hr", "repeat": 0, "status": "answered", "usd": 0.0002, "at": "2026-10-13T13:00:02+00:00",
              "seconds": {"total": 1.5}},
             {"role": "sales", "repeat": 1, "status": "answered", "usd": 0.0002, "at": "2026-10-13T13:00:08+00:00",
              "seconds": {"total": 7.0}},  # another role's line at the same cost
             {"role": "hr", "repeat": 1, "status": "answered", "usd": 0.0002, "at": "2026-10-13T13:00:01+00:00",
              "seconds": {"total": 8.0}},  # eight seconds before the answer: not its line
             {"role": "hr", "repeat": 1, "status": "answered", "usd": 0.0002, "at": "2026-10-13T13:00:08+00:00",
              "seconds": {"total": 2.5}}]
    assert baseline.seconds_of(answers, {"D1": "hr", "D2": "hr"}, spend) == [1.5, 2.5, 9.0]
    # Two answers alike in all but time each take their own line.
    twins = [{"id": "D3", "repeat": 0, "status": "answered", "usd": 0.0003, "at": "2026-10-13T13:00:30+00:00"},
             {"id": "D4", "repeat": 0, "status": "answered", "usd": 0.0003, "at": "2026-10-13T13:00:31+00:00"}]
    lines = [{"role": "hr", "repeat": 0, "status": "answered", "usd": 0.0003, "at": "2026-10-13T13:00:29+00:00",
              "seconds": {"total": 2.0}},
             {"role": "hr", "repeat": 0, "status": "answered", "usd": 0.0003, "at": "2026-10-13T13:00:30+00:00",
              "seconds": {"total": 3.0}}]
    assert baseline.seconds_of(twins, {"D3": "hr", "D4": "hr"}, lines) == [2.0, 3.0]
    hundred = [float(n) for n in range(1, 101)]
    assert (baseline.nearest_rank(hundred, 0.95), baseline.nearest_rank(hundred, 0.57)) == (95.0, 57.0)
    assert baseline.nearest_rank([3.0, 1.0, 2.0], 0.95) == 3.0


def test_the_baseline_report_names_every_provenance_field_its_runs_differ_on():
    # DR-F28: the report printed the test run's provenance as the baseline's and never compared the others.
    spec = importlib.util.spec_from_file_location("baseline", BENCH / "baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    same = {"commit": "a" * 40, "clean": True, "protocol": "p", "model": "m", "settings": {"temperature": 0},
            "manifest": {"run": 1}, "analyser": "x"}
    assert baseline.differences({"test": same, "dev": dict(same)}) == []
    assert baseline.differences({"test": same, "dev": dict(same), "test-guess": {**same, "commit": "b" * 40,
                                                                                "clean": False}}) == [
        "commit differs: test, dev at aaaaaaaaaaaa; test-guess at bbbbbbbbbbbb", "tree not clean when made: test-guess"]


def test_the_change_benches_read_what_each_sync_embedded_again_out_of_the_points():
    # PLAN section 1, number 5: the units one edit embeds again, out of the collection's points, read from the line the
    # watcher's sync prints, as cli.report_run writes it.
    spec = importlib.util.spec_from_file_location("watched", BENCH / "watched.py")
    watched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watched)
    import io

    from needtoknow.cli import report_run
    printed = io.StringIO()
    for embedded, seconds in ((1, 0.85), (0, 0.40)):
        report_run({"deleted": 1 - embedded, "embedded": embedded, "upserted": 1, "points_after": 501031, "units": 3366,
                    "seconds": seconds, "parts": {"embed": 0.30}}, printed)
    seen = watched.Watched()

    class Process:
        stdout = iter(printed.getvalue().splitlines(keepends=True))

    seen.process = Process()
    seen.read()
    assert [e["embedded"] for e in seen.events] == [1, 0]
    assert seen.embedded(min(e["began"] for e in seen.events), 60) == (1, 501031)


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


def test_the_before_recipe_runs_a_bench_its_commit_can_import():
    # MA-F13's recipe copied today's bench into a worktree at 094377a, whose prompt module has no reader_mark: it could
    # not run. The bench the recipe takes asks only for names that commit's modules define.
    import ast
    import subprocess

    def show(rev: str, path: str) -> str | None:
        done = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=BENCH.parent, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        return None if done.returncode else done.stdout

    def names(source: str) -> set[str]:
        found = set()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                found |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        return found

    readme = (BENCH / "README.md").read_text(encoding="utf-8")
    readme = readme[readme.index("## injection.py"):]
    readme = readme[:readme.index("\n## ", 1)]  # the section alone: other benches give worktrees of their own
    commit = re.search(r"git worktree add \.\./ntk-before (\w+)", readme)[1]
    line = next(line for line in readme.splitlines() if "injection.py" in line and line.endswith("../ntk-before/"))
    taken = re.match(r"git show (\w+):bench/injection\.py > \.\./ntk-before/bench/injection\.py", line)
    if show(commit, "pyproject.toml") is None or (taken and show(taken[1], "bench/injection.py") is None):
        pytest.skip("this clone's history holds neither commit")
    bench = show(taken[1], "bench/injection.py") if taken else (BENCH / "injection.py").read_text(encoding="utf-8")
    missing = []
    for node in ast.walk(ast.parse(bench)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("needtoknow."):
            path = "src/" + node.module.replace(".", "/")
            source = show(commit, path + ".py") or show(commit, path + "/__init__.py") or ""
            missing += [f"{node.module}.{alias.name}" for alias in node.names if alias.name not in names(source)
                        and show(commit, f"{path}/{alias.name}.py") is None]  # a submodule is a name too
    assert not missing, missing


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


def test_the_recall_bench_finds_a_golden_item_only_once_every_section_it_needs_is_hit():
    # PLAN section 1, number 3: an answerable item counts at the rank by which a section for each of its facts is hit,
    # so a two fact item is not found by its first fact alone.
    spec = importlib.util.spec_from_file_location("probe_recall", BENCH / "probe_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    a, b, c = ("D1", "S1"), ("D2", "S2"), ("D3", "S3")
    assert recall.found_at([{a}], [c, a, b]) == 2
    assert recall.found_at([{a}, {b, c}], [a, None, c]) == 3
    assert recall.found_at([{a}, {b}], [a, c, a]) is None


def test_the_recall_bench_reads_a_golden_item_by_the_units_holding_its_facts(monkeypatch):
    # RS-F31: a table row's section is a whole register, so an item is found and shown by the unit holding its fact or
    # row; a hit or an excerpt of another unit of the same section counts for nothing.
    spec = importlib.util.spec_from_file_location("probe_recall", BENCH / "probe_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    from needtoknow.eval import golden as G
    from needtoknow.eval import score as S

    class Units:
        def __init__(self, ctx):
            pass

        def units_of_fact(self, fact_id):
            return {"u7"}

        def units_of_row(self, row):
            return {"r9"}

    monkeypatch.setattr(G, "Context", lambda: None)
    monkeypatch.setattr(G, "golden_path", lambda split: split)
    monkeypatch.setattr(G, "load_items", lambda path: [
        {"id": "D1", "category": "A", "question": "q", "role": "sales", "expect": {"facts": [{"fact_id": "F1"}]}},
        {"id": "D2", "category": "I", "question": "q", "role": "sales", "row": {"doc_id": "X", "key": "K"}}])
    monkeypatch.setattr(S, "Sources", Units)
    hits = [{"unit_id": "u6"}, {"unit_id": "u7"}, {"unit_id": "r8"}]
    rows = recall.golden("dev", None, lambda client, question, role: hits, lambda found: [{"units": ["u6", "u7"]}])
    assert [(r["id"], r["rank"], r["shown"]) for r in rows] == [("D1", 2, True), ("D2", None, False)]


def test_the_recall_bench_reports_the_probe_arms_reach_by_family_from_a_saved_run():
    # MA-F19: the probe arm's misses by family and its best p, printed by a committed command and not by a scratch one.
    spec = importlib.util.spec_from_file_location("probe_recall", BENCH / "probe_recall.py")
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    probes = [{"id": f"P{n}", "family": family} for n, family in enumerate(["F1", "F1", "F2", "F3", "F3", "F4"])]
    rows = [{"rank": rank} for rank in (1, 12, None, 3, 2, 30)]  # P1 of F1, P2 of F2 and P5 of F4 outside the top 10
    arm = recall.reach_of(probes, rows)
    assert (arm["probes"], arm["misses"], arm["families"], arm["with_room"], arm["p"]) == (6, 3, 4, 3, 0.25)
    named = [{**row, "id": probe["id"]} for probe, row in zip(reversed(probes), reversed(rows))]
    assert recall.reach_of(probes, named) == arm  # a run whose rows name their probes, in any order


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
    # questions; each figure from its saved numbers is held to them, but no timing and no busy share, which a run
    # with the same arguments does not give again and the decision states with its date (MT-F13). A machine without
    # both runs skips.
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

        roles = [c for c, r in cases.items() if not r["bench_only"]]
        graph = {c: mean([r for r in rows if r["case"] == c and r["ef"] == 512 and r["split"] == "dev"]) for c in cases}
        prose = [r for r in results if r["kind"] in ("prose", "question") and r["class"] == "in domain"
                 and r["ef"] is None]
        assert all(r["tie_aware"] == 1.0 for r in results if r["ef"] is None)
        assert sum(r["foreign"] for r in results) == 0
        return {"first": {c: first(c) for c in cases}, "graph": graph, "roles": roles, "searches": len(results),
                "range": f"{min(graph[c] for c in roles):.3f} to {max(graph[c] for c in roles):.3f}",
                "annex": f"{100 * mean(prose, 'tables') / 10:.1f} percent"}

    units, questions = (figures(results) for results in runs)
    n51, n43 = section("N51"), section("N43")
    assert len(questions["roles"]) == 9 and not any(questions["first"][c] for c in questions["roles"])
    for phrase in (f"ran from {units['range']}", f"ran from {questions['range']} over the nine roles",
                   f"admin's fell from {units['graph']['admin']:.3f} to {questions['graph']['admin']:.3f}",
                   f"hr alone still reached it at ef {questions['first']['hr alone']}",
                   f"which reached it at ef {units['first']['outside counsel']} on unit prefixes, ended at "
                   f"{questions['graph']['outside counsel']:.3f}",
                   f"in {questions['searches']:,} searches"):
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
    for phrase in (f"top 10 for {at(selective, 10)} of the {len(selective)}: {at(same, 10)} when the question",
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
    # In the order the decision reads: the whole collection, its null, the view, its null; each found after the one
    # before, since two nulls can hold the same count.
    position = 0
    for quote in quoted:
        assert quote in n50[position:], (quote, quoted)
        position = n50.index(quote, position) + len(quote)


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
    # MT-F16: the filter bench's question mode, written beside DR-F25's fix, kept the old shape. Every bench that asks
    # probes names those it sets apart; one that reads the committed probes sets apart those whose gold unit the
    # manifest lacks now, and one that makes its own, as the draws bench does, those its generator found none for.
    asking = {path.name: path.read_text(encoding="utf-8") for path in sorted(BENCH.glob("*.py"))
              if "needtoknow.eval.probes import" in path.read_text(encoding="utf-8")}
    assert {"draws.py", "filter_bench.py", "permission_suite.py", "probe_recall.py"} <= set(asking), sorted(asking)
    for name, text in asking.items():
        assert "left out: {', '.join(" in text, name
        if "probes import load" in text:
            assert "lost(" in text, name


def test_each_freshness_edit_writes_a_value_no_earlier_run_wrote():
    # MA-F32: every run edited the fact to the same value, so the vector cache held the edited passage after the first
    # run ever, and no edit of a later run timed an embedding.
    sys.path.insert(0, str(BENCH))
    import freshness
    first, later = freshness.edit_values("3", 5, 1_761_000_000), freshness.edit_values("3", 5, 1_761_000_001)
    assert len(set(first)) == 5 and "3" not in first and not set(first) & set(later)
