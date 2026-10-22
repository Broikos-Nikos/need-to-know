"""The claims gate in the build (T5.6e2, P5's exit): every number the README's "What works today" states is printed by
a record of a command beside it, kept in docs/claims/ by bench/claims.py."""

import importlib.util
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench"


def claims():
    spec = importlib.util.spec_from_file_location("claims", BENCH / "claims.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_number_the_readme_states_is_printed_by_a_record_of_its_command():
    gate = claims()
    readme = (gate.REPO / "README.md").read_text(encoding="utf-8")
    assert sum(len(gate.claimed(bullet)) for bullet in gate.bullets(readme)) > 50  # the gate reads the claims at all
    assert gate.check(readme) == []
    # No record keeps a progress bar, whose counts would let a number pass that the command never claimed.
    assert not [p.name for p in gate.RECORDS.glob("*.txt") if "%|" in p.read_text(encoding="utf-8")]


def test_a_number_no_record_prints_and_a_bullet_with_no_record_are_named(tmp_path):
    gate = claims()
    readme = ("# x\n\n## What works today\n\n"
              "* Found 12 of 5,760 in 3.5 s, its top 10, recall@1, up to 20 a label, N43, RS-F6 and DP041: "
              "`uv run python bench/x.py --split test`.\n"
              "* Took 7 s: `ntk y`.\n\n## Run it\n")
    record = tmp_path / "python-bench-x.py-split-test.txt"
    record.write_text("command: uv run python bench/x.py --split test\n---\n12 of 5760 found in 3.5 s\n",
                      encoding="utf-8")
    problems = gate.check(readme, tmp_path)
    assert len(problems) == 1 and problems[0].startswith("Took 7 s") and "no record of uv run ntk y" in problems[0]
    record.write_text("command: uv run python bench/x.py --split test\n---\n12 of 5761 found in 3.5 s\n",
                      encoding="utf-8")
    assert any(p.endswith("prints 5,760") for p in gate.check(readme, tmp_path))


def test_a_number_only_a_records_head_holds_is_named(tmp_path):
    # DR-F32: the head's date and time were read as the command's output, so a figure equal to them passed.
    gate = claims()
    readme = "# x\n\n## What works today\n\n* Took 37 s: `uv run python bench/x.py`.\n\n## Run it\n"
    (tmp_path / "python-bench-x.py.txt").write_text(
        "command: uv run python bench/x.py\nrecorded 2026-10-18 23:37, run at 9e22fde\n---\nthe first in 1.62 s\n",
        encoding="utf-8")
    problems = gate.check(readme, tmp_path)
    assert len(problems) == 1 and problems[0].endswith("prints 37")


def test_the_decisions_and_the_bench_readme_say_where_each_number_comes_from(tmp_path):
    # MA-F24: 79 of 246 numbers in the decisions' prose and 7 of 56 in the bench README's were printed by no record,
    # held by no gate. A decision opens on when it was decided; a bench section's numbers are a record's, a parameter
    # in backticks, or dated where they stand.
    gate = claims()
    assert gate.check_docs() == []
    (tmp_path / "docs").mkdir()
    (tmp_path / "bench").mkdir()
    n = "N"  # spelt apart, so this file cites no decision the decisions page lacks
    (tmp_path / "docs" / "DECISIONS.md").write_text(f"# d\n\n## {n}1. Dated\n\n*Decided 2026-10-12, at tick 1.*\n\n"
                                                    f"Took 37 s.\n\n## {n}2. Undated\n\nTook 37 s.\n", encoding="utf-8")
    (tmp_path / "bench" / "README.md").write_text("# b\n\n## x.py\n\nTakes 37 s, `12` a label, at tick 9 then 8 s.\n\n"
                                                 "Asks 12 a label.\n", encoding="utf-8")
    problems = gate.check_docs(tmp_path / "records", tmp_path)
    assert len(problems) == 2 and f"{n}2. Undated" in problems[0] and problems[1].endswith(
        "prints 12, and nothing says when it was measured"), problems


def test_a_record_names_the_repository_as_a_dot_and_no_folder_of_this_machine():
    # SC-F5: a record kept a command's warnings as written, the full path of the checkout they came from among them, in
    # docs/claims/python-bench-draws.py.txt. The writer puts the repository's folder as `.`, in either slash.
    gate = claims()
    for root in (str(gate.REPO), gate.REPO.as_posix()):
        slash = "/" if "/" in root else chr(92)
        said = gate.written(chr(10).join([f"{root}{slash}src{slash}x.py:1: UserWarning", "kept 12", ""]))
        assert root not in said and said.startswith(".") and "kept 12" in said, said


def test_no_tracked_file_names_a_folder_of_this_machine():
    # SC-F5: whatever wrote it, a tracked file holds neither the checkout's own folder nor its user's home, in either
    # slash and any case, so the published history names no folder of the author's.
    import subprocess
    root = BENCH.parent
    forms = {form.lower() for place in (root, Path.home()) for form in (str(place), place.as_posix())}
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    found = []
    for name in tracked:
        try:
            text = (root / name).read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):  # a binary, a font or the picture
            continue
        found += [name for form in forms if form in text]
    assert not found, sorted(set(found))


def test_a_record_from_a_tree_holding_changes_names_the_code_that_differs():
    # MA-F29: the stats record said run at 0a52bf1, the tree holding changes, and that commit's code gave another
    # index size; the code the run used beyond its commit is named file by file, records and tests left out.
    tool = claims()
    status = " M src/needtoknow/index/collection.py\n M docs/claims/ntk-stats.txt\n?? bench/parts_check.py\n M tests/x.py\n"
    assert tool.changes(status) == ", with changes to bench/parts_check.py, src/needtoknow/index/collection.py"
    assert tool.changes(" M README.md\n") == ", the tree holding changes to no code"
    assert tool.changes("") == ""
