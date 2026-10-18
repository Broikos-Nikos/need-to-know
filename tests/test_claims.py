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
