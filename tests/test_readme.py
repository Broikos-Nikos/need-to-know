"""The README's claims that a test can hold (HE-F1 to HE-F3): every path it names exists, every command it gives
parses, it says the documents are synthetic, and it never says a network is unneeded without uv sync beside it."""

import re
import shlex

import pytest

from needtoknow.cli import build_parser
from needtoknow.settings import repo_root

README = (repo_root() / "README.md").read_text(encoding="utf-8")


def test_every_path_the_readme_names_exists():
    named = set(re.findall(r"`((?:[\w.-]+/)+[\w.-]*|\.env\.example)`", README))
    assert named, "no path found: the pattern no longer matches the README"
    missing = sorted(p for p in named if not (repo_root() / p).exists())
    assert not missing, missing


# A placeholder the README writes, and a value of its kind (MT-F17).
PLACEHOLDERS = {"question": "a question", "role": "sales", "file": "runs.jsonl"}


def ntk_commands(readme: str) -> list[str]:
    """Every ntk command the README gives, in a code span or a code block, its placeholders filled."""
    spans = re.findall(r"`([^`\n]+)`", readme)
    blocks = [line for block in re.findall(r"```\n(.*?)```", readme, re.S) for line in block.splitlines()]
    commands = []
    for text in spans + blocks:
        found = re.fullmatch(r"\s*(?:uv run )?ntk (.+?)\s*", text)
        if found:
            unknown = set(re.findall(r"<(\w+)>", found[1])) - set(PLACEHOLDERS)
            assert not unknown, (text, unknown)  # a placeholder of no known kind fails, it is not skipped
            commands.append(re.sub(r"<(\w+)>", lambda m: PLACEHOLDERS[m[1]], found[1]))
    return commands


def test_every_ntk_command_the_readme_gives_parses():
    # MT-F17: the gate parsed a command only when it held lower case letters, spaces and dashes, so the ones with a
    # placeholder, a quoted question or a path were skipped without a word, ntk ask and ntk eval run among them.
    commands = ntk_commands(README)
    assert any('"' in c for c in commands) and any("--as" in c for c in commands) and len(commands) >= 30
    for command in commands:
        try:
            build_parser().parse_args(shlex.split(command))
        except SystemExit as error:
            pytest.fail(f"ntk {command}: exit {error.code}")


def test_the_readme_says_which_commands_run_until_stopped():
    # HS-F23: ntk serve came before ntk ingest watch in the commands, and nothing said that both run until stopped.
    endless = ("ntk serve", "ntk ingest watch")
    assert all(f"uv run {command}" in README for command in endless)
    said = [s for s in re.split(r"(?<=[.;])\s+", README) if "run until stopped" in s]
    assert said and all(f"`{command}`" in said[0] for command in endless), said


def test_every_bench_the_readme_runs_is_in_the_repository():
    scripts = set(re.findall(r"python (bench/[\w]+\.py)", README))
    assert scripts and all((repo_root() / s).is_file() for s in scripts), scripts


def test_the_readme_says_what_the_documents_are_and_what_needs_a_network():
    assert "synthetic" in README.lower()
    for sentence in re.split(r"(?<=[.;])\s+", README):
        if "network" in sentence:
            assert "uv sync" in sentence or "uv sync" in README[:README.index(sentence)].rsplit("\n\n", 1)[-1], sentence
