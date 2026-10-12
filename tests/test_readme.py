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


def test_every_ntk_command_the_readme_gives_parses():
    commands = re.findall(r"`?(?:uv run )?ntk ([a-z][a-z -]*?)(?:`|\n|$)", README)
    assert len(commands) >= 5
    for command in commands:
        try:
            build_parser().parse_args(shlex.split(command))
        except SystemExit as error:
            pytest.fail(f"ntk {command}: exit {error.code}")


def test_every_bench_the_readme_runs_is_in_the_repository():
    scripts = set(re.findall(r"python (bench/[\w]+\.py)", README))
    assert scripts and all((repo_root() / s).is_file() for s in scripts), scripts


def test_the_readme_says_what_the_documents_are_and_what_needs_a_network():
    assert "synthetic" in README.lower()
    for sentence in re.split(r"(?<=[.;])\s+", README):
        if "network" in sentence:
            assert "uv sync" in sentence or "uv sync" in README[:README.index(sentence)].rsplit("\n\n", 1)[-1], sentence
