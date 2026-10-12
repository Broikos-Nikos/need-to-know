"""Every finding id the repository cites, in its files or its commit messages, resolves in docs/AUDITS.md (MA-F5)."""

import re
import subprocess

import pytest

from needtoknow.settings import repo_root

FINDING = re.compile(r"\b[A-Z]{1,6}\d?-[FK]\d+\b")


def git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if run.returncode != 0:
        pytest.skip("not a git checkout")
    return run.stdout


def test_every_cited_finding_id_has_a_row_in_docs_audits():
    root = repo_root()
    listed = set(re.findall(r"^\| `([A-Za-z0-9-]+)`", (root / "docs" / "AUDITS.md").read_text(encoding="utf-8"), re.M))
    cited: dict[str, str] = {}
    for name in git("ls-files").splitlines():
        if name in ("docs/AUDITS.md", "uv.lock") or not (root / name).is_file():
            continue
        for fid in FINDING.findall((root / name).read_bytes().decode("utf-8", errors="ignore")):
            cited.setdefault(fid, name)
    for fid in FINDING.findall(git("log", "--format=%B")):
        cited.setdefault(fid, "the commit log")
    missing = {fid: where for fid, where in cited.items() if fid not in listed}
    assert cited and not missing, f"cited but not listed in docs/AUDITS.md: {missing}"


def test_the_page_says_where_a_findings_numbers_come_from_and_every_command_it_names_is_here():
    # MA-F9: rows quoted probes the repository does not hold, as if they were its claims. The page now says whose
    # numbers they are, and where a command here measures the same thing, the row names it: it must be here.
    import shlex

    from needtoknow.cli import build_parser
    text = (repo_root() / "docs" / "AUDITS.md").read_text(encoding="utf-8")
    assert "A finding's numbers are what its audit measured when it filed it" in text
    clauses = re.findall(r"Measured by ([^|]*)[|]", text)
    assert len(clauses) >= 10
    for clause in clauses:
        for command in re.findall(r"`([^`]+)`", clause):
            words = shlex.split(command)
            if words[0] == "ntk":
                build_parser().parse_args(words[1:])
            elif words[:2] == ["uv", "run"]:
                assert words[2:3] == ["pytest"], command
            else:
                assert (repo_root() / words[0]).exists(), command
