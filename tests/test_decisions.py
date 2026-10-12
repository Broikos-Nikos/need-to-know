"""Every decision the repository cites, in its files or its commit messages, has an entry in docs/DECISIONS.md (MT-F6):
a reason cited is a reason only if the reader can open it."""

import re
import subprocess

import pytest

from needtoknow.settings import repo_root

DECISION = re.compile(r"[N][0-9]{1,3}(?![0-9])")
WORD = re.compile(r"(?<![A-Za-z0-9_-])" + DECISION.pattern)


def git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if run.returncode != 0:
        pytest.skip("not a git checkout")
    return run.stdout


def test_every_cited_decision_has_an_entry_in_docs_decisions():
    root = repo_root()
    page = (root / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^## (N[0-9]+)[.]", page, re.M))
    cited: dict[str, str] = {}
    for name in git("ls-files").splitlines():
        if name in ("docs/DECISIONS.md", "uv.lock") or name.startswith(("narrative/", "samples/")):
            continue
        if (root / name).is_file():
            for found in WORD.findall((root / name).read_bytes().decode("utf-8", errors="ignore")):
                cited.setdefault(found, name)
    for found in WORD.findall(git("log", "--format=%B")):
        cited.setdefault(found, "the commit log")
    missing = {decision: where for decision, where in cited.items() if decision not in listed}
    assert {"N50", "N51"} <= set(cited) and not missing, f"cited but not in docs/DECISIONS.md: {missing}"
