"""Every decision the repository cites, in its files or its commit messages, has an entry in docs/DECISIONS.md (MT-F6):
a reason cited is a reason only if the reader can open it."""

import re
import subprocess

import pytest

from needtoknow.settings import repo_root

DECISION = re.compile(r"[N][0-9]{1,3}(?![0-9])")
WORD = re.compile(r"(?<![A-Za-z0-9_-])" + DECISION.pattern)
BINARY = {".woff2", ".png", ".gif", ".webp", ".jpg"}  # files that hold no text, and so cite nothing


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
        if not (root / name).is_file():
            continue
        try:
            text = (root / name).read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is None:  # a font's bytes read as text cited decisions at random (tick 178)
            assert (root / name).suffix in BINARY, f"{name} is neither UTF-8 text nor a file that holds none"
            continue
        for found in WORD.findall(text):
            cited.setdefault(found, name)
    for found in WORD.findall(git("log", "--format=%B")):
        cited.setdefault(found, "the commit log")
    missing = {decision: where for decision, where in cited.items() if decision not in listed}
    assert {"N50", "N51"} <= set(cited) and not missing, f"cited but not in docs/DECISIONS.md: {missing}"


def test_the_decisions_page_says_what_the_task_ids_the_history_cites_are():
    # MT-F27: task ids like T6.2e and the house rules, cited in a hundred tracked files and the commit log, name the
    # project's working plan and log, which the repository does not hold, and no page said so. The decisions page does,
    # once, where it says what the research notes' citations are.
    task = re.compile(r"(?<![A-Za-z0-9_-])T[0-9]+[.][0-9]+[a-z0-9]*(?![0-9])")
    cited = [name for name in git("ls-files").splitlines() if (repo_root() / name).suffix in {".py", ".ts", ".mjs", ".md"}
             and task.search((repo_root() / name).read_text(encoding="utf-8", errors="replace"))]
    header = (repo_root() / "docs" / "DECISIONS.md").read_text(encoding="utf-8").split(chr(10) + "## ")[0]
    assert cited and "Task ids like `T6.2e`" in header and "working plan and log" in header, (len(cited), header[-400:])
