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
