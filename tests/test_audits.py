"""Every finding id the repository cites, in its files or its commit messages, resolves in docs/AUDITS.md (MA-F5)."""

import re
import subprocess

import pytest

from needtoknow.settings import repo_root

FINDING = re.compile(r"\b[A-Z]{1,6}\d?-[FK]\d+\b")
BINARY = {".woff2", ".png", ".gif", ".webp", ".jpg"}  # files that hold no text, and so cite nothing


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
        # The samples are corpus documents, some in the legacy encodings their metadata names, and cite nothing.
        if name in ("docs/AUDITS.md", "uv.lock") or name.startswith("samples/") or not (root / name).is_file():
            continue
        try:
            text = (root / name).read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is None:  # a font's bytes read as text could cite findings at random (tick 178)
            assert (root / name).suffix in BINARY, f"{name} is neither UTF-8 text nor a file that holds none"
            continue
        for fid in FINDING.findall(text):
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


def test_the_errata_name_each_message_kept_wrong_and_a_later_commit_that_mends_it():
    # MA-F23: 17a64d9's message counts a suite its own tree fails, and the history keeps it as written; the page says
    # so, and every commit it names is one this history holds, the mend after the message it corrects.
    text = (repo_root() / "docs" / "AUDITS.md").read_text(encoding="utf-8")
    assert "\n## Errata\n" in text, "docs/AUDITS.md names no message the history keeps although it is wrong"
    pairs = re.findall(r"^\* `([0-9a-f]{7,40})` .*`([0-9a-f]{7,40})` mends", text.partition("\n## Errata\n")[2], re.M)
    assert pairs, "the errata name no message and no commit that mends it"
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        pytest.skip("this clone holds only the last commits of the history")
    for wrong, mend in pairs:
        run = subprocess.run(["git", "merge-base", "--is-ancestor", wrong, mend], cwd=repo_root(), capture_output=True)
        assert run.returncode == 0, f"{wrong} is not a commit of this history before {mend}"
