import subprocess
from pathlib import Path

import pytest

from needtoknow import settings


def test_env_file_parsing_keeps_values_and_drops_noise():
    text = "\n".join([
        "# a comment",
        "",
        "PLAIN=value",
        'QUOTED="with spaces"',
        "SINGLE='x'",
        "export EXPORTED=yes",
        "URL=https://example.invalid/a=b",
        "EMPTY=",
        "no equals sign here",
    ])
    assert settings.parse_env_file(text) == {
        "PLAIN": "value",
        "QUOTED": "with spaces",
        "SINGLE": "x",
        "EXPORTED": "yes",
        "URL": "https://example.invalid/a=b",
        "EMPTY": "",
    }


@pytest.fixture
def fake_env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(settings, "env_file", lambda: path)
    return path


def test_the_environment_wins_over_the_file(fake_env_file, monkeypatch):
    fake_env_file.write_text("NTK_PROBE=from_file\n", encoding="utf-8")
    monkeypatch.setenv("NTK_PROBE", "from_environment")
    assert settings.setting("NTK_PROBE") == "from_environment"


def test_the_file_fills_in_when_the_environment_is_silent(fake_env_file, monkeypatch):
    fake_env_file.write_text("NTK_PROBE=from_file\n", encoding="utf-8")
    monkeypatch.delenv("NTK_PROBE", raising=False)
    assert settings.setting("NTK_PROBE") == "from_file"


def test_an_empty_value_counts_as_unset(fake_env_file, monkeypatch):
    fake_env_file.write_text("NTK_PROBE=\n", encoding="utf-8")
    monkeypatch.setenv("NTK_PROBE", "")
    assert settings.setting("NTK_PROBE", "fallback") == "fallback"


def test_a_relative_data_dir_is_under_the_repository(fake_env_file, monkeypatch):
    monkeypatch.setenv("NTK_DATA_DIR", "data")
    assert settings.data_dir() == settings.repo_root() / "data"


def test_an_absolute_data_dir_is_kept(fake_env_file, monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    assert settings.data_dir() == tmp_path


def test_git_ignores_the_env_file():
    # The key lives in .env. If git stops ignoring it, the next commit publishes it.
    root = settings.repo_root()
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", ".env"])
    assert result.returncode == 0, ".env is not ignored by git"
