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


def test_a_byte_order_mark_does_not_hide_the_first_key(fake_env_file, monkeypatch):
    # HS-F1: Windows PowerShell 5.1's Set-Content -Encoding UTF8 writes exactly this file.
    fake_env_file.write_bytes(b"\xef\xbb\xbfNTK_PROBE=from_file\r\n")
    monkeypatch.delenv("NTK_PROBE", raising=False)
    assert settings.setting("NTK_PROBE") == "from_file"


def test_an_inline_comment_is_not_part_of_an_unquoted_value():
    # HS-F3: the comment used to travel into the key and the provider refused it.
    text = 'KEY=sk-abc  # main key\nQUOTED="a # b"  # a note\nHASH=abc#def\n'
    assert settings.parse_env_file(text) == {"KEY": "sk-abc", "QUOTED": "a # b", "HASH": "abc#def"}


def test_spaces_around_an_environment_value_are_dropped(fake_env_file, monkeypatch):
    monkeypatch.setenv("NTK_PROBE", "  value with a space after  ")
    assert settings.setting("NTK_PROBE") == "value with a space after"


def test_a_home_relative_data_dir_expands_to_the_home_folder(fake_env_file, monkeypatch):
    # HS-F5: it used to land in a folder literally named ~ inside the repository.
    monkeypatch.setenv("NTK_DATA_DIR", "~/ntk-data")
    assert settings.data_dir() == settings.Path.home() / "ntk-data"


def test_a_tilde_that_cannot_expand_is_refused(fake_env_file, monkeypatch):
    monkeypatch.setenv("NTK_DATA_DIR", "data/~/x")
    with pytest.raises(settings.SettingsError, match="keeps a ~"):
        settings.data_dir()


def test_git_ignores_the_env_file():
    # The key lives in .env. If git stops ignoring it, the next commit publishes it.
    root = settings.repo_root()
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", ".env"])
    assert result.returncode == 0, ".env is not ignored by git"
