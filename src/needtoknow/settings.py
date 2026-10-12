"""Where things are: the repository, its .env file and the data directory.

The environment wins over .env, so a key set in the shell is never shadowed by a
stale file. Values are returned, never printed. A .env saved with a byte order
mark, an inline comment after an unquoted value, and spaces around a value are all
read the way the person who wrote them meant.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SettingsError(Exception):
    """A setting holds something that cannot be what its writer meant."""


def repo_root() -> Path:
    if not (ROOT / "pyproject.toml").is_file():
        raise RuntimeError(f"expected the repository at {ROOT}; run ntk from a checkout of need-to-know")
    return ROOT


def parse_env_file(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = re.split(r"\s#", value, maxsplit=1)[0].strip()
        values[name.strip()] = value
    return values


def env_file() -> Path:
    return repo_root() / ".env"


def setting(name: str, default: str | None = None) -> str | None:
    """The environment first, then .env, then the default. Empty counts as unset."""
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    path = env_file()
    if path.is_file():
        value = parse_env_file(path.read_text(encoding="utf-8-sig")).get(name)
        if value:
            return value
    return default


def data_dir() -> Path:
    raw = setting("NTK_DATA_DIR", "data")
    path = Path(raw).expanduser()
    if "~" in path.parts:
        raise SettingsError(f"NTK_DATA_DIR={raw} keeps a ~ that does not expand to a home folder; give a full path")
    return path if path.is_absolute() else repo_root() / path
