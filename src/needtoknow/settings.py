"""Where things are: the repository, its .env file and the data directory.

The environment wins over .env, so a key set in the shell is never shadowed by a
stale file. Values are returned, never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def env_file() -> Path:
    return repo_root() / ".env"


def setting(name: str, default: str | None = None) -> str | None:
    """The environment first, then .env, then the default. Empty counts as unset."""
    value = os.environ.get(name)
    if value:
        return value
    path = env_file()
    if path.is_file():
        value = parse_env_file(path.read_text(encoding="utf-8")).get(name)
        if value:
            return value
    return default


def data_dir() -> Path:
    raw = setting("NTK_DATA_DIR", "data")
    path = Path(raw)
    return path if path.is_absolute() else repo_root() / path
