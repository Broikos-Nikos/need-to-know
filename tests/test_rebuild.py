"""docs/REBUILD.md, the order a rebuild takes and the figures that say it is the same build (T7.7): the build's data
was deleted after the upload, so the page is the one thing between this repository and the build. Its commands parse
or name a file that is here, its pins are the configuration's own, and the records it names are committed."""

import re
import shlex
import tomllib

from needtoknow.cli import build_parser
from needtoknow.settings import repo_root

TEXT = (repo_root() / "docs" / "REBUILD.md").read_text(encoding="utf-8")


def test_every_command_the_rebuild_gives_parses_or_is_here():
    block = re.search(r"## The order\n\n```\n(.*?)```", TEXT, re.S)[1]
    commands = [line for line in block.splitlines() if line.strip()]
    assert len(commands) >= 10, commands
    for command in commands:
        words = shlex.split(command)
        if words[:3] == ["uv", "run", "ntk"]:
            build_parser().parse_args(words[3:])
        elif words[:3] == ["uv", "run", "python"]:
            assert (repo_root() / words[3]).is_file(), command
        else:
            assert words[0] in ("uv", "npm"), command


def test_the_rebuild_pins_what_the_configuration_pins():
    models = tomllib.loads((repo_root() / "config" / "models.toml").read_text(encoding="utf-8"))
    qdrant = tomllib.loads((repo_root() / "config" / "qdrant.toml").read_text(encoding="utf-8"))
    text = " ".join(TEXT.split())
    for part in ("embed", "rerank"):
        assert f"`{models[part]['repo']}` at `{models[part]['revision']}`" in text, part
    assert f"Qdrant `{qdrant['version']}`" in text, qdrant["version"]


def test_the_records_the_rebuild_names_are_committed():
    named = re.findall(r"`(docs/claims/[^`]+)`", TEXT)
    assert named and all((repo_root() / name).is_file() for name in named), named
