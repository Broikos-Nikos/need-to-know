"""The `ntk` command. Each subcommand arrives with the task that needs it."""

from __future__ import annotations

import argparse
import sys

from needtoknow import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntk",
        description="Retrieval over a large document base where every section names who may read it.",
    )
    parser.add_argument("--version", action="version", version=f"ntk {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not vars(args):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
