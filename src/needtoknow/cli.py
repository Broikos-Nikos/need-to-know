"""The `ntk` command. Each subcommand arrives with the task that needs it."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

from needtoknow import __version__
from needtoknow.providers import deepseek_account
from needtoknow.providers.catalog import load_catalog
from needtoknow.settings import setting

PROVIDERS = ("deepseek", "openai", "anthropic", "gemini")

# The network opener for live checks. Tests replace it, so no test touches the network.
OPENER = urllib.request.urlopen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntk",
        description="Retrieval over a large document base where every section names who may read it.",
    )
    parser.add_argument("--version", action="version", version=f"ntk {__version__}")
    commands = parser.add_subparsers(dest="command")
    providers = commands.add_parser(
        "providers",
        help="configured providers, whether each key is set, and DeepSeek's live models and balance",
    )
    providers.add_argument("--offline", action="store_true", help="skip the two live calls to DeepSeek")
    return parser


def cmd_providers(args: argparse.Namespace, out=None) -> int:
    """Exit 0 when the configured DeepSeek model is live and the balance is usable."""
    out = out or sys.stdout
    catalog = load_catalog()
    for name in PROVIDERS:
        entry = catalog[name]
        has_key = "yes" if setting(entry["key_env"]) else "no"
        model = entry.get("model") or "(not set)"
        print(f"{name:<10} key set: {has_key:<4} model: {model:<16} {entry['status']}", file=out)

    deepseek = catalog["deepseek"]
    price = deepseek["models"][deepseek["model"]]["price"]
    print(f"deepseek prices as of {price['as_of']}, from {price['source']}", file=out)
    if args.offline:
        print("live check skipped: --offline", file=out)
        return 0
    key = setting(deepseek["key_env"])
    if not key:
        print(f"live check skipped: {deepseek['key_env']} is not set in the environment or .env", file=out)
        return 1
    try:
        models = deepseek_account.list_models(deepseek["base_url"], key, OPENER)
        available, rows = deepseek_account.balance(deepseek["base_url"], key, OPENER)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"live check failed: {type(error).__name__}", file=out)
        return 1
    listed = deepseek["model"] in models
    print(f"live models: {', '.join(models)}; configured {deepseek['model']} listed: {'yes' if listed else 'no'}", file=out)
    amounts = ", ".join(f"{total} {currency}" for total, currency in rows) or "none reported"
    print(f"balance: {amounts}; usable: {'yes' if available else 'no'}", file=out)
    return 0 if listed and available else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "providers":
        return cmd_providers(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
