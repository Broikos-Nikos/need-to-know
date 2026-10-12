"""The `ntk` command. Each subcommand arrives with the task that needs it."""

from __future__ import annotations

import argparse
import sys
import urllib.request

from needtoknow import __version__, doctor
from needtoknow.index import server
from needtoknow.providers import deepseek_account
from needtoknow.providers.catalog import CatalogError, load_catalog
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
    qdrant = commands.add_parser("qdrant", help="start, stop or check the local Qdrant server")
    qdrant.add_argument("action", choices=("start", "stop", "status"))
    checkup = commands.add_parser("doctor", help="whether this machine can build and search the index, one line per check")
    checkup.add_argument("--offline", action="store_true", help="skip the live calls to DeepSeek")
    return parser


def cmd_qdrant(args: argparse.Namespace, out=None) -> int:
    """Exit 0 when the action succeeded; for status, when the server is running and ready."""
    out = out or sys.stdout
    try:
        config = server.load_config()
        where = f"{config['host']}:{config['http_port']}"
        if args.action == "start":
            pid, seconds, created = server.start(config)
            if created:
                print(f"wrote a new {config['key_env']} to .env", file=out)
            print(f"qdrant {config['version']} started as pid {pid} on {where}, ready in {seconds:.2f} s", file=out)
            return 0
        if args.action == "stop":
            pid = server.stop(config)
            print(f"stopped pid {pid}" if pid else "not running", file=out)
            return 0
        pid = server.running(config)
        ready = server.http_status(config, "/readyz") == 200
        print(f"running as pid {pid} on {where}, ready: {'yes' if ready else 'no'}" if pid else "not running", file=out)
        return 0 if pid and ready else 1
    except server.ServerError as error:
        print(f"qdrant: {error}", file=out)
        return 1


def cmd_providers(args: argparse.Namespace, out=None) -> int:
    """Exit 0 when the configured DeepSeek model is live and the balance is usable."""
    out = out or sys.stdout
    try:
        catalog = load_catalog()
    except CatalogError as error:
        print(f"providers: {error}", file=out)
        return 1
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
    passed, lines = deepseek_account.live_check(deepseek["base_url"], deepseek["model"], key, OPENER)
    for line in lines:
        print(line, file=out)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    # Half the corpus is Greek, and a pipe on Windows defaults to the locale's code
    # page, which cannot encode it (HS-F4): every command writes UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "providers":
        return cmd_providers(args)
    if args.command == "qdrant":
        return cmd_qdrant(args)
    if args.command == "doctor":
        return doctor.run(offline=args.offline, opener=OPENER)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
