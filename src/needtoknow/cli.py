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
    corpus = commands.add_parser("corpus", help="plan the corpus, assemble it from the writers' narrative, check it, "
                                                "measure who may read it, or change it")
    corpus.add_argument("action", choices=("plan", "assemble", "check", "coverage", "mutate", "samples"))
    corpus.add_argument("--count", type=int, default=9, help="mutate: how many changes to apply")
    corpus.add_argument("--seed", type=int, default=1, help="mutate: the seed that picks documents and values")
    ingest = commands.add_parser("ingest", help="cut the assembled corpus into units and record them in the manifest")
    ingest.add_argument("action", choices=("build", "scan"))
    return parser


def cmd_ingest(args: argparse.Namespace, out=None) -> int:
    from needtoknow.ingest.manifest import build, manifest_path, scan
    out = out or sys.stdout
    summary = build() if args.action == "build" else scan()
    changes = ", ".join(f"{k} {n}" for k, n in sorted(summary["changes"].items())) or "none"
    print(f"{summary['kind']} run {summary['run']}: files {summary['files']}, read {summary['files_read']}, changed {summary['files_changed']}, removed "
          f"{summary['files_removed']}; units added {summary['units_added']}, removed {summary['units_removed']}, "
          f"in the manifest {summary['units_total']} ({manifest_path()}); changes by kind: {changes}", file=out)
    if summary["unknown_audiences"]:
        print(f"audience names no label knows: {', '.join(summary['unknown_audiences'])}", file=out)
    return 1 if summary["unknown_audiences"] else 0


def cmd_coverage(out) -> int:
    """Units by label as ingestion reads them, against config/labels.toml (T2.4, KB3-K5, RS-F10)."""
    from needtoknow.ingest.segments import corpus_coverage
    from needtoknow.policy import UNLABELLED, load_labels
    from needtoknow.settings import data_dir
    full = data_dir() / "corpus" / "full"
    if not full.is_dir():
        print(f"no assembled corpus in {full}: run ntk corpus assemble first", file=out)
        return 1
    coverage = corpus_coverage(full)
    units = sum(coverage.first.values())
    layers = ", ".join(f"{layer} {count}" for layer, count in sorted(coverage.layers.items()))
    print(f"units {units} as the units step cuts them ({layers}); by first label, which sums to all "
          f"units, against the registered share, and the share each label can read", file=out)
    registered = load_labels()
    for label in [*registered, UNLABELLED]:
        target = f"{registered[label]['target_share']:.2f}" if label in registered else "none"
        print(f"  {label:<14} first {coverage.first[label] / max(units, 1):.3f}  registered {target}  "
              f"readable {coverage.readable[label] / max(units, 1):.3f}", file=out)
    print(f"audience names no label knows: {', '.join(sorted(set(coverage.unknown))) or 'none'}", file=out)
    for note, count in sorted(coverage.notes.items()):
        print(f"{note}: {count}", file=out)
    return 1 if coverage.unknown else 0


def cmd_corpus(args: argparse.Namespace, out=None) -> int:
    out = out or sys.stdout
    if args.action == "plan":
        from needtoknow.corpus.plan import build, summary
        path = build()
        print(f"plan written to {path}: {summary(path)}", file=out)
        return 0
    if args.action == "assemble":
        from needtoknow.corpus import narrative_dir
        from needtoknow.corpus.assemble import assemble
        from needtoknow.settings import data_dir
        report = assemble()
        if report["missing"]:
            print(f"nothing assembled: {len(report['missing'])} of {report['documents']} documents have no narrative in "
                  f"{narrative_dir(data_dir() / 'corpus')}, such as {', '.join(report['missing'][:3])}. "
                  f"The corpus in place, if any, is unchanged.", file=out)
            return 1
        print(f"assembled {report['written']} files, {report['bytes'] / 2**20:.1f} MB", file=out)
        units = sum(report["units"].values())
        if units:
            shares = ", ".join(f"{label} {report['units'][label] / units:.3f} ({share:.2f})"
                               for label, share in report["registered"].items())
            print(f"units {units}, a prose section or a group of rows as the units step cuts it; share by label, registered in "
                  f"brackets: {shares}", file=out)
        return 0
    if args.action == "samples":
        from needtoknow.corpus.assemble import samples
        from needtoknow.settings import repo_root
        for path in samples(repo_root() / "samples"):
            print(f"{path.name:<28} {path.stat().st_size:>8} bytes", file=out)
        return 0
    if args.action == "mutate":
        from needtoknow.corpus.mutate import mutate
        for entry in mutate(args.count, args.seed):
            print(f"{entry['seq']:>3} {entry['kind']:<17} {entry['doc_id']:<14} {entry['expected']}", file=out)
        return 0
    if args.action == "coverage":
        return cmd_coverage(out)
    from needtoknow.corpus.check import check
    result = check()
    for key, value in result["counts"].items():
        print(f"{key:<26} {value}", file=out)
    for line in result["problems"][:40]:
        print(f"problem  {line}", file=out)
    for line in result["invented_examples"]:
        print(f"value?   {line}", file=out)
    return 0 if not result["problems"] else 1


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
    if args.command == "corpus":
        return cmd_corpus(args)
    if args.command == "ingest":
        return cmd_ingest(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
