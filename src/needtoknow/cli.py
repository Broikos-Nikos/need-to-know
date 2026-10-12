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
    ingest.add_argument("action", choices=("build", "scan", "watch"))
    ingest.add_argument("--interval", type=float, default=1.0, help="watch: seconds between looks at the corpus folder")
    ingest.add_argument("--manifest-only", action="store_true",
                        help="watch: keep the manifest current and leave the index alone")
    index = commands.add_parser("index", help="create the Qdrant collection, or bring it to the manifest")
    index.add_argument("action", choices=("create", "sync"))
    find = commands.add_parser("search", help="the candidates a role's question finds, fused, best first")
    find.add_argument("question")
    find.add_argument("--as", dest="role", required=True, help="a role of config/policy.toml")
    find.add_argument("--form", choices=("labels", "label_set"), default="labels", help="the filter's form (KB1-F3)")
    find.add_argument("--limit", type=int, default=10)
    find.add_argument("--rerank", choices=("none", "multivector", "cross"), help="the second stage; config/models.toml "
                                                                                 "sets the default")
    embed = commands.add_parser("embed", help="embed every unit of the manifest whose passage the vector cache lacks")
    embed.add_argument("--budget", type=int, help="padded tokens per batch; config/models.toml sets the default")
    return parser


def report_run(summary: dict, out) -> None:
    from needtoknow.ingest.manifest import manifest_path
    if "error" in summary:
        print(f"ingest: {summary['error']}", file=out, flush=True)
        return
    if "upserted" in summary:
        print(f"index: deleted {summary['deleted']}, embedded {summary['embedded']}, upserted {summary['upserted']}; "
              f"points {summary['points_after']} for {summary['units']} units"
              + (f"; frequency table recounted from {summary['recounted']} points" if summary.get("recounted") else ""),
              file=out, flush=True)
        return
    changes = ", ".join(f"{k} {n}" for k, n in sorted(summary["changes"].items())) or "none"
    print(f"{summary['kind']} run {summary['run']}: files {summary['files']}, read {summary['files_read']}, "
          f"changed {summary['files_changed']}, removed {summary['files_removed']}; units added {summary['units_added']}, "
          f"removed {summary['units_removed']}, in the manifest {summary['units_total']} ({manifest_path()}); "
          f"changes by kind: {changes}", file=out, flush=True)
    if summary["unknown_audiences"]:
        print(f"audience names no label knows: {', '.join(summary['unknown_audiences'])}", file=out, flush=True)


def cmd_ingest(args: argparse.Namespace, out=None) -> int:
    from needtoknow.ingest.manifest import ManifestError, build, scan
    from needtoknow.settings import data_dir
    out = out or sys.stdout
    if args.action == "watch":
        from needtoknow.ingest.watch import watch
        full = data_dir() / "corpus" / "full"
        print(f"watching {full} every {args.interval:g} s; Ctrl+C stops", file=out, flush=True)

        def sync_index(doc_ids: list[str] | None) -> dict:
            from needtoknow.index.sync import sync
            return sync(server.client(), doc_ids=doc_ids)

        try:
            watch(full, interval=args.interval, report=lambda summary: report_run(summary, out),
                  sync_fn=None if args.manifest_only else sync_index)
        except KeyboardInterrupt:
            pass
        return 0
    try:
        summary = build() if args.action == "build" else scan()
    except ManifestError as error:
        report_run({"error": str(error)}, out)
        return 1
    report_run(summary, out)
    return 1 if summary["unknown_audiences"] else 0


def cmd_embed(args: argparse.Namespace, out=None) -> int:
    from needtoknow.embed.cache import CacheError, fill
    from needtoknow.embed.model import EmbedError
    out = out or sys.stdout
    try:
        summary = fill(budget=args.budget, report=lambda line: print(line, file=out, flush=True))
    except (EmbedError, CacheError) as error:
        print(f"embed: {error}", file=out, flush=True)
        return 1
    rate = f", {summary['tokens'] / summary['seconds']:,.0f} tokens/s" if summary["seconds"] else ""
    print(f"units {summary['units']}; passages cached before {summary['cached']}, missing {summary['missing']}, "
          f"embedded {summary['embedded']}: {summary['tokens']} tokens in {summary['seconds']:.0f} s{rate}; "
          f"{summary['model']} in {summary['path']}", file=out, flush=True)
    return 0


def cmd_index(args: argparse.Namespace, out=None, client=None) -> int:
    """Create the collection, freezing the lexical statistics over the manifest, or check the one there (T3.3);
    then, for sync, delete what the manifest no longer explains and upsert what the collection lacks (T3.4)."""
    import json

    from needtoknow.index.collection import COLLECTION, CollectionError, ensure
    from needtoknow.ingest.manifest import opened
    from needtoknow.ingest.units import passage
    out = out or sys.stdout
    try:
        client = client or server.client()
        with opened() as db:
            result = ensure(client, lambda: (passage(json.loads(p), t) for p, t in db.execute("SELECT path, text FROM units")))
    except (CollectionError, server.ServerError) as error:
        print(f"index: {error}", file=out, flush=True)
        return 1
    stats = result["lexical"]
    print(f"{COLLECTION}: {'created' if result['created'] else 'already there, and of this shape'}; lexical statistics "
          f"frozen over {stats['passages']} passages, avgdl {stats['avgdl']:.2f}", file=out, flush=True)
    if getattr(args, "action", "create") == "sync":
        from needtoknow.index.sync import sync
        done = sync(client, report=lambda line: print(line, file=out, flush=True))
        print(f"synced: units {done['units']}, points {done['points_before']} before and {done['points_after']} after; "
              f"deleted {done['deleted']}, embedded {done['embedded']}, upserted {done['upserted']} in {done['seconds']:.1f} s "
              f"(" + ", ".join(f"{part} {seconds:.1f}" for part, seconds in done["parts"].items()) + ")",
              file=out, flush=True)
        return 0 if done["points_after"] == done["units"] else 1
    return 0


def cmd_search(args: argparse.Namespace, out=None, client=None) -> int:
    """A role's question through the retrieval API: rank, fused score, document, labels and the start of the unit."""
    import json

    from needtoknow.index.search import search
    from needtoknow.ingest.manifest import opened
    from needtoknow.policy import PolicyError
    out = out or sys.stdout
    try:
        hits = search(client or server.client(), args.question, args.role, form=args.form,
                      rerank=getattr(args, "rerank", None))[:args.limit]
    except (PolicyError, server.ServerError) as error:
        print(f"search: {error}", file=out, flush=True)
        return 1
    with opened() as db:
        for hit in hits:
            doc_id, labels, text = db.execute("SELECT doc_id, labels, text FROM units WHERE unit_id = ?",
                                              (hit["unit_id"],)).fetchone()
            line = " ".join(text.split())
            print(f"{hit['rank']:>3} {hit['score']:.4f} {doc_id:<12} {'+'.join(json.loads(labels)):<14} {line[:90]}",
                  file=out, flush=True)
    return 0


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
    if len(result["problems"]) > 40:
        print(f"problem  and {len(result['problems']) - 40} more", file=out)
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
    if args.command == "embed":
        return cmd_embed(args)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "search":
        return cmd_search(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
