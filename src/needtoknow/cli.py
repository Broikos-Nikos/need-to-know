"""The `ntk` command. Each subcommand arrives with the task that needs it."""

from __future__ import annotations

import argparse
import sys
import time
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
    corpus.add_argument("action", choices=("plan", "assemble", "check", "coverage", "mutate", "samples", "draw"))
    corpus.add_argument("--count", type=int, default=9, help="mutate: how many changes to apply")
    corpus.add_argument("--seed", type=int, default=1, help="mutate: the seed that picks documents and values; draw: "
                                                            "the seed the values and annexes are drawn under")
    corpus.add_argument("--every", type=int, default=4, help="draw: one document in this many of the catalogue, 1 or "
                                                             "more")
    ingest = commands.add_parser("ingest", help="cut the assembled corpus into units and record them in the manifest")
    ingest.add_argument("action", choices=("build", "scan", "watch"))
    # Half a second: a change waits two intervals before its scan, and a look costs 9 ms (N53).
    ingest.add_argument("--interval", type=ranged(float, 0.1, 60.0), default=0.5,
                        help="watch: seconds between looks at the corpus folder, 0.1 to 60")
    ingest.add_argument("--manifest-only", action="store_true",
                        help="watch: keep the manifest current and leave the index alone")
    index = commands.add_parser("index", help="create the Qdrant collection, or bring it to the manifest")
    index.add_argument("action", choices=("create", "sync"))
    index.add_argument("--replace", action="store_true", help="delete the collection, its frequency table and its "
                                                              "lexical statistics first; the vector cache stays")
    index.add_argument("--force", action="store_true", help="sync: delete even when more than half of the collection "
                                                            "would go, which a sync from another data folder does")
    find = commands.add_parser("search", help="the candidates a role's question finds, fused, best first")
    find.add_argument("question")
    find.add_argument("--as", dest="role", required=True, help="a role of config/policy.toml")
    find.add_argument("--form", choices=("labels", "label_set"), default="labels", help="the filter's form (KB1-F3)")
    find.add_argument("--limit", type=ranged(int, 1, LIMIT), default=10, help=f"candidates to print, 1 to {LIMIT}")
    find.add_argument("--rerank", choices=("none", "multivector", "cross"), help="the second stage; config/models.toml "
                                                                                 "sets the default")
    reply = commands.add_parser("ask", help="a role's question answered from what it may read, with its citations, "
                                            "what the call cost and how long each stage took")
    reply.add_argument("question")
    reply.add_argument("--as", dest="role", required=True,
                       help="a role of config/policy.toml, on the caller's word: ntk is the operator's tool (AC-F7)")
    reply.add_argument("--provider", choices=("deepseek", "openai", "anthropic", "gemini"), default="deepseek",
                       help="the provider of config/providers.toml to ask; only DeepSeek has been run")
    reply.add_argument("--no-cache", action="store_true", help="ask the provider even when the cache holds the answer")
    evaluation = commands.add_parser("eval", help="draw the fact families' split by the protocol, or check the golden "
                                                  "items against it")
    evaluation.add_argument("action", choices=("split", "check", "draft", "rows", "twins", "score", "probes",
                                               "canaries", "run"))
    evaluation.add_argument("answers", nargs="?", help="score: a JSON lines file of answers, one per item, as parse "
                                                       "gives them with the item's id")
    evaluation.add_argument("--check", action="store_true", help="split, probes: make again and compare with the "
                                                                   "committed files")
    evaluation.add_argument("--rewrite", action="store_true",
                            help="split: write over eval/families.json although the draw differs from it")
    evaluation.add_argument("--purpose", choices=("development", "baseline", "experiment", "parity", "release"),
                            help="score: what the scores are for; the protocol refuses the test split for a purpose "
                                 "that chooses anything")
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev",
                            help="draft: the split whose families to show, planted sentences masked; run, score: "
                                 "the split asked or scored")
    evaluation.add_argument("--out", help="run: the JSON lines file the answers go to; one that exists is resumed")
    evaluation.add_argument("--repeats", type=int, help="run: how many times each item is asked; the protocol's "
                                                         "baseline count if not given")
    evaluation.add_argument("--items", type=int, help="run: ask only the first this many items, a smoke run")
    evaluation.add_argument("--closed-book", action="store_true",
                            help="run: ask with no excerpts under the answer rules, which leave only a refusal: the "
                                 "check that they hold with no documents")
    evaluation.add_argument("--guess", action="store_true",
                            help="run: ask with no excerpts for the model's best guess from its own knowledge, the "
                                 "control KB1-F7 asks for (DR-F27)")
    embed = commands.add_parser("embed", help="embed every unit of the manifest whose passage the vector cache lacks")
    embed.add_argument("--budget", type=int, help="padded tokens per batch; config/models.toml sets the default")
    commands.add_parser("stats", help="what the build holds: documents, their bytes, units, vectors and the index's "
                                      "size on disk")
    serve = commands.add_parser("serve", help="the local server, on 127.0.0.1 alone: a role signs in once, and each "
                                              "search answers with its stage trace")
    serve.add_argument("--port", type=ranged(int, 1024, 65535), help="the port to listen on; the server's own when "
                                                                     "not given")
    return parser


LIMIT = 100  # the search's CANDIDATES: no more come back to print


def ranged(kind, low, high):
    """An argument type that refuses a value outside low to high, by name, instead of taking it (HS-F12)."""
    def parse(text: str):
        value = kind(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{text} is outside {low:g} to {high:g}")
        return value
    parse.__name__ = kind.__name__
    return parse


def server_down(error: Exception, out, command: str) -> bool:
    """Whether an error came from the index's server not running, said in one line when it did (HS-F11)."""
    down = server.stopped()
    if down:
        print(f"{command}: the index's server is not running, so nothing reached it: ntk qdrant start "
              f"({type(error).__name__})", file=out, flush=True)
    return down


def report_run(summary: dict, out) -> None:
    from needtoknow.ingest.manifest import manifest_path
    if "error" in summary:
        print(f"ingest: {summary['error']}", file=out, flush=True)
        return
    if "upserted" in summary:
        parts = ", ".join(f"{part} {seconds:.2f}" for part, seconds in summary.get("parts", {}).items())
        print(f"index: deleted {summary['deleted']}, embedded {summary['embedded']}, upserted {summary['upserted']}; "
              f"points {summary['points_after']} for {summary['units']} units"
              + (f"; frequency table recounted from {summary['recounted']} points" if summary.get("recounted") else "")
              + f"; in {summary['seconds']:.2f} s ({parts})", file=out, flush=True)
        return
    changes = ", ".join(f"{k} {n}" for k, n in sorted(summary["changes"].items())) or "none"
    print(f"{summary['kind']} run {summary['run']}: files {summary['files']}, read {summary['files_read']}, "
          f"changed {summary['files_changed']}, removed {summary['files_removed']}; units added {summary['units_added']}, "
          f"removed {summary['units_removed']}, in the manifest {summary['units_total']} ({manifest_path()}); "
          f"changes by kind: {changes}; in {summary.get('seconds', 0.0):.2f} s", file=out, flush=True)
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

        if not args.manifest_only:  # an edited paragraph is embedded again: load the encoder now, not at the first edit
            try:
                from needtoknow.embed import model
                began = time.perf_counter()
                model.load()
                print(f"encoder loaded in {time.perf_counter() - began:.1f} s", file=out, flush=True)
            except Exception as error:  # no embed extra, no GPU: the first edit pays the load, or says why it cannot
                print(f"encoder not loaded, so the first edit loads it: {type(error).__name__}: {error}", file=out,
                      flush=True)
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

    from needtoknow.index.collection import COLLECTION, CollectionError, drop, ensure
    from needtoknow.lexical import LexicalError
    rebuild = "; ntk index create --replace rebuilds them with the collection, keeping the vector cache"
    from needtoknow.ingest.manifest import opened
    from needtoknow.ingest.units import passage
    from needtoknow.embed.model import EmbedError
    out = out or sys.stdout
    try:
        client = client or server.client()
        if getattr(args, "replace", False) and drop(client):
            print(f"{COLLECTION} deleted, to be built again; the vector cache kept", file=out, flush=True)
        with opened() as db:
            result = ensure(client, lambda: (passage(json.loads(p), t) for p, t in db.execute("SELECT path, text FROM units")))
    except (CollectionError, server.ServerError) as error:
        print(f"index: {error}", file=out, flush=True)
        return 1
    except LexicalError as error:  # said here, so lexical.py, whose bytes are the analyser's version, stays (MT-F7)
        print(f"index: {error}{rebuild}", file=out, flush=True)
        return 1
    except Exception as error:
        if server_down(error, out, "index"):
            return 1
        raise
    stats = result["lexical"]
    print(f"{COLLECTION}: {'created' if result['created'] else 'already there, and of this shape'}; lexical statistics "
          f"frozen over {stats['passages']} passages, avgdl {stats['avgdl']:.2f}", file=out, flush=True)
    if getattr(args, "action", "create") == "sync":
        from needtoknow.index.sync import SyncError, sync
        try:
            done = sync(client, report=lambda line: print(line, file=out, flush=True),
                        force=getattr(args, "force", False))
        except SyncError as error:  # most of the served index would go: asked before, never done (MT-F18)
            print(f"index: {error}", file=out, flush=True)
            return 1
        except EmbedError as error:
            print(f"index: {error}", file=out, flush=True)
            return 1
        except LexicalError as error:
            print(f"index: {error}{rebuild}", file=out, flush=True)
            return 1
        except Exception as error:
            if server_down(error, out, "index"):
                return 1
            raise
        print(f"synced: units {done['units']}, points {done['points_before']} before and {done['points_after']} after; "
              f"deleted {done['deleted']}, embedded {done['embedded']}, upserted {done['upserted']} in {done['seconds']:.1f} s "
              f"(" + ", ".join(f"{part} {seconds:.1f}" for part, seconds in done["parts"].items()) + ")",
              file=out, flush=True)
        return 0 if done["points_after"] == done["units"] else 1
    return 0


def cmd_search(args: argparse.Namespace, out=None, client=None) -> int:
    """A role's question through the retrieval API: rank, fused score, document, labels and the start of the unit."""
    import json

    from needtoknow.embed.model import EmbedError
    from needtoknow.index.search import QuestionError, search
    from needtoknow.ingest.manifest import opened
    from needtoknow.policy import PolicyError
    out = out or sys.stdout
    try:
        hits = search(client or server.client(), args.question, args.role, form=args.form,
                      rerank=getattr(args, "rerank", None))[:args.limit]
    except (PolicyError, server.ServerError, QuestionError, EmbedError) as error:
        print(f"search: {error}", file=out, flush=True)
        return 1
    except Exception as error:
        if server_down(error, out, "search"):
            return 1
        raise
    with opened() as db:
        for hit in hits:
            row = db.execute("SELECT doc_id, labels, text FROM units WHERE unit_id = ?", (hit["unit_id"],)).fetchone()
            if row is None:  # a scan between the search and this line dropped it (DR-F15)
                continue
            doc_id, labels, text = row
            line = " ".join(text.split())
            print(f"{hit['rank']:>3} {hit['score']:.4f} {doc_id:<12} {'+'.join(json.loads(labels)):<14} {line[:90]}",
                  file=out, flush=True)
    if len(hits) < args.limit:  # said, so a short list is not read as a cut (HS-F12)
        print(f"{len(hits)} candidates in all, fewer than the {args.limit} asked", file=out, flush=True)
    return 0


def cmd_ask(args: argparse.Namespace, out=None, client=None) -> int:
    """A role's question answered from what it may read (T4.5): the answer, its citations, any citation no excerpt could
    take (RS-F14), and the model, the tokens, the cost and each stage's seconds. The role is the caller's word, as for
    search: a served path takes it from the session, never from the request (AC-F7)."""
    import json

    from needtoknow.answer.pipeline import ask
    from needtoknow.embed.model import EmbedError
    from needtoknow.index.search import QuestionError
    from needtoknow.policy import PolicyError
    from needtoknow.providers.call import ProviderError
    out = out or sys.stdout
    try:
        result = ask(client or server.client(), args.question, args.role, provider=args.provider,
                     use_cache=not args.no_cache)
    except (PolicyError, server.ServerError, QuestionError, EmbedError, ProviderError) as error:
        print(f"ask: {error}", file=out, flush=True)
        return 1
    except Exception as error:
        if server_down(error, out, "ask"):
            return 1
        raise
    if result.get("cache_set_aside"):  # HS-F17
        print(f"the answer cache could not be read and was set aside as {result['cache_set_aside']}, which can be "
              "deleted; this answer did without it", file=out)
    if result.get("cache_busy"):  # DR-F31
        print("the answer cache was held by another writer past its wait, so this answer did without it and left "
              "it as it was", file=out)
    if result["status"] == "invalid":
        print(f"the reply could not be read: {result['error']}", file=out)
    else:
        print(f"{result['status']}: {result['answer']}", file=out)
    for number, citation in enumerate(result["citations"], 1):
        print(f"  [{number}] {citation['doc_id']}, {citation['section']}", file=out)
    if result["dropped"]:
        print(f"  cited, but no excerpt shown could take it: {', '.join(json.dumps(d) for d in result['dropped'])}",
              file=out)
    used, took = result["usage"], result["seconds"]
    # How many tokens the provider had cached is kept in the spend log, which only the operator reads: printed, it
    # would tell a role that someone asked over the same text (KB5-K12).
    source = ("from the cache" if result["cached"] else
              f"{used['input_hit'] + used['input_miss']} tokens in, {used['output']} out")
    cost = "no price configured" if result["cost_usd"] is None else f"{result['cost_usd']:.6f} USD"
    load = f"the models loaded in {took['load']:.1f} s, then " if took.get("load", 0.0) >= 0.05 else ""
    print(f"{result['model']}, {result['stop_reason']}; {source}; {cost}; {load}search {took['search']:.2f} s, "
          f"provider {took['provider']:.2f} s, in all {took['total']:.2f} s", file=out, flush=True)
    return 1 if result["status"] == "invalid" else 0


def cmd_eval(args: argparse.Namespace, out=None) -> int:
    """The fact families' split (T5.1b, N56): written to eval/families.json, or checked against it; once written, never
    written over by a draw that differs from it unless asked (MT-F12). Also the writer's view and the table rows."""
    import json

    from needtoknow.eval.protocol import ProtocolError
    from needtoknow.eval.split import SplitError, split, split_path, write
    out = out or sys.stdout
    if args.action == "check":
        return cmd_eval_check(out)
    if args.action == "draft":
        from needtoknow.eval.golden import Context, GoldenError, draft_view
        try:
            lines = draft_view(args.split, Context())
        except (GoldenError, ProtocolError, SplitError) as error:
            print(f"eval draft: {error}", file=out, flush=True)
            return 1
        print("\n".join(lines), file=out, flush=True)
        return 0
    if args.action == "score":
        return cmd_eval_score(args, out)
    if args.action == "probes":
        return cmd_eval_probes(args, out)
    if args.action == "canaries":
        return cmd_eval_canaries(out)
    if args.action == "run":
        return cmd_eval_run(args, out)
    if args.action == "twins":
        from needtoknow.eval.golden import GoldenError, draw_twins, golden_path, load_items
        from needtoknow.eval.protocol import load_protocol
        try:
            dev, test = load_items(golden_path("dev")), load_items(golden_path("test"))
            drawn = draw_twins(dev, test, load_protocol()["items"]["paraphrase_twins"])
        except (GoldenError, ProtocolError) as error:
            print(f"eval twins: {error}", file=out, flush=True)
            return 1
        by_id = {item["id"]: item for item in dev}
        for name in drawn:
            item = by_id[name]
            print(f"{name} {item['category']} {','.join(item.get('families') or []) or '-'} {item['role']} {item['lang']}: "
                  f"{item['question']}", file=out)
        print(f"{len(drawn)} items drawn for a twin", file=out, flush=True)
        return 0
    if args.action == "rows":
        from needtoknow.eval.golden import GoldenError, draw_rows
        try:
            rows = draw_rows()
        except GoldenError as error:
            print(f"eval rows: {error}", file=out, flush=True)
            return 1
        print("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), file=out, flush=True)
        return 0
    try:
        result = split()
    except (SplitError, ProtocolError) as error:
        print(f"eval split: {error}", file=out, flush=True)
        return 1
    gaps = ", ".join(f"{name} {gap:+.3f}" for name, gap in result["draws"][-1]["smd"].items())
    line = (f"{result['families']} families from {result['registry_families']} registry families: "
            f"{result['test_families']} test, {result['families'] - result['test_families']} dev; balanced at draw "
            f"{len(result['draws'])}, seed {result['seed']}: {gaps}, each under {result['limit']}")
    if args.check:
        path = split_path()
        same = path.is_file() and json.loads(path.read_text(encoding="utf-8")) == result
        print(line + ("; eval/families.json matches it" if same else
                      "; eval/families.json does not match it: ntk eval split writes it"), file=out, flush=True)
        return 0 if same else 1
    path = split_path()
    if path.is_file() and not args.rewrite:
        committed = json.loads(path.read_text(encoding="utf-8"))
        if committed != result:  # the items were written against the registered split (MT-F12)
            was = {family["id"]: family["split"] for family in committed["split"]}
            moved = sorted(family["id"] for family in result["split"] if was.get(family["id"]) != family["split"])
            print(line + f"; it differs from the registered eval/families.json, {len(moved)} families changing split"
                  + (f" ({', '.join(moved)})" if moved else "") + ": not written. --rewrite writes it, and the "
                  "change belongs in eval/protocol_history.toml", file=out, flush=True)
            return 1
    write(result)
    print(line + "; written to eval/families.json", file=out, flush=True)
    return 0


def cmd_eval_run(args: argparse.Namespace, out) -> int:
    """Every item of a split asked as its role, each repeat a fresh call (T5.5c); the protocol asked first."""
    from pathlib import Path

    from needtoknow.eval.golden import GoldenError
    from needtoknow.eval.protocol import ProtocolError, load_protocol
    from needtoknow.eval.run import RunError, run
    from needtoknow.index import server
    from needtoknow.lines import LineError
    if not args.out or not args.purpose:
        print("eval run: give --out and --purpose", file=out, flush=True)
        return 1
    repeats = args.repeats or load_protocol()["baseline"]["repeats"]
    try:
        summary = run(args.split, args.purpose, repeats, Path(args.out), server.client(), args.closed_book,
                      report=lambda line: print(line, file=out, flush=True), first=args.items, guess=args.guess)
    except (GoldenError, ProtocolError, RunError, LineError) as error:
        print(f"eval run: {error}", file=out, flush=True)
        return 1
    book = ", the best guess with no excerpts" if args.guess else ", closed book" if args.closed_book else ""
    mended = "; a last line a stopped write left was cut off and asked again" if summary["mended"] else ""
    print(f"{summary['items']} items, {summary['repeats']} repeats{book}: {summary['asked']} asked now, "
          f"{summary['kept']} already in {args.out}{mended}; {summary['usd']:.6f} USD; answered by "
          f"{', '.join(summary['models']) or 'no call'}", file=out, flush=True)
    return 0


def cmd_eval_canaries(out) -> int:
    """Every spend log line, every answer the cache holds and every recorded page trace, searched for canaries of
    labels the asking role may not read (T5.4e); exits 1 on any, and on a scan that found nothing to read (AC-F15)."""
    from needtoknow.answer.cache import cache_path
    from needtoknow.answer.pipeline import spend_path
    from needtoknow.canaries import ScanError, scan
    from needtoknow.lines import LineError
    try:
        report = scan(spend_path(), cache_path())
    except (ScanError, LineError) as error:
        print(f"eval canaries: {error}", file=out, flush=True)
        return 1
    for leak in report["leaks"]:
        print(f"leak: {' '.join(leak.split())}", file=out)  # one line each, a canary a line break split too
    print(f"{report['lines']} spend log lines, {report['answers']} cached answers read through them, "
          f"{report['unnamed']} cached answers no line names, {report['traces']} recorded page traces; canaries of a "
          f"label the role may not read: {len(report['leaks'])}", file=out, flush=True)
    if not (report["lines"] or report["unnamed"] or report["traces"]):  # a scan of nothing clears nothing (AC-F15)
        print("eval canaries: nothing to scan, no spend log line, no cached answer and no recorded page trace; a "
              "run that asked anything leaves them", file=out, flush=True)
        return 1
    return 1 if report["leaks"] else 0


def cmd_eval_probes(args: argparse.Namespace, out) -> int:
    """The retrieval probes (T5.4b): two a fact from each split's own families, written to eval/probes/, or made
    again and compared; the share sharing none, one or more of their planted sentence's rarest terms printed (RS-F4)."""
    from collections import Counter

    from needtoknow.eval import probes as P
    from needtoknow.eval.golden import Context, GoldenError
    from needtoknow.eval.protocol import ProtocolError
    from needtoknow.eval.split import SplitError
    try:
        ctx = Context()
        made = P.generate(ctx)
    except (GoldenError, ProtocolError, SplitError) as error:
        print(f"eval probes: {error}", file=out, flush=True)
        return 1
    for split, rows in made.items():
        shared = ", ".join(f"{n} sharing {k}" for k, n in sorted(Counter(r["shared_rare"] for r in rows).items()))
        print(f"{split}: {len(rows)} probes over {len({r['fact_id'] for r in rows})} facts; of their planted sentences' "
              f"rarest terms, {shared}", file=out)
    wrong = P.problems(made, ctx)
    for problem in wrong:
        print(f"  {problem}", file=out)
    if args.check:
        same = all(P.load(split) == rows for split, rows in made.items())
        print("eval/probes/ matches" if same else "eval/probes/ does not match: ntk eval probes writes it", file=out,
              flush=True)
        return 0 if same and not wrong else 1
    if wrong:
        print(f"{len(wrong)} problems: not written", file=out, flush=True)
        return 1
    P.write(made)
    print("written to eval/probes/", file=out, flush=True)
    return 0


def cmd_eval_score(args: argparse.Namespace, out) -> int:
    """A run's answers scored against one split's items (T5.2b): the four figures apart, then partial credit, the
    citation measures and every figure by category, layer and language, with the scorer's hash. The protocol is asked
    first, so the test split serves no purpose that chooses anything."""
    from pathlib import Path

    from needtoknow.eval.golden import Context, GoldenError, golden_path, load_items
    from needtoknow.eval.protocol import ProtocolError, check_use
    from needtoknow.eval.run import answers as run_answers
    from needtoknow.eval.score import over_repeats, score
    from needtoknow.eval.split import SplitError
    from needtoknow.lines import LineError
    if not args.answers or not args.purpose:
        print("eval score: give the answers file and --purpose", file=out, flush=True)
        return 1
    if not Path(args.answers).is_file():  # read as empty, it scored every item unanswered (DR-F24)
        print(f"eval score: no answers file at {args.answers}", file=out, flush=True)
        return 1
    try:
        check_use(args.split, args.purpose)
        answers = run_answers(Path(args.answers))  # each line what a run writes, by file and line (HS-F21)
        items = load_items(golden_path(args.split))
        repeats = sorted({answer.get("repeat", 0) for answer in answers})
        if len(repeats) > 1:
            return print_repeats(over_repeats(items, answers, Context()), args, out)
        results, figures = score(items, answers, Context())
    except (GoldenError, ProtocolError, SplitError, LineError, OSError) as error:
        print(f"eval score: {error}", file=out, flush=True)
        return 1

    def line(name: str, rate: dict) -> str:
        share = "no items" if rate["rate"] is None else f"{rate['rate']:.3f}"
        return f"{name}: {rate['correct']} of {rate['items']}, {share}"

    print(line("answerable accuracy", figures["answerable"]), file=out)
    print(line("refusal correctness", figures["refusals"]), file=out)
    print(f"leaks: {figures['leaks']}; a forbidden value a readable fact also states, not a leak: "
          f"{figures['stated_elsewhere']}", file=out)
    print(line("overall accuracy", figures["overall"]), file=out)
    print(line("grounded accuracy", figures["grounded"]), file=out)
    if figures["twins"]:
        twins = figures["twins"]
        gap = "none" if twins["gap"] is None else f"{twins['gap']:+.3f}"
        print(f"twins: {line('reworded', twins['twins'])}; {line('their originals', twins['originals'])}; gap {gap}",
              file=out)
    for name in ("partial", "citation_precision", "citation_recall"):
        value = figures[name]
        print(f"{name.replace('_', ' ')}: {'none' if value is None else f'{value:.3f}'}", file=out)
    for axis, groups in figures["by"].items():
        print(f"by {axis}: " + "; ".join(f"{key} {g['correct']}/{g['items']}" for key, g in groups.items()), file=out)
    print(f"{len(results)} items scored, {args.split} split, for {args.purpose}; scorer {figures['scorer'][:16]}",
          file=out, flush=True)
    return 0


def print_repeats(run: dict, args: argparse.Namespace, out) -> int:
    """A run's repeats: each one's four figures, then the means over items of each item's mean, and the unstable."""
    for repeat, figures in zip(run["repeats"], run["figures"]):
        parts = [f"{name} {figures[name]['correct']} of {figures[name]['items']}"
                 for name in ("answerable", "refusals", "overall")]
        print(f"repeat {repeat}: " + ", ".join(parts) + f", leaks {figures['leaks']}", file=out)
    for name in ("answerable", "refusals", "overall", "grounded"):
        share = "no items" if run[name]["rate"] is None else f"{run[name]['rate']:.3f}"
        print(f"{name}, each item's mean over {len(run['repeats'])} repeats: {share} over {run[name]['items']} items",
              file=out)
    for axis, groups in run["by"].items():
        print(f"by {axis}, means: " + "; ".join(f"{key} {g['rate']:.3f} of {g['items']}" for key, g in groups.items()
                                               if g["rate"] is not None), file=out)
    print(f"unstable, a verdict that changed between repeats: {len(run['unstable'])}"
          + (f" ({', '.join(run['unstable'])})" if run["unstable"] else "") + f"; leaks in all repeats: {run['leaks']}",
          file=out)
    print(f"{args.split} split, for {args.purpose}; scorer {run['figures'][0]['scorer'][:16]}", file=out, flush=True)
    return 0


def cmd_eval_check(out) -> int:
    """The golden items against their categories' rules and the protocol's counts (T5.1c2), one line a problem."""
    import sqlite3

    from needtoknow.eval.golden import Context, GoldenError, check
    from needtoknow.eval.protocol import ProtocolError
    from needtoknow.eval.split import SplitError
    try:
        ctx = Context()
        summary, problems = check(ctx)
    except (GoldenError, ProtocolError, SplitError, sqlite3.OperationalError) as error:
        print(f"eval check: {error}", file=out, flush=True)
        return 1
    for split, seen in summary.items():
        categories = " ".join(f"{code} {count}" for code, count in seen["categories"].items()) or "none"
        layers = ", ".join(f"{name} {count}" for name, count in sorted(seen["layers"].items())) or "none"
        spared = f"; none for {', '.join(seen['exempt'])} (N58)" if seen.get("exempt") else ""
        print(f"{split}: {seen['items']} items, {seen['greek']} in Greek, {seen['family_items']} on a family{spared}; "
              f"by category {categories}; by layer {layers}", file=out)
    for problem in problems:
        print(f"  {problem}", file=out)
    print(f"{len(problems)} problems" if problems else
          f"every item keeps its rules, and the counts are version {ctx.protocol['version']}'s", file=out, flush=True)
    return 1 if problems else 0


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
    """A corpus action; one that needs the plan where none was made says so in one line, where it was a traceback
    (HS-F22)."""
    from needtoknow.corpus import NoPlan
    from needtoknow.lines import LineError
    out = out or sys.stdout
    try:
        return corpus_action(args, out)
    except (NoPlan, LineError) as error:
        print(f"corpus {args.action}: {error}", file=out, flush=True)
        return 1


def corpus_action(args: argparse.Namespace, out) -> int:
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
    if args.action == "draw":
        from needtoknow.corpus.draw import DrawError, draw
        try:
            report = draw(args.seed, args.every)
        except (DrawError, OSError) as error:
            print(f"corpus draw: {error}", file=out, flush=True)
            return 1
        # Counted over the drawn documents alone, never the plan's facts beside their sentences (DR-F29).
        print(f"draw {report['seed']}: {report['documents']} documents, one in {args.every}, planting "
              f"{report['facts']} facts, {report['values']} of them drawn a new value; {report['sentences']} planted "
              f"sentences rewritten; {report['bytes']:,} bytes in draws/{report['seed']}, a data folder of its own",
              file=out, flush=True)
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


def cmd_stats(out=None, client=None) -> int:
    """What the build holds, PLAN section 1's number 6 bar the build's time, which bench/full_build.py measures: the
    documents, their bytes and units from the manifest, the collection's vectors from the server, and on disk the
    index's storage, the manifest and the vector cache, as `index.footprint` reads them for the server too."""
    from needtoknow.index.footprint import NoBuild, footprint, vectors
    out = out or sys.stdout
    try:
        held = footprint()
    except NoBuild as error:
        print(f"stats: {error}", file=out, flush=True)
        return 1
    try:
        counted = f"{vectors(client or server.client()):,} vectors in {held['collection']}"
    except Exception as error:
        if not server_down(error, out, "stats"):
            raise
        counted = "the vectors not counted"
    disk = held["disk"]
    print(f"{held['documents']} documents, {held['bytes']:,} bytes; {held['units']:,} units ("
          + ", ".join(f"{layer} {count:,}" for layer, count in held["layers"].items()) + f"); {counted}; on disk, the "
          f"index {disk['index']:,} bytes, the manifest {disk['manifest']:,}, the vector cache {disk['vector_cache']:,}",
          file=out, flush=True)
    return 0


def cmd_serve(args: argparse.Namespace, out=None, run=None) -> int:
    """The local server (T6.1), its models loaded before it listens, so no question waits for them (PF-F7). It
    listens on 127.0.0.1 alone, and nothing here can make it listen anywhere else. The port is taken first, so a
    port another program holds is said at once, not after the models load, and the line that says it serves comes
    once it listens (HS-F18)."""
    import socket

    from needtoknow import api
    from needtoknow.answer.pipeline import warm
    from needtoknow.embed.model import EmbedError
    out = out or sys.stdout
    port = args.port or api.PORT
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((api.HOST, port))
        listener.listen()
    except OSError as error:
        listener.close()
        print(f"serve: port {port} on {api.HOST} is taken, by another ntk serve or another program: --port names "
              f"another ({error.strerror})", file=out, flush=True)
        return 1
    try:
        config = server.load_config()
        if server.running(config) is None:
            listener.close()
            print("serve: the index's server is not running: ntk qdrant start", file=out, flush=True)
            return 1
        client = server.client(config)
        loaded = warm()
    except (server.ServerError, EmbedError) as error:
        listener.close()
        print(f"serve: {error}", file=out, flush=True)
        return 1
    print(f"the models loaded in {loaded:.1f} s; serving on http://{api.HOST}:{port} until stopped", file=out,
          flush=True)
    (run or served)(api.create(client), listener)
    return 0


def served(app, listener) -> None:
    """The app on the socket `ntk serve` holds, with no access log and no proxy's headers believed, until stopped."""
    import uvicorn
    config = uvicorn.Config(app, access_log=False, proxy_headers=False, log_level="warning")
    uvicorn.Server(config).run(sockets=[listener])


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
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "stats":
        return cmd_stats()
    if args.command == "serve":
        return cmd_serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
