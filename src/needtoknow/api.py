"""The local server (T6.1): on the served path the index's only client, listening on 127.0.0.1 alone.

A role reaches it once, at sign in: `POST /session` takes a role of `config/policy.toml` on
the caller's word, as `ntk --as` does, and answers with a token. Every query carries the
token and never a role: a body that names one is refused, and so is a query without a
session (`specs/architecture.md` section 10, the store). A deployment puts its identity
provider where the sign in stands. The role's labels are read from the policy at each
query, so a change to it holds from the next one (`03-retrieval-rag/11` section 9).

It answers only requests addressed to 127.0.0.1 or localhost, so a page served elsewhere
cannot reach it through a name of its own that resolves here, and it keeps no access log.
`ntk serve` loads the encoder and the reranker before it listens, so no question waits for
them (PF-F7); one search runs at a time, as the GPU holds one batch, and an answer waits on its
provider outside that lock (DR-F34).

`POST /search` answers with the question's stage trace (N64): the role's filter, its labels
and the units it passes and holds back, counted; each search's candidates with its own
scores; their fusion; the reranker's order; the excerpts a question would be shown; each
stage's seconds; and the document, section, labels and text of every unit named. Every unit
in it passed the role's filter; of the units held back it has the count alone. `POST /ask`
answers with the same trace, the excerpts as they were sent, and the answer with its
citations. It asks the provider every time, reading and keeping no cached answer: one read
back comes in milliseconds, and its time would tell a role that someone with its access
asked the same (N68); the call's tokens and price stay in the spend log, the operator's
(KB5-K12). A dev golden question asked as its own role is scored beside its answer by the
registered scorer, its seconds among the stages'; `eval/protocol.toml` registers the test
split for the baseline and the release alone, so a test question is answered here and never
scored. The verdict names no category and, for a refusal, no leak: the category says
whether a refused answer exists where the role may not read, E, or nowhere, J and N, and
the leak flag compares the answer with the hidden values themselves, so either would tell a
role what its refusal is worded to hide (AC-F16).

`GET /doc/<id>` is the role's view of a document: the segments it may read, in document
order, each with its heading path, labels, how many units it holds and its first units'
text, and nothing of the others, not their count; a document it may read nothing of is
refused in the words an unknown one is, so the refusal tells nothing (`specs/architecture.md`
section 9, step 7). Nor does its time: the view reads the segments kept under the role's labels
and their first units alone, the same work whatever the others hold (AC-F18). A document is
named by its id, which says its owner and kind (N65).
`GET /stats` answers with what the build holds, as `ntk stats` prints it: the whole
collection's figures, the same for every role. `GET /events?after=<time>` is the watcher's
change feed as the role may see it: each finished scan that added units it may read, with
the documents and how many, keyed by the time it finished. A scan that added none of them
is left out whole, so a change to text the role may not read shows as nothing, not even its
time; the scans' own numbers stay in the server, whose gaps would count those changes; and a
removal shows as nothing, a removed unit's labels having gone with it (N66). The feed reads
the rows its role's labels keep and no others, so its time says nothing of such a scan either
(AC-F19).

`GET /roles` names the policy's roles, which the sign in needs and the public policy file
holds, and asks no session. When the page is built, `web/dist`, it is served at `/` beside
the routes, so the page's calls stay on one origin.

What the page asks over and over costs little: a poll of `/events` without `after` starts
after the last scan the session was shown, a place the server keeps and never serves
(PF-F11); and the build's figures are kept a minute and served with the time they were
measured (PF-F12).
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from needtoknow.answer.pipeline import ask
from needtoknow.answer.prompt import section_of
from needtoknow.embed.model import EmbedError
from needtoknow.eval.golden import Context, GoldenError, golden_path, load_items
from needtoknow.eval.protocol import ProtocolError, check_use
from needtoknow.eval.score import REFUSALS, Sources, score_item
from needtoknow.eval.split import SplitError
from needtoknow.index import server
from needtoknow.index.expand import excerpts
from needtoknow.index.footprint import NoBuild, footprint, vectors
from needtoknow.index.search import QuestionError, search
from needtoknow.ingest.manifest import opened, read
from needtoknow.policy import PolicyError, load_policy, readable
from needtoknow.providers.call import ProviderError
from needtoknow.settings import repo_root

HOST = "127.0.0.1"
NAMES = [HOST, "localhost"]
PORT = 6435  # beside the index's server, on 6433 and 6434
SESSIONS = 256  # sign ins kept, the oldest dropped first
STAGES = ("dense", "sparse", "fused", "reranked")
# What a served answer holds; never whether the cache answered, the tokens or the price (KB5-K12).
ANSWERED = ("status", "answer", "citations", "dropped", "error", "model", "stop_reason")
PREVIEW = 3  # the units of a segment's text a document's outline shows
EVENTS = 50  # the latest events a feed answers with
FIGURES = 60.0  # seconds the build's figures are kept before they are measured again (PF-F12)
# What a served verdict holds of the scorer's result: nothing that reads text the role may not read (AC-F16).
VERDICT = ("correct", "grounded", "partial", "shown_facts")
NO_SESSION = "no session: sign in with POST /session and send its token; a query names no role"
STOPPED = "the index's server is not running, so nothing reached it: ntk qdrant start"


class SignIn(BaseModel):
    """A role of the policy, on the caller's word: where a deployment's identity provider would stand."""
    model_config = ConfigDict(extra="forbid")
    role: str


class Question(BaseModel):
    """A question, and nothing else: the session holds the role."""
    model_config = ConfigDict(extra="forbid")
    question: str


def described(unit_ids: list[str]) -> dict[str, dict]:
    """Each named unit's document, section, labels and text, from the manifest; one a scan dropped since its search is
    left out (DR-F15)."""
    found: dict[str, dict] = {}
    with opened() as db:
        for start in range(0, len(unit_ids), 900):
            part = unit_ids[start:start + 900]
            for unit, doc_id, labels, path, text in db.execute(
                    f"SELECT unit_id, doc_id, labels, path, text FROM units WHERE unit_id IN ({','.join('?' * len(part))})",
                    part):
                found[unit] = {"doc_id": doc_id, "section": " > ".join(json.loads(path)), "labels": json.loads(labels),
                               "text": text}
    return found


def among(reads: frozenset[str]) -> str:
    """The placeholders of a reader's labels, for the rows kept under them."""
    return ",".join("?" * len(reads))


def outline(doc_id: str, reads: frozenset[str]) -> list[dict]:
    """The document's segments a reader of `reads` may read, from the manifest, in document order: each one's key,
    heading path, labels, units and first units' text. A segment holds one label set, so it is read whole or not at
    all, by the rule the index's filter applies: any of its labels read. The segments are found under the reader's
    labels and the units read are their first ones, so the view does no work for a segment the reader may not read,
    whatever it holds (AC-F18)."""
    labels_read = sorted(reads)
    shown = []
    with read() as db:  # the segments and their first units from one snapshot, which no scan moves (DR-F41)
        found = {(key, labels): units for key, labels, units in db.execute(
            f"SELECT section_key, labels, units FROM segments WHERE doc_id = ? AND label IN ({among(reads)})",
            (doc_id, *labels_read))}
        for (key, labels), units in found.items():
            first = db.execute(f"SELECT start, unit_id, path, text FROM units WHERE doc_id = ? AND section_key = ? AND "
                               f"labels = ? ORDER BY start, unit_id LIMIT {PREVIEW}", (doc_id, key, labels)).fetchall()
            shown.append((first[0][:2], {"key": key, "section": " > ".join(json.loads(first[0][2])),
                                         "labels": json.loads(labels), "units": units, "preview": [row[3] for row in first]}))
    return [segment for _, segment in sorted(shown, key=lambda pair: pair[0])]


def events(after: str, reads: frozenset[str], limit: int = EVENTS) -> list[dict]:
    """The watcher's finished scans after the time `after`, each as the documents where it added units a reader of
    `reads` may read, and how many; a scan that added none is left out."""
    return scanned(after, reads, limit)[0]


def scanned(after: str | None, reads: frozenset[str], limit: int = EVENTS, run: int = 0) -> tuple[list[dict], int]:
    """The events after the scan `run`, and after the time `after` when one is given, and the last scan they show:
    where a session's next poll starts, which the server keeps and never serves (PF-F11, N66). The units a scan added
    are counted under each of their labels as they are written, so a poll reads the rows of the reader's labels alone
    and never a scan it cannot see, not even its time (AC-F19). Scans run one at a time, so their ids keep the order
    they finished in, and two that end in one second are two places, not one (DR-F36)."""
    labels_read = sorted(reads)
    with opened() as db:
        rows = db.execute(
            f"SELECT a.run_id, r.finished, a.doc_id, a.labels, a.units FROM added a JOIN runs r ON r.run_id = a.run_id "
            f"WHERE a.label IN ({among(reads)}) AND a.run_id > ? AND r.finished IS NOT NULL"
            + ("" if after is None else " AND r.finished > ?"),
            (*labels_read, run, *([] if after is None else [after]))).fetchall()
    # A label set the reader reads under two of its labels is one count, read twice.
    counted = {(run_id, finished, doc_id, labels): units for run_id, finished, doc_id, labels, units in rows}
    scans: dict[int, dict] = {}
    for (run_id, finished, doc_id, _), units in sorted(counted.items()):
        documents = scans.setdefault(run_id, {"at": finished, "documents": {}})["documents"]
        documents[doc_id] = documents.get(doc_id, 0) + units
    found = [{"at": scan["at"], "documents": [{"doc_id": doc, "units": n} for doc, n in scan["documents"].items()]}
             for _, scan in sorted(scans.items())]
    return found[-limit:], max(scans, default=run)


def keep_latest(table: OrderedDict, key: str, value, limit: int = SESSIONS) -> None:
    """A value under its key, moved last, and the oldest dropped past the limit without iterating the table: a plain
    route runs on the thread pool, and two polls dropping at once met mid iteration (DR-F40); popitem is one call."""
    table.pop(key, None)
    table[key] = value
    while len(table) > limit:
        table.popitem(last=False)


def verdict(question: str, role: str, answered: dict, items=None, ctx=None, sources=None) -> dict | None:
    """The registered scorer's verdict on an answer to a dev golden question asked as its own role, None for any other
    question (T6.1f). Its context is built for each verdict, a tenth of a second, so no SQLite connection crosses the
    server's threads. `items`, `ctx` and `sources` stand in for the committed dev items and the build's context."""
    if items is None:
        check_use("dev", "development")
        items = load_items(golden_path("dev"))
    asked = " ".join(question.split())
    item = next((i for i in items if i["role"] == role and " ".join(i["question"].split()) == asked), None)
    if item is None:
        return None
    ctx = ctx or Context()
    result = score_item(item, answered, ctx, sources or Sources(ctx))
    # One verdict for every refusal: E, J and N alike, answered by the answer the role sees and nothing else.
    return {"item": item["id"], "refusal": item["category"] in REFUSALS, **{key: result[key] for key in VERDICT}}


@contextmanager
def refused():
    """A query's errors as the answers that say them: no traceback reaches the caller."""
    try:
        yield
    except QuestionError as error:
        raise HTTPException(400, str(error)) from None
    except PolicyError as error:  # the role left the policy after its sign in
        raise HTTPException(403, str(error)) from None
    except ProviderError as error:
        raise HTTPException(502, str(error)) from None
    except (EmbedError, NoBuild) as error:
        raise HTTPException(503, str(error)) from None
    except Exception:
        if server.stopped():
            raise HTTPException(503, STOPPED) from None
        raise


def staged(role: str, trace: dict, shown: list[dict], seconds: dict, began: float) -> dict:
    """The stage trace as served: the filter, each stage's units, ranks and scores, the excerpts, the text of every
    unit named, and each stage's seconds."""
    clock = time.perf_counter()
    units = described(list(dict.fromkeys(hit["unit_id"] for stage in STAGES for hit in trace[stage])))
    seconds = {**seconds, "texts": time.perf_counter() - clock, "total": time.perf_counter() - began}
    return {"role": role, "filter": trace["filter"],
            **{stage: [{"unit_id": hit["unit_id"], "rank": hit["rank"], "score": float(hit["score"])}
                       for hit in trace[stage]] for stage in STAGES},
            "excerpts": [{"doc_id": e["doc_id"], "section": section_of(e), "labels": e["labels"], "units": e["units"],
                          "rank": e["rank"], "text": e["text"]} for e in shown],
            "units": units, "seconds": {stage: round(value, 4) for stage, value in seconds.items()}}


def create(client, *, policy: dict | None = None, encode=None, rerank: str | None = None, judge=verdict,
           page: Path | None = None) -> FastAPI:
    """The server's app over the index's client. `policy`, `encode` and `rerank` reach the search as they would from
    `ntk`: None reads the policy at each query and takes the configured models. `judge` scores an answer. `page`, the
    built page's folder, `web/dist` unless given, is served at `/` when it holds one."""
    sessions: OrderedDict[str, str] = OrderedDict()
    places: OrderedDict[str, int] = OrderedDict()  # each session's place in the change feed, a scan's id (PF-F11)
    lock = threading.Lock()
    app = FastAPI(title="need-to-know", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=NAMES)

    def role_of(request: Request) -> str:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        role = sessions.get(token) if scheme.lower() == "bearer" else None
        if role is None:
            raise HTTPException(401, NO_SESSION, headers={"WWW-Authenticate": "Bearer"})
        return role

    # Async, so every sign in runs on the event loop's one thread and the sessions change in one place.
    @app.post("/session")
    async def sign_in(body: SignIn) -> dict:
        try:
            readable(policy or load_policy(), body.role)
        except PolicyError as error:
            raise HTTPException(400, str(error)) from None
        token = secrets.token_urlsafe(32)
        keep_latest(sessions, token, body.role)
        return {"token": token, "role": body.role}

    # Each dependency is a default, not an annotation: the module's annotations are strings, which cannot name a
    # function local to create().
    @app.post("/search")
    def find(body: Question, role: str = Depends(role_of)) -> dict:
        began = time.perf_counter()
        trace: dict = {}
        with lock, refused():
            search(client, body.question, role, policy, encode=encode, rerank=rerank, trace=trace)
            clock = time.perf_counter()
            shown = excerpts(trace["reranked"])
        return staged(role, trace, shown, {**trace["seconds"], "expand": time.perf_counter() - clock}, began)

    @app.get("/doc/{doc_id}")
    def document(doc_id: str, role: str = Depends(role_of)) -> dict:
        with refused():
            sections = outline(doc_id, readable(policy or load_policy(), role))
        if not sections:  # unknown, or nothing of it the role may read: one answer for both
            raise HTTPException(404, f"no document {doc_id[:40]!r}")
        return {"doc_id": doc_id, "sections": sections}

    @app.get("/roles")
    def roles() -> dict:
        return {"roles": list((policy or load_policy())["roles"])}

    # Without `after`, a poll starts after the last scan the session was shown; the place is the server's and is never
    # served (N66).
    @app.get("/events")
    def feed(request: Request, after: str | None = None, role: str = Depends(role_of)) -> dict:
        token = request.headers["authorization"].partition(" ")[2]
        place = places.get(token, 0)
        reads = readable(policy or load_policy(), role)
        with refused():
            found, until = scanned(after, reads) if after is not None else scanned(None, reads, run=place)
        keep_latest(places, token, max(until, place))
        return {"events": found}

    # The figures walk the index's folder, half a second warm and seconds cold, and move only when a sync does: kept a
    # minute, and served with the time they were measured (PF-F12).
    kept: dict = {}

    @app.get("/stats")
    def figures(role: str = Depends(role_of)) -> dict:
        if not kept or time.monotonic() - kept["at"] > FIGURES:
            with refused():
                measured = {**footprint(), "vectors": vectors(client)}
            kept.update(at=time.monotonic(), figures=measured,
                        measured=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return {**kept["figures"], "measured": kept["measured"]}

    @app.post("/ask")
    def answer(body: Question, role: str = Depends(role_of)) -> dict:
        began = time.perf_counter()
        trace: dict = {}
        # The lock is the GPU's, held for the search alone, not the provider's call (DR-F34). No cached answer is read
        # or kept: one read back comes in milliseconds, and its time would tell the asker that someone with its access
        # asked the same before (N68).
        with refused():
            result = ask(client, body.question, role, policy, encode=encode, rerank=rerank, trace=trace, hold=lock,
                         use_cache=False)
        taken = result["seconds"]
        clock = time.perf_counter()
        try:
            judged = judge(body.question, role, result)
        except (GoldenError, ProtocolError, SplitError, OSError, sqlite3.Error) as error:  # the answer stands
            judged = {"error": f"no verdict: {error}"}
        served = staged(role, trace, trace["shown"], {**trace["seconds"], "expand": taken["expand"],
                                                      "provider": taken["provider"], "parse": taken["parse"],
                                                      "verdict": time.perf_counter() - clock}, began)
        return {**served, "answer": {key: result.get(key) for key in ANSWERED}, "verdict": judged}

    built = page or repo_root() / "web" / "dist"
    if (built / "index.html").is_file():  # mounted last, so every route above answers before the page's files
        app.mount("/", StaticFiles(directory=built, html=True), name="page")
    return app
