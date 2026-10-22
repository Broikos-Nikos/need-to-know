"""The query (T3.5): a role's filter inside both searches, the two lists fused in the engine.

A question becomes a dense vector, BGE-M3 with the question cut at max_length, and a
sparse one, the lexical analyser's terms each weighed by BM25's IDF over the label sets
the role may read, from the frequency table, so text the role may not read moves no
weight (N50, AC-F5). Both searches run inside the role's
filter, taken from `policy.readable()` and nowhere else (N31). The filter is part of each
search, never a pass over its results, so a unit the role may not read can neither be
returned nor take a candidate's place. Qdrant fuses the two lists by reciprocal rank,
1 / (60 + rank) summed as the base's M9 did (`03-retrieval-rag/12` section 5), and returns
the fused order with fused scores only. KB1-F4's first condition kept the raw lexical
scores in the server; since N50 made them a function of the role's view, the served
trace shows each search's own scores, which the noninterference test holds (N64). The
dense search is exact under the role's filter, every vector the
filter passes scored in float32, as Qdrant's exact path does whatever quantization is
asked: the filter bench found Qdrant's graph short of the exact top 10 under every
role's filter at every ef up to 512, and the exact path at 1.000 for every role in a
median 26 to 83 ms (N51). Naming an `ef` asks for the graph instead, its int8 codes
rescored against the originals, for the bench and experiments.

The filter takes two forms, which the filter bench compares (KB1-F3): `labels`, any of
the labels the role reads, and `label_set`, any label set present in the collection that
holds one of them.
"""

from __future__ import annotations

import time

from needtoknow import lexical
from needtoknow.index import frequencies
from needtoknow.index.collection import COLLECTION, DENSE, SPARSE
from needtoknow.policy import load_policy, readable

CANDIDATES = 100
# The longest question asked: the encoder reads 512 tokens of it, and the lexical side would weigh every term of a
# megabyte, one SQL variable each, until SQLite refused (HS-F10).
MAX_QUESTION = 4000


class QuestionError(ValueError):
    """A question the retrieval API will not ask: empty, or too long."""
# The base's k, with ranks counted from 1. Qdrant counts them from 0, so it is sent as RRF_K + 1.
RRF_K = 60
FORMS = ("labels", "label_set")
# Qdrant hands back points of equal fused score in an order that changes from call to call, and the reranker's half
# precision scores tie often, so on one build a tie at the eighth place moved an item's excerpts between runs
# (RS-F29): the fusion is asked for TIES more than the limit, ordered by score and then by unit id, and cut there.
TIES = 20


def label_sets(client) -> list[str]:
    """Every label set the collection holds."""
    hits = client.facet(COLLECTION, key="label_set", limit=100_000, exact=True).hits
    return sorted(hit.value for hit in hits)


def role_filter(policy: dict, role: str, form: str = "labels", sets: list[str] | None = None):
    """The filter a role's queries send, in either form."""
    from qdrant_client import models

    reads = readable(policy, role)
    if form == "labels":
        allowed = sorted(reads)
    elif form == "label_set":
        allowed = [s for s in sets or [] if reads & set(s.split("+"))]
    else:
        raise ValueError(f"no filter form {form!r}; the forms are {', '.join(FORMS)}")
    return models.Filter(must=[models.FieldCondition(key=form, match=models.MatchAny(any=allowed or ["\x00"]))])


def search(client, question: str, role: str, policy: dict | None = None, form: str = "labels", limit: int = CANDIDATES,
           ef: int | None = None, encode=None, sides: tuple[str, ...] = ("dense", "sparse"),
           rerank: str | None = None, trace: dict | None = None) -> list[dict]:
    """The role's candidates for the question, best first: unit id, rank and score, nothing else.

    `sides` leaves one search out, for the evaluation's run of each path alone and for the side channel bench;
    whatever runs, runs inside the role's filter. `rerank` reorders the fused candidates (T3.6); None takes
    `config/models.toml`'s switch, and the score is then the reranker's.

    `trace`, a dict, is filled with the stages as the served path shows them (T6.1, N64): the role's labels and the
    units its filter passes and holds back, counted; each search's own candidates and scores, asked again on their
    own, since the engine hands back only their fusion; the fused list; the reranked one; and each stage's seconds.
    """
    from qdrant_client import models

    if not any(c.isalnum() for c in question):  # a blank form answered with whatever lies nearest nothing (HS-F13)
        raise QuestionError("a question needs at least one word")
    if len(question) > MAX_QUESTION:
        raise QuestionError(f"a question is at most {MAX_QUESTION:,} characters; this one has {len(question):,}")
    try:  # JSON carries half a surrogate pair, which the tokenizer met as the server's one 500 (HS-F19)
        question.encode("utf-8")
    except UnicodeEncodeError as error:
        raise QuestionError(f"a question must be text: character {error.start + 1:,} is half of a surrogate pair "
                            "alone") from None
    if encode is None:
        from needtoknow.embed.model import encode
    policy = policy or load_policy()
    reads = readable(policy, role)
    flt = role_filter(policy, role, form, label_sets(client) if form == "label_set" else None)
    seconds: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        seconds[stage], clock = now - clock, now

    prefetch = []
    if "dense" in sides:
        dense = encode([question])[0]
        lap("encode")
        # No ef: the exact path (N51), every vector the filter passes scored in float32; Qdrant ignores quantization
        # there, so none is sent. An ef asks for the graph, its int8 codes rescored against the originals.
        params = (models.SearchParams(exact=True) if ef is None else
                  models.SearchParams(hnsw_ef=ef, quantization=models.QuantizationSearchParams(rescore=True)))
        prefetch.append(models.Prefetch(query=dense.tolist(), using=DENSE, filter=flt, limit=limit, params=params))
    indices, _ = lexical.query(question)
    if indices and "sparse" in sides:  # a question of stopwords alone has no lexical side
        with frequencies.opened(COLLECTION) as counts:
            view = [s for (s,) in counts.execute("SELECT label_set FROM sets") if reads & set(s.split("+"))]
            values = frequencies.weights(counts, indices, view)
        prefetch.append(models.Prefetch(query=models.SparseVector(indices=indices, values=values), using=SPARSE,
                                        filter=flt, limit=limit))
        lap("weights")
    found = client.query_points(COLLECTION, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=RRF_K + 1)),
                                query_filter=flt, limit=limit + TIES, with_payload=["unit_id"]).points
    found = sorted(found, key=lambda point: (-point.score, point.payload["unit_id"]))[:limit]
    lap("fusion")
    # A point whose unit the manifest no longer holds awaits its sync, and its labels may be the ones a relabel has
    # just narrowed: the manifest decides what exists, so such a point is dropped here, before the reranker reads
    # passages, and a narrowing takes effect at the scan (DR-F15).
    kept = in_manifest([point.payload["unit_id"] for point in found])
    hits = [{"unit_id": unit, "rank": rank, "score": score} for rank, (unit, score) in
            enumerate(((p.payload["unit_id"], p.score) for p in found if p.payload["unit_id"] in kept), 1)]
    lap("manifest")
    if trace is not None:
        passes = client.count(COLLECTION, count_filter=flt, exact=True).count
        # The whole collection's size, which `ntk stats` prints too: a count, and no unit of those held back.
        trace.update(filter={"form": form, "labels": sorted(reads), "passes": passes,
                             "holds_back": client.count(COLLECTION, exact=True).count - passes},
                     **apart(client, prefetch), fused=hits)
        lap("trace")
    from needtoknow.index import rerank as second
    mode = rerank or second.load_config()["mode"]
    if mode == "none" or not hits:
        result = hits
    else:
        # A unit a scan dropped since the manifest's check above awaits its sync as one never held does (DR-F41).
        passages = passages_of([hit["unit_id"] for hit in hits])
        held = [hit for hit in hits if hit["unit_id"] in passages]
        result = second.rerank(question, held, [passages[hit["unit_id"]] for hit in held], mode)
    if trace is not None:
        lap("rerank")
        trace.update(reranked=result, seconds=seconds)
    return result


def apart(client, prefetch: list) -> dict:
    """Each search's own candidates and scores, as the fusion's prefetches ask them, asked again on their own: the same
    query, filter, parameters and limit, in one round trip. In the engine's order, units of equal score as it placed
    them, since the fusion ranks them so: ordered by unit id instead, as the fused list is (RS-F29), tied units showed
    ranks the fusion had not given them (tick 163). A unit the manifest no longer holds is left out, as from the
    fusion (DR-F15)."""
    from qdrant_client import models

    asked = client.query_batch_points(COLLECTION, requests=[
        models.QueryRequest(query=p.query, using=p.using, filter=p.filter, params=p.params, limit=p.limit,
                            with_payload=["unit_id"]) for p in prefetch]) if prefetch else []
    found = {p.using: [(point.payload["unit_id"], point.score) for point in response.points]
             for p, response in zip(prefetch, asked)}
    kept = in_manifest(sorted({unit for pairs in found.values() for unit, _ in pairs}))
    return {side: [{"unit_id": unit, "rank": rank, "score": score} for rank, (unit, score) in
                   enumerate(((u, s) for u, s in found.get(name, []) if u in kept), 1)]
            for side, name in (("dense", DENSE), ("sparse", SPARSE))}


def in_manifest(unit_ids: list[str]) -> set[str]:
    """Those of the given units the manifest holds."""
    from needtoknow.ingest.manifest import opened

    held: set[str] = set()
    with opened() as db:
        for start in range(0, len(unit_ids), 900):
            part = unit_ids[start:start + 900]
            held.update(u for (u,) in db.execute(
                f"SELECT unit_id FROM units WHERE unit_id IN ({','.join('?' * len(part))})", part))
    return held


def passages_of(unit_ids: list[str]) -> dict[str, str]:
    """The passages of those of the given units the manifest holds, as the encoders read them, by unit: one a scan took
    since its search found it is not there, where every unit was assumed to be and a KeyError ended the search
    (DR-F41)."""
    import json

    from needtoknow.ingest.manifest import opened
    from needtoknow.ingest.units import passage

    with opened() as db:
        rows = db.execute(f"SELECT unit_id, path, text FROM units WHERE unit_id IN ({','.join('?' * len(unit_ids))})",
                          unit_ids).fetchall()
    return {unit: passage(json.loads(path), text) for unit, path, text in rows}
