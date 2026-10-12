"""The query (T3.5): a role's filter inside both searches, the two lists fused in the engine.

A question becomes a dense vector, BGE-M3 with the question cut at max_length, and a
sparse one, the lexical analyser's terms each weighed by BM25's IDF over the label sets
the role may read, from the frequency table, so text the role may not read moves no
weight (N50, AC-F5). Both searches run inside the role's
filter, taken from `policy.readable()` and nowhere else (N31). The filter is part of each
search, never a pass over its results, so a unit the role may not read can neither be
returned nor take a candidate's place. Qdrant fuses the two lists by reciprocal rank,
1 / (60 + rank) summed as the base's M9 did (`03-retrieval-rag/12` section 5), and returns
the fused order with fused scores only: no raw lexical score leaves the server (KB1-F4's
first condition). The dense search is exact under the role's filter, every vector the
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

from needtoknow import lexical
from needtoknow.index import frequencies
from needtoknow.index.collection import COLLECTION, DENSE, SPARSE
from needtoknow.policy import load_policy, readable

CANDIDATES = 100
# The base's k, with ranks counted from 1. Qdrant counts them from 0, so it is sent as RRF_K + 1.
RRF_K = 60
FORMS = ("labels", "label_set")


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
           rerank: str | None = None) -> list[dict]:
    """The role's candidates for the question, best first: unit id, rank and score, nothing else.

    `sides` leaves one search out, for the evaluation's run of each path alone and for the side channel bench;
    whatever runs, runs inside the role's filter. `rerank` reorders the fused candidates (T3.6); None takes
    `config/models.toml`'s switch, and the score is then the reranker's.
    """
    from qdrant_client import models

    if encode is None:
        from needtoknow.embed.model import encode
    policy = policy or load_policy()
    flt = role_filter(policy, role, form, label_sets(client) if form == "label_set" else None)
    prefetch = []
    if "dense" in sides:
        dense = encode([question])[0]
        # No ef: the exact path (N51), every vector the filter passes scored in float32; Qdrant ignores quantization
        # there, so none is sent. An ef asks for the graph, its int8 codes rescored against the originals.
        params = (models.SearchParams(exact=True) if ef is None else
                  models.SearchParams(hnsw_ef=ef, quantization=models.QuantizationSearchParams(rescore=True)))
        prefetch.append(models.Prefetch(query=dense.tolist(), using=DENSE, filter=flt, limit=limit, params=params))
    indices, _ = lexical.query(question)
    if indices and "sparse" in sides:  # a question of stopwords alone has no lexical side
        reads = readable(policy, role)
        with frequencies.opened(COLLECTION) as counts:
            view = [s for (s,) in counts.execute("SELECT label_set FROM sets") if reads & set(s.split("+"))]
            values = frequencies.weights(counts, indices, view)
        prefetch.append(models.Prefetch(query=models.SparseVector(indices=indices, values=values), using=SPARSE,
                                        filter=flt, limit=limit))
    found = client.query_points(COLLECTION, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=RRF_K + 1)),
                                query_filter=flt, limit=limit, with_payload=["unit_id"]).points
    hits = [{"unit_id": point.payload["unit_id"], "rank": rank, "score": point.score} for rank, point in enumerate(found, 1)]
    from needtoknow.index import rerank as second
    mode = rerank or second.load_config()["mode"]
    if mode == "none" or not hits:
        return hits
    return second.rerank(question, hits, passages_of([hit["unit_id"] for hit in hits]), mode)


def passages_of(unit_ids: list[str]) -> list[str]:
    """The passages of the given units, as the encoders read them, from the manifest."""
    import json

    from needtoknow.ingest.manifest import opened
    from needtoknow.ingest.units import passage

    with opened() as db:
        rows = {u: (p, t) for u, p, t in db.execute(
            f"SELECT unit_id, path, text FROM units WHERE unit_id IN ({','.join('?' * len(unit_ids))})", unit_ids)}
    return [passage(json.loads(rows[u][0]), rows[u][1]) for u in unit_ids]
