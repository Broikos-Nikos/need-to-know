"""The query (T3.5): a role's filter inside both searches, the two lists fused in the engine.

A question becomes a dense vector, BGE-M3 with the question cut at max_length, and a
sparse one, the lexical analyser's terms each once. Both searches run inside the role's
filter, taken from `policy.readable()` and nowhere else (N31). The filter is part of each
search, never a pass over its results, so a unit the role may not read can neither be
returned nor take a candidate's place. Qdrant fuses the two lists by reciprocal rank,
1 / (60 + rank) summed as the base's M9 did (`03-retrieval-rag/12` section 5), and returns
the fused order with fused scores only: no raw lexical score leaves the server (KB1-F4's
first condition). The dense search rescores its int8 candidates against the float32
originals.

The filter takes two forms, which the filter bench compares (KB1-F3): `labels`, any of
the labels the role reads, and `label_set`, any label set present in the collection that
holds one of them.
"""

from __future__ import annotations

from needtoknow import lexical
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
           ef: int | None = None, encode=None) -> list[dict]:
    """The role's fused candidates for the question, best first: unit id, rank and fused score, nothing else."""
    from qdrant_client import models

    if encode is None:
        from needtoknow.embed.model import encode
    policy = policy or load_policy()
    flt = role_filter(policy, role, form, label_sets(client) if form == "label_set" else None)
    dense = encode([question])[0]
    params = models.SearchParams(hnsw_ef=ef, quantization=models.QuantizationSearchParams(rescore=True))
    prefetch = [models.Prefetch(query=dense.tolist(), using=DENSE, filter=flt, limit=limit, params=params)]
    indices, values = lexical.query(question)
    if indices:  # a question of stopwords alone has no lexical side
        prefetch.append(models.Prefetch(query=models.SparseVector(indices=indices, values=values), using=SPARSE,
                                        filter=flt, limit=limit))
    found = client.query_points(COLLECTION, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=RRF_K + 1)),
                                query_filter=flt, limit=limit, with_payload=["unit_id"]).points
    return [{"unit_id": point.payload["unit_id"], "rank": rank, "score": point.score} for rank, point in enumerate(found, 1)]
