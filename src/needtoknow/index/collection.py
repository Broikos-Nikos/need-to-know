"""The Qdrant collection (T3.3): every unit once, its dense and its lexical vector under one filter.

The settings are the architecture's (section 8): the dense vector, 1024 dimensions and
cosine, HNSW with m 16, int8 scalar quantization kept in RAM with the float32 originals
on disk, which a query rescores against; the sparse vector `bm25`, holding BM25's term
frequency part, without Qdrant's IDF modifier, which would count every label set: the
question brings IDF weights over the asking role's view instead (N50, AC-F5); and
keyword payload indexes on `label_set`, the tenant key that keeps each set's points
together, and on `labels`, `doc_id` and `lang`. Filters may name `labels` or
`label_set`, and the filter bench measures both (KB1-F3); no search walks the graph by
`doc_id` or `lang`, so their indexes build no edges in it (KB4-K10).

A point's id is a UUID made from its unit id, so an upsert replays over itself. Its
payload holds what a filter or a delete needs; the text and the offsets stay in the
manifest, which keeps them current, since offsets move without the id changing.

`ensure` creates the collection when it is missing, and freezes the lexical statistics
over the passages with it, since an index's sparse vectors are written at one average
length (KB1-F15), and starts its frequency table afresh. Otherwise it checks that the collection is the one this code builds
and refuses any other, so no run writes into an index of another shape; payload
indexes are asserted every time, which is a no op when they exist.
"""

from __future__ import annotations

import hashlib
import uuid

from needtoknow import lexical
from needtoknow.index import frequencies

# v2 since tick 63: the sparse vector lost the IDF modifier (N50).
COLLECTION = "documents_v2"
DENSE, SPARSE = "dense", "bm25"
DIMENSIONS = 1024
HNSW_M = 16
# Qdrant's indexing threshold, in kilobytes of vectors, its default named here: a segment above it is indexed, and a
# bulk load, which defers the indexing to its end, puts this back (T7.5).
INDEXING_THRESHOLD = 10_000


class CollectionError(RuntimeError):
    """A collection of another shape under this project's name."""


def label_set(labels) -> str:
    """The tenant key: a unit's labels, sorted and joined."""
    return "+".join(sorted(labels))


def point_id(unit_id: str) -> str:
    return str(uuid.UUID(hashlib.sha256(unit_id.encode("utf-8")).hexdigest()[:32]))


def payload(unit_id: str, doc_id: str, section_key: str, layer: str, labels, path, language: str) -> dict:
    return {"unit_id": unit_id, "doc_id": doc_id, "section_key": section_key, "layer": layer, "labels": list(labels),
            "label_set": label_set(labels), "lang": language, "path": list(path)}


def config():
    from qdrant_client import models

    dense = models.VectorParams(
        size=DIMENSIONS, distance=models.Distance.COSINE, on_disk=True, hnsw_config=models.HnswConfigDiff(m=HNSW_M),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)))
    return {DENSE: dense}, {SPARSE: models.SparseVectorParams()}


def indexes() -> dict:
    from qdrant_client import models

    keyword = models.PayloadSchemaType.KEYWORD
    unwalked = models.KeywordIndexParams(type="keyword", enable_hnsw=False)
    return {"label_set": models.KeywordIndexParams(type="keyword", is_tenant=True), "labels": keyword,
            "doc_id": unwalked, "lang": unwalked}


def problems(info) -> list[str]:
    """How a collection differs from the one this code builds; empty when it is that one."""
    from qdrant_client import models

    found = []
    dense = (info.config.params.vectors or {}).get(DENSE) if isinstance(info.config.params.vectors, dict) else None
    if dense is None:
        return [f"no named vector {DENSE!r}"]
    if (dense.size, dense.distance) != (DIMENSIONS, models.Distance.COSINE):
        found.append(f"{DENSE} is {dense.size} dimensions by {dense.distance}, not {DIMENSIONS} by cosine")
    m = (dense.hnsw_config.m if dense.hnsw_config and dense.hnsw_config.m else info.config.hnsw_config.m)
    if m != HNSW_M:
        found.append(f"{DENSE} has HNSW m {m}, not {HNSW_M}")
    scalar = getattr(dense.quantization_config, "scalar", None)
    if not dense.on_disk or scalar is None or scalar.type != models.ScalarType.INT8 or not scalar.always_ram:
        found.append(f"{DENSE} is not int8 in RAM over float32 originals on disk")
    sparse = (info.config.params.sparse_vectors or {}).get(SPARSE)
    if sparse is None:
        found.append(f"no sparse vector {SPARSE!r}")
    elif sparse.modifier not in (None, models.Modifier.NONE):
        found.append(f"{SPARSE} carries Qdrant's IDF modifier, which counts every label set (AC-F5)")
    return found


def drop(client) -> bool:
    """Delete the collection, so that the next ensure() builds it again with statistics frozen afresh and an empty
    frequency table; the vector cache stays, its vectors owing nothing to the index (MT-F7). Whether one was there."""
    there = client.collection_exists(COLLECTION)
    if there:
        client.delete_collection(COLLECTION)
    return there


def create(client, name: str = COLLECTION) -> None:
    """A collection made as the index's is: its vectors, its indexing threshold and its payload indexes. The ties bench
    makes its own here too; a copy of this call there missed the threshold T7.5 named (MA-F30)."""
    from qdrant_client import models

    vectors, sparse = config()
    client.create_collection(name, vectors_config=vectors, sparse_vectors_config=sparse,
                             optimizers_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD))
    for field, schema in indexes().items():
        client.create_payload_index(name, field, field_schema=schema)


def ensure(client, passages) -> dict:
    """The collection, created with its frozen lexical statistics when missing; refused when of another shape.

    `passages` is called only on creation, and gives every passage the index will hold.
    """
    created = not client.collection_exists(COLLECTION)
    if created:  # the statistics first: a run killed in between leaves no collection without them
        lexical.freeze(lexical.measure(passages()))
        for leftover in frequencies.table_path().parent.glob(frequencies.table_path().name + "*"):
            leftover.unlink()  # the new collection holds no point, and its table counts none
        create(client)
    else:
        found = problems(client.get_collection(COLLECTION))
        if found:
            raise CollectionError(f"{COLLECTION} is not the collection this code builds: {'; '.join(found)}; "
                                  f"ntk index create --replace builds it again, keeping the vector cache")
        for field, schema in indexes().items():  # an index this code added since the collection was made
            client.create_payload_index(COLLECTION, field, field_schema=schema)
    return {"created": created, "lexical": lexical.frozen()}
