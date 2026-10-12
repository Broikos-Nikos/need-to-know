"""Qdrant's in process local mode, which the tests use instead of a server: it must hold the index's schema."""

from qdrant_client import QdrantClient, models


def test_local_mode_holds_named_dense_and_idf_sparse_vectors_under_both_filter_forms():
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "units",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upsert("units", points=[
        models.PointStruct(id=1, payload={"labels": ["hr"], "label_set": "hr"},
                           vector={"dense": [1.0, 0.0, 0.0, 0.0], "bm25": models.SparseVector(indices=[7, 9], values=[1.0, 1.0])}),
        models.PointStruct(id=2, payload={"labels": ["general", "legal"], "label_set": "general+legal"},
                           vector={"dense": [0.9, 0.1, 0.0, 0.0], "bm25": models.SparseVector(indices=[7], values=[1.0])}),
    ])
    by_labels = models.Filter(must=[models.FieldCondition(key="labels", match=models.MatchAny(any=["general", "legal"]))])
    by_label_set = models.Filter(must=[models.FieldCondition(key="label_set", match=models.MatchAny(any=["general+legal", "legal"]))])
    for flt in (by_labels, by_label_set):
        dense = client.query_points("units", query=[1.0, 0.0, 0.0, 0.0], using="dense", query_filter=flt, limit=10).points
        sparse = client.query_points("units", query=models.SparseVector(indices=[7], values=[1.0]), using="bm25",
                                     query_filter=flt, limit=10).points
        assert [p.id for p in dense] == [2] and [p.id for p in sparse] == [2]
