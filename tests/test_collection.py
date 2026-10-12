"""The Qdrant collection (T3.3), in Qdrant's local mode: created once with the architecture's settings and its
frozen lexical statistics, checked every time after, and refused when of another shape."""

import pytest
from qdrant_client import QdrantClient, models

from needtoknow import lexical
from needtoknow.cli import cmd_index
from needtoknow.index import collection
from needtoknow.index.collection import COLLECTION, CollectionError, ensure, label_set, payload, point_id, problems

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
PASSAGES = ["Procedure: Sales > SECTION 1  Scope\nThe team checks every case.", "SKU | Product\nRS1 | devices"]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    return QdrantClient(location=":memory:")


def test_the_collection_is_created_once_with_its_settings_and_frozen_statistics(client):
    calls = []
    first = ensure(client, lambda: calls.append(1) or iter(PASSAGES))
    assert first["created"] and calls == [1] and first["lexical"] == lexical.measure(PASSAGES)
    assert problems(client.get_collection(COLLECTION)) == []
    second = ensure(client, lambda: calls.append(1) or iter(PASSAGES))
    assert not second["created"] and calls == [1] and second["lexical"] == first["lexical"]


@pytest.mark.parametrize("change, named", [
    ({"size": 768}, "768 dimensions"),
    ({"distance": models.Distance.DOT}, "by Dot"),
    ({"hnsw_config": models.HnswConfigDiff(m=32)}, "HNSW m 32"),
    ({"on_disk": False}, "not int8 in RAM"),
    ({"quantization_config": None}, "not int8 in RAM"),
])
def test_a_collection_of_another_shape_is_refused_by_what_differs(client, change, named):
    vectors, sparse = collection.config()
    vectors = {collection.DENSE: vectors[collection.DENSE].model_copy(update=change)}
    client.create_collection(COLLECTION, vectors_config=vectors, sparse_vectors_config=sparse)
    with pytest.raises(CollectionError, match=named):
        ensure(client, lambda: iter(PASSAGES))


def test_a_collection_whose_sparse_vector_counts_idf_over_everything_is_refused(client):
    # N50, AC-F5: Qdrant's IDF modifier counts every label set; the weights come from the role's view instead.
    vectors, _ = collection.config()
    client.create_collection(COLLECTION, vectors_config=vectors,
                             sparse_vectors_config={collection.SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
    with pytest.raises(CollectionError, match="IDF modifier, which counts every label set"):
        ensure(client, lambda: iter(PASSAGES))


def test_ids_and_payloads():
    assert point_id("abc") == point_id("abc") != point_id("abc#1")
    assert label_set(["returns", "legal"]) == label_set(["legal", "returns"]) == "legal+returns"
    body = payload("abc", "DOC", "key", "prose", ("returns", "legal"), ("Title", "Section"), "en")
    assert body["label_set"] == "legal+returns" and body["labels"] == ["returns", "legal"]
    assert not {"text", "start", "end"} & set(body)  # the manifest keeps them current (T3.7)


def test_ntk_index_create_freezes_the_statistics_over_the_manifest(client, capsys):
    from needtoknow.ingest.manifest import opened
    with opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC_1', 'doc.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.execute("""INSERT INTO units VALUES ('u1', 'DOC_1', 's', 'prose', '["sales"]', '["Title"]', 'cat dog dog', 0, 1, 'en', 1)""")
    assert cmd_index(None, client=client) == 0
    assert "documents_v2: created; lexical statistics frozen over 1 passages, avgdl 4.00" in capsys.readouterr().out
    assert cmd_index(None, client=client) == 0
    assert "already there, and of this shape" in capsys.readouterr().out


def test_a_run_killed_while_measuring_leaves_no_collection_without_its_statistics(client):
    def killed():
        raise RuntimeError("killed")
        yield

    with pytest.raises(RuntimeError, match="killed"):
        ensure(client, killed)
    assert not client.collection_exists(COLLECTION)
