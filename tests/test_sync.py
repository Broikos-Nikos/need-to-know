"""The load stage (T3.4) in Qdrant's local mode, the model replaced by the stand in of test_embed_cache: vectors from
the cache by passage, deletes before upserts, a killed sync finished by the next, and two in a row changing nothing."""

import json

import numpy as np
import pytest
from qdrant_client import QdrantClient
from test_embed_cache import Gpu, fake_tokenize

from needtoknow import lexical
from needtoknow.embed import cache, model
from needtoknow.index import sync as loader
from needtoknow.index.collection import COLLECTION, DENSE, SPARSE, ensure
from needtoknow.ingest.manifest import opened as manifest_opened
from needtoknow.ingest.units import passage

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
ROWS = [("u1", ["sales"], ["Procedure: Sales", "SECTION 1  Scope"], "The team checks every case against the procedure."),
        ("u2", ["general"], ["SECTION 2  Shops"], "Shop | Region | Manager")]


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    gpu = Gpu()
    monkeypatch.setattr(model, "tokenize", fake_tokenize)
    monkeypatch.setattr(model, "states", gpu.states)
    client = QdrantClient(location=":memory:")
    calls = []
    for name in ("delete", "upsert"):
        real = getattr(client, name)
        monkeypatch.setattr(client, name, lambda *a, _real=real, _name=name, **k: calls.append(_name) or _real(*a, **k))
    manifest(ROWS)
    ensure(client, lambda: (passage(p, t) for _, _, p, t in ROWS))
    return client, gpu, calls


def manifest(rows):
    """The manifest holds exactly these (unit_id, labels, path, text) rows, all in one document."""
    with manifest_opened() as db:
        db.execute("INSERT OR IGNORE INTO files VALUES ('DOC_1', 'doc.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.execute("DELETE FROM units")
        db.executemany("INSERT INTO units VALUES (?, 'DOC_1', 's', 'prose', ?, ?, ?, 0, 1, 'en', 1)",
                       [(unit, json.dumps(labels), json.dumps(path), text) for unit, labels, path, text in rows])


def points(client):
    records, _ = client.scroll(COLLECTION, limit=1000, with_payload=True, with_vectors=True)
    return {record.payload["unit_id"]: record for record in records}


def test_a_sync_fills_the_collection_from_the_cache_and_a_second_changes_nothing(world):
    client, gpu, _ = world
    first = loader.sync(client)
    assert (first["deleted"], first["embedded"], first["upserted"], first["points_after"]) == (0, 2, 2, 2)
    stats = lexical.frozen()
    for unit, labels, path, text in ROWS:
        point = points(client)[unit]
        with cache.opened() as db:
            stored = cache.read(db, [cache.passage_key(passage(path, text))])[0]
        assert np.allclose(point.vector[DENSE], stored, atol=1e-6)
        indices, values = lexical.document(passage(path, text), stats["avgdl"])
        assert point.vector[SPARSE].indices == indices and np.allclose(point.vector[SPARSE].values, values)
        assert point.payload["labels"] == labels and "text" not in point.payload
    second = loader.sync(client)
    assert (second["deleted"], second["embedded"], second["upserted"], gpu.calls) == (0, 0, 0, [2])


def test_a_relabel_that_keeps_its_passage_deletes_first_and_embeds_nothing(world):
    client, gpu, calls = world
    loader.sync(client)
    calls.clear()
    manifest([ROWS[0], ("u2b", ["general", "legal"], ROWS[1][2], ROWS[1][3])])  # the same passage, new labels
    summary = loader.sync(client)
    assert (summary["deleted"], summary["upserted"], summary["embedded"], gpu.calls) == (1, 1, 0, [2])
    assert calls == ["delete", "upsert"]  # never the old, wider point beside the new one
    assert set(points(client)) == {"u1", "u2b"} and points(client)["u2b"].payload["label_set"] == "general+legal"


def test_a_relabel_that_cuts_the_path_embeds_its_new_passage(world):
    client, gpu, _ = world
    loader.sync(client)
    manifest([("u1b", ["sales", "returns"], ["SECTION 1  Scope"], ROWS[0][3]), ROWS[1]])  # the title left the path
    summary = loader.sync(client)
    assert (summary["deleted"], summary["upserted"], summary["embedded"], gpu.calls) == (1, 1, 1, [2, 1])


def test_a_sync_killed_midway_is_finished_by_the_next(world, monkeypatch):
    client, _, _ = world
    rows = [(f"u{n}", ["sales"], ["Procedure: Sales"], f"Case {n} is checked against the procedure.") for n in range(600)]
    manifest(rows)
    real, made = client.upsert, []

    def dies_on_the_second(*args, **kwargs):
        made.append(1)
        if len(made) == 2:
            raise RuntimeError("killed")
        return real(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", dies_on_the_second)
    with pytest.raises(RuntimeError, match="killed"):
        loader.sync(client)
    assert client.count(COLLECTION, exact=True).count == loader.BATCH
    monkeypatch.setattr(client, "upsert", real)
    finished = loader.sync(client)
    assert (finished["upserted"], finished["points_after"]) == (600 - loader.BATCH, 600)
    assert loader.sync(client)["upserted"] == 0


def test_no_sync_without_the_frozen_statistics(world):
    client, _, _ = world
    lexical.stats_path().unlink()
    with pytest.raises(lexical.LexicalError, match="does not exist"):
        loader.sync(client)
    assert client.count(COLLECTION, exact=True).count == 0


def test_ntk_index_sync_says_what_it_did_and_fails_when_the_counts_disagree(world, capsys, monkeypatch):
    import argparse

    from needtoknow.cli import cmd_index
    client, _, _ = world
    assert cmd_index(argparse.Namespace(action="sync"), client=client) == 0
    assert "synced: units 2, points 0 before and 2 after; deleted 0, embedded 2, upserted 2" in capsys.readouterr().out
    monkeypatch.setattr(loader, "held", lambda client, doc_ids=None: set())  # a collection that seems empty: every unit upserted again
    monkeypatch.setattr(client, "count", lambda *a, **k: type("Count", (), {"count": 3})())
    assert cmd_index(argparse.Namespace(action="sync"), client=client) == 1


def test_a_sync_of_some_documents_compares_those_alone(world):
    client, _, _ = world
    with manifest_opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC_2', 'two.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.execute("""INSERT INTO units VALUES ('u3', 'DOC_2', 's', 'prose', '["hr"]', '[]', 'Pay bands', 0, 1, 'en', 1)""")
    assert loader.sync(client)["points_after"] == 3
    with manifest_opened() as db:
        db.execute("DELETE FROM units WHERE doc_id = 'DOC_2'")
    untouched = loader.sync(client, doc_ids=["DOC_1"])
    assert (untouched["deleted"], untouched["points_after"], untouched["documents"]) == (0, 3, 1)
    assert loader.sync(client, doc_ids=["DOC_2"])["deleted"] == 1


def test_one_sync_writes_at_a_time(world):
    # OP-F5: two syncs at once both counted the same new points in.
    from needtoknow.corpus import CorpusBusy, exclusive
    from needtoknow.settings import data_dir
    client, _, _ = world
    (data_dir() / "index").mkdir(parents=True, exist_ok=True)
    with exclusive(data_dir() / "index"):
        with pytest.raises(CorpusBusy, match="writer.lock is held"):
            loader.sync(client)


def test_a_table_a_killed_sync_left_wrong_is_recounted_by_the_next(world):
    # OP-F5: a sync killed between the index and the table left ghosts no later sync noticed.
    from needtoknow.index import frequencies
    client, _, _ = world
    loader.sync(client)
    with frequencies.opened(COLLECTION) as db:
        frequencies.count(db, [("sales", [1, 2, 3])] * 6, 1)  # six ghosts, as the probe left them
    summary = loader.sync(client)
    assert summary["drift"] == {"sales": (1, 7)} and summary["recounted"] == 2
    with frequencies.opened(COLLECTION) as db:
        assert frequencies.drift(db, client, COLLECTION) == {}
