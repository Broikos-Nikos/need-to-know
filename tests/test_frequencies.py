"""Document frequencies per label set and the weights of a role's view (T3.12, N50, AC-F5): text a role may not read
moves none of its weights, the table follows the index through syncs, and a recount gives the same table."""

import json
import math

import pytest
from qdrant_client import QdrantClient

from needtoknow import lexical
from needtoknow.index import frequencies
from needtoknow.index import sync as loader
from needtoknow.index.collection import COLLECTION, ensure
from needtoknow.ingest.manifest import opened as manifest_opened
from needtoknow.ingest.units import passage

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")


def table(db):
    return (sorted(db.execute("SELECT label_set, passages FROM sets WHERE passages > 0")),
            sorted(db.execute("SELECT s.label_set, d.term, d.n FROM df d JOIN sets s ON s.id = d.set_id WHERE d.n > 0")))


def test_a_word_frequent_only_in_hidden_text_weighs_what_the_view_gives_it(monkeypatch, tmp_path):
    # AC-F5: as sales, two words seen once each weighed apart, because legal text held one of them 40 times.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    with frequencies.opened(COLLECTION) as db:
        frequencies.count(db, [("sales", [1, 9]), ("sales", [2, 9]), ("general", [9])] + [("legal", [1])] * 40, 1)
        seen_by_sales = frequencies.weights(db, [1, 2], ["general", "sales"])
        seen_by_admin = frequencies.weights(db, [1, 2], ["general", "legal", "sales"])
        assert seen_by_sales[0] == seen_by_sales[1] == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))
        assert seen_by_admin[0] < seen_by_admin[1]  # the administrator reads the legal text, so for it the count is right
        frequencies.count(db, [("legal", [1])] * 40, -1)
        assert frequencies.weights(db, [1, 2], ["general", "legal", "sales"]) == pytest.approx(
            frequencies.weights(db, [1, 2], ["general", "sales"]))


def test_a_table_kept_for_another_collection_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    with frequencies.opened("documents_v1"):
        pass
    with pytest.raises(frequencies.FrequencyError, match="counts documents_v1, not documents_v2"):
        with frequencies.opened(COLLECTION):
            pass


def test_the_table_follows_the_index_through_syncs_and_a_recount_agrees(monkeypatch, tmp_path):
    from test_embed_cache import Gpu, fake_tokenize

    from needtoknow.embed import model
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(model, "tokenize", fake_tokenize)
    monkeypatch.setattr(model, "states", Gpu().states)
    rows = [("u1", ["sales"], ["Policy"], "The ceiling for claims is set each quarter."),
            ("u2", ["legal"], ["Counsel"], "The ceiling for settlements is set by counsel."),
            ("u3", ["general"], [], "Shops open at nine.")]

    def manifest(units):
        with manifest_opened() as db:
            db.execute("INSERT OR IGNORE INTO files VALUES ('DOC', 'd.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
            db.execute("DELETE FROM units")
            db.executemany("INSERT INTO units VALUES (?, 'DOC', 's', 'prose', ?, ?, ?, 0, 1, 'en', 1)",
                           [(u, json.dumps(labels), json.dumps(path), text) for u, labels, path, text in units])

    client = QdrantClient(location=":memory:")
    manifest(rows)
    ensure(client, lambda: (passage(p, t) for _, _, p, t in rows))
    loader.sync(client)
    manifest([rows[0], ("u2b", ["legal", "sales"], ["Counsel"], rows[1][3])])  # a relabel, and a unit gone
    loader.sync(client)
    with frequencies.opened(COLLECTION) as db:
        kept = table(db)
        assert frequencies.recount(db, client, COLLECTION) == 2
        assert table(db) == kept and kept[0] == [("legal+sales", 1), ("sales", 1)]
    avgdl = lexical.frozen()["avgdl"]
    assert avgdl > 0
