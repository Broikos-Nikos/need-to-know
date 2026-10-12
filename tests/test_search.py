"""The query (T3.5) in Qdrant's local mode, with a stand in for the question encoder: every role's filter in both
forms, the filter inside each search rather than after it, fusion by reciprocal rank, and nothing but ranks out."""

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from needtoknow.index import frequencies
from needtoknow.index.collection import COLLECTION, DENSE, SPARSE, ensure, label_set, payload, point_id
from needtoknow.index.search import FORMS, RRF_K, role_filter, search
from needtoknow.lexical import document, frozen
from needtoknow.policy import PolicyError, load_policy, readable

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
QUESTION = np.random.default_rng(1).normal(size=1024).astype(np.float32)
QUESTION /= np.linalg.norm(QUESTION)


def near(seed: int, distance: float) -> np.ndarray:
    """A unit vector at roughly the given distance from the question."""
    noise = np.random.default_rng(seed).normal(size=1024).astype(np.float32)
    vector = QUESTION + distance * noise / np.linalg.norm(noise)
    return vector / np.linalg.norm(vector)


UNITS = ([(f"secret{n}", ["legal"], "Legal note on matters of counsel.", near(100 + n, 0.1)) for n in range(50)]
         + [("open", ["general"], "Shops open at nine on weekdays.", near(2, 0.8)),
            ("sales", ["sales"], "The settlement ceiling for sales claims is set each quarter.", near(3, 0.9)),
            ("lost", ["unlabelled"], "Text under an audience no label knows.", near(4, 0.2))])


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    client = QdrantClient(location=":memory:")
    ensure(client, lambda: (text for _, _, text, _ in UNITS))
    avgdl = frozen()["avgdl"]
    points = []
    for unit, labels, text, vector in UNITS:
        indices, values = document(text, avgdl)
        points.append(models.PointStruct(id=point_id(unit), vector={DENSE: vector.tolist(),
                                         SPARSE: models.SparseVector(indices=indices, values=values)},
                                         payload=payload(unit, "DOC", "key", "prose", labels, (), "en")))
    client.upsert(COLLECTION, points=points)
    with frequencies.opened(COLLECTION) as db:  # as the sync counts them (N50)
        frequencies.count(db, [(label_set(labels), document(text, avgdl)[0]) for _, labels, text, _ in UNITS], 1)
    return client


def ask(client, question="settlement ceiling", role="sales", **kwargs):
    return search(client, question, role, encode=lambda texts: np.vstack([QUESTION] * len(texts)),
                  **{"rerank": "none", **kwargs})


@pytest.mark.parametrize("form", FORMS)
def test_no_role_gets_a_unit_it_may_not_read_in_either_filter_form(client, form):
    policy = load_policy()
    labels = {unit: set(unit_labels) for unit, unit_labels, _, _ in UNITS}
    for role in policy["roles"]:
        found = {hit["unit_id"] for hit in ask(client, role=role, form=form)}
        allowed = {unit for unit, unit_labels in labels.items() if unit_labels & readable(policy, role)}
        assert found == allowed, (role, form)
    assert len(found) == len(UNITS)  # the last role is the administrator, who reads everything by the policy


def test_the_filter_is_inside_the_search_so_closer_forbidden_units_take_no_place(client):
    # Fifty legal units sit closer to the question than anything sales may read, and a question of stopwords leaves
    # the dense side alone: with the filter applied after its top 1, sales would get nothing.
    assert [hit["unit_id"] for hit in ask(client, question="the and of", role="sales", limit=1)] == ["open"]


def test_every_search_and_the_fusion_carry_the_roles_filter(client, monkeypatch):
    # Qdrant also pushes the outer filter into each search; the code does not lean on that.
    sent = {}
    real = client.query_points
    monkeypatch.setattr(client, "query_points", lambda *args, **kwargs: sent.update(kwargs) or real(*args, **kwargs))
    ask(client, question="settlement ceiling", role="sales")
    expected = role_filter(load_policy(), "sales")
    assert len(sent["prefetch"]) == 2 and all(p.filter == expected for p in sent["prefetch"])
    assert sent["query_filter"] == expected


def test_a_unit_both_searches_find_comes_before_one_only_the_dense_side_finds(client):
    hits = ask(client, question="settlement ceiling", role="sales")
    assert [hit["unit_id"] for hit in hits] == ["sales", "open"]  # open is nearer, sales is found by both sides
    assert hits[0]["score"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))


def test_only_ranks_and_fused_scores_leave_the_search(client):
    hits = ask(client)
    assert all(set(hit) == {"unit_id", "rank", "score"} for hit in hits) and [h["rank"] for h in hits] == [1, 2]
    assert all(hit["score"] <= 2 / (RRF_K + 1) for hit in hits)


def test_a_question_of_stopwords_alone_still_searches_the_dense_side(client):
    assert [hit["unit_id"] for hit in ask(client, question="the and of", role="sales")] == ["open", "sales"]


def test_a_role_the_policy_does_not_define_is_refused(client):
    with pytest.raises(PolicyError, match="no role named 'board'"):
        ask(client, role="board")


def test_ntk_search_prints_what_the_role_may_read_and_refuses_an_unknown_role(client, monkeypatch, capsys):
    import argparse
    import json

    from needtoknow.cli import cmd_search
    from needtoknow.embed import model
    from needtoknow.ingest.manifest import opened
    with opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC', 'doc.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'key', 'prose', ?, '[]', ?, 0, 1, 'en', 1)",
                       [(unit, json.dumps(labels), text) for unit, labels, text, _ in UNITS])
    monkeypatch.setattr(model, "encode", lambda texts: np.vstack([QUESTION] * len(texts)))
    question = argparse.Namespace(question="settlement ceiling", role="sales", form="labels", limit=10, rerank="none")
    assert cmd_search(question, client=client) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and "sales" in lines[0] and "settlement ceiling" in lines[0] and "general" in lines[1]
    assert cmd_search(argparse.Namespace(question="x", role="board", form="labels", limit=10, rerank="none"),
                      client=client) == 1


def test_a_questions_lexical_weights_are_counted_over_the_roles_view_alone(client, monkeypatch):
    # AC-F5, N50: weights counted over the whole collection let ranks tell which words hidden text holds.
    sent = {}
    real = client.query_points
    monkeypatch.setattr(client, "query_points", lambda *args, **kwargs: sent.update(kwargs) or real(*args, **kwargs))
    ask(client, question="settlement ceiling", role="sales")
    sparse = sent["prefetch"][1].query
    with frequencies.opened(COLLECTION) as db:
        assert sparse.values == pytest.approx(frequencies.weights(db, sparse.indices, ["general", "sales"]))
        assert sparse.values != pytest.approx(frequencies.weights(db, sparse.indices, ["general", "legal", "sales",
                                                                                         "unlabelled"]))


def test_the_dense_search_is_exact_unless_an_ef_asks_for_the_graph(client, monkeypatch):
    # N51: under a filter that matches any of several labels, the graph missed up to half of the exact top 10 at any
    # ef. The exact path scores every vector the filter passes in float32 and ignores quantization, so none is sent:
    # asked for int8 alone, it gave float32's ids and scores for 30 questions of 30 (tick 71). The filter bench names
    # an ef, and the graph's int8 codes are rescored.
    sent = []
    real = client.query_points
    monkeypatch.setattr(client, "query_points", lambda *args, **kwargs: sent.append(kwargs) or real(*args, **kwargs))
    ask(client, role="sales")
    ask(client, role="sales", ef=128)
    exact, graph = (next(p.params for p in kwargs["prefetch"] if p.using == DENSE) for kwargs in sent)
    assert exact.exact and exact.hnsw_ef is None and exact.quantization is None
    assert not graph.exact and graph.hnsw_ef == 128 and graph.quantization.rescore


def test_one_side_alone_runs_inside_the_roles_filter_too(client):
    assert [hit["unit_id"] for hit in ask(client, role="sales", sides=("sparse",))] == ["sales"]
    assert [hit["unit_id"] for hit in ask(client, role="sales", sides=("dense",))] == ["open", "sales"]


def test_the_rerank_switch_hands_the_reranker_the_candidates_own_passages(client, monkeypatch):
    # T3.6: the reranker reads each fused candidate's passage from the manifest and reorders those, none other.
    import json

    from needtoknow.index import rerank
    from needtoknow.ingest.manifest import opened
    with opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC', 'doc.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'key', 'prose', ?, '[\"Policy\"]', ?, 0, 1, 'en', 1)",
                       [(unit, json.dumps(labels), text) for unit, labels, text, _ in UNITS])
    seen = []
    monkeypatch.setattr(rerank, "cross_scores", lambda question, passages: seen.append(passages) or [0.0, 1.0])
    hits = search(client, "settlement ceiling", "sales", encode=lambda texts: np.vstack([QUESTION] * len(texts)), rerank="cross")
    assert [h["unit_id"] for h in hits] == ["open", "sales"]  # the reranker preferred the fused second
    assert seen == [["Policy\nThe settlement ceiling for sales claims is set each quarter.",
                     "Policy\nShops open at nine on weekdays."]]
