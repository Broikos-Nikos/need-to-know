"""The query (T3.5) in Qdrant's local mode, with a stand in for the question encoder: every role's filter in both
forms, the filter inside each search rather than after it, fusion by reciprocal rank, and nothing but ranks out."""

import json

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from needtoknow.index import frequencies
from needtoknow.index.collection import COLLECTION, DENSE, SPARSE, ensure, label_set, payload, point_id
from needtoknow.index.search import FORMS, RRF_K, role_filter, search
from needtoknow.ingest.manifest import opened
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


def indexed(monkeypatch, tmp_path, units: list) -> QdrantClient:
    """These units in a local collection, with the manifest and the frequencies a sync leaves beside it."""
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    client = QdrantClient(location=":memory:")
    ensure(client, lambda: (text for _, _, text, _ in units))
    avgdl = frozen()["avgdl"]
    points = []
    for unit, labels, text, vector in units:
        indices, values = document(text, avgdl)
        points.append(models.PointStruct(id=point_id(unit), vector={DENSE: vector.tolist(),
                                         SPARSE: models.SparseVector(indices=indices, values=values)},
                                         payload=payload(unit, "DOC", "key", "prose", labels, (), "en")))
    client.upsert(COLLECTION, points=points)
    with opened() as db:  # the manifest holds what the index holds, as a sync leaves them (DR-F15)
        db.execute("INSERT INTO files VALUES ('DOC', 'doc.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'key', 'prose', ?, ?, ?, 0, 1, 'en', 1)",
                       [(unit, json.dumps(labels), json.dumps(["Policy"]), text) for unit, labels, text, _ in units])
    with frequencies.opened(COLLECTION) as db:  # as the sync counts them (N50)
        frequencies.count(db, [(label_set(labels), document(text, avgdl)[0]) for _, labels, text, _ in units], 1)
    return client


@pytest.fixture
def client(monkeypatch, tmp_path):
    return indexed(monkeypatch, tmp_path, UNITS)


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
    monkeypatch.setattr(model, "encode", lambda texts: np.vstack([QUESTION] * len(texts)))
    question = argparse.Namespace(question="settlement ceiling", role="sales", form="labels", limit=10, rerank="none")
    assert cmd_search(question, client=client) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and "sales" in lines[0] and "settlement ceiling" in lines[0] and "general" in lines[1]
    assert lines[2] == "2 candidates in all, fewer than the 10 asked"  # HS-F12: a short list is not a cut
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
    seen = []
    monkeypatch.setattr(rerank, "cross_scores", lambda question, passages: seen.append(passages) or [0.0, 1.0])
    hits = search(client, "settlement ceiling", "sales", encode=lambda texts: np.vstack([QUESTION] * len(texts)), rerank="cross")
    assert [h["unit_id"] for h in hits] == ["open", "sales"]  # the reranker preferred the fused second
    assert seen == [["Policy\nThe settlement ceiling for sales claims is set each quarter.",
                     "Policy\nShops open at nine on weekdays."]]


def test_a_blank_or_a_megabyte_question_is_refused_by_name_and_a_question_of_stopwords_is_not(client):
    # HS-F13: a blank question returned three shop directory rows. HS-F10: a megabyte of distinct words ended in
    # "too many SQL variables" after 12 s.
    from needtoknow.index.search import MAX_QUESTION, QuestionError
    for blank in ("", "   ", "?! ..."):
        with pytest.raises(QuestionError, match="at least one word"):
            ask(client, question=blank)
    long = " ".join(f"word{n}" for n in range(MAX_QUESTION))
    with pytest.raises(QuestionError, match="at most 4,000 characters"):
        ask(client, question=long)
    assert ask(client, question="the and of", role="sales")  # stopwords alone still ask the dense side


def test_the_weights_of_more_terms_than_sqlite_takes_in_one_statement_are_counted_every_one(tmp_path):
    # HS-F10's second half: one SQL variable a term, and a statement holds 32,766 here, 999 in older SQLite. A term
    # far past the first chunk must weigh what its count says, not what an absent term would.
    import math
    with frequencies.opened("chunks", path=tmp_path / "chunks.db") as db:
        frequencies.count(db, [("general", [1, 35000]), ("general", [35000]), ("general", [2])], 1)
        values = frequencies.weights(db, list(range(1, 40001)), ["general"])
    assert len(values) == 40000
    assert values[35000 - 1] == pytest.approx(math.log(1 + 1.5 / 2.5))  # in 2 of 3 passages
    assert values[1 - 1] == pytest.approx(math.log(1 + 2.5 / 1.5)) and values[3 - 1] == pytest.approx(math.log(8))


def test_a_stopped_server_or_a_missing_gpu_is_one_line_not_a_traceback(client, monkeypatch, capsys):
    # HS-F11: both ended ntk search in a traceback, the gRPC one 1 KB long.
    import argparse

    from needtoknow import cli
    from needtoknow.embed import model
    question = argparse.Namespace(question="settlement ceiling", role="sales", form="labels", limit=3, rerank="none")
    monkeypatch.setattr(model, "encode", lambda texts: (_ for _ in ()).throw(model.EmbedError("no CUDA device is visible to torch")))
    assert cli.cmd_search(question, client=client) == 1
    assert capsys.readouterr().out == "search: no CUDA device is visible to torch" + chr(10)
    monkeypatch.setattr(model, "encode", lambda texts: np.vstack([QUESTION] * len(texts)))
    monkeypatch.setattr(client, "query_points", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused")))
    monkeypatch.setattr(cli.server, "running", lambda config: None)
    assert cli.cmd_search(question, client=client) == 1
    assert capsys.readouterr().out.startswith("search: the index's server is not running, so nothing reached it: ntk qdrant start")


def test_numbers_out_of_range_are_refused_by_the_parser():
    # HS-F12: --limit -5 printed 95 results, --limit 0 nothing, --interval -1 a traceback, --interval 0 a busy loop.
    from needtoknow.cli import LIMIT, build_parser
    from needtoknow.index.search import CANDIDATES
    assert LIMIT == CANDIDATES
    for args in (["search", "q", "--as", "sales", "--limit", "0"], ["search", "q", "--as", "sales", "--limit", "-5"],
                 ["search", "q", "--as", "sales", "--limit", "101"], ["ingest", "watch", "--interval", "0"],
                 ["ingest", "watch", "--interval", "-1"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(args)
    assert build_parser().parse_args(["search", "q", "--as", "sales", "--limit", "100"]).limit == 100


def test_a_point_the_manifest_has_dropped_is_never_returned_reranked_or_printed(client, monkeypatch, capsys):
    # DR-F15: between a scan that drops a unit and the sync that deletes its point, the index still returned it, the
    # reranked search raised KeyError and ntk search TypeError. A unit's id holds its labels, so the point that
    # outlives its unit is the one a relabel narrowed.
    import argparse

    from needtoknow.cli import cmd_search
    from needtoknow.embed import model
    from needtoknow.index import rerank
    with opened() as db:
        db.execute("DELETE FROM units WHERE unit_id = 'open'")  # the scan has run, the sync has not
    monkeypatch.setattr(rerank, "cross_scores", lambda question, passages: [0.0] * len(passages))
    for mode in ("none", "cross"):
        hits = ask(client, role="sales", rerank=mode)
        assert "open" not in {h["unit_id"] for h in hits}
        assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1)) and hits
    monkeypatch.setattr(model, "encode", lambda texts: np.vstack([QUESTION] * len(texts)))
    question = argparse.Namespace(question="settlement ceiling", role="sales", form="labels", limit=10, rerank="none")
    assert cmd_search(question, client=client) == 0
    assert "Shops open at nine" not in capsys.readouterr().out


def test_ntk_search_skips_a_hit_a_scan_dropped_between_the_search_and_its_printing(client, monkeypatch, capsys):
    # DR-F15's race: the search kept to the manifest, and a scan dropped the unit before the line was printed.
    import argparse

    from needtoknow.cli import cmd_search
    from needtoknow.index import search as retrieval
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: [{"unit_id": "ghost", "rank": 1, "score": 0.1}])
    question = argparse.Namespace(question="settlement ceiling", role="sales", form="labels", limit=1, rerank="none")
    assert cmd_search(question, client=client) == 0 and "ghost" not in capsys.readouterr().out


def test_points_of_equal_score_come_back_in_one_order_whatever_order_the_engine_gives(client, monkeypatch):
    # RS-F29: Qdrant returned tied fused scores in an order that changed from call to call, and the cut at the limit
    # with them; ten runs of one question gave ten fused lists.
    from needtoknow.index import search as retrieval
    real = client.query_points

    returned, asked = [], []

    def shuffled(*args, **kwargs):
        asked.append(kwargs["limit"])
        response = real(*args, **kwargs)
        returned.extend(p.payload["unit_id"] for p in response.points)
        tied = [models.ScoredPoint(id=p.id, version=0, score=0.5, payload=p.payload) for p in response.points]
        response.points = list(reversed(tied))
        return response

    monkeypatch.setattr(client, "query_points", shuffled)
    first = [hit["unit_id"] for hit in ask(client, role="admin", limit=3)]
    assert asked == [3 + retrieval.TIES] and len(returned) > 3  # asked past the limit, so a tie at the cut is whole
    assert first == sorted(returned)[:3]  # by unit id among equal scores, then cut


def test_the_reranker_breaks_a_tie_by_unit_id_not_by_the_order_it_was_given(monkeypatch):
    # RS-F29: its half precision scores tie often, and a stable sort kept the fused list's changing order for them.
    from needtoknow.index import rerank as second
    monkeypatch.setattr(second, "cross_scores", lambda question, passages: [2.0 if p == "u9" else 1.0 for p in passages])
    hits = [{"unit_id": u, "rank": n, "score": 0.0} for n, u in enumerate(("u3", "u9", "u1", "u2"), 1)]
    order = [hit["unit_id"] for hit in second.rerank("q", hits, [h["unit_id"] for h in hits], "cross")]
    backwards = list(reversed(hits))
    again = [hit["unit_id"] for hit in second.rerank("q", backwards, [h["unit_id"] for h in backwards], "cross")]
    assert order == again == ["u9", "u1", "u2", "u3"]
