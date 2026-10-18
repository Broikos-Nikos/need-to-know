"""The local server (T6.1) over Qdrant's local mode, with test_search's units and stand in encoder: a role reaches it
once, at sign in, and a query that names one or holds no session is refused; it answers only requests addressed to
the loopback's names; a search's trace shows each search's own candidates, which fused as the engine fuses them give
its fused list, and counts the units the role's filter passes and holds back; and `ntk serve` loads the models before
it listens, on 127.0.0.1 alone."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_search import QUESTION, UNITS, client, indexed, near  # noqa: F401, the fixture

from needtoknow import api, cli
from needtoknow.embed.model import EmbedError
from needtoknow.index.search import RRF_K
from needtoknow.policy import load_policy, readable

pytestmark = [pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant"),
              pytest.mark.filterwarnings("ignore:Local mode performs exact")]  # the trace asks its searches apart
ASKED = {"question": "settlement ceiling"}
LABELS = {unit: set(labels) for unit, labels, _, _ in UNITS}


def served(client, encode=lambda texts: np.vstack([QUESTION] * len(texts))) -> TestClient:
    return TestClient(api.create(client, encode=encode, rerank="none"), base_url="http://127.0.0.1")


def signed(http: TestClient, role: str) -> dict:
    response = http.post("/session", json={"role": role})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def fusion_of(body: dict) -> list[float]:
    """Each fused unit's score as reciprocal rank fusion of its ranks in the two searches as the trace shows them."""
    ranks = [{hit["unit_id"]: hit["rank"] for hit in body[side]} for side in ("dense", "sparse")]
    return [sum(1 / (RRF_K + side[hit["unit_id"]]) for side in ranks if hit["unit_id"] in side) for hit in body["fused"]]


def named(body: dict) -> set[str]:
    """Every unit a served search names: in a stage, among the units described, or in an excerpt."""
    return ({hit["unit_id"] for stage in api.STAGES for hit in body[stage]} | set(body["units"])
            | {unit for excerpt in body["excerpts"] for unit in excerpt["units"]})


def test_a_query_needs_a_session_and_names_no_role(client):
    http = served(client)
    assert http.post("/search", json=ASKED).status_code == 401
    assert http.post("/search", json=ASKED, headers={"Authorization": "Bearer guessed"}).status_code == 401
    headers = signed(http, "sales")
    widened = http.post("/search", json={**ASKED, "role": "legal"}, headers=headers)
    assert widened.status_code == 422 and "role" in widened.text
    body = http.post("/search", json=ASKED, headers=headers).json()
    assert body["role"] == "sales" and body["filter"]["labels"] == sorted(readable(load_policy(), "sales"))
    blank = http.post("/search", json={"question": "  ?  "}, headers=headers)
    assert blank.status_code == 400 and blank.json()["detail"] == "a question needs at least one word"


def test_a_role_the_policy_does_not_define_cannot_sign_in(client):
    response = served(client).post("/session", json={"role": "board"})
    assert response.status_code == 400 and "no role named 'board'" in response.json()["detail"]


def test_each_session_is_shown_its_roles_units_and_no_other(client):
    http = served(client)
    for role, secrets in (("sales", False), ("legal", True)):
        reads = readable(load_policy(), role)
        body = http.post("/search", json=ASKED, headers=signed(http, role)).json()
        units = named(body)
        assert units and all(LABELS[unit] & reads for unit in units), role
        assert any(unit.startswith("secret") for unit in units) == secrets, role
        assert all(body["units"][unit]["text"] == text for unit, _, text, _ in UNITS if unit in units)


def test_a_request_addressed_by_another_name_is_refused(client):
    # A page elsewhere whose name resolves to this machine reaches the loopback with that name as its host.
    app = served(client).app
    assert TestClient(app, base_url="http://elsewhere.example").post("/session", json={"role": "sales"}).status_code == 400
    assert TestClient(app, base_url=f"http://localhost:{api.PORT}").post("/session", json={"role": "sales"}).status_code == 200


def test_the_trace_shows_each_search_as_the_fusion_took_it_and_counts_the_filter(client):
    http = served(client)
    body = http.post("/search", json=ASKED, headers=signed(http, "sales")).json()
    vectors = {unit: vector for unit, _, _, vector in UNITS}
    assert [hit["unit_id"] for hit in body["dense"]] == ["open", "sales"]
    assert [hit["score"] for hit in body["dense"]] == pytest.approx([float(vectors[u] @ QUESTION) for u in ("open", "sales")])
    assert [hit["unit_id"] for hit in body["sparse"]] == ["sales"] and body["sparse"][0]["score"] > 0
    assert [hit["unit_id"] for hit in body["fused"]] == ["sales", "open"]
    assert [hit["score"] for hit in body["fused"]] == pytest.approx(fusion_of(body), rel=1e-6)
    assert body["reranked"] == body["fused"]
    reads = readable(load_policy(), "sales")
    passes = sum(bool(labels & reads) for labels in LABELS.values())
    assert body["filter"] == {"form": "labels", "labels": sorted(reads), "passes": passes,
                              "holds_back": len(UNITS) - passes}
    assert [excerpt["units"] for excerpt in body["excerpts"]] == [["sales"], ["open"]]  # two audiences, two excerpts
    assert set(body["seconds"]) >= {"encode", "weights", "fusion", "trace", "rerank", "expand", "texts", "total"}


def test_units_of_equal_score_keep_the_ranks_the_fusion_gave_them(monkeypatch, tmp_path):
    # Six units of one text tie on the sparse side, and the engine places them in an order of its own, which is the
    # order the fusion ranks them in: sorted by unit id, the trace showed them at ranks the fusion had not given them
    # (tick 163, on the server). Local mode puts ties last stored first, so they are stored in the ids' order.
    tied = [(f"tie{c}", ["sales"], "The settlement ceiling for sales claims.", near(10 + n, 0.3 + 0.05 * n))
            for n, c in enumerate("abcdef")]
    client = indexed(monkeypatch, tmp_path, tied)
    answered = []
    real = client.query_batch_points
    monkeypatch.setattr(client, "query_batch_points", lambda *a, **k: answered.append(real(*a, **k)) or answered[-1])
    http = served(client)
    body = http.post("/search", json=ASKED, headers=signed(http, "sales")).json()
    engine = [point.payload["unit_id"] for point in answered[-1][1].points]  # the sparse search, asked second
    assert len({hit["score"] for hit in body["sparse"]}) == 1 and len(engine) == 6
    assert engine != sorted(engine), "the engine orders the tied units by id, so this test cannot tell the two apart"
    assert [hit["unit_id"] for hit in body["sparse"]] == engine
    assert [hit["score"] for hit in body["fused"]] == pytest.approx(fusion_of(body), rel=1e-6)


def test_a_missing_gpu_or_a_stopped_index_is_a_503_that_names_it(client, monkeypatch):
    def lost(texts):
        raise EmbedError("no CUDA device is visible to torch")
    http = served(client, encode=lost)
    response = http.post("/search", json=ASKED, headers=signed(http, "sales"))
    assert response.status_code == 503 and response.json()["detail"] == "no CUDA device is visible to torch"
    http = served(client)
    headers = signed(http, "sales")
    monkeypatch.setattr(client, "query_points", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused")))
    monkeypatch.setattr(api.server, "running", lambda config: None)
    response = http.post("/search", json=ASKED, headers=headers)
    assert response.status_code == 503 and response.json()["detail"].startswith("the index's server is not running")


def test_ntk_serve_loads_the_models_first_and_listens_on_the_loopback_alone(monkeypatch, capsys):
    from needtoknow.answer import pipeline
    monkeypatch.setattr(cli.server, "load_config", lambda: {})
    monkeypatch.setattr(cli.server, "running", lambda config: 4242)
    monkeypatch.setattr(cli.server, "client", lambda config: None)
    monkeypatch.setattr(pipeline, "warm", lambda: 31.4)
    ran = {}
    args = cli.build_parser().parse_args(["serve"])
    assert cli.cmd_serve(args, run=lambda app, **kwargs: ran.update(kwargs)) == 0
    assert (ran["host"], ran["port"], ran["access_log"]) == ("127.0.0.1", api.PORT, False)
    assert capsys.readouterr().out == f"the models loaded in 31.4 s; serving on http://127.0.0.1:{api.PORT} until stopped\n"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--host", "0.0.0.0"])
    monkeypatch.setattr(cli.server, "running", lambda config: None)
    monkeypatch.setattr(pipeline, "warm", lambda: pytest.fail("the models loaded for a server that cannot search"))
    assert cli.cmd_serve(args, run=lambda app, **kwargs: pytest.fail("it listened")) == 1
    assert capsys.readouterr().out == "serve: the index's server is not running: ntk qdrant start\n"
