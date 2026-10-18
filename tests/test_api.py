"""The local server (T6.1) over Qdrant's local mode, with test_search's units and stand in encoder: a role reaches it
once, at sign in, and a query that names one or holds no session is refused; it answers only requests addressed to
the loopback's names; a search's trace shows each search's own candidates, which fused as the engine fuses them give
its fused list, and counts the units the role's filter passes and holds back; and `ntk serve` loads the models before
it listens, on 127.0.0.1 alone."""

import json
import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_search import QUESTION, UNITS, client, indexed, near  # noqa: F401, the fixture

from needtoknow import api, cli
from needtoknow.embed.model import EmbedError
from needtoknow.index.search import RRF_K
from needtoknow.policy import load_policy, readable
from needtoknow.providers.call import Completion, ProviderError

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


def echoing(system, messages, config=None, **settings):
    """A provider that answers with the whole prompt it read, so whatever the prompt holds shows in the answer."""
    return Completion(text=json.dumps({"status": "answered", "answer": messages[0]["content"], "citations": [1]}),
                      reasoning="", usage={"input_hit": 7, "input_miss": 300, "output": 20, "reasoning": 0},
                      stop_reason="stop", seconds=0.01, cost_usd=0.00006, model="deepseek-flash", attempts=1)


def test_ask_answers_with_the_trace_the_excerpts_sent_and_nothing_of_the_cache(client, monkeypatch):
    from needtoknow.answer import pipeline
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", echoing)
    monkeypatch.setattr(pipeline, "warm", lambda encode=None, rerank=None: 0.0)
    http = served(client)
    for role, secrets in (("sales", False), ("legal", True)):
        body = http.post("/ask", json=ASKED, headers=signed(http, role)).json()
        reads = readable(load_policy(), role)
        assert named(body) and all(LABELS[unit] & reads for unit in named(body)), role
        said = body["answer"]["answer"]
        # The provider read the role's excerpts, the ones the trace shows, and no other text.
        assert ("Legal note on matters of counsel." in said) == secrets, role
        assert body["excerpts"] and all(excerpt["text"] in said for excerpt in body["excerpts"]), role
        assert said.count("<document number=") == len(body["excerpts"]), role  # every one sent, and only those
    assert set(body) == {"role", "filter", *api.STAGES, "excerpts", "units", "seconds", "answer"}
    assert set(body["answer"]) == {"status", "answer", "citations", "dropped", "error", "model", "stop_reason"}
    assert body["answer"]["citations"]
    assert {"provider", "parse", "expand", "total"} <= set(body["seconds"])
    # From the cache now: the asker cannot tell, the tokens, the price and the cache's part staying in the spend log.
    again = http.post("/ask", json=ASKED, headers=signed(http, "legal")).json()
    assert again["answer"] == body["answer"] and set(again) == set(body)
    lines = [json.loads(line) for line in pipeline.spend_path().read_text(encoding="utf-8").splitlines()]
    assert [line["cached"] for line in lines] == [False, False, True]

    def refused(system, messages, config=None, **settings):
        raise ProviderError("deepseek: the key was refused")
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", refused)
    response = http.post("/ask", json={"question": "settlement ceiling for claims"}, headers=signed(http, "sales"))
    assert response.status_code == 502 and response.json()["detail"] == "deepseek: the key was refused"


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


def test_a_question_utf8_cannot_carry_is_refused_by_name(client):
    # HS-F19: JSON carries half a surrogate pair as an escape, and the tokenizer met it as the server's one 500.
    http = served(client)
    body = json.dumps({"question": "return " + chr(0xD800) + " days"})  # escaped, as JSON carries it
    response = http.post("/search", content=body, headers={**signed(http, "sales"), "Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"] == "a question must be text: character 8 is half of a surrogate pair alone"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_ntk_serve_holds_its_port_then_loads_the_models_and_listens_on_the_loopback_alone(monkeypatch, capsys):
    from needtoknow.answer import pipeline
    monkeypatch.setattr(cli.server, "load_config", lambda: {})
    monkeypatch.setattr(cli.server, "running", lambda config: 4242)
    monkeypatch.setattr(cli.server, "client", lambda config: None)
    port = free_port()
    args = cli.build_parser().parse_args(["serve", "--port", str(port)])
    # HS-F18: a port another program holds is said at once, before any model loads, and nothing claims to serve.
    monkeypatch.setattr(pipeline, "warm", lambda: pytest.fail("the models loaded for a port it could not hold"))
    with socket.socket() as other:
        other.bind(("127.0.0.1", port))
        other.listen()
        assert cli.cmd_serve(args, run=lambda app, listener: pytest.fail("it listened")) == 1
    assert capsys.readouterr().out.startswith(f"serve: port {port} on 127.0.0.1 is taken, by another ntk serve or "
                                              "another program: --port names another (")
    # The port free: held on the loopback, the models loaded, and then the line that it serves.
    monkeypatch.setattr(pipeline, "warm", lambda: 31.4)
    held = {}

    def listening(app, listener):
        held["at"] = listener.getsockname()
        listener.close()

    assert cli.cmd_serve(args, run=listening) == 0
    assert held["at"] == ("127.0.0.1", port)
    assert capsys.readouterr().out == f"the models loaded in 31.4 s; serving on http://127.0.0.1:{port} until stopped\n"
    assert cli.build_parser().parse_args(["serve"]).port is None and api.PORT == 6435
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--host", "0.0.0.0"])
    # The index's server down: nothing loads, and the port is let go.
    monkeypatch.setattr(cli.server, "running", lambda config: None)
    assert cli.cmd_serve(args, run=lambda app, listener: pytest.fail("it listened")) == 1
    assert capsys.readouterr().out == "serve: the index's server is not running: ntk qdrant start\n"
    with socket.socket() as again:
        again.bind(("127.0.0.1", port))
