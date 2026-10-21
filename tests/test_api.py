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


def test_a_documents_view_holds_the_segments_the_role_may_read_and_nothing_of_the_others(client):
    # T6.1c: the role's view of a document; one it may read nothing of is refused as an unknown one is.
    from needtoknow.ingest.manifest import opened
    with opened() as db:  # a second document, all of it legal's
        db.execute("INSERT INTO files VALUES ('LEG', 'leg.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.execute("INSERT INTO units VALUES ('legal1', 'LEG', 'k9', 'prose', ?, ?, 'Privileged advice.', 0, 1, 'en', 1)",
                   (json.dumps(["legal"]), json.dumps(["Counsel"])))
    http = served(client)
    sales, legal = signed(http, "sales"), signed(http, "legal")
    assert http.get("/doc/DOC").status_code == 401
    assert http.get("/doc/DOC", headers=sales).json() == {"doc_id": "DOC", "sections": [
        {"key": "key", "section": "Policy", "labels": ["general"], "units": 1, "preview": ["Shops open at nine on weekdays."]},
        {"key": "key", "section": "Policy", "labels": ["sales"], "units": 1,
         "preview": ["The settlement ceiling for sales claims is set each quarter."]}]}
    sections = http.get("/doc/DOC", headers=legal).json()["sections"]
    assert [(s["labels"], s["units"], len(s["preview"])) for s in sections] == [(["general"], 1, 1), (["legal"], 50, 3)]
    assert http.get("/doc/LEG", headers=legal).json()["sections"][0]["preview"] == ["Privileged advice."]
    hidden, unknown = http.get("/doc/LEG", headers=sales), http.get("/doc/NOPE", headers=sales)
    assert (hidden.status_code, unknown.status_code) == (404, 404)
    assert hidden.json()["detail"] == "no document 'LEG'" and unknown.json()["detail"] == "no document 'NOPE'"


def test_the_change_feed_shows_a_role_only_what_it_may_read_and_nothing_of_the_rest(client):
    # T6.1j: a scan that added only text the role may not read, a removal and an unfinished scan show as nothing,
    # and no scan's number is served, whose gaps would count them.
    from needtoknow.ingest.manifest import opened
    at = {run: f"2026-10-20T10:{5 * (run - 1):02d}:00+00:00" for run in range(1, 6)}
    with opened() as db:
        db.executemany("INSERT INTO runs (run_id, kind, started, finished) VALUES (?, 'scan', ?, ?)",
                       [(run, at[run], None if run == 5 else at[run]) for run in range(1, 6)])
        db.executemany("INSERT INTO changes VALUES (?, 'DOC', 'changed', ?, ?)",
                       [(1, len(UNITS), 0), (2, 1, 0), (3, 1, 0), (4, 0, 1), (5, 1, 0)])
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'key', 'prose', ?, ?, ?, 0, 1, 'en', ?)",
                       [("secret50", json.dumps(["legal"]), json.dumps(["Policy"]), "A later note.", 2),
                        ("open2", json.dumps(["general"]), json.dumps(["Policy"]), "Shops close at six.", 3),
                        ("sales2", json.dumps(["sales"]), json.dumps(["Policy"]), "Not yet scanned.", 5)])
    http = served(client)
    assert http.get("/events").status_code == 401
    sales, legal = signed(http, "sales"), signed(http, "legal")
    assert http.get("/events", headers=sales).json() == {"events": [
        {"at": at[1], "documents": [{"doc_id": "DOC", "units": 2}]}, {"at": at[3], "documents": [{"doc_id": "DOC", "units": 1}]}]}
    assert [e["documents"][0]["units"] for e in http.get("/events", headers=legal).json()["events"]] == [51, 1, 1]
    assert http.get("/events", params={"after": at[1]}, headers=sales).json()["events"] == [
        {"at": at[3], "documents": [{"doc_id": "DOC", "units": 1}]}]


def test_stats_are_the_builds_figures_the_same_for_every_role(client):
    # T6.1h: what ntk stats prints, for the whole collection, behind a session like every other route.
    http = served(client)
    assert http.get("/stats").status_code == 401
    seen = [http.get("/stats", headers=signed(http, role)).json() for role in ("sales", "legal")]
    assert seen[0] == seen[1]
    assert (seen[0]["documents"], seen[0]["units"], seen[0]["layers"], seen[0]["vectors"]) == (
        1, len(UNITS), {"prose": len(UNITS)}, len(UNITS))


def test_the_roles_answer_without_a_session_and_the_built_page_is_served_beside_the_routes(client, tmp_path):
    # T6.2a: the page signs in by a role's name, so the names come first, from the public policy; the built page is
    # mounted after every route, which keep answering.
    page = tmp_path / "dist"
    page.mkdir()
    (page / "index.html").write_text("<!doctype html><title>the page</title>", encoding="utf-8")
    http = TestClient(api.create(client, encode=lambda texts: np.vstack([QUESTION] * len(texts)), rerank="none",
                                 page=page), base_url="http://127.0.0.1")
    assert http.get("/roles").json() == {"roles": list(load_policy()["roles"])}
    assert "the page" in http.get("/").text
    assert http.post("/search", json=ASKED).status_code == 401  # the routes before the page's files
    unbuilt = TestClient(api.create(client, page=tmp_path / "unbuilt"), base_url="http://127.0.0.1")
    assert unbuilt.get("/").status_code == 404 and unbuilt.get("/roles").status_code == 200  # no page, routes still


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
    assert set(body) == {"role", "filter", *api.STAGES, "excerpts", "units", "seconds", "answer", "verdict"}
    assert body["verdict"] is None  # no golden question
    assert set(body["answer"]) == {"status", "answer", "citations", "dropped", "error", "model", "stop_reason"}
    assert body["answer"]["citations"]
    assert {"provider", "parse", "expand", "total"} <= set(body["seconds"])
    # Asked again, it is asked of the provider again, and the tokens and the price stay in the spend log (N68).
    again = http.post("/ask", json=ASKED, headers=signed(http, "legal")).json()
    assert again["answer"] == body["answer"] and set(again) == set(body)
    lines = [json.loads(line) for line in pipeline.spend_path().read_text(encoding="utf-8").splitlines()]
    assert [line["cached"] for line in lines] == [False, False, False]

    def refused(system, messages, config=None, **settings):
        raise ProviderError("deepseek: the key was refused")
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", refused)
    response = http.post("/ask", json={"question": "settlement ceiling for claims"}, headers=signed(http, "sales"))
    assert response.status_code == 502 and response.json()["detail"] == "deepseek: the key was refused"


def test_a_question_asked_again_takes_a_calls_time_so_it_tells_no_one_who_asked_before(client, monkeypatch):
    # T6.1k: an answer read from the cache came back in milliseconds where a call takes seconds, so the trace's seconds,
    # and the response's time on any clock, told a role that someone with its access had asked the same (N68).
    import time as clock

    from needtoknow.answer import pipeline
    calls = []

    def slow(system, messages, config=None, **settings):
        calls.append(1)
        clock.sleep(0.2)
        return echoing(system, messages, config, **settings)
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", slow)
    monkeypatch.setattr(pipeline, "warm", lambda encode=None, rerank=None: 0.0)
    http = served(client)
    first, second = (http.post("/ask", json=ASKED, headers=signed(http, "sales")).json() for _ in range(2))
    assert second["answer"] == first["answer"]
    assert len(calls) == 2 and min(first["seconds"]["provider"], second["seconds"]["provider"]) >= 0.2, (
        len(calls), first["seconds"]["provider"], second["seconds"]["provider"])


def test_a_dev_golden_question_asked_as_its_role_is_scored_beside_its_answer_and_no_other(client, monkeypatch):
    # T6.1f: the registered scorer's verdict on the answer; asked by another role, or a question of the test split,
    # it has none, eval/protocol.toml keeping the test split for the baseline and the release.
    import types

    from needtoknow.answer import pipeline
    from needtoknow.eval import score as S
    from needtoknow.eval.golden import GoldenError, golden_path, load_items
    from needtoknow.eval.split import SplitError
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", echoing)
    monkeypatch.setattr(pipeline, "warm", lambda encode=None, rerank=None: 0.0)
    golden = [{"id": "D01", "category": "A", "role": "sales", "lang": "en", "question": "settlement  ceiling",
               "expect": {"status": "answered", "facts": [{"fact_id": "F01", "forms": ["each quarter"]}]}}]
    ctx = types.SimpleNamespace(facts={"F01": {"id": "F01", "family": "F01", "values": ["each quarter"] * 2}})
    sources = S.Sources(ctx, given={"F01": {("DOC", "Policy")}}, units={"F01": {"sales"}})
    judge = lambda question, role, answered: api.verdict(question, role, answered, golden, ctx, sources)  # noqa: E731
    http = TestClient(api.create(client, encode=lambda texts: np.vstack([QUESTION] * len(texts)), rerank="none",
                                 judge=judge), base_url="http://127.0.0.1")
    body = http.post("/ask", json=ASKED, headers=signed(http, "sales")).json()
    assert body["verdict"] == {"item": "D01", "refusal": False, "correct": True, "grounded": True, "partial": 1.0,
                               "shown_facts": True}
    assert http.post("/ask", json=ASKED, headers=signed(http, "legal")).json()["verdict"] is None
    first = load_items(golden_path("test"))[0]  # the committed test split, asked as its own role
    assert api.verdict(first["question"], first["role"], {"status": "answered", "answer": "", "citations": []}) is None

    def unbuilt(question, role, answered):  # the plan missing raises a SplitError, the split a GoldenError
        raise (SplitError if question.startswith("settlement") else GoldenError)("no plan in data: run ntk corpus plan first")
    http = TestClient(api.create(client, encode=lambda texts: np.vstack([QUESTION] * len(texts)), rerank="none",
                                 judge=unbuilt), base_url="http://127.0.0.1")
    response = http.post("/ask", json=ASKED, headers=signed(http, "sales"))  # the answer stands, and says why
    assert response.status_code == 200
    assert response.json()["verdict"] == {"error": "no verdict: no plan in data: run ntk corpus plan first"}


def test_the_page_names_every_step_the_server_times_and_draws_the_stages_in_its_order(client, monkeypatch):
    # T6.2b: the page's stage map, web/src/stages.json, names each step a trace's seconds hold, so none of them goes
    # unshown, and gives each to one stage at most; the steps no stage owns are the view's own, shown as such.
    from needtoknow.answer import pipeline
    from needtoknow.settings import repo_root
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", echoing)
    monkeypatch.setattr(pipeline, "warm", lambda encode=None, rerank=None: 0.0)
    plan = json.loads((repo_root() / "web" / "src" / "stages.json").read_text(encoding="utf-8"))
    http = served(client)
    headers = signed(http, "sales")
    searched = set(http.post("/search", json=ASKED, headers=headers).json()["seconds"]) - {"total"}
    asked = set(http.post("/ask", json=ASKED, headers=headers).json()["seconds"]) - {"total"}
    assert set(plan["seconds"]) == asked and searched < asked, (set(plan["seconds"]) ^ asked, searched - asked)
    owned = [step for stage in plan["stages"] for step in stage["timed"]]
    assert len(owned) == len(set(owned)) and set(owned) <= asked
    assert {"verdict"} <= set(owned)  # the scoring timed with the rest
    assert [stage["key"] for stage in plan["stages"]] == ["filter", *api.STAGES, "excerpts", "answer", "verdict"]


def test_the_readmes_calls_to_the_server_answer_as_it_says(client, monkeypatch):
    # MT-F20: the README named the routes and not how to call one; each curl line it gives is sent here to the app as
    # written, the token the sign in answers with in place of <token>.
    import shlex

    from needtoknow.answer import pipeline
    from needtoknow.settings import repo_root
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", echoing)
    monkeypatch.setattr(pipeline, "warm", lambda encode=None, rerank=None: 0.0)
    lines = [line.strip() for line in (repo_root() / "README.md").read_text(encoding="utf-8").splitlines()
             if line.strip().startswith("curl ")]
    assert [shlex.split(line)[shlex.split(line).index("-X") + 2].rsplit("/", 1)[-1] for line in lines] == [
        "session", "search", "ask"]
    http = served(client)
    token = None
    for line in lines:
        words = shlex.split(line)
        url = next(word for word in words if word.startswith(f"http://{api.HOST}:{api.PORT}/"))
        headers = dict(words[i + 1].split(": ", 1) for i, word in enumerate(words) if word == "-H")
        if token:
            headers = {name: value.replace("<token>", token) for name, value in headers.items()}
        response = http.request(words[words.index("-X") + 1], url.removeprefix(f"http://{api.HOST}:{api.PORT}"),
                                headers=headers, content=words[words.index("-d") + 1])
        assert response.status_code == 200, (line[:60], response.text[:200])
        token = token or response.json()["token"]


def test_a_refusals_verdict_reads_the_same_whether_its_answer_exists_hidden_or_nowhere():
    # AC-F16: the category said E, the answer exists where the asker may not read, or J and N, nowhere; and the leak
    # flag, comparing the answer with the hidden value, would confirm a guess at it.
    import types

    from needtoknow.eval import score as S
    ctx = types.SimpleNamespace(facts={"F01": {"id": "F01", "family": "F01", "values": ["15 percent"] * 2}})
    sources = S.Sources(ctx, given={}, units={})
    refusals = [{"id": f"D0{n}", "category": category, "role": "sales", "lang": "en", "question": "the discount",
                 "expect": {"status": "not_found",
                            "forbidden": [{"fact_id": "F01", "forms": ["15 percent"]}] if category == "E" else []}}
                for n, category in enumerate("EJN", 1)]
    for answered in ({"status": "not_found", "answer": "No document here says so.", "citations": []},
                     {"status": "answered", "answer": "The discount is 15 percent.", "citations": []}):
        verdicts = [api.verdict("the discount", "sales", answered, [item], ctx, sources) for item in refusals]
        assert [v.pop("item") for v in verdicts] == ["D01", "D02", "D03"]
        assert verdicts[0] == verdicts[1] == verdicts[2] and verdicts[0]["refusal"], verdicts
        assert not {"category", "leak"} & set(verdicts[0])
    assert S.score_item(refusals[0], answered, ctx, sources)["leak"]  # the scorer saw the guess: the server says nothing


def test_a_search_does_not_wait_for_an_answers_provider(client, monkeypatch):
    # DR-F34: the GPU's lock was held through the provider's network call, so every search queued behind an answer.
    import threading

    from needtoknow.answer import pipeline
    waiting, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow(system, messages, config=None, **settings):
        waiting.set()
        release.wait(3)
        finished.set()
        return echoing(system, messages, config, **settings)

    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", slow)
    monkeypatch.setattr(pipeline, "warm", lambda encode=None, rerank=None: 0.0)
    http = served(client)
    headers = signed(http, "sales")
    answered = []
    asking = threading.Thread(target=lambda: answered.append(http.post("/ask", json=ASKED, headers=headers)))
    asking.start()
    assert waiting.wait(10)
    found = http.post("/search", json=ASKED, headers=headers)  # while the answer waits on its provider
    early = not finished.is_set()
    release.set()
    asking.join(10)
    assert found.status_code == 200 and early and answered[0].status_code == 200


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
