"""One question end to end (T4.4), with the search, the excerpts and the provider stood in for: the answer cache keyed
by the prompt as sent, so a narrowed section is never answered from it (AC-F8); only normal replies kept; the settings
read from the catalogue; and a spend log that holds numbers and hashes and no text."""

import json

import pytest

from needtoknow.answer import pipeline
from needtoknow.providers.call import Completion

QUESTION = "What is the settlement ceiling?"
SECRET = {"doc_id": "LEG_SET_004", "section_key": "k1", "labels": ["legal", "sales"], "path": ["Settlements", "Ceiling"],
          "units": ["u1"], "rank": 1, "text": "The settlement ceiling is 2,000,000 EUR."}
PUBLIC = {"doc_id": "SAL_PRO_033", "section_key": "k2", "labels": ["sales"], "path": ["Sales procedure", "Authorisation"],
          "units": ["u2"], "rank": 2, "text": "Return authorisation is issued by the returns team."}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """What each role is shown, the provider's calls, and the stop reason it gives."""
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    state = {"shown": {"sales": [SECRET, PUBLIC], "legal": [SECRET, PUBLIC]}, "calls": [], "stop": "stop"}
    monkeypatch.setattr(pipeline, "search", lambda client, question, role, policy=None, **kw: [{"role": role}])
    monkeypatch.setattr(pipeline, "expand", lambda hits: [dict(e) for e in state["shown"][hits[0]["role"]]])

    def provider(system, messages, config=None, **settings):
        state["calls"].append({"messages": messages, **settings})
        found = "2,000,000" in messages[0]["content"]
        reply = {"status": "answered", "answer": "2,000,000 EUR.", "citations": [1]} if found else \
            {"status": "not_found", "answer": "", "citations": []}
        return Completion(text=json.dumps(reply), reasoning="", usage={"input_hit": 0, "input_miss": 300, "output": 20,
                                                                       "reasoning": 0},
                          stop_reason=state["stop"], seconds=0.01, cost_usd=0.00006, model="deepseek-flash", attempts=1)

    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", provider)
    return state


def test_a_narrowed_section_is_never_answered_from_the_cache(world):
    first = pipeline.ask(None, QUESTION, "sales")
    again = pipeline.ask(None, QUESTION, "sales")
    assert (first["status"], first["citations"][0]["doc_id"], first["cached"]) == ("answered", "LEG_SET_004", False)
    assert again["cached"] and len(world["calls"]) == 1
    world["shown"]["sales"] = [PUBLIC]  # the ceiling's section narrowed to legal alone
    after = pipeline.ask(None, QUESTION, "sales")
    assert (after["status"], after["cached"], len(world["calls"])) == ("not_found", False, 2)
    assert "2,000,000" not in json.dumps(after)
    other = pipeline.ask(None, QUESTION, "legal")  # shown the very text sales was shown first, reading other labels
    assert not other["cached"] and other["status"] == "answered" and len(world["calls"]) == 3


def test_a_cached_answer_is_shared_only_by_roles_that_read_the_same_labels(world):
    # AC-F11: from the cache, a reply would tell a role that someone with other access asked over the same text.
    policy = {"roles": {"sales": {"reads": ["general", "sales", "returns"]},
                        "seller": {"reads": ["returns", "sales", "general"]}, "legal": {"reads": ["general", "legal"]}}}
    world["shown"]["seller"] = world["shown"]["sales"]
    first = pipeline.ask(None, QUESTION, "sales", policy)
    same = pipeline.ask(None, QUESTION, "seller", policy)
    other = pipeline.ask(None, QUESTION, "legal", policy)
    assert (first["cached"], same["cached"], other["cached"], len(world["calls"])) == (False, True, False, 2)
    sent = [call["messages"][0]["content"].split("\n")[0] for call in world["calls"]]
    assert sent[0] != sent[1] and all(line.startswith("Reader: ") for line in sent)


def test_each_repeat_of_an_evaluation_is_asked_once_and_read_back_after(world):
    # KB5-K13: with the repeat outside the key, repeats two and three came back from the cache and looked stable.
    for repeat in (0, 1, 2):
        pipeline.ask(None, QUESTION, "sales", repeat=repeat)
    assert len(world["calls"]) == 3
    again = [pipeline.ask(None, QUESTION, "sales", repeat=repeat)["cached"] for repeat in (0, 1, 2)]
    assert again == [True, True, True] and len(world["calls"]) == 3
    lines = [json.loads(line) for line in pipeline.spend_path().read_text(encoding="utf-8").splitlines()]
    assert [(line["repeat"], line["cached"]) for line in lines] == [(0, False), (1, False), (2, False), (0, True),
                                                                     (1, True), (2, True)]


def test_an_entry_goes_stale_when_its_alias_answers_with_another_model(world, monkeypatch):
    # KB5-K14: the provider moves an alias under its callers, and an old model's answers must not outlive the move.
    pipeline.ask(None, QUESTION, "sales")
    assert pipeline.ask(None, QUESTION, "sales")["cached"]
    from dataclasses import replace
    real = pipeline.ADAPTERS["deepseek"]
    monkeypatch.setitem(pipeline.ADAPTERS, "deepseek", lambda *args, **kwargs: replace(real(*args, **kwargs),
                                                                                        model="deepseek-next"))
    pipeline.ask(None, "Who issues the authorisation?", "sales")  # a fresh reply names the new model
    moved = pipeline.ask(None, QUESTION, "sales")
    assert (moved["cached"], moved["model"]) == (False, "deepseek-next")


def test_only_a_reply_that_ended_normally_is_kept_and_the_settings_come_from_the_catalogue(world):
    world["stop"] = "length"
    pipeline.ask(None, QUESTION, "sales")
    world["stop"] = "stop"
    pipeline.ask(None, QUESTION, "sales")
    assert len(world["calls"]) == 2  # the cut reply was not kept
    sent = world["calls"][0]
    assert (sent["thinking"], sent["as_json"], sent["max_tokens"], sent["temperature"]) == (False, True, 1024, 0.0)


def test_the_spend_log_holds_numbers_and_no_text_and_sums(world):
    pipeline.ask(None, QUESTION, "sales")
    pipeline.ask(None, QUESTION, "sales")
    world["shown"]["sales"] = [PUBLIC]
    pipeline.ask(None, QUESTION, "sales")
    raw = pipeline.spend_path().read_text(encoding="utf-8")
    for text in (QUESTION, SECRET["text"], PUBLIC["text"], "2,000,000", "Settlements", "settlement"):
        assert text not in raw, text
    lines = [json.loads(line) for line in raw.splitlines()]
    assert [line["cached"] for line in lines] == [False, True, False]
    assert set(lines[0]["seconds"]) == {"search", "expand", "provider", "parse", "total"}
    assert (lines[1]["usd"], lines[1]["attempts"]) == (0.0, 0)  # nothing billed for a cached answer
    total = pipeline.spend()
    assert (total["questions"], total["cached"], total["usd"]) == (3, 1, pytest.approx(0.00012))
    assert total["tokens"]["input_miss"] == 600 and total["p95_seconds"] >= total["median_seconds"]
