"""`ntk ask` (T4.5): the answer, its citations, what was cited that no excerpt could take (RS-F14), and the call's
model, tokens, cost and seconds; refusals in one line; the role on the caller's word, said in the help (AC-F7)."""

import argparse
import io

from needtoknow import cli
from needtoknow.answer import pipeline
from needtoknow.index.search import QuestionError
from needtoknow.providers.call import ProviderError

RESULT = {"status": "answered", "answer": "Within 30 days.", "dropped": [9, "x"],
          "citations": [{"doc_id": "RET_POL_011", "section": "Returns policy > Window"}],
          "excerpts": [], "cached": False, "model": "deepseek-flash", "stop_reason": "stop",
          "usage": {"input_hit": 128, "input_miss": 160, "output": 20, "reasoning": 0}, "cost_usd": 0.000036,
          "attempts": 1, "seconds": {"search": 0.61, "expand": 0.01, "provider": 0.9, "parse": 0.0, "total": 1.53},
          "prompt": "ab" * 32}


def run(monkeypatch, result=None, raises=None) -> tuple[int, str]:
    def fake(client, question, role, provider="deepseek", use_cache=True):
        if raises:
            raise raises
        return result
    monkeypatch.setattr(pipeline, "ask", fake)
    out = io.StringIO()
    code = cli.cmd_ask(argparse.Namespace(question="How many days?", role="returns", provider="deepseek",
                                          no_cache=False), out=out, client=object())
    return code, out.getvalue()


def test_a_cache_set_aside_is_said_in_one_line(monkeypatch):
    # HS-F17: the line names the file and says it can be deleted.
    aside = "C:/data/llm-cache/answers.db.unreadable-20261019-000207"
    code, text = run(monkeypatch, {**RESULT, "cache_set_aside": aside})
    assert code == 0 and text.splitlines()[0] == (f"the answer cache could not be read and was set aside as {aside}, "
                                                  "which can be deleted; this answer did without it")
    assert "set aside" not in run(monkeypatch, {**RESULT, "cache_set_aside": None})[1]


def test_an_answer_prints_its_citations_what_could_not_be_cited_and_its_cost(monkeypatch):
    code, text = run(monkeypatch, RESULT)
    assert code == 0
    assert text.splitlines()[:3] == ["answered: Within 30 days.", "  [1] RET_POL_011, Returns policy > Window",
                                     '  cited, but no excerpt shown could take it: 9, "x"']
    assert "288 tokens in, 20 out; 0.000036 USD" in text and "in all 1.53 s" in text
    assert "cached" not in text  # the provider's cache count stays in the spend log (KB5-K12)
    assert "from the cache" in run(monkeypatch, {**RESULT, "cached": True, "dropped": []})[1]
    assert "no price configured" in run(monkeypatch, {**RESULT, "cost_usd": None})[1]


def test_a_refusal_an_unreadable_reply_and_a_failed_call_each_say_so(monkeypatch):
    refused = {**RESULT, "status": "not_found", "answer": "The documents you can read do not answer this question.",
               "citations": [], "dropped": []}
    code, text = run(monkeypatch, refused)
    assert code == 0 and text.startswith("not_found: The documents you can read do not answer this question.")
    code, text = run(monkeypatch, {**RESULT, "status": "invalid", "answer": "", "citations": [], "dropped": [],
                                   "error": "not JSON: Expecting value"})
    assert code == 1 and text.startswith("the reply could not be read: not JSON")
    for error in (QuestionError("a question needs at least one word"), ProviderError("DeepSeek refused the call")):
        code, text = run(monkeypatch, raises=error)
        assert code == 1 and text.strip().splitlines() == [f"ask: {error}"]


def test_the_help_says_the_role_is_the_callers_word():
    help_text = cli.build_parser()._subparsers._group_actions[0].choices["ask"].format_help()
    assert "on the caller's word" in " ".join(help_text.split()) and "operator's tool" in help_text
