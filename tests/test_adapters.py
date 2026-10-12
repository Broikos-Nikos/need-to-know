"""The OpenAI, Anthropic and Gemini adapters (T4.3, N13), on replies shaped as each provider's reference documents
them (OpenAI's Responses API, Anthropic's Messages API, Gemini's generateContent, read 2026-10-13). No call was made:
no key for these providers exists here, and nothing in these tests claims more than the request and the reading."""

import io
import json
import urllib.error

import pytest

from needtoknow import settings
from needtoknow.providers import anthropic, gemini, openai
from needtoknow.providers.call import ProviderError
from needtoknow.providers.catalog import load_catalog

KEYS = {"OPENAI_API_KEY": "sk-openai-test-0123456789abcdef", "ANTHROPIC_API_KEY": "sk-ant-test-0123456789abcdef",
        "GEMINI_API_KEY": "AIza-test-0123456789abcdef"}
ANSWER = '{"status": "answered", "answer": "Within 30 days.", "citations": [1]}'
MESSAGES = [{"role": "user", "content": "How many days?"}, {"role": "assistant", "content": "Which device?"},
            {"role": "user", "content": "A phone."}]

OPENAI_REPLY = {
    "id": "resp_test", "object": "response", "status": "completed", "model": "gpt-test-2026",
    "output": [{"type": "reasoning", "id": "rs_test", "summary": [{"type": "summary_text", "text": "Excerpt 1 says it."}]},
               {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]}],
    "usage": {"input_tokens": 3000, "input_tokens_details": {"cached_tokens": 1024}, "output_tokens": 300,
              "output_tokens_details": {"reasoning_tokens": 120}, "total_tokens": 3300}}
ANTHROPIC_REPLY = {
    "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test-2026",
    "content": [{"type": "thinking", "thinking": "Excerpt 1 says it.", "signature": "sig"},
                {"type": "text", "text": ANSWER}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 50, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 2800,
              "output_tokens": 300}}
GEMINI_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Excerpt 1 says it.", "thought": True},
                                                           {"text": ANSWER}]},
                    "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 3000, "cachedContentTokenCount": 2048, "candidatesTokenCount": 40,
                      "thoughtsTokenCount": 260, "totalTokenCount": 3300},
    "modelVersion": "gemini-test-001"}


class Scripted:
    """An opener answering from a script: a dict is a reply, an int an HTTP error whose message echoes the caller's
    key masked in part, as some providers' errors do; a scrub of the whole key would let it through."""

    def __init__(self, *script):
        self.script, self.sent = list(script), []

    def __call__(self, request, timeout=None):
        self.sent.append(request)
        step = self.script.pop(0)
        if isinstance(step, int):
            sent = [v for v in KEYS.values() if v in str(request.header_items())][0]
            said = {"error": {"message": f"invalid key {sent[:12]}****{sent[-4:]}"}}
            raise urllib.error.HTTPError(request.full_url, step, "Test", {}, io.BytesIO(json.dumps(said).encode()))
        return io.BytesIO(json.dumps(step).encode("utf-8"))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "env_file", lambda: tmp_path / ".env")  # never the real .env
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    catalog = load_catalog()
    for provider, model in (("openai", "gpt-test"), ("anthropic", "claude-test"), ("gemini", "gemini-test")):
        catalog[provider]["model"] = model
    return catalog


def ask(module, config, *script, **options):
    opener = Scripted(*script)
    done = module.complete("Rules. Reply with one JSON object.", MESSAGES, config=config, opener=opener,
                           sleep=lambda _seconds: None, **options)
    return done, opener


def test_openai_asks_the_responses_api_statelessly_and_reads_its_usage(config):
    done, opener = ask(openai, config, OPENAI_REPLY, as_json=True, thinking=True, max_tokens=800)
    sent = json.loads(opener.sent[0].data)
    assert opener.sent[0].full_url == "https://api.openai.com/v1/responses"
    assert opener.sent[0].get_header("Authorization") == "Bearer " + KEYS["OPENAI_API_KEY"]
    assert sent["instructions"] == "Rules. Reply with one JSON object." and sent["input"] == MESSAGES
    assert sent["store"] is False and sent["text"] == {"format": {"type": "json_object"}}
    assert "response_format" not in sent  # ignored on Responses: no JSON would be enforced
    assert (sent["reasoning"], sent["max_output_tokens"]) == ({"effort": "high"}, 800)
    assert (done.text, done.reasoning, done.stop_reason, done.model) == (ANSWER, "Excerpt 1 says it.", "stop",
                                                                         "gpt-test-2026")
    assert done.usage == {"input_hit": 1024, "input_miss": 1976, "output": 300, "reasoning": 120}
    assert done.cost_usd is None  # no price configured for the model
    off, opener = ask(openai, config, OPENAI_REPLY, thinking=False)
    left, opener_left = ask(openai, config, OPENAI_REPLY)
    assert json.loads(opener.sent[0].data)["reasoning"] == {"effort": "none"}
    assert "reasoning" not in json.loads(opener_left.sent[0].data)


def test_openai_reads_a_cut_reply_as_length_and_a_refusal_as_refusal(config):
    cut = {**OPENAI_REPLY, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    refused = {**OPENAI_REPLY, "output": [{"type": "message", "role": "assistant", "status": "completed",
                                           "content": [{"type": "refusal", "refusal": "I can't help with that."}]}]}
    assert ask(openai, config, cut)[0].stop_reason == "length"
    done = ask(openai, config, refused)[0]
    assert (done.stop_reason, done.text) == ("refusal", "")


def test_anthropic_sends_its_version_and_a_max_tokens_and_reads_cache_reads_as_hits(config):
    done, opener = ask(anthropic, config, ANTHROPIC_REPLY, as_json=True, thinking=True)
    request = opener.sent[0]
    sent = json.loads(request.data)
    assert request.full_url == "https://api.anthropic.com/v1/messages"
    assert request.get_header("X-api-key") == KEYS["ANTHROPIC_API_KEY"] and request.get_header("Authorization") is None
    assert request.get_header("Anthropic-version") == "2023-06-01"
    assert sent["system"] == "Rules. Reply with one JSON object." and sent["messages"] == MESSAGES
    assert sent["max_tokens"] == 4096 and sent["thinking"] == {"type": "adaptive"}
    assert not {"temperature", "response_format", "output_config"} & set(sent)
    assert (done.text, done.reasoning, done.stop_reason) == (ANSWER, "Excerpt 1 says it.", "stop")
    assert done.usage == {"input_hit": 2800, "input_miss": 250, "output": 300, "reasoning": None}
    off, opener = ask(anthropic, config, {**ANTHROPIC_REPLY, "stop_reason": "max_tokens"}, thinking=False,
                      temperature=0.0)
    assert json.loads(opener.sent[0].data)["thinking"] == {"type": "disabled"} and off.stop_reason == "length"
    assert json.loads(opener.sent[0].data)["temperature"] == 0.0


def test_anthropic_reads_a_refusal_that_arrives_as_a_200_and_retries_an_overloaded_server(config):
    refused = {**ANTHROPIC_REPLY, "content": [], "stop_reason": "refusal"}
    done, opener = ask(anthropic, config, 529, refused)
    assert (done.stop_reason, done.text, done.attempts, len(opener.sent)) == ("refusal", "", 2, 2)


def test_gemini_keeps_the_key_out_of_the_url_and_bills_thoughts_as_output(config):
    done, opener = ask(gemini, config, GEMINI_REPLY, as_json=True, thinking=True, max_tokens=800, temperature=0.0)
    request = opener.sent[0]
    sent = json.loads(request.data)
    assert request.full_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    assert KEYS["GEMINI_API_KEY"] not in request.full_url
    assert request.get_header("X-goog-api-key") == KEYS["GEMINI_API_KEY"]
    assert sent["systemInstruction"] == {"parts": [{"text": "Rules. Reply with one JSON object."}]}
    assert [c["role"] for c in sent["contents"]] == ["user", "model", "user"]
    assert sent["generationConfig"] == {"responseMimeType": "application/json", "maxOutputTokens": 800,
                                        "temperature": 0.0,
                                        "thinkingConfig": {"thinkingLevel": "high", "includeThoughts": True}}
    assert (done.text, done.reasoning, done.stop_reason, done.model) == (ANSWER, "Excerpt 1 says it.", "stop",
                                                                         "gemini-test-001")
    assert done.usage == {"input_hit": 2048, "input_miss": 952, "output": 300, "reasoning": 260}
    left, opener = ask(gemini, config, GEMINI_REPLY)
    assert "generationConfig" not in json.loads(opener.sent[0].data)


def test_gemini_refuses_to_turn_thinking_off_before_any_call_and_reads_a_blocked_prompt(config):
    opener = Scripted(GEMINI_REPLY)
    with pytest.raises(ProviderError, match="no thinking level that turns it off"):
        gemini.complete("Rules.", MESSAGES, thinking=False, config=config, opener=opener)
    assert opener.sent == []
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 40}}
    done = ask(gemini, config, blocked)[0]
    assert (done.stop_reason, done.text, done.usage["output"]) == ("content_filter", "", 0)
    assert ask(gemini, config, {**GEMINI_REPLY, "candidates": [{**GEMINI_REPLY["candidates"][0],
                                                                "finishReason": "MAX_TOKENS"}]})[0].stop_reason == "length"


@pytest.mark.parametrize("module", [openai, anthropic, gemini])
def test_no_key_reaches_an_error_and_a_missing_model_is_named(module, config, monkeypatch):
    for status in (400, 401, 404):
        with pytest.raises(ProviderError) as caught:
            ask(module, config, status)
        key = KEYS[f"{module.__name__.rsplit('.', 1)[-1].upper()}_API_KEY"]
        assert not any(key[i:i + 8] in str(caught.value) for i in range(len(key) - 7)), status
    name = module.__name__.rsplit(".", 1)[-1]
    config[name]["model"] = ""
    with pytest.raises(ProviderError, match=f"{name}.model"):
        ask(module, config, OPENAI_REPLY)


def test_a_price_in_the_catalogue_is_charged_and_none_without_one(config):
    config["anthropic"]["models"] = {"claude-test": {"price": {"off_peak": {"input_hit": 0.5, "input_miss": 5.0,
                                                                           "output": 25.0}}}}
    done = ask(anthropic, config, ANTHROPIC_REPLY)[0]
    assert done.cost_usd == pytest.approx((2800 * 0.5 + 250 * 5.0 + 300 * 25.0) / 1_000_000)
    assert ask(gemini, config, GEMINI_REPLY)[0].cost_usd is None
