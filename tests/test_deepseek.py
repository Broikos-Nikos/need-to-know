"""The DeepSeek adapter (T4.2) against a scripted opener: the request it sends, the usage and cost it reads back, what
it tries again and what it refuses, and that the key never reaches an error."""

import datetime as dt
import io
import json
import urllib.error

import pytest

from needtoknow import settings
from needtoknow.providers import deepseek
from needtoknow.providers.catalog import load_catalog

FAKE_KEY = "sk-test-0123456789abcdef0123456789"
OFF_PEAK = dt.datetime(2026, 10, 12, 23, 0, tzinfo=dt.timezone.utc)  # a Monday, outside both windows
PEAK = dt.datetime(2026, 10, 13, 2, 0, tzinfo=dt.timezone.utc)  # a Tuesday, inside 01:00 to 04:00 UTC


def reply(content='{"status": "answered", "answer": "Within 30 days.", "citations": [1]}', finish="stop", hit=1000,
          miss=2000, out=300, reasoning=0, reasoning_content=None) -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {"model": "deepseek-flash", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": hit + miss, "prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": miss,
                      "completion_tokens": out, "total_tokens": hit + miss + out,
                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}


class Fake:
    """An opener that answers each call from a script: a dict is a reply, bytes a raw body, an int an HTTP error with
    that status, an exception a failed connection."""

    def __init__(self, *script):
        self.script, self.sent = list(script), []

    def __call__(self, request, timeout=None):
        self.sent.append(request)
        step = self.script.pop(0)
        if isinstance(step, int):
            said = {"error": {"message": f"Authentication Fails, your api key {FAKE_KEY} is invalid", "type": "x"}}
            raise urllib.error.HTTPError(request.full_url, step, "Fake", {}, io.BytesIO(json.dumps(said).encode()))
        if isinstance(step, Exception):
            raise step
        return io.BytesIO(step if isinstance(step, bytes) else json.dumps(step).encode("utf-8"))


@pytest.fixture(autouse=True)
def keyed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "env_file", lambda: tmp_path / ".env")  # never the real .env and its key
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    return monkeypatch


def call(fake: Fake, when: dt.datetime = OFF_PEAK, **options):
    slept = []
    done = deepseek.complete("Rules. Reply with one JSON object.", [{"role": "user", "content": "How many days?"}],
                             opener=fake, now=lambda: when, sleep=slept.append, **options)
    return done, slept


def test_thinking_is_a_top_level_field_off_unless_asked():
    # N13: thinking is the API's default, and inside extra_body a raw JSON body leaves it on.
    fake = Fake(reply(), reply(reasoning=19, reasoning_content="The first excerpt says 30 days."))
    off, _ = call(fake, as_json=True)
    on, _ = call(fake, thinking=True, max_tokens=4096, temperature=0.3)
    first, second = (json.loads(request.data) for request in fake.sent)
    assert first["thinking"] == {"type": "disabled"} and "extra_body" not in first
    assert second["thinking"] == {"type": "enabled"}
    assert first["response_format"] == {"type": "json_object"} and "response_format" not in second
    assert "max_tokens" not in first and "temperature" not in first  # the API's own defaults
    assert (second["max_tokens"], second["temperature"]) == (4096, 0.3)
    assert first["model"] == "deepseek-flash" and first["messages"][0]["role"] == "system"
    assert fake.sent[0].full_url == "https://api.deepseek.com/chat/completions"
    assert fake.sent[0].get_header("Authorization") == "Bearer " + FAKE_KEY
    assert (off.reasoning, on.reasoning, on.usage["reasoning"]) == ("", "The first excerpt says 30 days.", 19)


def test_usage_is_billed_as_deepseek_prices_it_and_peak_hours_cost_double():
    rates = load_catalog()["deepseek"]["models"]["deepseek-flash"]["price"]["off_peak"]
    assert rates["input_hit"] < rates["input_miss"]  # else the split below would prove nothing
    off, _ = call(Fake(reply(hit=1000, miss=2000, out=300)))
    peak, _ = call(Fake(reply(hit=1000, miss=2000, out=300)), when=PEAK)
    assert off.usage == {"input_hit": 1000, "input_miss": 2000, "output": 300, "reasoning": 0}
    assert off.cost_usd == pytest.approx((1000 * rates["input_hit"] + 2000 * rates["input_miss"]
                                          + 300 * rates["output"]) / 1_000_000)
    assert peak.cost_usd == pytest.approx(2 * off.cost_usd)
    assert (off.stop_reason, off.model, off.attempts) == ("stop", "deepseek-flash", 1)


def test_a_busy_server_is_tried_again_and_a_refusal_is_not():
    done, slept = call(Fake(429, 503, urllib.error.URLError("connection reset"), reply()))
    assert done.attempts == 4 and slept == [1.0, 2.0, 4.0]
    for status, words in ((402, "the balance is used up"), (401, "DeepSeek refused the key"),
                          (422, "a parameter is invalid")):
        fake = Fake(status, reply())
        with pytest.raises(deepseek.ProviderError, match=words):
            call(fake)
        assert len(fake.sent) == 1, status  # never tried again
    with pytest.raises(deepseek.ProviderError, match="HTTP 503"):
        call(Fake(503, 503, 503, 503))


def test_a_body_cut_off_mid_read_is_tried_again_and_then_named_without_the_key():
    # DR-F18: http.client's IncompleteRead is no OSError, and escaped the retries after one attempt as itself.
    import http.client
    done, slept = call(Fake(http.client.IncompleteRead(b'{"choices'), reply()))
    assert (done.attempts, slept) == (2, [1.0])
    fake = Fake(*[http.client.IncompleteRead(b"{") for _ in range(4)])
    with pytest.raises(deepseek.ProviderError, match="could not be reached: IncompleteRead") as raised:
        call(fake)
    assert len(fake.sent) == 4 and FAKE_KEY not in str(raised.value)


def test_a_reply_cut_short_by_the_server_is_billed_and_tried_again_and_one_cut_by_length_is_returned():
    done, slept = call(Fake(reply(content="", finish="insufficient_system_resource", out=5), reply()))
    assert (done.attempts, done.stop_reason, slept) == (2, "stop", [1.0])
    assert (done.usage["input_miss"], done.usage["output"]) == (4000, 305)  # both attempts were billed
    cut, _ = call(Fake(reply(content='{"status": "answ', finish="length")))
    assert (cut.stop_reason, cut.attempts, cut.text) == ("length", 1, '{"status": "answ')


def test_the_key_never_reaches_an_error_and_a_missing_one_is_named(keyed):
    for script in ((401,), (400,), (422,), (b"<html>Sign in to the network</html>",)):
        with pytest.raises(deepseek.ProviderError) as caught:
            call(Fake(*script))
        assert FAKE_KEY not in str(caught.value) and FAKE_KEY[3:15] not in str(caught.value), script
    keyed.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(deepseek.ProviderError, match="DEEPSEEK_API_KEY"):
        call(Fake(reply()))


def test_a_completion_of_the_wrong_shape_is_refused_or_read_as_empty():
    # DR-F17's class: choices as an object raised a KeyError, a message given as text an AttributeError.
    with pytest.raises(deepseek.ProviderError, match="not with a completion"):
        call(Fake({"choices": {"0": {}}}))
    done, _ = call(Fake({"model": "deepseek-flash", "choices": [{"message": "x", "finish_reason": 5}],
                         "usage": {"prompt_tokens": "many", "completion_tokens_details": []}}))
    assert (done.text, done.stop_reason) == ("", "")
    assert done.usage == {"input_hit": 0, "input_miss": 0, "output": 0, "reasoning": 0}
