import argparse
import io
import json

import pytest

from needtoknow import cli, settings

FAKE_KEY = "sk-test-0123456789abcdef0123456789"


def fake_opener(models=("deepseek-flash", "deepseek-v4-pro"), available=True):
    def opener(request, timeout=None):
        if request.full_url.endswith("/models"):
            body = {"data": [{"id": m} for m in models]}
        elif request.full_url.endswith("/user/balance"):
            body = {"is_available": available, "balance_infos": [{"currency": "USD", "total_balance": "6.96"}]}
        else:
            raise AssertionError(f"unexpected url {request.full_url}")
        assert request.get_header("Authorization") == "Bearer " + FAKE_KEY
        return io.BytesIO(json.dumps(body).encode("utf-8"))
    return opener


def refusing_opener(request, timeout=None):
    raise AssertionError("the offline path must not touch the network")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    # Never read the real .env, which holds the real key.
    monkeypatch.setattr(settings, "env_file", lambda: tmp_path / ".env")
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def run(offline=False):
    out = io.StringIO()
    code = cli.cmd_providers(argparse.Namespace(offline=offline), out=out)
    return code, out.getvalue()


def test_live_check_reports_models_and_balance_and_never_the_key(isolated):
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", fake_opener())
    code, text = run()
    assert code == 0
    assert "configured deepseek-flash listed: yes" in text
    assert "6.96 USD" in text
    assert FAKE_KEY not in text
    assert FAKE_KEY[3:15] not in text


def test_a_configured_model_that_vanished_fails(isolated):
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", fake_opener(models=("deepseek-v4-pro",)))
    code, text = run()
    assert code == 1
    assert "listed: no" in text


def test_an_unusable_balance_fails(isolated):
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", fake_opener(available=False))
    code, text = run()
    assert code == 1
    assert "usable: no" in text


def test_no_key_fails_without_calling_out(isolated):
    isolated.setattr(cli, "OPENER", refusing_opener)
    code, text = run()
    assert code == 1
    assert "DEEPSEEK_API_KEY is not set" in text


def test_offline_never_calls_out(isolated):
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", refusing_opener)
    code, text = run(offline=True)
    assert code == 0
    assert "skipped: --offline" in text
    assert FAKE_KEY not in text


def test_every_provider_is_listed_with_its_key_state(isolated):
    isolated.setenv("OPENAI_API_KEY", "set-but-not-shown")
    code, text = run(offline=True)
    lines = {line.split()[0]: line for line in text.splitlines() if "key set:" in line}
    assert set(lines) == set(cli.PROVIDERS)
    assert "key set: yes" in lines["openai"]
    assert "key set: no" in lines["deepseek"]
    assert "set-but-not-shown" not in text
