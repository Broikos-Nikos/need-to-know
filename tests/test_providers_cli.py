import argparse
import io
import json
import urllib.error

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


def opener_answering(body: bytes):
    def opener(request, timeout=None):
        return io.BytesIO(body)
    return opener


def test_an_html_page_instead_of_json_is_named_not_a_traceback(isolated):
    # HS-F2: a captive portal or an outage page answers 200 with HTML.
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", opener_answering(b"<html><body>Sign in to Wi-Fi</body></html>"))
    code, text = run()
    assert code == 1 and "not JSON" in text and "<html>" in text and FAKE_KEY not in text


def test_a_refused_key_says_so_with_its_status_and_never_shows_it(isolated):
    # HS-F3: it used to print only "HTTPError".
    def unauthorized(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", unauthorized)
    code, text = run()
    assert code == 1 and "HTTP 401 Unauthorized: DeepSeek refused the key" in text and FAKE_KEY not in text


def test_a_model_item_without_an_id_is_skipped(isolated):
    # HS-F7: it used to raise KeyError: 'id'.
    body = {"data": [{"object": "model"}, {"id": "deepseek-flash"}], "is_available": True, "balance_infos": []}
    isolated.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    isolated.setattr(cli, "OPENER", opener_answering(json.dumps(body).encode("utf-8")))
    code, text = run()
    assert code == 0 and "configured deepseek-flash listed: yes" in text


def test_a_model_name_with_a_typo_is_named_with_its_file_and_key(isolated, tmp_path):
    # HS-F8: it used to raise KeyError before any network call, even offline.
    from needtoknow.providers import catalog
    broken = tmp_path / "providers.toml"
    broken.write_text(catalog.catalog_path().read_text(encoding="utf-8").replace(
        'model = "deepseek-flash"', 'model = "deepseek-flash-2"', 1), encoding="utf-8")
    isolated.setattr(catalog, "catalog_path", lambda: broken)
    code, text = run(offline=True)
    assert code == 1 and "deepseek.model is 'deepseek-flash-2'" in text
