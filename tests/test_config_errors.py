"""A config file a stranger mistypes is named in one line, never a traceback (the class behind HS-F8)."""

import argparse
import io

from needtoknow import cli
from needtoknow.index import server
from needtoknow.providers import catalog


def test_malformed_providers_toml_is_named(monkeypatch, tmp_path):
    broken = tmp_path / "providers.toml"
    broken.write_text('[deepseek\nmodel = "deepseek-flash"\n', encoding="utf-8")
    monkeypatch.setattr(catalog, "catalog_path", lambda: broken)
    out = io.StringIO()
    assert cli.cmd_providers(argparse.Namespace(offline=True), out=out) == 1
    assert "config/providers.toml is not valid TOML" in out.getvalue()


def test_malformed_qdrant_toml_is_named(monkeypatch, tmp_path):
    broken = tmp_path / "qdrant.toml"
    broken.write_text("http_port = = 6433\n", encoding="utf-8")
    real = server.load_config
    monkeypatch.setattr(server, "load_config", lambda path=None: real(broken))
    out = io.StringIO()
    assert cli.cmd_qdrant(argparse.Namespace(action="status"), out=out) == 1
    assert "config/qdrant.toml is not valid TOML" in out.getvalue()
