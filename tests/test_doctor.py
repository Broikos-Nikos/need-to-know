"""ntk doctor: Greek survives a pipe, a broken check is one line, and no key is ever printed.

Nothing here asserts what this machine has: a test that expects a GPU or a running
server would be a measurement of the laptop, not of the doctor.
"""

import io
import os
import subprocess
import sys
from collections import namedtuple

import pytest

from needtoknow import doctor

FAKE_KEY = "sk-test-doctor-0123456789abcdef"


def test_greek_survives_a_pipe_whatever_the_code_page():
    # HS-F4: piped on Windows, stdout used the locale's code page and Greek raised.
    env = {name: value for name, value in os.environ.items() if name not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONUTF8"] = "0"
    result = subprocess.run([sys.executable, "-m", "needtoknow.cli", "doctor", "--offline"],
                            capture_output=True, env=env)
    assert b"Traceback" not in result.stderr, result.stderr.decode("utf-8", "replace")
    assert doctor.GREEK in result.stdout.decode("utf-8")


def test_a_check_that_raises_is_one_failed_line_not_a_traceback(monkeypatch):
    def broken(_context):
        raise RuntimeError("the cable is out")
    monkeypatch.setattr(doctor, "CHECKS", [("python", doctor.check_python), ("broken", broken)])
    out = io.StringIO()
    assert doctor.run(offline=True, out=out) == 1
    assert "FAIL  broken    RuntimeError: the cable is out" in out.getvalue()
    assert "doctor: 1 of 2 ok" in out.getvalue()


def test_the_doctor_never_prints_a_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv("QDRANT_API_KEY", FAKE_KEY + "-qdrant")
    out = io.StringIO()
    doctor.run(offline=True, out=out)
    assert FAKE_KEY not in out.getvalue() and FAKE_KEY[8:20] not in out.getvalue()


def test_too_little_free_disk_fails_the_data_check(monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda _path: usage(100 * 2**30, 99 * 2**30, 1 * 2**30))
    ok, detail = doctor.check_data({})
    assert not ok and f"under the {doctor.min_free_gb()} GB" in detail


def test_the_disk_floor_is_a_measured_full_builds_peak_half_again():
    # MA-F3: the floor was 15 GB with nothing measured behind it; it is now the data directory's peak during a full
    # build, recorded beside the command that printed it.
    import math
    import tomllib

    from needtoknow.settings import repo_root
    text = (repo_root() / "config" / "build.toml").read_text(encoding="utf-8")
    assert "bench/full_build.py --fresh" in text
    assert doctor.min_free_gb() == math.ceil(tomllib.loads(text)["measured"]["peak_data_dir_gb"] * 1.5)


def test_a_missing_model_comes_with_the_command_that_fetches_its_pinned_revision(monkeypatch):
    # SC-F3: the doctor names what is missing and how to get exactly that.
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: None)
    from needtoknow.embed.model import load_config
    from needtoknow.index.rerank import load_config as rerank_config
    ok, line = doctor.check_model({})
    config, reranker = load_config(), rerank_config()
    assert not ok and f"uv run hf download {config['repo']} --revision {config['revision']};" in line
    assert line.endswith(f"uv run hf download {reranker['repo']} --revision {reranker['revision']}")
