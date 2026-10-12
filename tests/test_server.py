"""The local Qdrant wrapper: loopback only, the key never on a command line or in a log, a pinned binary,
anonymous requests refused, data kept across a hard stop, and no other process ever stopped."""

import os
import secrets
import socket
import subprocess

import pytest

import needtoknow.settings
from needtoknow.index import server

WINDOWS = os.name == "nt"
CONFIG = server.load_config()
LIVE = WINDOWS and server.binary_path(CONFIG).is_file()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A throwaway data directory, .env and key, with the real pinned binary and two free ports."""
    monkeypatch.setenv("NTK_QDRANT_BIN", str(server.binary_path(CONFIG)))
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QDRANT_API_KEY", secrets.token_urlsafe(32))
    config = dict(CONFIG, http_port=free_port(), grpc_port=free_port())
    yield config
    if server.running(config):
        server.stop(config)


def test_the_server_listens_on_the_loopback_interface_only(tmp_path):
    env = server.server_env(CONFIG, "key", tmp_path)
    assert CONFIG["host"] == env["QDRANT__SERVICE__HOST"] == "127.0.0.1"


def test_inherited_qdrant_settings_cannot_override_the_wrapper(monkeypatch, tmp_path):
    monkeypatch.setenv("QDRANT__SERVICE__HOST", "0.0.0.0")
    monkeypatch.setenv("QDRANT__SERVICE__ENABLE_CORS", "true")
    env = server.server_env(CONFIG, "key", tmp_path)
    assert env["QDRANT__SERVICE__HOST"] == "127.0.0.1"
    assert "QDRANT__SERVICE__ENABLE_CORS" not in env


def test_telemetry_is_off_and_storage_stays_in_the_data_directory(tmp_path):
    env = server.server_env(CONFIG, "key", tmp_path)
    assert env["QDRANT__TELEMETRY_DISABLED"] == "true"
    assert env["QDRANT__STORAGE__STORAGE_PATH"].startswith(str(tmp_path))
    assert env["QDRANT__STORAGE__SNAPSHOTS_PATH"].startswith(str(tmp_path))


def test_a_binary_with_another_hash_is_refused(tmp_path):
    fake = tmp_path / "qdrant.exe"
    fake.write_bytes(b"not qdrant")
    with pytest.raises(server.ServerError, match="refusing to run it"):
        server.check_binary(fake, CONFIG["exe_sha256"])


def test_a_missing_binary_says_where_it_looked_and_how_to_get_it(tmp_path):
    with pytest.raises(server.ServerError, match="no Qdrant binary at") as error:
        server.check_binary(tmp_path / "qdrant.exe", CONFIG["exe_sha256"])
    # SC-F3: the asset to download and the digest it is checked against.
    assert CONFIG["asset"] in str(error.value) and CONFIG["asset_sha256"] in str(error.value)


def test_a_missing_key_is_written_to_env_once(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("NTK_DATA_DIR=./data", encoding="utf-8")  # no final newline
    monkeypatch.setattr(needtoknow.settings, "env_file", lambda: env_path)
    monkeypatch.setattr(server, "env_file", lambda: env_path)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    key, created = server.ensure_key(CONFIG)
    again, created_again = server.ensure_key(CONFIG)
    assert created and not created_again and again == key and len(key) >= 40
    assert env_path.read_text(encoding="utf-8") == f"NTK_DATA_DIR=./data\nQDRANT_API_KEY={key}\n"


@pytest.mark.skipif(not WINDOWS, reason="process identity is read through the Windows API")
def test_stop_never_touches_a_process_that_is_not_this_projects_qdrant(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    server.pid_file().parent.mkdir(parents=True)
    server.pid_file().write_text(str(os.getpid()))  # this test's own Python process
    stopped = []
    monkeypatch.setattr(server, "terminate", stopped.append)
    with pytest.raises(server.ServerError, match="not this project's Qdrant"):
        server.stop(CONFIG)
    assert stopped == []


def test_a_stale_pid_file_is_cleared_without_stopping_anything(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    server.pid_file().parent.mkdir(parents=True)
    server.pid_file().write_text("4194300")
    stopped = []
    monkeypatch.setattr(server, "terminate", stopped.append)
    assert server.stop(CONFIG) is None and stopped == [] and not server.pid_file().exists()


@pytest.mark.skipif(not LIVE, reason="needs the pinned qdrant.exe on Windows")
@pytest.mark.filterwarnings("ignore:Api key is used with an insecure connection")
def test_live_server_refuses_anonymous_requests_and_keeps_data_across_a_hard_stop(isolated, tmp_path):
    from qdrant_client import QdrantClient, models

    config, key = isolated, os.environ["QDRANT_API_KEY"]
    pid, seconds, created = server.start(config)
    assert not created and seconds < 30
    assert server.http_status(config, "/collections") == 401
    assert server.http_status(config, "/collections", "wrong" + key) == 401

    listening = [line.split() for line in subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout.splitlines()
                 if "LISTENING" in line and line.split()[-1] == str(pid)]
    assert listening and all(row[1].startswith("127.0.0.1:") for row in listening), listening
    command_line = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                                  capture_output=True, text=True).stdout
    assert command_line.strip() and key not in command_line

    client = QdrantClient(url=f"http://127.0.0.1:{config['http_port']}", api_key=key, prefer_grpc=False, timeout=10)
    client.create_collection("smoke", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE))
    client.upsert("smoke", wait=True, points=[models.PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"labels": ["hr"]})])
    assert server.stop(config) == pid
    assert server.http_status(config, "/readyz") is None

    server.start(config)
    (point,) = client.retrieve("smoke", ids=[1])
    assert point.payload == {"labels": ["hr"]}
    server.stop(config)
    assert key not in (tmp_path / "logs" / "qdrant.log").read_text(encoding="utf-8", errors="replace")


@pytest.mark.skipif(not LIVE, reason="needs the pinned qdrant.exe on Windows")
def test_live_server_that_answers_without_the_key_is_stopped(isolated, monkeypatch):
    keyed = server.server_env
    monkeypatch.setattr(server, "server_env", lambda *args: {k: v for k, v in keyed(*args).items() if k != "QDRANT__SERVICE__API_KEY"})
    with pytest.raises(server.ServerError, match="answered without the API key"):
        server.start(isolated)
    assert server.running(isolated) is None and server.http_status(isolated, "/readyz") is None
