"""The local Qdrant server: a pinned binary on the loopback interface, behind an API key.

Qdrant runs as a detached process with its storage under NTK_DATA_DIR/qdrant. Before
every start the binary is checked against its pinned sha256; the key reaches the
server through its environment, never its command line; and once the server is
ready, start confirms it refuses a request without the key. Stop ends a process only
after confirming the pid file still points at this project's own binary, so another
Qdrant on the same machine is never touched. A hard stop is safe: Qdrant replays its
write ahead log on the next start.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import signal
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from needtoknow.settings import data_dir, env_file, repo_root, setting

# Detached and in its own process group, so the server outlives the command that started it.
DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
# Loopback calls never go through a proxy, whatever the machine's proxy settings say.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ServerError(Exception):
    """The binary, the key, the port or the process is not what it must be."""


def load_config(path: Path | None = None) -> dict:
    try:
        with open(path or repo_root() / "config" / "qdrant.toml", "rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ServerError(f"config/qdrant.toml is not valid TOML: {error}") from None


def binary_path(config: dict) -> Path:
    """NTK_QDRANT_BIN when set, else the pinned version's folder under NTK_DATA_DIR/bin."""
    override = setting("NTK_QDRANT_BIN")
    return Path(override) if override else data_dir() / "bin" / f"qdrant-{config['version']}" / "qdrant.exe"


def pid_file() -> Path:
    return data_dir() / "qdrant" / "qdrant.pid"


def check_binary(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise ServerError(f"no Qdrant binary at {path}; unpack qdrant.exe there from the asset in config/qdrant.toml")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ServerError(f"{path} has sha256 {digest[:12]}, not the pinned {expected_sha256[:12]}; refusing to run it")


def server_env(config: dict, key: str, storage: Path) -> dict[str, str]:
    """Qdrant's settings as environment variables, so neither a file nor an argument carries the key."""
    inherited = {name: value for name, value in os.environ.items() if not name.upper().startswith("QDRANT__")}
    return inherited | {
        "QDRANT__SERVICE__HOST": config["host"],
        "QDRANT__SERVICE__HTTP_PORT": str(config["http_port"]),
        "QDRANT__SERVICE__GRPC_PORT": str(config["grpc_port"]),
        "QDRANT__SERVICE__API_KEY": key,
        "QDRANT__STORAGE__STORAGE_PATH": str(storage / "storage"),
        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(storage / "snapshots"),
        "QDRANT__TELEMETRY_DISABLED": "true",
    }


def ensure_key(config: dict) -> tuple[str, bool]:
    """The API key from the environment or .env, and whether a new random one had to be appended to .env."""
    name = config["key_env"]
    key = setting(name)
    if key:
        return key, False
    key = secrets.token_urlsafe(32)
    path = env_file()
    existing = path.read_text(encoding="utf-8-sig") if path.is_file() else ""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(("\n" if existing and not existing.endswith("\n") else "") + f"{name}={key}\n")
    return key, True


def http_status(config: dict, path: str, key: str | None = None, timeout: float = 2.0) -> int | None:
    """The status of a GET on the server, or None when nothing answers."""
    request = urllib.request.Request(f"http://{config['host']}:{config['http_port']}{path}", headers={"api-key": key} if key else {})
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return None


def image_path(pid: int) -> Path | None:
    """The full path of a running process's executable, or None if no such process is running."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
            return None
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return Path(buffer.value)
    finally:
        kernel32.CloseHandle(handle)


def is_ours(pid: int, config: dict) -> bool:
    found = image_path(pid)
    return found is not None and os.path.normcase(found.resolve()) == os.path.normcase(binary_path(config).resolve())


def terminate(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows: a hard stop, which the write ahead log makes safe


def running(config: dict) -> int | None:
    """The pid of this project's Qdrant when it is running, else None."""
    path = pid_file()
    if not path.is_file():
        return None
    pid = int(path.read_text().strip())
    return pid if is_ours(pid, config) else None


def start(config: dict, popen=subprocess.Popen, timeout: float = 30.0) -> tuple[int, float, bool]:
    """Start the server. Returns its pid, the seconds until it was ready, and whether a new key was written."""
    if os.name != "nt":
        raise ServerError("the pinned binary is the Windows build; elsewhere run Qdrant yourself, or use local mode")
    binary = binary_path(config)
    check_binary(binary, config["exe_sha256"])
    if running(config):
        raise ServerError(f"Qdrant is already running as pid {running(config)}")
    if http_status(config, "/readyz") is not None:
        raise ServerError(f"something else answers on {config['host']}:{config['http_port']}; change the ports in config/qdrant.toml")
    key, created = ensure_key(config)
    storage, logs = data_dir() / "qdrant", data_dir() / "logs"
    storage.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    with open(logs / "qdrant.log", "ab") as log:
        process = popen([str(binary)], cwd=storage, env=server_env(config, key, storage), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, creationflags=DETACHED, close_fds=True)
    pid_file().write_bytes(str(process.pid).encode("ascii"))
    while http_status(config, "/readyz") != 200:
        if time.perf_counter() - began > timeout:
            stop(config)
            raise ServerError(f"Qdrant was not ready within {timeout:.0f} seconds; see {logs / 'qdrant.log'}")
        time.sleep(0.05)
    seconds = time.perf_counter() - began
    if http_status(config, "/collections") != 401 or http_status(config, "/collections", key) != 200:
        stop(config)
        raise ServerError("Qdrant answered without the API key, or refused it, so it was stopped")
    return process.pid, seconds, created


def stop(config: dict, timeout: float = 10.0) -> int | None:
    """Stop this project's server. Returns the pid stopped, or None when none was running."""
    path = pid_file()
    if not path.is_file():
        return None
    pid = int(path.read_text().strip())
    found = image_path(pid)
    if found is None:
        path.unlink()
        return None
    if not is_ours(pid, config):
        raise ServerError(f"pid {pid} in {path} is {found}, not this project's Qdrant; not stopping it")
    terminate(pid)
    began = time.perf_counter()
    while image_path(pid) is not None and time.perf_counter() - began < timeout:
        time.sleep(0.05)
    path.unlink()
    return pid
