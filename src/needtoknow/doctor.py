"""ntk doctor: whether this machine can build and search the index, one line per check.

A check returns whether it passed and one line of detail, and an exception inside a
check becomes that check's failure, never a traceback. Nothing here prints a key,
and nothing loads the embedding model: the doctor checks that the pieces are
present, and the test suite checks that they work.
"""

from __future__ import annotations

import math
import shutil
import sys
import tomllib
import urllib.request

from needtoknow import policy
from needtoknow.eval.protocol import load_protocol
from needtoknow.index import server
from needtoknow.providers import deepseek_account
from needtoknow.providers.catalog import load_catalog
from needtoknow.settings import data_dir, repo_root, setting

GREEK = "Ελληνικά"


def min_free_gb() -> int:
    """The free disk a full build needs: the data directory's peak during one, as config/build.toml records it from
    bench/full_build.py, half again for headroom (MA-F3)."""
    measured = tomllib.loads((repo_root() / "config" / "build.toml").read_text(encoding="utf-8"))["measured"]
    return math.ceil(measured["peak_data_dir_gb"] * 1.5)


def check_python(_context: dict) -> tuple[bool, str]:
    version = sys.version_info
    return version[:2] == (3, 12), f"{version.major}.{version.minor}.{version.micro}"


def check_console(context: dict) -> tuple[bool, str]:
    encoding = getattr(context["out"], "encoding", None) or "unknown"
    return encoding.lower().replace("-", "") == "utf8", f"{encoding}; Greek prints as {GREEK}"


def check_data(_context: dict) -> tuple[bool, str]:
    path = data_dir()
    existing = next(p for p in (path, *path.parents) if p.exists())
    free, floor = shutil.disk_usage(existing).free / 2**30, min_free_gb()
    need = f", {floor} GB a full build needs" if free >= floor else f", under the {floor} GB the full build needs"
    return free >= floor, f"{path}, {free:.0f} GB free{need}"


def check_config(_context: dict) -> tuple[bool, str]:
    labels = policy.load_labels()
    roles = policy.load_policy(labels=labels)["roles"]
    protocol = load_protocol()
    load_catalog()
    server.load_config()
    from needtoknow.embed.model import load_config as embed_config

    embed_config()
    return True, f"providers, {len(labels)} labels, {len(roles)} roles, protocol version {protocol['version']}, qdrant, models"


def check_keys(_context: dict) -> tuple[bool, str]:
    catalog = load_catalog()
    deepseek = bool(setting(catalog["deepseek"]["key_env"]))
    qdrant = bool(setting(server.load_config()["key_env"]))
    optional = [catalog[name]["key_env"] for name in ("openai", "anthropic", "gemini") if not setting(catalog[name]["key_env"])]
    return deepseek, (
        f"DEEPSEEK_API_KEY {'set' if deepseek else 'missing'}; "
        f"QDRANT_API_KEY {'set' if qdrant else 'not yet, ntk qdrant start writes one'}; "
        f"{len(optional)} optional keys unset"
    )


def check_qdrant(_context: dict) -> tuple[bool, str]:
    config = server.load_config()
    server.check_binary(server.binary_path(config), config["exe_sha256"])
    pid = server.running(config)
    return True, f"{config['version']} matches its pin; " + (f"running as pid {pid}" if pid else "not running, ntk qdrant start")


def check_gpu(_context: dict) -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed: uv sync --extra embed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} sees no CUDA device"
    card = torch.cuda.get_device_properties(0)
    return True, f"{card.name}, {card.total_memory // 2**20} MiB, CUDA {torch.version.cuda}, torch {torch.__version__}"


def check_model(_context: dict) -> tuple[bool, str]:
    """The embedding model and the reranker (T3.6), each at its pinned revision, in the local cache."""
    from needtoknow.embed.model import load_config as embed_config
    from needtoknow.index.rerank import load_config as rerank_config

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False, "huggingface_hub is not installed: uv sync --extra embed"
    models = [embed_config(), rerank_config()]
    missing = [c for c in models if not isinstance(try_to_load_from_cache(c["repo"], "config.json", revision=c["revision"]), str)]
    if not missing:
        return True, " and ".join(f"{c['repo']} at {c['revision'][:8]}" for c in models) + " in the local cache"
    return False, "; ".join(f"{c['repo']} at {c['revision'][:8]} is not in the local cache: "
                            f"uv run hf download {c['repo']} --revision {c['revision']}" for c in missing)


def check_deepseek(context: dict) -> tuple[bool, str]:
    if context["offline"]:
        return True, "live check skipped: --offline"
    deepseek = load_catalog()["deepseek"]
    key = setting(deepseek["key_env"])
    if not key:
        return False, f"{deepseek['key_env']} is not set"
    passed, lines = deepseek_account.live_check(deepseek["base_url"], deepseek["model"], key, context["opener"])
    return passed, "; ".join(lines)


CHECKS = [
    ("python", check_python), ("console", check_console), ("data", check_data), ("config", check_config),
    ("keys", check_keys), ("qdrant", check_qdrant), ("gpu", check_gpu), ("model", check_model),
    ("deepseek", check_deepseek),
]


def run(offline: bool = False, out=None, opener=urllib.request.urlopen) -> int:
    """Print one line per check and a summary; 0 when every check passed."""
    out = out or sys.stdout
    context = {"offline": offline, "out": out, "opener": opener}
    passed = 0
    for name, check in CHECKS:
        try:
            ok, detail = check(context)
        except Exception as error:  # a check that breaks is a failed check, said in one line
            ok, detail = False, f"{type(error).__name__}: {error}"
        passed += ok
        print(f"{'ok' if ok else 'FAIL':<5} {name:<9} {detail}", file=out)
    print(f"doctor: {passed} of {len(CHECKS)} ok", file=out)
    return 0 if passed == len(CHECKS) else 1
