"""DeepSeek's account endpoints: which models exist, and how much is left to spend.

Read only and free. The opener is a parameter so tests never touch the network.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable

Opener = Callable[..., object]


def get_json(base_url: str, path: str, key: str, opener: Opener = urllib.request.urlopen, timeout: float = 20) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    with opener(request, timeout=timeout) as response:
        return json.load(response)


def list_models(base_url: str, key: str, opener: Opener = urllib.request.urlopen) -> list[str]:
    body = get_json(base_url, "/models", key, opener)
    return sorted(item["id"] for item in body.get("data", []))


def balance(base_url: str, key: str, opener: Opener = urllib.request.urlopen) -> tuple[bool, list[tuple[str, str]]]:
    body = get_json(base_url, "/user/balance", key, opener)
    rows = [(info.get("total_balance", "?"), info.get("currency", "?")) for info in body.get("balance_infos", [])]
    return bool(body.get("is_available")), rows
