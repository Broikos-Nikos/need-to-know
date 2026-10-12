"""DeepSeek's account endpoints: which models exist, and how much is left to spend.

Read only and free. The opener is a parameter so tests never touch the network. An
answer that is not JSON, such as a captive portal's sign in page, becomes an
AccountError that says what came back, never a traceback.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

Opener = Callable[..., object]

# What an HTTP status from these endpoints most likely means, in the user's terms.
HTTP_HINTS = {
    401: "DeepSeek refused the key",
    402: "the balance is used up",
    403: "the key may not call this endpoint",
    429: "too many requests, try again shortly",
}


class AccountError(Exception):
    """DeepSeek answered, but not with what its account endpoints promise."""


def get_json(base_url: str, path: str, key: str, opener: Opener = urllib.request.urlopen, timeout: float = 20) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    with opener(request, timeout=timeout) as response:
        body = response.read()
        kind = response.headers.get("Content-Type", "") if hasattr(response, "headers") else ""
    try:
        return json.loads(body)
    except ValueError:
        raise AccountError(f"{path} answered {kind or 'with no content type'}, not JSON: {body[:60]!r}") from None


def list_models(base_url: str, key: str, opener: Opener = urllib.request.urlopen) -> list[str]:
    body = get_json(base_url, "/models", key, opener)
    return sorted(item["id"] for item in body.get("data", []) if isinstance(item, dict) and "id" in item)


def balance(base_url: str, key: str, opener: Opener = urllib.request.urlopen) -> tuple[bool, list[tuple[str, str]]]:
    body = get_json(base_url, "/user/balance", key, opener)
    rows = [(info.get("total_balance", "?"), info.get("currency", "?")) for info in body.get("balance_infos", [])]
    return bool(body.get("is_available")), rows


def live_check(base_url: str, model: str, key: str, opener: Opener = urllib.request.urlopen) -> tuple[bool, list[str]]:
    """Whether the configured model is listed and the balance usable, as lines that never hold the key."""
    try:
        models = list_models(base_url, key, opener)
        available, rows = balance(base_url, key, opener)
    except urllib.error.HTTPError as error:
        hint = HTTP_HINTS.get(error.code)
        return False, [f"live check failed: HTTP {error.code} {error.reason}" + (f": {hint}" if hint else "")]
    except AccountError as error:
        return False, [f"live check failed: {error}"]
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", None)
        return False, [f"live check failed: {type(error).__name__}" + (f" ({reason})" if reason else "")]
    listed = model in models
    amounts = ", ".join(f"{total} {currency}" for total, currency in rows) or "none reported"
    return listed and available, [
        f"live models: {', '.join(models)}; configured {model} listed: {'yes' if listed else 'no'}",
        f"balance: {amounts}; usable: {'yes' if available else 'no'}",
    ]
