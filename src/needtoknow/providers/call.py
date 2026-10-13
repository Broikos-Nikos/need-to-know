"""What every adapter shares: the result of a call, the error it raises, and the post that tries again.

The envelope is shared and the behaviour is not (`09-production/05`): each adapter builds
its own request and reads its own reply, since reasoning switches, JSON output, stop
reasons and token counts differ by provider and mostly fail silently when carried across.
This module only posts: it tries again on 429, 5xx and a dropped connection after 1, 2
and 4 seconds, and on a reply its adapter calls unusable; it turns any other HTTP error
into a ProviderError that says what went wrong and never holds the key.
"""

from __future__ import annotations

import datetime as dt
import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from needtoknow.providers.catalog import cost_usd

RETRIED = (429, 500, 502, 503, 504, 529)  # 529: Anthropic's "overloaded"
BACKOFF = (1.0, 2.0, 4.0)  # seconds before each retry: four attempts at most
HINTS = {400: "the request body is malformed", 401: "the key was refused", 403: "the key may not call this endpoint",
         404: "no such model or endpoint", 422: "a parameter is invalid", 429: "too many requests"}
EXPLAINED = (400, 404, 422)  # the codes whose own message names what is wrong; no other reply's words are kept


def records(value) -> list[dict]:
    """The objects of a reply's list field; a reply of another shape gives none, so it reads as empty, never raises."""
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def record(value) -> dict:
    """A reply's object field, or an empty one when the field is anything else."""
    return value if isinstance(value, dict) else {}


def words(value) -> str:
    """A reply's text field, or nothing when it is not text."""
    return value if isinstance(value, str) else ""


def tokens(value) -> int:
    """A token count as the reply gives it, or 0 when it gives none or no whole number."""
    if isinstance(value, bool):
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


class ProviderError(Exception):
    """A provider did not complete the call; the message says why and never holds the key."""


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning: str  # the thinking a provider returns beside the answer, where it returns any
    usage: dict  # input_hit, input_miss, output and reasoning tokens over every billed reply; None where not reported
    stop_reason: str  # stop, length, refusal, content_filter, or the provider's own word for anything else
    seconds: float
    cost_usd: float | None  # None when config/providers.toml prices no such model
    model: str  # the model that answered, as the reply names it
    attempts: int


def priced(entry: dict, model: str, usage: dict, when: dt.datetime) -> float | None:
    """USD at the price config/providers.toml gives the model, in force at `when`; None where it gives none."""
    price = ((entry.get("models") or {}).get(model) or {}).get("price")
    if not price:
        return None
    return cost_usd(price, usage, when, entry.get("peak") or {"weekdays": [], "windows_utc": []})


def started(now: Callable[[], dt.datetime] | None) -> dt.datetime:
    return (now or (lambda: dt.datetime.now(dt.timezone.utc)))()


def refusal(error: urllib.error.HTTPError, key: str, hints: dict) -> str:
    """An HTTP error in the user's terms. The provider's own words are kept only where they name a parameter, and only
    if they hold no run of eight characters of the key: a reply that echoes a key masked in part would pass a scrub
    of the whole key, so none of such a reply is kept."""
    said = ""
    if error.code in EXPLAINED:
        try:
            body = json.loads(error.read() or b"{}")
            said = str(body.get("error", {}).get("message", "") if isinstance(body, dict) else "")
        except (ValueError, AttributeError, OSError):
            said = ""
        if any(key[i:i + 8] in said for i in range(max(len(key) - 7, 1))):
            said = ""
        said = said[:200]
    hint = hints.get(error.code)
    return f"HTTP {error.code} {error.reason}" + (f": {hint}" if hint else "") + (f" ({said})" if said else "")


def post(name: str, url: str, headers: dict, body: dict, key: str, opener: Callable, sleep: Callable[[float], None],
         timeout: float, hints: dict = HINTS, unusable: Callable[[dict], bool] | None = None) -> tuple[list[dict], int]:
    """Post until a usable reply. Returns every reply that came back, each of them billed, and the attempts made."""
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json", **headers}
    replies: list[dict] = []
    last = len(BACKOFF) + 1
    for attempt in range(1, last + 1):
        request = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with opener(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRIED or attempt == last:
                raise ProviderError(f"{name} refused the call: {refusal(error, key, hints)}") from None
            sleep(BACKOFF[attempt - 1])
            continue
        # A body cut off mid read raises http.client's IncompleteRead, an HTTPException and no OSError (DR-F18).
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            if attempt == last:
                raise ProviderError(f"{name} could not be reached: {type(error).__name__}") from None
            sleep(BACKOFF[attempt - 1])
            continue
        try:
            reply = json.loads(raw)
        except ValueError:
            raise ProviderError(f"{name} answered, but not with JSON: {raw[:60]!r}") from None
        if not isinstance(reply, dict):
            raise ProviderError(f"{name} answered, but not with an object: {raw[:60]!r}")
        replies.append(reply)
        if unusable is not None and unusable(reply) and attempt < last:
            sleep(BACKOFF[attempt - 1])
            continue
        return replies, attempt
    raise ProviderError(f"{name} gave no reply in {last} attempts")
