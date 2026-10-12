"""One question through the DeepSeek adapter, live: thinking off, on, and off the wrong way (T4.2, N13).

    uv run python bench/deepseek_call.py

Needs a DeepSeek key in the environment or `.env`, and the network; no index and no GPU.
It builds T4.1's prompt over two fixed excerpts and a question the first of them answers,
and asks it three ways: thinking disabled at the top level of the body, the configured
default; enabled; and disabled inside `extra_body`, as the OpenAI SDK's syntax writes it
and a raw post keeps it. For each it prints the stop reason, the attempts, the model that
answered, the tokens as DeepSeek bills them, cache hits apart, the reasoning tokens among
the output, the seconds, the USD at the price in force, and the answer as `parse` reads
it. It exits 1 when thinking off at the top level spent a reasoning token, when the
wrapped form spent none, which would mean the API now reads it, or when an answer is not
the first excerpt's, cited alone.
"""

from __future__ import annotations

import sys

from needtoknow.answer.prompt import build, parse
from needtoknow.providers import deepseek

EXCERPTS = [{"doc_id": "RET_POL_011", "path": ["Returns policy", "Window"],
             "text": "A customer may return a device within 30 days of purchase, with its receipt."},
            {"doc_id": "SAL_PRO_033", "path": ["Sales procedure", "Authorisation"],
             "text": "Return authorisation is issued by the returns team within two working days."}]
QUESTION = "How many days does a customer have to return a device?"
REAL_BODY = deepseek.request_body


def wrapped_body(*args, **kwargs) -> dict:
    """The adapter's body with `thinking` moved inside `extra_body`, which the adapter itself never sends."""
    body = REAL_BODY(*args, **kwargs)
    body["extra_body"] = {"thinking": body.pop("thinking")}
    return body


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    system, user = build(QUESTION, EXCERPTS)
    failed = False
    for name, thinking, wrapped in (("off", False, False), ("on", True, False), ("off inside extra_body", False, True)):
        deepseek.request_body = wrapped_body if wrapped else REAL_BODY
        try:
            done = deepseek.complete(system, [{"role": "user", "content": user}], thinking=thinking, as_json=True)
        finally:
            deepseek.request_body = REAL_BODY
        answer = parse(done.text, EXCERPTS, QUESTION)
        used = done.usage
        cited = [c["doc_id"] for c in answer["citations"]]
        print(f"thinking {name}: {done.stop_reason}, attempts {done.attempts}, {done.model}; input "
              f"{used['input_hit']} cached and {used['input_miss']} not, output {used['output']} of which reasoning "
              f"{used['reasoning']}; {done.seconds:.1f} s; {done.cost_usd:.6f} USD; {answer['status']}: "
              f"{answer['answer']!r}, citing {', '.join(cited) or 'nothing'}", flush=True)
        if name == "off" and used["reasoning"]:
            print("thinking off at the top level still spent reasoning tokens", file=sys.stderr)
            failed = True
        if wrapped and not used["reasoning"]:
            print("thinking off inside extra_body spent no reasoning token: the API now reads it, and N13's note is "
                  "stale", file=sys.stderr)
            failed = True
        if answer["status"] != "answered" or cited != ["RET_POL_011"]:
            print(f"thinking {name}: not the first excerpt's answer, cited alone: {answer}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
