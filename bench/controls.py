"""The negative controls behind the gates, run from the repository (MA-F16).

    uv run python bench/controls.py             every control in bench/controls.toml
    uv run python bench/controls.py canary      those whose name holds the word

Each control breaks what its gate guards, by replacing text in one file or more, in order; runs the one test it names,
with the GPU hidden and no bytecode written; and restores every file it touched byte for byte, and the files its
`then` command writes, which it runs again once restored. A control whose test passes is a gate that cannot fail; one
whose text its file no longer holds once is stale. It prints each control's first error line, since a gate can fail
for its fixture rather than its subject, and exits 1 on either.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTROLS = REPO / "bench" / "controls.toml"


def run(control: dict) -> tuple[str, str]:
    """The control's outcome, failed, passed or stale, and the test's first error line."""
    touched = {REPO / change["file"] for change in control["change"]} | {REPO / keep for keep in control.get("keeps", [])}
    saved = {path: path.read_bytes() for path in touched if path.is_file()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    try:
        for change in control["change"]:
            path = REPO / change["file"]
            text = path.read_bytes().decode("utf-8")
            if text.count(change["old"]) != 1:
                return "stale", f"{change['file']} holds its text {text.count(change['old'])} times"
            path.write_bytes(text.replace(change["old"], change["new"]).encode("utf-8"))
        if control.get("then"):
            subprocess.run([sys.executable, *control["then"]], cwd=REPO, env=env, check=True, capture_output=True)
        done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", control["test"]], cwd=REPO,
                              env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        reason = next((line[1:].strip() for line in done.stdout.splitlines() if line.startswith("E ")), "")
        return ("failed" if done.returncode else "passed"), reason
    finally:
        for path, data in saved.items():
            path.write_bytes(data)
        if control.get("then"):
            subprocess.run([sys.executable, *control["then"]], cwd=REPO, env=env, check=True, capture_output=True)
        changed = [path for path, data in saved.items() if path.read_bytes() != data]
        if changed:
            raise SystemExit(f"controls: {changed[0]} was not restored byte for byte")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    words = argv if argv is not None else sys.argv[1:]
    controls = [c for c in tomllib.loads(CONTROLS.read_text(encoding="utf-8"))["control"]
                if not words or any(word in c["name"] for word in words)]
    outcomes = {"failed": 0, "passed": 0, "stale": 0}
    for control in controls:
        outcome, reason = run(control)
        outcomes[outcome] += 1
        verdict = {"failed": "fails, as it must", "passed": "PASSES: the gate cannot fail", "stale": "STALE"}[outcome]
        print(f"{control['name']}: {verdict} | {reason[:140]}", flush=True)
    print(f"{len(controls)} controls: {outcomes['failed']} fail their gate as they must, {outcomes['passed']} pass it, "
          f"{outcomes['stale']} stale", flush=True)
    return 1 if outcomes["passed"] or outcomes["stale"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
