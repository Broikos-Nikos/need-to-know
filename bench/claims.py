"""Each number the README states, held to the recorded output of the command beside it (T5.5e2's successor, P5's exit:
the claims gate in the build).

    uv run python bench/claims.py record "uv run ntk stats"
    uv run python bench/claims.py record "uv run python bench/draws.py" --log draws-t146.log
    uv run python bench/claims.py check

A record is one command's output kept in `docs/claims/`, a file a command, headed by the command and by when and at
which commit it ran, or by the log of an earlier run it was taken from, a log `NTK_DATA_DIR/logs` holds; a run that
takes an hour on the GPU is recorded from its log rather than run again. `record` runs a command the README names
the way the README names it, `uv run ntk ...` or `uv run python bench/...`, in this interpreter.

`check` reads the README's "What works today" and "What fails": for each bullet that states a number, the commands it names, and each
number of its prose, outside its backticks, which must be a number some record of those commands prints. It names
every number no record prints and every bullet with numbers and no record, and exits 1 on either; the test suite runs
the same check (`tests/test_claims.py`). Not claims: identifiers (N43, RS-F6, T12, F07V04, DP041), and the
parameters of a measure, the 10 of a top 10, the 1 of recall@1, the 20 of up to 20 a label. A thousands separator
reads as nothing, on both sides.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs" / "claims"
NUMBER = re.compile(r"(?<![\w.@])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")
PARAMETER = re.compile(r"(?:\btop|\bup to)\s+$")
COMMAND = re.compile(r"`((?:uv run )?(?:ntk|python bench/)[^`]*)`")


def normal(number: str) -> str:
    return number.replace(",", "")


def command_of(text: str) -> str:
    """A command as the README names it, `ntk ...` and `uv run ntk ...` alike."""
    text = " ".join(text.split())
    return text if text.startswith("uv run ") else "uv run " + text


def record_path(command: str) -> Path:
    return RECORDS / (re.sub(r"[^A-Za-z0-9._]+", "-", command_of(command).removeprefix("uv run ")).strip("-") + ".txt")


def bullets(readme: str) -> list[str]:
    """The bullets of the README's "What works today" and "What fails", whose numbers are held alike."""
    found = []
    for heading in ("## What works today", "## What fails"):
        if heading == "## What fails" and heading not in readme:  # the README's own test asks for it
            continue
        section = readme[readme.index(heading):]
        section = section[:section.index("\n## ", 1)]
        found += [part for part in re.split(r"\n\* ", section)[1:] if part.strip()]
    return found


def claimed(bullet: str) -> list[str]:
    """The numbers a bullet's prose states, outside its backticks, identifiers and a measure's parameters left out."""
    # A digit after a letter starts no number here, so N43, RS-F6 and DP041 are never read as claims.
    prose = re.sub(r"`[^`]*`", " ", bullet)
    return [m.group(0) for m in NUMBER.finditer(prose) if not PARAMETER.search(prose[:m.start()])]


def printed(text: str) -> set[str]:
    return {normal(m.group(0)) for m in NUMBER.finditer(text)}


def check(readme: str, records: Path = RECORDS) -> list[str]:
    """Every number of a bullet that no record of its commands prints, and every bullet with numbers and no record."""
    problems = []
    for bullet in bullets(readme):
        numbers = claimed(bullet)
        if not numbers:
            continue
        commands = [command_of(c) for c in COMMAND.findall(bullet)]
        # Below the head alone: its date and time are no output, and passed a figure equal to them (DR-F32).
        texts = [path.read_text(encoding="utf-8").partition("\n---\n")[2]
                 for path in (records / record_path(c).name for c in commands) if path.is_file()]
        first = " ".join(bullet.split())[:60]
        if not texts:
            problems.append(f"{first}: {len(numbers)} numbers, and no record of {', '.join(commands) or 'any command'}")
            continue
        seen = set().union(*(printed(text) for text in texts))
        missing = [n for n in numbers if normal(n) not in seen]
        if missing:
            problems.append(f"{first}: no record of {', '.join(commands)} prints {', '.join(missing)}")
    return problems


def run(command: str) -> str:
    """A README command run the way it names it, from the repository, its output and errors together."""
    words = shlex.split(command_of(command))[2:]  # after "uv run"
    if words[0] == "ntk":
        argv = [str(Path(sys.executable).with_name("ntk.exe" if sys.platform == "win32" else "ntk")), *words[1:]]
    elif words[0] == "python":
        argv = [sys.executable, *words[1:]]
    else:
        raise SystemExit(f"{command}: record runs `uv run ntk ...` or `uv run python bench/...`")
    done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode:
        raise SystemExit(f"{command} exited {done.returncode}: {done.stdout[-500:]}{done.stderr[-500:]}")
    return done.stdout + done.stderr


def head(command: str, log: Path | None) -> str:
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    if log is not None:
        written = dt.datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"command: {command_of(command)}\nrecorded {when} from the log {log.name}, written {written}\n---\n"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout
    clean = not subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return (f"command: {command_of(command)}\nrecorded {when}, run at {commit.strip()}"
            f"{'' if clean else ', the tree holding changes'}\n---\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("record", "check"))
    parser.add_argument("command", nargs="?", help="record: the command as the README names it")
    parser.add_argument("--log", help="record: take the output from this log in NTK_DATA_DIR/logs instead of a run")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.action == "check":
        problems = check((REPO / "README.md").read_text(encoding="utf-8"))
        for problem in problems:
            print(problem)
        claims = sum(len(claimed(b)) for b in bullets((REPO / "README.md").read_text(encoding="utf-8")))
        print(f"{claims} numbers in the README's What works today and What fails; {len(problems)} problems")
        return 1 if problems else 0
    if not args.command:
        raise SystemExit("record: name the command, as the README names it")
    log = None
    if args.log:
        from needtoknow.settings import data_dir
        log = data_dir() / "logs" / args.log
        output = log.read_text(encoding="utf-8", errors="replace")
    else:
        output = run(args.command)
    # A progress bar's counts are no claim, and would let any number pass: its lines are left out. This machine's data
    # folder is written as the setting that names it.
    from needtoknow.settings import data_dir
    output = "\n".join(line for line in output.replace("\r\n", "\n").split("\n") if "%|" not in line)
    for form in {str(data_dir()), data_dir().as_posix()}:
        output = output.replace(form, "NTK_DATA_DIR")
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = record_path(args.command)
    path.write_bytes((head(args.command, log) + output).encode("utf-8"))
    print(f"{path.relative_to(REPO).as_posix()}: {len(printed(output))} numbers printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
