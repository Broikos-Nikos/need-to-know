"""The README's claims that a test can hold (HE-F1 to HE-F3): every path it names exists, every command it gives
parses, it says the documents are synthetic, and it never says a network is unneeded without uv sync beside it."""

import re
import shlex

import pytest

from needtoknow.cli import build_parser
from needtoknow.settings import repo_root

README = (repo_root() / "README.md").read_text(encoding="utf-8")


MADE = {"web/dist": "npm run build"}  # a path a README command makes, and the command


def test_every_path_the_readme_names_exists():
    # T7.2's preflight: the test asked the working tree, where an earlier build had left web/dist, and a fresh clone,
    # the runner's among them, has none until npm run build. A path the README names is one the repository tracks, or
    # one a command the README gives makes.
    import subprocess
    named = set(re.findall(r"`((?:[\w.-]+/)+[\w.-]*|\.env\.example)`", README))
    assert named, "no path found: the pattern no longer matches the README"
    tracked = subprocess.run(["git", "ls-files"], cwd=repo_root(), capture_output=True, text=True, check=True).stdout.split()
    held = set(tracked) | {"/".join(path.split("/")[:n]) + "/" for path in tracked for n in range(1, path.count("/") + 1)}
    missing = sorted(p for p in named if p not in held and p.rstrip("/") not in held and p.rstrip("/") not in MADE)
    assert not missing, missing
    assert all(command in README for path, command in MADE.items() if path in {p.rstrip("/") for p in named}), MADE


# A placeholder the README writes, and a value of its kind (MT-F17).
PLACEHOLDERS = {"question": "a question", "role": "sales", "file": "runs.jsonl"}


def ntk_commands(readme: str) -> list[str]:
    """Every ntk command the README gives, in a code span or a code block, its placeholders filled."""
    spans = re.findall(r"`([^`\n]+)`", readme)
    blocks = [line for block in re.findall(r"```\n(.*?)```", readme, re.S) for line in block.splitlines()]
    commands = []
    for text in spans + blocks:
        found = re.fullmatch(r"\s*(?:uv run )?ntk (.+?)\s*", text)
        if found:
            unknown = set(re.findall(r"<(\w+)>", found[1])) - set(PLACEHOLDERS)
            assert not unknown, (text, unknown)  # a placeholder of no known kind fails, it is not skipped
            commands.append(re.sub(r"<(\w+)>", lambda m: PLACEHOLDERS[m[1]], found[1]))
    return commands


def test_every_ntk_command_the_readme_gives_parses():
    # MT-F17: the gate parsed a command only when it held lower case letters, spaces and dashes, so the ones with a
    # placeholder, a quoted question or a path were skipped without a word, ntk ask and ntk eval run among them.
    commands = ntk_commands(README)
    assert any('"' in c for c in commands) and any("--as" in c for c in commands) and len(commands) >= 30
    for command in commands:
        try:
            build_parser().parse_args(shlex.split(command))
        except SystemExit as error:
            pytest.fail(f"ntk {command}: exit {error.code}")


def test_the_readme_says_which_commands_run_until_stopped():
    # HS-F23: ntk serve came before ntk ingest watch in the commands, and nothing said that both run until stopped.
    endless = ("ntk serve", "ntk ingest watch")
    assert all(f"uv run {command}" in README for command in endless)
    said = [s for s in re.split(r"(?<=[.;])\s+", README) if "run until stopped" in s]
    assert said and all(f"`{command}`" in said[0] for command in endless), said


def test_every_bench_the_readme_runs_is_in_the_repository():
    scripts = set(re.findall(r"python (bench/[\w]+\.py)", README))
    assert scripts and all((repo_root() / s).is_file() for s in scripts), scripts


def test_the_readme_says_what_the_documents_are_and_what_needs_a_network():
    assert "synthetic" in README.lower()
    for sentence in re.split(r"(?<=[.;])\s+", README):
        if "network" in sentence:
            assert "uv sync" in sentence or "uv sync" in README[:README.index(sentence)].rsplit("\n\n", 1)[-1], sentence


def test_the_about_box_opens_with_the_readmes_first_sentence_and_carries_numbers_it_holds():
    # RC-F7: on GitHub the README starts under the list of files, so the About box holds the first screen's only
    # sentence. docs/PUBLISH.md's description opens with the README's own first sentence, fits GitHub's 350
    # characters, and every number in it is one the README's claims, which the claims check holds, state; ten topics.
    publish = (repo_root() / "docs" / "PUBLISH.md").read_text(encoding="utf-8")
    description = " ".join(line[2:].strip() for line in publish.splitlines() if line.startswith("> "))
    first = re.split(r"(?<=[.!?])\s", " ".join(README.split(chr(10) * 2)[1].split()))[0]
    assert description.startswith(first + " ") and len(description) <= 350, (first, len(description))
    claims = " ".join(README[README.index("## What works today"):README.index("## Run it")].split())
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", description)
    assert numbers and all(re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d])", claims) for n in numbers), numbers
    topics = re.search(r"\*\*Topics\*\*.*?```\n(.*?)```", publish, re.S)[1].split()
    assert len(topics) == len(set(topics)) == 10 and all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", t) for t in topics), topics


def test_the_readme_opens_with_what_it_does_for_a_person():
    # RC-F1: it opened on "A vector database RAG project", the plain idea forty words in.
    opening = README.split(chr(10) * 2)[1]
    first = re.split(r"(?<=[.!?])\s", " ".join(opening.split()))[0]
    jargon = [word for word in ("RAG", "vector", "embedding", "retrieval", "LLM", "rerank") if word.lower() in first.lower()]
    assert not jargon and len(first.split()) <= 25, (first, jargon)


def test_the_readme_names_every_case_the_permission_suite_runs_beyond_the_roles():
    # MA-F12: it called a user with only general and a removed membership "combinations of roles".
    from needtoknow.policy import load_policy
    record = (repo_root() / "docs" / "claims" / "python-bench-permission_suite.py.txt").read_text(encoding="utf-8")
    cases = re.findall(r"^  (\S.*?)\s{2,}foreign units:", record, re.M)
    beyond = [case for case in cases if case not in load_policy()["roles"]]
    assert len(beyond) == 3, cases
    text = " ".join(README.split())
    bullet = text[text.index("no unit a role may not read came back"):]
    named = {"sales and hr": "sales and hr held together", "general alone": "general alone",
             "sales, returns removed": "sales with returns removed"}
    assert all(named[case] in bullet[:600] for case in beyond), beyond


def test_the_readme_says_what_fails_and_the_claims_gate_reads_it():
    # T6.5, house rule 2: a project publishes a failure, measured; its numbers are held as the rest are.
    import importlib.util
    spec = importlib.util.spec_from_file_location("claims", repo_root() / "bench" / "claims.py")
    claims = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(claims)
    assert "## What fails" in README
    section = README[README.index("## What fails"):]
    section = section[:section.index("\n## ", 1)]
    failing = [part for part in re.split(r"\n\* ", section)[1:] if part.strip()]
    read = claims.bullets(README)
    assert len(failing) >= 2 and all(bullet in read and claims.claimed(bullet) for bullet in failing), failing


def test_the_readme_gives_the_relabel_it_fails_in_both_directions():
    # T6.5: the bullet gave the widening's figures alone, not the narrowing's, the direction that takes access away.
    record = (repo_root() / "docs" / "claims" / "python-bench-revocation.py.txt").read_text(encoding="utf-8")
    pairs = re.findall(r"median (\d+\.\d) s, largest (\d+\.\d) s", record)
    text = " ".join(README.split())
    bullet = text[text.index("A relabel of the largest document"):]
    bullet = bullet[:bullet.index("bench/revocation.py")]
    assert pairs and all(f"{median} s at the median and {largest} s at most" in bullet for median, largest in pairs), pairs


def test_the_readme_says_what_the_injection_bench_tried_and_how_often_it_held():
    # MA-F10: the README said a planted instruction is quoted rather than followed, on two instructions in one excerpt
    # and no count; it now gives the recorded run's asks under a planted instruction and how many moved the answer.
    record = (repo_root() / "docs" / "claims" / "python-bench-injection.py-repeats-20.txt").read_text(encoding="utf-8")
    asked, moved = re.search(r"the planted instructions: (\d+) asked, (\d+) answered otherwise", record).groups()
    text = " ".join(README.split())
    bullet = text[text.index("Two instructions planted in a permitted excerpt"):]
    assert f"moved the answer in {moved} of {asked} asks" in bullet[:400], (moved, asked)
    assert "uv run python bench/injection.py --repeats 20" in bullet[:600]


def test_the_readme_gives_the_node_the_page_build_needs():
    # MT-F22: it said Node 22 or later, where vite asks for ^20.19.0 || >=22.12.0, and package.json declared nothing.
    import json
    package = json.loads((repo_root() / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((repo_root() / "package-lock.json").read_text(encoding="utf-8"))
    wanted = lock["packages"]["node_modules/vite"]["engines"]["node"]
    assert package["engines"]["node"] == wanted, (package.get("engines"), wanted)
    (low, low_minor), (high, high_minor) = re.findall(r"(\d+)\.(\d+)\.0", wanted)  # ^20.19.0 || >=22.12.0
    said = f"Node {low}.{low_minor}, or {high}.{high_minor} or later"
    assert said in " ".join(README.split()), (wanted, said)


def test_the_readme_shows_the_page_right_after_its_claim():
    # RC-F2: nothing in the first screen could be seen or tried. The house order is claim, picture, number, method: the
    # recording of the page follows the opening, it moves, it is the width GitHub draws it at, and it says what made it.
    import json

    from needtoknow.policy import load_policy
    picture = re.fullmatch(r"!\[([^\]]{80,})\]\((docs/[\w.-]+\.gif)\)", README.split(chr(10) * 2)[2].strip())
    assert picture, README.split(chr(10) * 2)[2][:120]
    gif = (repo_root() / picture[2]).read_bytes()
    frames = gif.count(bytes((0x21, 0xF9, 0x04)))  # a graphic control extension opens each frame
    assert gif[:6] == b"GIF89a" and int.from_bytes(gif[6:8], "little") == 880 and frames > 20, frames
    made = json.loads((repo_root() / "docs" / "capture.json").read_text(encoding="utf-8"))
    assert made["ffmpeg"] and set(made["looked"]["every_role"]) == set(load_policy()["roles"])
    assert "npm run capture" in README and "ffmpeg" in README


def test_the_readme_gives_the_change_feed_the_poll_the_page_makes():
    # MA-F25: the README said the page asks the feed every five seconds, FEED = 5000 in main.ts, held by nothing.
    page = (repo_root() / "web" / "src" / "main.ts").read_text(encoding="utf-8")
    seconds = int(re.search(r"^const FEED = (\d+)", page, re.M)[1]) // 1000
    said = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 10: "ten"}.get(seconds, str(seconds))
    assert f"asked every {said} seconds" in " ".join(README.split()), seconds


def test_the_readme_gives_the_python_the_project_pins():
    # P6's exit: the README's one number outside its claims and its tests, Python 3.12, held to the pins it restates.
    pinned = (repo_root() / ".python-version").read_text(encoding="utf-8").strip()
    project = (repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    requires = re.search(r'^requires-python = ">=([\d.]+)', project, re.M)[1]
    assert pinned == requires and f"You need Python {pinned} and" in README, (pinned, requires)


def test_the_build_runs_the_projects_own_gates():
    # P6's exit, the workspace's first condition of a project done enough to leave: npm run build passes and runs the
    # project's own gates, the claims check and the page data check, as the other projects' builds run theirs.
    import json
    scripts = json.loads((repo_root() / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert scripts["build"].split(" && ") == ["tsc --noEmit", "npm run check", "vite build"], scripts["build"]
    assert {"bench/claims.py check", "bench/page_data.py --check"} <= {part.removeprefix("uv run python ")
                                                                       for part in scripts["check"].split(" && ")}


def test_the_ties_bullet_and_n50_name_the_collection_the_move_was_measured_in():
    # MA-F27: both gave the bulk of 15,000 written into the collection and not the 30,000 units it held, the size the
    # optimizer's work that moved the order came with.
    record = (repo_root() / "docs" / "claims" / "python-bench-ties.py.txt").read_text(encoding="utf-8")
    size, labels = re.search(r"ties_bench: ([\d,]+) filler units over (\d+) labels", record).groups()
    said = f"{size} units of all {labels} labels"
    text = " ".join(README.split())
    bullet = text[text.index("The order of units of exactly equal score"):]
    decisions = " ".join((repo_root() / "docs" / "DECISIONS.md").read_text(encoding="utf-8").split())
    amendment = decisions[decisions.index("*Amended 2026-10-21, tick 214 (AC-F12):*"):]
    assert said in bullet[:bullet.index("bench/ties.py")], said
    assert said in amendment[:amendment.index("bench/ties.py")], said


def test_run_it_says_the_count_of_passed_tests_is_partly_the_machines_and_the_workflow_lists_the_skips():
    # MA-F2: "122 tests pass" read as the code's, and a machine without the GPU, the models or the Qdrant binary counts
    # fewer; each skip says what it needed, and the README and the workflow's suite both list them.
    run_it = " ".join(README[README.index("## Run it"):].split())
    assert "`uv run pytest -rs` lists: how many pass depends on the machine as well as the code" in run_it
    workflow = (repo_root() / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert re.search(r"^ +- run: uv run --frozen pytest -q -rs$", workflow, re.M), "the workflow's suite lists no skips"
