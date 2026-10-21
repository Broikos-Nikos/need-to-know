"""The instrument page (T6.2): its fonts draw every character the narrative, the golden questions and the page's own
words use, so no Greek falls back to another face, the house style's fourth rule for an instrument; each ships with its
licence and keeps its copyright and licence records; and the page writes what the server sends, the documents' text
and the model's answer among it, as text, never through the HTML parser."""

import importlib.util
import json
import re

import pytest
from fontTools.ttLib import TTFont

from needtoknow.settings import repo_root

FONTS = repo_root() / "web" / "public" / "fonts"
# Every way a script hands a string to the HTML parser, or writes a page of its own.
MARKUP = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|createContextualFragment|DOMParser"
                    r"|srcdoc|setHTMLUnsafe|parseHTMLUnsafe")


def page_sources() -> list:
    web = repo_root() / "web"
    return [web / "index.html", *sorted(path for path in (web / "src").rglob("*") if path.is_file())]


def shown_characters() -> set[str]:
    root = repo_root()
    texts = [path.read_text(encoding="utf-8") for path in sorted((root / "narrative").glob("*.txt"))]
    texts += [path.read_text(encoding="utf-8") for path in sorted((root / "eval" / "golden").glob("*.jsonl"))]
    texts += [path.read_text(encoding="utf-8") for path in page_sources()]
    return {character for text in texts for character in text if ord(character) > 32}


@pytest.mark.parametrize("font", ["manrope-var.woff2", "robotomono-var.woff2"])
def test_the_page_fonts_draw_every_character_the_corpus_and_the_page_use(font):
    characters = shown_characters()
    assert any(0x370 <= ord(character) <= 0x3FF for character in characters)  # Greek is there to draw
    drawn = TTFont(FONTS / font).getBestCmap()
    missing = sorted(character for character in characters if ord(character) not in drawn)
    assert not missing, (font, missing[:20])


def test_each_font_ships_with_its_licence_and_keeps_its_name_records():
    for font, licence in (("manrope-var.woff2", "Manrope-OFL.txt"), ("robotomono-var.woff2", "RobotoMono-LICENSE.txt")):
        assert (FONTS / licence).is_file(), licence
        assert {0, 13} <= {record.nameID for record in TTFont(FONTS / font)["name"].names}, font


def test_nothing_moves_or_waits_to_appear_when_the_reader_asks_for_less_motion():
    # T6.2b: slow motion's rows arrive with a movement and a delay each; with reduced motion set, the stages still take
    # their turns, and within one nothing moves and nothing is held back.
    style = (repo_root() / "web" / "src" / "style.css").read_text(encoding="utf-8")
    assert re.search(r"animation:|transition:", style)
    block = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}", style, re.S)
    assert block, "no reduced motion block"
    for held in ("animation-duration: 0.01ms", "animation-delay: 0s", "transition-duration: 0.01ms",
                 "transition-delay: 0s"):
        assert f"{held} !important" in block.group(1), held


def test_the_page_writes_text_and_never_markup():
    # T6.2b: an excerpt is a document's text and an answer is a model's, and a section the role may read can hold
    # markup; the page writes both with textContent, so a tag in them shows as the characters it is made of.
    sources = [path for path in page_sources() if path.suffix in {".ts", ".js", ".html"}]
    assert any(path.suffix == ".ts" for path in sources)
    found = [f"{path.name}:{number}" for path in sources
             for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if MARKUP.search(line)]
    assert not found, found


def page_data():
    spec = importlib.util.spec_from_file_location("page_data", repo_root() / "bench" / "page_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numbers(value) -> list[float]:
    """Every number a JSON value holds, its strings left out: they are copied whole from a record."""
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    items = value.values() if isinstance(value, dict) else value
    return [n for item in items for n in numbers(item)]


def test_the_pages_recorded_data_is_what_the_records_print():
    # T6.2e: the page replays the change benches' runs and shows the build's figures from the records the README's
    # claims are held to. What it ships is what bench/page_data.py reads from them, every number in it is one a record
    # prints, and every run a bench counted is there.
    data = page_data()
    built = data.built()
    out = repo_root() / "web" / "public" / "recorded"
    for name, content in built.items():
        assert (out / name).read_bytes() == content, f"web/public/recorded/{name} is stale: run bench/page_data.py"
    names = [record for record, _ in data.BENCHES.values()] + [data.STATS]
    printed = {float(n.replace(",", "")) for name in names
               for n in re.findall(r"\d[\d,]*(?:\.\d+)?", (data.RECORDS / name).read_text(encoding="utf-8"))}
    shown = {n for content in built.values() for n in numbers(json.loads(content))}
    assert shown <= printed, sorted(shown - printed)[:10]
    for change in json.loads(built["changes.json"])["changes"]:
        counted = {int(n) for line in change["said"] for n in re.findall(r"over (\d+) runs", line)}
        assert len(counted) == 1 and len(change["runs"]) == 2 * counted.pop(), change["bench"]
        assert {run["change"] for run in change["runs"]} == set(change["kinds"]), change["bench"]
        # MA-F20, MA-F22: a timing names the commit it ran at and the load it ran under
        assert change["commit"] and change["before"] is not None and change["during"] is not None, change["bench"]


def test_a_timing_record_that_names_no_commit_or_load_is_refused(tmp_path, monkeypatch):
    # MA-F22: a record taken from a log kept the log's name and time, and the page replayed runs of no named code.
    data = page_data()
    for name in [record for record, _ in data.BENCHES.values()] + [data.STATS]:
        (tmp_path / name).write_bytes((data.RECORDS / name).read_bytes())
    monkeypatch.setattr(data, "RECORDS", tmp_path)
    path = tmp_path / data.BENCHES["freshness"][0]
    head, sep, body = path.read_bytes().decode("utf-8").partition(chr(10) + "---" + chr(10))
    kept = [line for line in body.split(chr(10)) if not line.startswith("at ")]
    path.write_bytes((head + sep + chr(10).join(kept)).encode("utf-8"))
    with pytest.raises(data.RecordError, match="no commit and no load"):
        data.built()


def test_a_record_line_the_page_cannot_read_is_an_error(tmp_path, monkeypatch):
    # T6.2e: a bench that prints a new kind of line must not drop out of the page unseen.
    data = page_data()
    for name in [record for record, _ in data.BENCHES.values()] + [data.STATS]:
        (tmp_path / name).write_bytes((data.RECORDS / name).read_bytes())
    monkeypatch.setattr(data, "RECORDS", tmp_path)
    assert data.built()  # the copies read as the records do
    path = tmp_path / data.BENCHES["revocation"][0]
    path.write_bytes(path.read_bytes() + b"  run 6, the widening: scan at 1.04 s took 2.54 s (1 read)" + bytes([10]))  # never seen
    with pytest.raises(data.RecordError, match="none of its forms"):
        data.built()


def test_a_recorded_trace_keeps_the_rows_the_page_draws_and_the_text_they_name(monkeypatch):
    # T6.2c: a stage's rows past the page's first ones, and the text of units nothing drawn names, stay out of the
    # recording; how many rows each stage held stays in it.
    spec = importlib.util.spec_from_file_location("page_traces", repo_root() / "bench" / "page_traces.py")
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    monkeypatch.setattr(recorder, "SHOWN", 1)
    trace = {stage: [{"unit_id": unit, "rank": rank, "score": 1 / rank} for rank, unit in enumerate("abcd", 1)]
             for stage in recorder.STAGES}
    trace.update(role="sales", excerpts=[{"units": ["c"], "labels": ["sales"], "text": "c"}],
                 units={unit: {"doc_id": "DOC", "section": "", "labels": ["sales"], "text": unit} for unit in "abcd"})
    kept = recorder.trimmed(trace)
    assert kept["counts"] == dict.fromkeys(recorder.STAGES, 4)
    assert all([hit["unit_id"] for hit in kept[stage]] == ["a"] for stage in recorder.STAGES)
    assert set(kept["units"]) == {"a", "c"}  # the first row, and the excerpt


def test_the_recorder_keeps_the_rows_the_page_draws():
    # MT-F23: both were ten by copy; raised in the page alone, a replayed trace of ten rows said "the first 20 shown".
    spec = importlib.util.spec_from_file_location("page_traces", repo_root() / "bench" / "page_traces.py")
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    page = (repo_root() / "web" / "src" / "main.ts").read_text(encoding="utf-8")
    assert re.search(r"^const SHOWN = (\d+)$", page, re.M)[1] == str(recorder.SHOWN)


def test_the_recorded_traces_hold_each_roles_view_and_say_when_and_with_what():
    # T6.2c: what the page replays where no server answers: each recorded question asked as every role, with when, at
    # which commit and with which models; every unit and excerpt in a trace one its role may read, a verdict for the
    # question's own role alone, and no stage holding more rows than the page draws.
    from needtoknow.policy import load_policy, readable
    folder = repo_root() / "web" / "public" / "recorded" / "traces"
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    assert {"recorded", "commit", "clean", "models", "embedder", "reranker"} <= set(index) and index["models"]
    policy = load_policy()
    assert index["roles"] == list(policy["roles"])
    dev = {json.loads(line)["id"] for line in (repo_root() / "eval" / "golden" / "dev.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()}
    asked = [question["id"] for question in index["questions"]]
    assert asked and set(asked) <= dev  # never the test split
    assert sorted(path.name for path in folder.glob("*.json")) == sorted(["index.json", *(f"{q}.json" for q in asked)])
    shown = int(re.search(r"^const SHOWN = (\d+)$", (repo_root() / "web" / "src" / "main.ts").read_text(
        encoding="utf-8"), re.M)[1])
    for question in index["questions"]:
        kept = json.loads((folder / f"{question['id']}.json").read_text(encoding="utf-8"))
        assert kept["question"] == question["id"] and set(kept["traces"]) == set(policy["roles"]), question["id"]
        for role, trace in kept["traces"].items():
            where = (question["id"], role)
            reads = readable(policy, role)
            assert trace["role"] == role and trace["answer"]["status"], where
            assert all(set(unit["labels"]) & reads for unit in trace["units"].values()), where
            assert all(set(excerpt["labels"]) & reads for excerpt in trace["excerpts"]), where
            assert (trace["verdict"] is not None) == (role == question["role"]), where
            assert "error" not in (trace["verdict"] or {}), where
            for stage in ("dense", "sparse", "fused", "reranked"):
                assert len(trace[stage]) == min(shown, trace["counts"][stage]), (*where, stage)
