"""The evaluation run (T5.5c): every item asked as its role with the cache off, each repeat its own call, every answer
written as it comes with the model that gave it, and a run cut short resumed without asking anything twice."""

import json

import pytest

from needtoknow.eval.golden import golden_path, load_items
from needtoknow.eval.protocol import ProtocolError
from needtoknow.eval import run as R
from needtoknow.eval.run import run
from needtoknow.lines import LineError


class Asker:
    """Stands in for answer.pipeline.ask, recording how it was called; it can be told to fail after some calls."""

    def __init__(self, fail_after: int | None = None):
        self.calls, self.fail_after = [], fail_after

    def __call__(self, client, question, role, use_cache=True, repeat=0, closed_book=False, guess=False):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("killed")
        self.calls.append({"role": role, "use_cache": use_cache, "repeat": repeat, "closed_book": closed_book,
                           "guess": guess})
        shown = [] if closed_book else [{"doc_id": "D1", "section": "S", "units": ["u1", "u2"]}]
        return {"status": "answered", "answer": "7 percent", "citations": [{"doc_id": "D1", "section": "S"}],
                "dropped": [], "model": "deepseek-v4-flash", "cost_usd": 0.0005, "excerpts": shown,
                "seconds": {"search": 0.25, "provider": 1.2504, "total": 1.6}}


def made_by(head="a" * 40):
    return lambda split, purpose, repeats, closed_book, guess=False: {"commit": head, "split": split, "repeats": repeats}


@pytest.fixture(autouse=True)
def recorded(monkeypatch):
    """The provenance stood in for, so a test needs no repository or manifest."""
    monkeypatch.setattr(R, "provenance", made_by())


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_item_is_asked_as_its_role_once_a_repeat_with_the_cache_off(tmp_path):
    items = load_items(golden_path("dev"))
    asker, out = Asker(), tmp_path / "dev.jsonl"
    summary = run("dev", "development", 2, out, None, ask=asker, tables=lambda units: len(units) - 1)
    assert (summary["asked"], summary["models"]) == (2 * len(items), ["deepseek-v4-flash"])
    assert {call["use_cache"] for call in asker.calls} == {False}  # the protocol's cache "off"
    assert [call["role"] for call in asker.calls[:len(items)]] == [item["role"] for item in items]
    written = lines(out)
    assert {(r["id"], r["repeat"]) for r in written} == {(i["id"], n) for i in items for n in (0, 1)}
    assert {(r["model"], r["units"], r["tables_shown"], r["closed_book"]) for r in written} == {
        ("deepseek-v4-flash", 2, 1, False)}
    # Each answer keeps its stages' seconds, so the run's ledger holds its latency as well as its cost (number 7).
    assert {json.dumps(r["seconds"], sort_keys=True) for r in written} == {
        json.dumps({"provider": 1.25, "search": 0.25, "total": 1.6}, sort_keys=True)}


def test_a_run_cut_short_resumes_without_asking_anything_twice(tmp_path):
    items = load_items(golden_path("dev"))
    out = tmp_path / "dev.jsonl"
    with pytest.raises(RuntimeError):
        run("dev", "development", 1, out, None, ask=Asker(fail_after=10), tables=lambda units: 0)
    assert len(lines(out)) == 10
    again = Asker()
    summary = run("dev", "development", 1, out, None, ask=again, tables=lambda units: 0)
    assert (summary["kept"], summary["asked"], len(again.calls)) == (10, len(items) - 10, len(items) - 10)
    assert sorted(r["id"] for r in lines(out)) == sorted(i["id"] for i in items)


def test_a_run_writes_its_provenance_first_and_is_resumed_only_at_its_commit(tmp_path):
    out = tmp_path / "dev.jsonl"
    with pytest.raises(RuntimeError):
        run("dev", "development", 1, out, None, ask=Asker(fail_after=3), tables=lambda u: 0, made_by=made_by())
    written = json.loads((tmp_path / "dev.jsonl.provenance.json").read_text(encoding="utf-8"))
    assert written == {"commit": "a" * 40, "split": "dev", "repeats": 1}
    with pytest.raises(R.RunError, match="started at aaaaaaaaaaaa and this is bbbbbbbbbbbb"):
        run("dev", "development", 1, out, None, ask=Asker(), tables=lambda u: 0, made_by=made_by("b" * 40))
    assert len(lines(out)) == 3  # nothing asked at the other commit


def test_a_baseline_run_starts_only_on_a_clean_tree(tmp_path):
    # Tick 139: the test and closed book runs began while the report was being written beside them, and their
    # provenance says so; a published run's commit must be all that made it.
    dirty = lambda split, purpose, repeats, closed_book, guess=False: {"commit": "c" * 40, "clean": False}
    with pytest.raises(R.RunError, match="commit them first"):
        run("dev", "baseline", 1, tmp_path / "dev.jsonl", None, ask=Asker(), tables=lambda u: 0, made_by=dirty)
    assert not (tmp_path / "dev.jsonl").exists()
    run("dev", "development", 1, tmp_path / "dev.jsonl", None, ask=Asker(), tables=lambda u: 0, made_by=dirty)


def test_a_closed_book_run_asks_with_no_excerpts_and_the_protocol_is_asked_first(tmp_path):
    asker = Asker()
    run("dev", "development", 1, tmp_path / "closed.jsonl", None, closed_book=True, ask=asker, tables=lambda u: 0)
    assert {call["closed_book"] for call in asker.calls} == {True}
    assert {(r["closed_book"], r["units"]) for r in lines(tmp_path / "closed.jsonl")} == {(True, 0)}
    with pytest.raises(ProtocolError):
        run("test", "development", 1, tmp_path / "test.jsonl", None, ask=Asker())


def test_a_best_guess_run_asks_with_no_excerpts_and_says_so_in_every_line(tmp_path):
    asker = Asker()
    run("dev", "development", 1, tmp_path / "guess.jsonl", None, guess=True, ask=asker, tables=lambda u: 0)
    assert {(call["closed_book"], call["guess"]) for call in asker.calls} == {(True, True)}
    assert {(r["closed_book"], r["guess"], r["units"]) for r in lines(tmp_path / "guess.jsonl")} == {(True, True, 0)}


def test_a_run_resumes_only_with_the_split_purpose_and_book_it_started_with(tmp_path):
    # DR-F28: resumed closed book, a file begun open book held both kinds of answer under a provenance naming one.
    out = tmp_path / "dev.jsonl"
    started = lambda split, purpose, repeats, closed_book, guess=False: {
        "commit": "a" * 40, "split": split, "purpose": purpose, "closed_book": closed_book, "guess": guess}
    with pytest.raises(RuntimeError):
        run("dev", "development", 1, out, None, ask=Asker(fail_after=5), tables=lambda u: 0, made_by=started)
    for purpose, book in (("development", {"closed_book": True}), ("development", {"guess": True}), ("baseline", {})):
        with pytest.raises(R.RunError, match="one file holds one kind of answer"):
            run("dev", purpose, 1, out, None, ask=Asker(), tables=lambda u: 0, made_by=started, **book)
    assert len(lines(out)) == 5
    summary = run("dev", "development", 1, out, None, ask=Asker(), tables=lambda u: 0, made_by=started)
    assert summary["kept"] == 5 and {r["closed_book"] for r in lines(out)} == {False}


def test_a_last_line_a_stopped_write_left_is_cut_off_and_asked_again(tmp_path):
    # DR-F28: a line cut mid write made asked() raise, so the run its docstring says resumes could not.
    out = tmp_path / "dev.jsonl"
    with pytest.raises(RuntimeError):
        run("dev", "development", 1, out, None, ask=Asker(fail_after=4), tables=lambda u: 0)
    cut = '{"id": "D001", "repeat": 0, "answer": "Τα έγ'.encode("utf-8")[:-1]  # stopped inside a Greek letter
    out.write_bytes(out.read_bytes() + cut)
    assert len(R.asked(out)) == 4
    summary = run("dev", "development", 1, out, None, ask=Asker(), tables=lambda u: 0)
    assert summary["mended"] and summary["kept"] == 4
    assert sorted(r["id"] for r in lines(out)) == sorted(item["id"] for item in load_items(golden_path("dev")))


def test_a_folder_or_a_line_no_run_wrote_is_refused_before_anything_is_asked_or_written(tmp_path):
    # HS-F21: --out naming a folder failed only when the first answer, paid for, was appended; a bad line in a file
    # resumed ended in a traceback after a fresh provenance sidecar had been written beside it.
    asker = Asker()
    with pytest.raises(R.RunError, match="is a folder: --out names the file the answers go to"):
        run("dev", "development", 1, tmp_path, None, ask=asker, tables=lambda u: 0)
    out = tmp_path / "dev.jsonl"
    out.write_bytes(b'{"id": "D001", "repeat": 0, "status": "answered", "answer": ["7"], "citations": []}\n')
    with pytest.raises(LineError, match='dev.jsonl line 1: answer is \\["7"\\], where a run writes text'):
        run("dev", "development", 1, out, None, ask=asker, tables=lambda u: 0)
    assert asker.calls == [] and not out.with_name("dev.jsonl.provenance.json").exists()


def test_the_scorer_refuses_a_line_no_run_wrote_by_file_and_line(tmp_path, capsys):
    # HS-F21: lines of an id alone scored as wrong answers with exit 0, and an answer that was a list was a traceback.
    from needtoknow.cli import main
    bare = tmp_path / "bare.jsonl"
    bare.write_bytes(b'{"id": "D001"}\n')
    assert main(["eval", "score", str(bare), "--purpose", "development"]) == 1
    assert capsys.readouterr().out == "eval score: bare.jsonl line 1: no repeat, which every answer a run writes holds\n"
