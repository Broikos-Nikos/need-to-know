"""The evaluation run (T5.5c): every item asked as its role with the cache off, each repeat its own call, every answer
written as it comes with the model that gave it, and a run cut short resumed without asking anything twice."""

import json

import pytest

from needtoknow.eval.golden import golden_path, load_items
from needtoknow.eval.protocol import ProtocolError
from needtoknow.eval import run as R
from needtoknow.eval.run import run


class Asker:
    """Stands in for answer.pipeline.ask, recording how it was called; it can be told to fail after some calls."""

    def __init__(self, fail_after: int | None = None):
        self.calls, self.fail_after = [], fail_after

    def __call__(self, client, question, role, use_cache=True, repeat=0, closed_book=False):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("killed")
        self.calls.append({"role": role, "use_cache": use_cache, "repeat": repeat, "closed_book": closed_book})
        shown = [] if closed_book else [{"doc_id": "D1", "section": "S", "units": ["u1", "u2"]}]
        return {"status": "answered", "answer": "7 percent", "citations": [{"doc_id": "D1", "section": "S"}],
                "dropped": [], "model": "deepseek-v4-flash", "cost_usd": 0.0005, "excerpts": shown}


def made_by(head="a" * 40):
    return lambda split, purpose, repeats, closed_book: {"commit": head, "split": split, "repeats": repeats}


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


def test_a_closed_book_run_asks_with_no_excerpts_and_the_protocol_is_asked_first(tmp_path):
    asker = Asker()
    run("dev", "development", 1, tmp_path / "closed.jsonl", None, closed_book=True, ask=asker, tables=lambda u: 0)
    assert {call["closed_book"] for call in asker.calls} == {True}
    assert {(r["closed_book"], r["units"]) for r in lines(tmp_path / "closed.jsonl")} == {(True, 0)}
    with pytest.raises(ProtocolError):
        run("test", "development", 1, tmp_path / "test.jsonl", None, ask=Asker())
