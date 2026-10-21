"""Canaries (T5.4e, `specs/evaluation.md` section 6, item 4): a plain string search over what each role was told, for
the canaries of labels it may not read."""

import json
import sqlite3

from needtoknow import canaries
from needtoknow.cli import main
from needtoknow.corpus.plan import build
from needtoknow.settings import data_dir


def test_every_planted_canary_names_its_sections_label(monkeypatch, tmp_path):
    # The scan reads a canary's label from its suffix alone, so the plan must go on writing it there.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    build()
    docs = json.loads((data_dir() / "corpus" / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    planted = [(s["canary"], s["label"]) for doc in docs for s in doc["sections"] if s["canary"]]
    names = canaries.suffixes()
    assert len(planted) > 40
    for canary, label in planted:
        match = canaries.PATTERN.fullmatch(canary)
        assert match and names[match.group(1)] == label, canary


def test_a_canary_counts_only_where_its_label_is_not_read():
    names = canaries.suffixes()
    text = "CANARY_0A1B2C_CUSTOMERCARE, then CANARY_FFFFFF_LEGAL, then CANARY_123456_NOSUCHLABEL"
    assert canaries.found(text, frozenset({"general", "customer-care"}), names) == [
        "CANARY_FFFFFF_LEGAL", "CANARY_123456_NOSUCHLABEL"]
    assert canaries.found(text, frozenset({"customer-care", "legal"}), names) == ["CANARY_123456_NOSUCHLABEL"]


def answer_cache(path, rows):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE answers (key TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, "
               "text TEXT NOT NULL, reasoning TEXT NOT NULL, usage TEXT NOT NULL, stop_reason TEXT NOT NULL, "
               "stored REAL NOT NULL)")
    db.executemany("INSERT INTO answers VALUES (?, 'deepseek', 'deepseek-flash', ?, ?, '{}', 'stop', 0)", rows)
    db.commit()
    db.close()


def test_the_scan_reads_each_cached_answer_for_every_role_its_log_names(tmp_path):
    # Roles shown the same text share an entry (AC-F11), so an entry is read for each of them; the reasoning counts,
    # and an entry no line names counts any canary.
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    answer_cache(tmp_path / "answers.db", [(a, '{"answer": "CANARY_00000A_RETURNS"}', ""),
                                           (b, '{"status": "not_found"}', "it says CANARY_00000B_LEGAL"),
                                           (c, "CANARY_00000C_HR", "")])
    lines = [{"role": "returns", "prompt": a[:16]}, {"role": "sales", "prompt": a[:16]},
             {"role": "legal", "prompt": a[:16]}, {"role": "legal", "prompt": b[:16]}, {"role": "hr", "prompt": b},
             {"role": "sales", "prompt": "d" * 16, "status": "CANARY_00000D_FINANCE"}]
    (tmp_path / "spend.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    report = canaries.scan(tmp_path / "spend.jsonl", tmp_path / "answers.db")
    assert (report["lines"], report["answers"], report["unnamed"]) == (6, 5, 1)
    assert sorted(report["leaks"]) == sorted([f"cached answer {a[:16]}, legal: CANARY_00000A_RETURNS",
                                              f"cached answer {b[:16]}, hr: CANARY_00000B_LEGAL",
                                              f"cached answer {c[:16]}, no role: CANARY_00000C_HR",
                                              "spend log line 6, sales: CANARY_00000D_FINANCE"])


def test_the_scan_reads_whole_lines_and_says_when_it_cannot_read_the_cache(monkeypatch, tmp_path, capsys):
    # DR-F30: a last line a stopped write cut ended the scan in a traceback, and so did a cache it could not read,
    # which must clear nothing.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(canaries, "PAGE", tmp_path / "no-page.json")
    (tmp_path / "logs").mkdir()
    (tmp_path / "llm-cache").mkdir()
    answer_cache(tmp_path / "llm-cache" / "answers.db", [("e" * 64, "CANARY_00000E_HR", "")])
    whole = json.dumps({"role": "hr", "prompt": "e" * 16}).encode("utf-8")
    (tmp_path / "logs" / "spend.jsonl").write_bytes(whole + b"\n" + whole[:20])
    assert main(["eval", "canaries"]) == 0
    assert capsys.readouterr().out.startswith("1 spend log lines, 1 cached answers read through them")
    (tmp_path / "llm-cache" / "answers.db").write_bytes(b"not a database")
    assert main(["eval", "canaries"]) == 1
    assert capsys.readouterr().out.startswith("eval canaries: the answer cache at ")


def test_ntk_eval_canaries_exits_1_on_a_leak_and_0_on_none(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(canaries, "PAGE", tmp_path / "no-page.json")
    (tmp_path / "logs").mkdir()
    (tmp_path / "llm-cache").mkdir()
    answer_cache(tmp_path / "llm-cache" / "answers.db", [("e" * 64, "CANARY_00000E_HR", "")])
    spend = tmp_path / "logs" / "spend.jsonl"
    spend.write_text(json.dumps({"role": "hr", "prompt": "e" * 16}) + "\n", encoding="utf-8")
    assert main(["eval", "canaries"]) == 0
    assert "canaries of a label the role may not read: 0" in capsys.readouterr().out
    spend.write_text(json.dumps({"role": "sales", "prompt": "e" * 16}) + "\n", encoding="utf-8")
    assert main(["eval", "canaries"]) == 1
    assert "leak: cached answer eeeeeeeeeeeeeeee, sales: CANARY_00000E_HR" in capsys.readouterr().out


def test_a_canary_is_found_in_any_case_and_with_any_separator():
    # AC-F14: lower case, spaces, dashes, a line break and mixed case each passed the scan; a model that repeats a
    # canary need not keep its form.
    names = canaries.suffixes()
    shapes = ["CANARY_0A1B2C_LEGAL", "canary_0a1b2c_legal", "CANARY 0A1B2C LEGAL", "CANARY_0A1B2C_" + chr(10) + "LEGAL",
              "Canary-0a1B2c-Legal", "CANARY__0A1B2C - LEGAL"]
    for shape in shapes:
        assert canaries.found(f"the reply: {shape}.", frozenset({"general", "sales"}), names) == [shape], shape
        assert canaries.found(f"the reply: {shape}.", frozenset({"legal"}), names) == [], shape
    assert canaries.found("canary_0a1b2c_customercare", frozenset({"customer-care"}), names) == []


def test_a_scan_of_nothing_says_so_and_fails(monkeypatch, tmp_path, capsys):
    # AC-F15: with no spend log, no cache and no recorded page, the scan printed zeros and exited 0, as after a clean
    # scan of hundreds of answers.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(canaries, "PAGE", tmp_path / "no-page.json")
    assert main(["eval", "canaries"]) == 1
    assert "eval canaries: nothing to scan" in capsys.readouterr().out


def test_the_scan_reads_each_recorded_page_trace_for_the_role_it_was_asked_as(monkeypatch, tmp_path, capsys):
    # The page's recorded traces are what a stranger reads: each is read for its own role, a canary split by a line
    # break within one of its texts too.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    page = tmp_path / "traces"
    page.mkdir()
    split = "CANARY_00000F_" + chr(10) + "LEGAL"
    traces = {"legal": {"answer": {"answer": "legal may read CANARY_00000F_LEGAL"}},
              "sales": {"excerpts": [{"text": f"and sales may not: {split}"}]}}
    (page / "index.json").write_text(json.dumps({"questions": [{"id": "D1"}]}), encoding="utf-8")
    (page / "D1.json").write_text(json.dumps({"question": "D1", "traces": traces}), encoding="utf-8")
    monkeypatch.setattr(canaries, "PAGE", page)
    assert main(["eval", "canaries"]) == 1
    out = capsys.readouterr().out
    assert "leak: page trace D1, sales: CANARY_00000F_ LEGAL" + chr(10) in out and "page trace D1, legal" not in out
    assert "2 recorded page traces" in out
