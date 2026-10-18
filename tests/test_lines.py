"""The append only line files (DR-F30): a last line a stopped write cut is left out by every reader, even inside a Greek
letter, and cut off by every writer before it appends; a whole line that is not a record is named by file and line."""

import json

import pytest

from needtoknow.lines import LineError, append, mend, records

GREEK = json.dumps({"seq": 1, "new": "Ελληνικά"}, ensure_ascii=False).encode("utf-8")


def cut_inside_a_greek_letter(path):
    path.write_bytes(GREEK + b"\n" + GREEK[:GREEK.index("Ελ".encode("utf-8")) + 3])


def test_a_last_line_a_stopped_write_cut_is_left_out_even_inside_a_greek_letter(tmp_path):
    path = tmp_path / "journal.jsonl"
    cut_inside_a_greek_letter(path)
    assert records(path) == [{"seq": 1, "new": "Ελληνικά"}]
    assert records(tmp_path / "none.jsonl") == []


def test_a_writer_cuts_the_stopped_line_off_so_the_next_record_starts_its_own(tmp_path):
    path = tmp_path / "spend.jsonl"
    cut_inside_a_greek_letter(path)
    append(path, {"seq": 2})
    assert path.read_bytes().split(b"\n") == [GREEK, b'{"seq": 2}', b""]
    assert not mend(path) and [r["seq"] for r in records(path)] == [1, 2]


def test_a_whole_line_that_is_not_a_record_is_named_by_file_and_line(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_bytes(GREEK + b"\n\nnot json\n")
    with pytest.raises(LineError, match="run.jsonl line 3 is not JSON"):
        records(path)
    path.write_bytes(GREEK + b"\n[1, 2]\n")
    with pytest.raises(LineError, match="run.jsonl line 2 is JSON but not a record"):
        records(path)
    path.write_bytes(GREEK + b"\n")
    with pytest.raises(LineError, match="run.jsonl line 1: no answer"):
        records(path, check=lambda record: None if "answer" in record else "no answer")


def test_a_shared_files_writer_waits_for_its_lock_before_it_mends(tmp_path):
    # DR-F33: the spend log has three writers, and a mend cannot tell a line in flight from a stopped one, so a writer
    # takes the file's own lock first.
    import threading
    import time

    from needtoknow.corpus import exclusive
    path = tmp_path / "spend.jsonl"
    held, let_go = threading.Event(), threading.Event()

    def other_writer():
        with exclusive(tmp_path, name="spend.jsonl.lock"):
            path.write_bytes(GREEK[:9])  # its line, half written
            held.set()
            let_go.wait(5)
            with path.open("ab") as handle:
                handle.write(GREEK[9:] + b"\n")

    thread = threading.Thread(target=other_writer)
    thread.start()
    held.wait(5)
    threading.Timer(0.3, let_go.set).start()
    began = time.perf_counter()
    append(path, {"seq": 2}, shared=True)
    thread.join()
    assert time.perf_counter() - began >= 0.25
    assert records(path) == [{"seq": 1, "new": "Ελληνικά"}, {"seq": 2}]
