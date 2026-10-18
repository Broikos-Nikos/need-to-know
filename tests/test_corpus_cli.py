"""The corpus commands through the command line, the way a stranger runs them on a fresh clone (MT-F1, MT-F4)."""

import json

import pytest

from needtoknow.cli import main
from needtoknow.corpus import assemble, narrative_dir
from needtoknow.corpus.check import check
from needtoknow.corpus.plan import header
from needtoknow.settings import repo_root

PROSE = {"en": "The team checks every case against the procedure and records the outcome before it moves on. " * 3,
         "el": "Η ομάδα ελέγχει κάθε υπόθεση με βάση τη διαδικασία και καταγράφει το αποτέλεσμα πριν προχωρήσει. " * 3}
SHAPES = [("en", "txt"), ("el", "txt"), ("en", "md"), ("el", "md"), ("en", "html"), ("el", "html")]


@pytest.fixture
def small(monkeypatch, tmp_path):
    """Six documents, one per language and format, with their narrative written."""
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    docs = []
    for n, (lang, fmt) in enumerate(SHAPES, 1):
        owner = ["sales", "logistics", "returns", "legal", "finance", "hr"][n - 1]
        doc = {"id": f"{owner[:3].upper()}_PRO_{n:03d}", "type": "PRO", "type_name": "Procedure", "owner": owner,
               "language": lang, "format": fmt, "target_mb": 0.02, "version": "1.2",
               "sections": [{"label": owner, "topic_en": "Scope", "topic_el": "Πεδίο", "facts": [], "canary": None}]}
        docs.append(doc)
        text = "\n".join(header(doc, 1, doc["sections"][0]) + ["", PROSE[lang], ""])
        (base / "narrative" / f"{doc['id']}.txt").write_bytes(text.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps(docs, ensure_ascii=False).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    return base, docs


def state(base):
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in (base / "full").rglob("*") if p.is_file()}


def test_plan_then_assemble_without_a_narrative_names_the_gap_without_a_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    assert main(["corpus", "plan"]) == 0
    assert "160 documents" in capsys.readouterr().out
    # An empty narrative folder in the data directory takes the place of the committed one.
    (tmp_path / "corpus" / "narrative").mkdir()
    assert main(["corpus", "assemble"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("nothing assembled: 160 of 160 documents have no narrative") and "unchanged" in out
    assert not (tmp_path / "corpus" / "full").exists()


@pytest.mark.parametrize("action", [["assemble"], ["check"], ["samples"], ["mutate", "--count", "1"], ["draw"]])
def test_a_corpus_command_without_a_plan_says_so_in_one_line(monkeypatch, tmp_path, capsys, action):
    # HS-F22: draw named the missing file as an Errno; assemble, check, samples and mutate ended in a traceback.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    assert main(["corpus", *action]) == 1
    assert capsys.readouterr().out == (f"corpus {action[0]}: no plan in {tmp_path / 'corpus' / 'plan'}: run ntk corpus "
                                       "plan first\n")


def test_a_fresh_clone_finds_the_committed_narrative_and_it_carries_the_whole_plan(monkeypatch, tmp_path):
    # MT-F2: the plan is code and the narrative is committed, so a clone has the corpus's whole source.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    assert main(["corpus", "plan"]) == 0
    assert narrative_dir(tmp_path / "corpus") == repo_root() / "narrative"
    result = check()
    assert result["problems"] == ["full/: no assembled corpus"]
    counts = result["counts"]
    assert counts["documents"] == 160 and counts["facts found"] == counts["facts planned"] > 0
    assert counts["canaries found"] == counts["canaries planned"] > 0


def test_assemble_and_check_through_the_command_line(small, capsys):
    base, docs = small
    assert main(["corpus", "assemble"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"assembled {len(docs)} files") and "share by label" in out
    assert main(["corpus", "check"]) == 0
    assert capsys.readouterr().out.split("documents in full")[1].split()[0] == "6"


def test_a_missing_narrative_leaves_the_corpus_in_place_untouched(small, capsys):
    base, docs = small
    assert main(["corpus", "assemble"]) == 0
    before = state(base)
    (base / "narrative" / f"{docs[2]['id']}.txt").unlink()
    assert main(["corpus", "assemble"]) == 1
    assert f"1 of 6 documents have no narrative" in capsys.readouterr().out
    assert state(base) == before
    assert not any((base / name).exists() for name in ("full.building", "full.retired"))


def test_samples_replace_their_own_files_and_keep_anything_added_by_hand(small, tmp_path):
    out = tmp_path / "samples"
    out.mkdir()
    (out / "README.md").write_bytes(b"what these files are\n")
    (out / f"{small[1][0]['id']}.txt").write_bytes(b"an old sample\n")
    written = assemble.samples(out)
    assert (out / "README.md").read_bytes() == b"what these files are\n"
    assert len(written) == 13 and (out / f"{small[1][0]['id']}.txt").read_bytes() != b"an old sample\n"


def test_an_empty_catalogue_assembles_to_nothing_without_dividing_by_zero(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    (tmp_path / "corpus" / "plan").mkdir(parents=True)
    (tmp_path / "corpus" / "plan" / "catalogue.json").write_bytes(b"[]")
    assert main(["corpus", "assemble"]) == 0
    assert capsys.readouterr().out.startswith("assembled 0 files, 0.0 MB")


def test_the_check_says_how_many_problems_it_does_not_print(monkeypatch, capsys):
    from needtoknow.corpus import check as checker
    monkeypatch.setattr(checker, "check", lambda: {"counts": {}, "problems": [f"p{n}" for n in range(45)],
                                                   "invented_examples": []})
    assert main(["corpus", "check"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 41 and lines[-1] == "problem  and 5 more"
