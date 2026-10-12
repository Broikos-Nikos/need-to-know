"""The scan (T2.8) and the change bench (P2's exit): only what changed is read again, renames and touches cost no
cut, a version bump marks every file, and every kind of change the simulator makes gives exactly its unit diff."""

import json
import os
from pathlib import Path

import pytest

from needtoknow.corpus import assemble, mutate
from needtoknow.corpus.plan import header
from needtoknow.ingest import manifest, pipeline, segments
from needtoknow.ingest.manifest import build, opened, scan

FACT = {"en": "Returned items are inspected within 3 working days of arrival at the warehouse.",
        "el": "Τα επιστρεφόμενα είδη ελέγχονται εντός 3 εργάσιμων ημερών από την άφιξή τους στην αποθήκη."}
PROSE = {"en": "The team checks every case against the procedure and records the outcome before it moves on. " * 3,
         "el": "Η ομάδα ελέγχει κάθε υπόθεση με βάση τη διαδικασία και καταγράφει το αποτέλεσμα πριν προχωρήσει. " * 3}
SHAPES = [("returns", "en", "txt"), ("returns", "el", "md"), ("returns", "en", "html"), ("logistics", "en", "txt")]


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    docs = []
    for n, (owner, lang, fmt) in enumerate(SHAPES, 1):
        doc = {"id": f"{owner[:3].upper()}_PRO_{n:03d}", "type": "PRO", "type_name": "Procedure", "owner": owner,
               "language": lang, "format": fmt, "target_mb": 0.03, "version": "1.2",
               "sections": [{"label": owner, "topic_en": "Scope", "topic_el": "Πεδίο", "facts": ["F1"], "canary": None}]}
        docs.append(doc)
        body = ["", PROSE[lang] + FACT[lang], "", PROSE[lang].replace("records", "notes"), "", PROSE[lang].replace("records", "keeps"), ""]
        (base / "narrative" / f"{doc['id']}.txt").write_bytes("\n".join(header(doc, 1, doc["sections"][0]) + body).encode("utf-8"))
    facts = [{"id": "F1", "family": "F1", "label": "returns", "kind": "single", "values": ["3", "3"], **FACT,
              "placements": [d["id"] for d in docs]}]
    (base / "plan" / "catalogue.json").write_bytes(json.dumps(docs, ensure_ascii=False).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(json.dumps(facts, ensure_ascii=False).encode("utf-8"))
    assert assemble.assemble()["written"] == len(docs)
    build()
    return base, docs


def doc_units(doc_id):
    with opened() as db:
        return {row[0]: (row[1], json.loads(row[2]), row[3], row[4]) for row in
                db.execute("SELECT unit_id, text, labels, layer, section_key FROM units WHERE doc_id = ?", (doc_id,))}


def all_units():
    with opened() as db:
        return {row[0]: row[1] for row in db.execute("SELECT unit_id, doc_id FROM units")}


def counting_reads(monkeypatch):
    reads = []
    real = pipeline.document_units

    def counted(file, meta, unknown=None):
        reads.append(meta["doc_id"])
        return real(file, meta, unknown)

    monkeypatch.setattr(pipeline, "document_units", counted)
    return reads


def test_a_scan_of_an_unchanged_corpus_reads_nothing(corpus, monkeypatch):
    reads = counting_reads(monkeypatch)
    summary = scan()
    assert reads == [] and summary["files_read"] == 0 and summary["changes"] == {}


def test_a_touched_file_is_read_once_and_changes_no_unit(corpus, monkeypatch):
    base, docs = corpus
    target = next((base / "full").glob(f"{docs[0]['id']}.txt"))
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    reads = counting_reads(monkeypatch)
    summary = scan()
    assert reads == [docs[0]["id"]] and summary["changes"] == {"touched": 1} and summary["units_added"] == 0
    assert scan()["files_read"] == 0  # the new mtime is recorded


def test_a_version_bump_reads_every_file(corpus, monkeypatch):
    monkeypatch.setattr(pipeline, "VERSIONS", pipeline.VERSIONS + ", bumped")
    reads = counting_reads(monkeypatch)
    summary = scan()
    assert sorted(reads) == sorted(d["id"] for d in corpus[1]) and summary["units_added"] == summary["units_removed"] == 0


def test_a_sidecar_default_audience_change_relabels_without_a_text_change(corpus):
    base, docs = corpus
    meta_path = base / "full" / f"{docs[3]['id']}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    before = {u[0]: u[1] for u in doc_units(docs[3]["id"]).values()}
    meta["default_audience"] = "finance"
    meta_path.write_bytes(json.dumps(meta).encode("utf-8"))
    summary = scan()
    assert summary["changes"] == {"changed": 1} and summary["units_added"] == summary["units_removed"] >= 1
    after = {u[0]: u[1] for u in doc_units(docs[3]["id"]).values()}
    assert any(labels == ["finance"] for labels in after.values()) and set(after) == set(before)


def test_every_kind_of_change_gives_exactly_its_unit_diff(corpus):
    """P2's exit: the change bench. Each change of the simulator, one at a time, then a scan."""
    base, docs = corpus
    checked = []
    for seed in range(1, 1 + len(mutate.KINDS)):
        before_all = all_units()
        (entry,) = mutate.mutate(count=1, seed=seed)
        doc = entry["doc_id"]
        before = {uid: v for uid, v in doc_units(doc).items()} if entry["kind"] != "new_file" else {}
        before_texts = {uid: v[0] for uid, v in before.items()}
        summary = scan()
        after = doc_units(doc)
        added = {uid: after[uid] for uid in set(after) - set(before)}
        removed = {uid: before[uid] for uid in set(before) - set(after)}
        others_before = {u: d for u, d in before_all.items() if d != doc}
        others_after = {u: d for u, d in all_units().items() if d != doc}
        assert others_before == others_after, (entry["kind"], "a unit of another document changed")
        kind = entry["kind"]
        checked.append(kind)
        if kind == "value_update":
            assert removed and len(added) == len(removed)
            assert all(entry["old_value"] in v[0] for v in removed.values())
            assert all(entry["sentence"] in v[0] for v in added.values())
        elif kind == "insert_paragraph":
            assert any(entry["paragraph"] in v[0] for v in added.values()) and len(removed) <= 1
            assert len({v[3] for v in [*added.values(), *removed.values()]}) == 1  # one section
        elif kind == "delete_paragraph":
            assert removed and any(entry["paragraph_start"][:40] in v[0] for v in removed.values()) and len(added) <= 1
        elif kind == "relabel":
            assert sorted(v[0] for v in removed.values()) == sorted(v[0] for v in added.values()) and added
            assert all(entry["to_labels"][1] in v[1] for v in added.values())
        elif kind in ("rename", "re_encode"):
            assert not added and not removed
            assert summary["changes"] == {"renamed" if kind == "rename" else "touched": 1}
            assert summary["files_read"] == (0 if kind == "rename" else 1)  # a move costs no read at all
        elif kind == "annex_refresh":
            assert added and len(added) == len(removed) <= 2 * entry["rows_changed"]
            assert all(v[2] == "table" for v in [*added.values(), *removed.values()])
        elif kind == "new_file":
            assert added and not removed and summary["changes"] == {"added": 1}
        elif kind == "delete_file":
            assert removed and not after and summary["changes"] == {"removed": 1}
    assert checked == mutate.KINDS  # every kind, each as asked, none as a fallback


def test_every_ingest_module_that_cuts_is_in_the_version_the_scan_compares():
    here = Path(pipeline.__file__).parent
    cutting = {p.name for p in here.glob("*.py")} - {"__init__.py", "manifest.py", "watch.py"}
    assert cutting == set(pipeline.STEP_FILES)


def test_the_version_moves_with_any_step_or_label_name_and_not_with_the_manifest(tmp_path):
    here = Path(pipeline.__file__).parent
    for name in pipeline.STEP_FILES + ("manifest.py",):
        (tmp_path / name).write_bytes((here / name).read_bytes())
    before = pipeline.code_digest(tmp_path)
    assert before == pipeline.code_digest()
    (tmp_path / "manifest.py").write_bytes(b"# a change outside the steps")
    assert pipeline.code_digest(tmp_path) == before
    (tmp_path / "segments.py").write_bytes((here / "segments.py").read_bytes() + b"# an access rule changed")
    assert pipeline.code_digest(tmp_path) != before
    renamed = dict(pipeline.LABEL_EN, sales="Sales Department")
    assert pipeline.code_digest(names=(renamed, pipeline.LABEL_EL, "unlabelled")) != pipeline.code_digest()


def test_a_rule_change_relabels_through_scan_and_through_build(corpus, monkeypatch):
    base, docs = corpus
    returns = docs[0]["id"]
    assert any(labels == ["returns"] for _, labels, _, _ in doc_units(returns).values())
    monkeypatch.setitem(segments.NAMES, segments.fold("Returns"), "legal")  # the same line now names legal
    assert scan()["units_added"] == 0  # the code on disk is unchanged, so the version is too
    monkeypatch.setattr(pipeline, "VERSIONS", pipeline.VERSIONS + " with the rule changed")  # as the new code gives it
    moved = scan()
    assert moved["units_added"] == moved["units_removed"] > 0
    assert not any(labels == ["returns"] for _, labels, _, _ in doc_units(returns).values())
    monkeypatch.setitem(segments.NAMES, segments.fold("Returns"), "returns")
    again = build()  # build alone, with the version left where it is: it trusts the fresh cut, not the hashes
    assert again["units_added"] == again["units_removed"] == moved["units_added"]
    assert any(labels == ["returns"] for _, labels, _, _ in doc_units(returns).values())


def test_a_missing_corpus_folder_is_refused_and_the_manifest_kept(corpus, capsys):
    from needtoknow.cli import main
    base, docs = corpus
    total = len(all_units())
    (base / "full").rename(base / "full.retired")  # as a swap killed between its renames leaves it
    with pytest.raises(manifest.ManifestError, match="is not there"):
        scan()
    assert main(["ingest", "scan"]) == 1
    assert "ingest: " in capsys.readouterr().out and len(all_units()) == total
