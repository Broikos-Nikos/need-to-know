"""Expansion (T3.7): a hit widened to its neighbours in document order, never across its segment's boundary, overlaps
merged inside one segment only, and offsets that stay current when a unit moves (DR-F12)."""

import json

import pytest

from needtoknow.index.expand import excerpts
from needtoknow.ingest.manifest import opened

# One section, two segments: the second half was narrowed to legal by a later audience line.
UNITS = [("a", "returns", 0, "First."), ("b", "returns", 10, "Second."), ("c", "returns", 20, "Third."),
         ("d", "returns", 30, "Fourth."), ("x", "legal", 40, "Legal only."), ("y", "returns", 50, "After.")]


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    with opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC', 'd.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.executemany("INSERT INTO units VALUES (?, 'DOC', 'key', 'prose', ?, '[\"Policy\"]', ?, ?, ?, 'en', 1)",
                       [(u, json.dumps([label]), text, start, start + 5) for u, label, start, text in UNITS])


def hit(unit, rank):
    return {"unit_id": unit, "rank": rank, "score": 1.0 / rank}


def test_a_hit_is_widened_to_its_neighbours_in_document_order(manifest):
    (only,) = excerpts([hit("b", 1)])
    assert only["units"] == ["a", "b", "c"] and only["text"] == "First.\n\nSecond.\n\nThird."
    assert (only["doc_id"], only["labels"], only["path"], only["rank"]) == ("DOC", ["returns"], ["Policy"], 1)


def test_a_neighbour_across_the_segment_boundary_is_never_taken(manifest):
    # "x" sits between "d" and "y" in the file, under legal alone: a returns hit may not pull it in.
    (around_d,) = excerpts([hit("d", 1)])
    assert around_d["units"] == ["c", "d", "y"] and "Legal only." not in around_d["text"]
    (legal,) = excerpts([hit("x", 1)])
    assert legal["units"] == ["x"]


def test_overlapping_hits_merge_inside_a_segment_and_nowhere_else(manifest):
    found = excerpts([hit("b", 1), hit("c", 2), hit("x", 3)])
    assert [(e["units"], e["rank"]) for e in found] == [(["a", "b", "c", "d"], 1), (["x"], 3)]


def test_only_the_top_hits_are_widened_and_a_hit_the_manifest_lost_is_left_out(manifest):
    found = excerpts([hit("gone", 1), hit("a", 2), hit("y", 3)], top=2)
    assert [e["units"] for e in found] == [["a", "b"]]


def test_a_hits_segment_is_found_through_its_own_index_not_its_whole_document(manifest):
    # PF-F5: the segment was read through units_by_doc, every unit of the document, 31 to 43 ms for the top 8.
    from needtoknow.index.expand import SEGMENT
    with opened() as db:
        plan = " ".join(row[3] for row in db.execute("EXPLAIN QUERY PLAN " + SEGMENT, ("DOC", "key", "[]")))
    assert "USING INDEX units_by_segment" in plan
