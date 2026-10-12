"""P2's third exit criterion, end to end: a section planted with an audience no label knows comes out of the whole
ingestion into the manifest unlabelled, and of every role in the policy only the administrator may read it.

The other two criteria are held where they are built: two runs over an unchanged corpus give the same ids and
counts (tests/test_manifest.py), and every kind of change gives exactly its unit diff (tests/test_scan.py)."""

import json

from needtoknow.corpus import assemble
from needtoknow.corpus.plan import header
from needtoknow.ingest.manifest import build, opened
from needtoknow.policy import UNLABELLED, load_policy, readable

PROSE = "The board reviews the annual figures and decides on the dividend before the general meeting. " * 3


def test_a_planted_unlabelled_section_is_readable_by_the_administrator_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    doc = {"id": "FIN_POL_001", "type": "POL", "type_name": "Policy", "owner": "finance", "language": "en", "format": "md",
           "target_mb": 0.01, "version": "1.0",
           "sections": [{"label": "finance", "topic_en": "Open", "topic_el": "Ανοιχτό", "facts": [], "canary": None},
                        {"label": "finance", "topic_en": "Board only", "topic_el": "Μόνο", "facts": [], "canary": None}]}
    heads = [header(doc, n, s) for n, s in enumerate(doc["sections"], 1)]
    heads[1][1] = "Audience: Board of Directors"  # a name no label knows: planted unlabelled
    text = "\n".join(heads[0] + ["", PROSE.replace("board", "team"), ""] + heads[1] + ["", PROSE, ""])
    (base / "narrative" / "FIN_POL_001.txt").write_bytes(text.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps([doc]).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    assemble.assemble()
    summary = build()
    assert summary["unknown_audiences"] == ["Board of Directors"]
    with opened() as db:
        units = [(json.loads(labels), text) for labels, text in db.execute("SELECT labels, text FROM units")]
    planted = [labels for labels, text in units if "board reviews" in text]
    assert planted and all(labels == [UNLABELLED] for labels in planted)
    policy = load_policy()
    readers = {role for role in policy["roles"] for labels in planted if readable(policy, role) & set(labels)}
    assert readers == {"admin"}
    open_units = [labels for labels, text in units if "team reviews" in text]
    assert open_units and all(labels == ["finance"] for labels in open_units)
