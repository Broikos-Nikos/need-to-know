"""Expansion (T3.7): the top units widened to their neighbours, inside their own segment.

A unit is a paragraph, a piece of a long one, or a group of table rows, and an answer
often needs the sentence before it or the rows after. Each of the top hits is widened to
`window` units on either side, from the manifest, in document order, among the units
that share its document, its section and its labels: its segment (architecture section
9, step 5). A neighbour across the segment's boundary may be readable by other roles than
the hit, so it is never taken, whatever it says. Two widened hits that share a unit
become one excerpt, and only inside one segment, so no excerpt mixes audiences. A hit
the manifest no longer holds, a point awaiting its sync, is left out.
"""

from __future__ import annotations

import json

from needtoknow.ingest.manifest import opened

TOP = 8
WINDOW = 1
# A segment's units in document order, through the manifest's units_by_segment index (PF-F5).
SEGMENT = 'SELECT unit_id FROM units WHERE doc_id = ? AND section_key = ? AND labels = ? ORDER BY start, unit_id'


def excerpts(hits: list[dict], top: int = TOP, window: int = WINDOW) -> list[dict]:
    """The top hits as excerpts, best first: document, section, labels, heading path, units in order, text, rank."""
    found: list[dict] = []
    with opened() as db:
        for hit in hits[:top]:
            row = db.execute("SELECT doc_id, section_key, labels, path FROM units WHERE unit_id = ?",
                             (hit["unit_id"],)).fetchone()
            if row is None:
                continue
            doc_id, key, labels, path = row
            segment = [u for (u,) in db.execute(SEGMENT, (doc_id, key, labels))]
            at = segment.index(hit["unit_id"])
            span = segment[max(0, at - window):at + window + 1]
            same = next((e for e in found if (e["doc_id"], e["section_key"], e["labels"]) == (doc_id, key, json.loads(labels))
                         and set(e["units"]) & set(span)), None)
            if same is None:
                found.append({"doc_id": doc_id, "section_key": key, "labels": json.loads(labels), "path": json.loads(path),
                              "units": span, "rank": hit["rank"], "order": segment})
            else:
                same["units"] = [u for u in segment if u in set(same["units"]) | set(span)]
        for excerpt in found:
            texts = dict(db.execute(f"SELECT unit_id, text FROM units WHERE unit_id IN ({','.join('?' * len(excerpt['units']))})",
                                    excerpt["units"]))
            excerpt["text"] = "\n\n".join(texts[u] for u in excerpt.pop("order") if u in texts and u in excerpt["units"])
    return sorted(found, key=lambda e: e["rank"])
