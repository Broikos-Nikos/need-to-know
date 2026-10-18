"""The load stage (T3.4): the collection brought to the manifest, by comparing two states.

A sync reads every unit the manifest holds and every point id the collection holds. It
deletes the points no unit explains, then embeds what the cache lacks, then upserts the
units no point holds. Deletes go first: a relabel is a delete and an upsert, since a
point's id comes from its unit id and that holds the labels, and deleting first means
the old points, readable by the old labels, never stand beside the new ones; for a
moment the section is missing instead, which fails closed.

Vectors come from the cache by passage (DR-F10): a relabel that keeps its passage costs
no embedding, and one whose safe path lost a heading costs one. Sparse vectors are
written at the frozen average length (KB1-F15), and every point's terms are counted
into the frequency table per label set as it goes in, and out, read from the point
itself, before it is deleted (N50). Being a comparison of states rather
than a replay of a log, a sync killed at any point is finished by the next one, and
two in a row change nothing. One sync writes at a time, under an operating system
lock, and every sync ends by checking the frequency table's passages per label set
against the index's own counts, recounting from the index when they differ: a sync
killed between the two stores leaves the table wrong otherwise, and no later sync,
finding index and manifest agreed, would notice (OP-F5). Given the documents a scan changed, it compares those
alone, through the payload index on `doc_id`, which is what the watcher asks for: a
whole comparison reads every unit and every point id. Its seconds count from the start,
and `ntk index sync` prints them by part: the reading, the deletes, the embedding, the
upserts and the closing check (MA-F8).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from needtoknow import lexical
from needtoknow.corpus import exclusive
from needtoknow.embed import cache
from needtoknow.index import frequencies
from needtoknow.settings import data_dir
from needtoknow.index.collection import COLLECTION, DENSE, SPARSE, label_set, payload, point_id
from needtoknow.ingest.units import passage

BATCH = 256


class SyncError(RuntimeError):
    """A sync that would delete most of the collection, which one asked for only when it is meant."""


def held(client, doc_ids: list[str] | None = None) -> set[str]:
    """Every point id in the collection, or in the given documents."""
    from qdrant_client import models

    only = None if doc_ids is None else models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=sorted(doc_ids) or ["\x00"]))])
    ids: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(COLLECTION, scroll_filter=only, limit=10_000, offset=offset, with_payload=False,
                                        with_vectors=False)
        ids.update(str(record.id) for record in records)
        if offset is None:
            return ids


def sync(client, manifest: Path | None = None, report=None, doc_ids: list[str] | None = None,
         force: bool = False) -> dict:
    """Delete the points the manifest no longer explains, then upsert the units the collection lacks; in every
    document, or in the given ones. One sync at a time. A sync that would delete more than half of the collection
    refuses unless `force`: the server and its key are the same whatever NTK_DATA_DIR names, so a sync from another
    data folder, a corpus draw's, would empty the served index (MT-F18)."""
    folder = data_dir() / "index"
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder):
        return _sync(client, manifest, report, doc_ids, force)


def _sync(client, manifest: Path | None, report, doc_ids: list[str] | None, force: bool = False) -> dict:
    from qdrant_client import models

    from needtoknow.ingest.manifest import opened as manifest_opened

    began = mark = time.perf_counter()
    parts: dict[str, float] = {}  # seconds by part, from the start: a quiet sync is mostly its reading (MA-F8)

    def lap(name: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        parts[name] = now - mark
        mark = now

    stats = lexical.frozen()
    with manifest_opened(manifest) as db:
        query = 'SELECT unit_id, doc_id, section_key, layer, labels, path, text, start, "end", language FROM units'
        if doc_ids is None:
            rows = db.execute(query).fetchall()
        else:
            rows = [row for doc_id in sorted(set(doc_ids)) for row in db.execute(query + " WHERE doc_id = ?", (doc_id,))]
    wanted = {point_id(row[0]): row for row in rows}
    present = held(client, None if doc_ids is None else sorted(set(doc_ids)))
    stale = sorted(present - set(wanted))
    todo = [wanted[pid] for pid in sorted(set(wanted) - present)]
    # Measured against the whole collection, so the watcher's sync of the documents it changed never trips it.
    total = len(present) if doc_ids is None else client.count(COLLECTION, exact=True).count
    if stale and 2 * len(stale) > total and not force:
        raise SyncError(f"this sync would delete {len(stale):,} of the collection's {total:,} points, more than half, "
                        f"for the manifest in {data_dir()}: a data folder other than the one the index was built "
                        "from, a corpus draw's say, reaches the same server. ntk index sync --force deletes them "
                        "when that is meant (MT-F18)")
    summary = {"units": len(wanted), "points_before": len(present), "deleted": len(stale), "upserted": 0,
               "embedded": 0, "seconds": 0.0, "parts": parts}
    lap("read")
    with frequencies.opened(COLLECTION) as counts, cache.opened() as vectors:
        for start in range(0, len(stale), BATCH * 4):
            part = stale[start:start + BATCH * 4]
            gone = client.retrieve(COLLECTION, ids=part, with_payload=["label_set"], with_vectors=[SPARSE])
            client.delete(COLLECTION, points_selector=models.PointIdsList(points=part), wait=True)
            frequencies.count(counts, [(r.payload["label_set"], r.vector[SPARSE].indices) for r in gone], -1)
            counts.commit()
        lap("deletes")
        texts = [passage(json.loads(row[5]), row[6]) for row in todo]
        keys = [cache.passage_key(text) for text in texts]
        absent = cache.missing(vectors, keys)
        if absent:  # passages new to the cache: a changed text, or a path a relabel cut
            first = {}
            for key, row, text in zip(keys, todo, texts):
                if key in absent:
                    first.setdefault(key, (row[0], text))
            summary["embedded"] = cache.embed(vectors, list(first.items()))["embedded"]
        lap("embedding")
        for start in range(0, len(todo), BATCH):
            part, part_texts = todo[start:start + BATCH], texts[start:start + BATCH]
            dense = cache.read(vectors, keys[start:start + BATCH])
            sparse = [lexical.document(text, stats["avgdl"]) for text in part_texts]
            client.upsert(COLLECTION, wait=True, points=models.Batch(
                ids=[point_id(row[0]) for row in part],
                vectors={DENSE: dense.tolist(),
                         SPARSE: [models.SparseVector(indices=i, values=v) for i, v in sparse]},
                payloads=[payload(row[0], row[1], row[2], row[3], json.loads(row[4]), json.loads(row[5]), row[9])
                          for row in part]))
            frequencies.count(counts, [(label_set(json.loads(row[4])), indices) for row, (indices, _) in zip(part, sparse)], 1)
            counts.commit()
            summary["upserted"] += len(part)
            if report and (summary["upserted"] % (BATCH * 200) == 0 or summary["upserted"] == len(todo)):
                report(f"upserted {summary['upserted']} of {len(todo)} points, {time.perf_counter() - began:.0f} s")
        lap("upserts")
    with frequencies.opened(COLLECTION) as counts:
        summary["drift"] = frequencies.drift(counts, client, COLLECTION)
        summary["recounted"] = frequencies.recount(counts, client, COLLECTION) if summary["drift"] else 0
    summary["points_after"] = client.count(COLLECTION, exact=True).count
    lap("check")
    summary["seconds"] = time.perf_counter() - began
    if doc_ids is not None:
        summary["documents"] = len(set(doc_ids))
    return summary
