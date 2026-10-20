"""What the build holds (PLAN section 1, number 6, bar the build's time, which `bench/full_build.py` measures): the
documents, their bytes and units from the manifest, and on disk the index's own storage, the manifest and the vector
cache; and the collection's vectors, counted by the index's server. `ntk stats` prints them, and the local server's
`GET /stats` answers with them (T6.1h)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from needtoknow.index.collection import COLLECTION


class NoBuild(RuntimeError):
    """No manifest: nothing has been built in this data folder."""


def folder_bytes(path: Path) -> int:
    """The bytes of every file under a folder, none when it is not there."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.is_dir() else 0


def footprint() -> dict:
    """The manifest's documents, bytes and units by layer, and the bytes on disk of the index's own storage, the
    manifest and the vector cache; NoBuild when there is no manifest."""
    from needtoknow.embed.cache import cache_path
    from needtoknow.embed.model import load_config
    from needtoknow.ingest.manifest import manifest_path
    from needtoknow.settings import data_dir

    path = manifest_path()
    if not path.is_file():
        raise NoBuild(f"no manifest at {path}: ntk ingest build makes it")
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        documents, size = db.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM files").fetchone()
        layers = dict(db.execute("SELECT layer, COUNT(*) FROM units GROUP BY layer").fetchall())
    finally:
        db.close()
    cache = cache_path(load_config())
    return {"documents": documents, "bytes": size, "units": sum(layers.values()), "layers": dict(sorted(layers.items())),
            "collection": COLLECTION,
            "disk": {"index": folder_bytes(data_dir() / "qdrant" / "storage" / "collections" / COLLECTION),
                     "manifest": path.stat().st_size, "vector_cache": cache.stat().st_size if cache.is_file() else 0}}


def vectors(client) -> int:
    """The collection's points, counted exactly by the index's server."""
    return client.count(COLLECTION, exact=True).count
