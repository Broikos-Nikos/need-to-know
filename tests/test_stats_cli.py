"""`ntk stats` (PLAN section 1, number 6): the documents, their bytes and units from the manifest, the vectors from the
server, and on disk the index's own storage, the manifest and the vector cache."""

import types

from needtoknow.cli import cmd_stats, main
from needtoknow.embed.cache import cache_path
from needtoknow.embed.model import load_config
from needtoknow.index.collection import COLLECTION
from needtoknow.ingest.manifest import manifest_path, opened
from needtoknow.settings import data_dir


class Counter:
    """Stands in for the server: the collection's points, counted exactly."""

    def count(self, collection, exact=False):
        assert (collection, exact) == (COLLECTION, True)
        return types.SimpleNamespace(count=3)


def test_stats_reads_the_manifest_the_server_and_the_index_own_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    with opened() as db:
        db.executemany("INSERT INTO files VALUES (?, ?, ?, 1, 1, 1, 'h', 'txt', 'utf-8', 'en', 'v', 1)",
                       [("D1", "D1.txt", 1000), ("D2", "D2.txt", 2500)])
        db.executemany("""INSERT INTO units VALUES (?, ?, 'k', ?, '["general"]', '[]', 't', 0, 1, 'en', 1)""",
                       [("u1", "D1", "prose"), ("u2", "D1", "table"), ("u3", "D2", "table")])
    collections = data_dir() / "qdrant" / "storage" / "collections"
    (collections / COLLECTION / "segments").mkdir(parents=True)
    (collections / COLLECTION / "segments" / "a.bin").write_bytes(b"x" * 700)
    (collections / COLLECTION / "config.json").write_bytes(b"y" * 300)
    (collections / "another").mkdir()  # another collection's storage is not this index's size
    (collections / "another" / "b.bin").write_bytes(b"w" * 4000)
    cache = cache_path(load_config())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"z" * 50)
    assert cmd_stats(client=Counter()) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"2 documents, 3,500 bytes; 3 units (prose 1, table 2); 3 vectors in {COLLECTION}; "), line
    assert line.endswith(f"on disk, the index 1,000 bytes, the manifest {manifest_path().stat().st_size:,}, the vector "
                         "cache 50"), line


def test_stats_without_a_manifest_says_how_to_make_one(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    assert main(["stats"]) == 1
    assert "no manifest at" in capsys.readouterr().out
