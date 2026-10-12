"""The vector cache (T3.1) with the model replaced by a stand in, so no GPU is needed: what a
passage is, what a run embeds, what a killed run keeps, and what is refused."""

import json

import numpy as np
import pytest

from needtoknow.embed import cache, model
from needtoknow.ingest.manifest import opened as manifest_opened
from needtoknow.ingest.units import passage


def fake_tokenize(texts, cut=False):
    return [[0] + [ord(c) % 250 + 3 for c in text] + [2] for text in texts]


def fake_state(ids):
    return np.random.default_rng(sum(i * (n + 1) for n, i in enumerate(ids))).normal(size=1024).astype(np.float16)


class Gpu:
    """Stands in for the model on the card and counts what reaches it."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[int] = []
        self.fail_on_call = fail_on_call

    def states(self, ids, budget=None):
        self.calls.append(len(ids))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("killed")
        return np.vstack([fake_state(x) for x in ids])


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    stand_in = Gpu()
    monkeypatch.setattr(model, "tokenize", fake_tokenize)
    monkeypatch.setattr(model, "states", stand_in.states)
    return stand_in


def manifest(rows):
    """A manifest holding (unit_id, labels, path, text) rows, all in one document."""
    with manifest_opened() as db:
        db.execute("INSERT INTO files VALUES ('DOC_1', 'doc.md', 1, 1, 1, 1, 'h', 'md', 'utf-8', 'en', 'v', 1)")
        db.executemany("INSERT INTO units VALUES (?, 'DOC_1', 's', 'prose', ?, ?, ?, 0, 1, 'en', 1)",
                       [(unit, json.dumps(labels), json.dumps(path), text) for unit, labels, path, text in rows])


def stored():
    with cache.opened() as db:
        return {key for (key,) in db.execute("SELECT key FROM vectors")}


def test_a_passage_is_the_heading_path_on_a_line_above_the_text():
    assert passage(["Procedure: Sales", "SECTION 5  Annex A"], "Date | Shop") == \
        "Procedure: Sales > SECTION 5  Annex A\nDate | Shop"
    assert passage([], "Date | Shop") == "Date | Shop"


def test_a_run_embeds_each_missing_passage_once_and_the_next_run_nothing(gpu):
    manifest([("u1", ["finance"], ["Policy", "Returns"], "Text A"),
              ("u2", ["legal"], ["Policy", "Returns"], "Text A"),     # a relabel's twin: the same passage
              ("u3", ["finance"], ["Policy", "Fees"], "Text A"),      # another heading: another passage
              ("u4", ["finance"], [], "Text B")])
    first = cache.fill()
    assert (first["units"], first["missing"], first["embedded"], gpu.calls) == (4, 3, 3, [3])
    second = cache.fill()
    assert (second["cached"], second["missing"], second["embedded"], gpu.calls) == (3, 0, 0, [3])


def test_read_gives_back_exactly_the_normalised_state_that_was_stored(gpu):
    manifest([("u1", ["finance"], ["Policy"], "Text A"), ("u2", ["finance"], ["Policy"], "Text B")])
    cache.fill()
    texts = [passage(["Policy"], "Text B"), passage(["Policy"], "Text A")]
    with cache.opened() as db:
        rows = cache.read(db, [cache.passage_key(text) for text in texts])
    expected = model.normalise(np.vstack([fake_state(ids) for ids in fake_tokenize(texts)]))
    assert rows.dtype == np.float32 and np.array_equal(rows, expected)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-6)


def test_a_killed_run_keeps_its_finished_shards_and_the_next_run_does_only_the_rest(gpu, monkeypatch):
    manifest([(f"u{n}", ["finance"], ["Policy"], f"Text {n}") for n in range(5)])
    gpu.fail_on_call = 2
    with pytest.raises(RuntimeError, match="killed"):
        cache.fill(shard=2)
    assert len(stored()) == 2
    rerun = Gpu()
    monkeypatch.setattr(model, "states", rerun.states)
    summary = cache.fill(shard=2)
    assert (summary["cached"], summary["embedded"], rerun.calls) == (2, 3, [2, 1])
    assert len(stored()) == 5


def test_a_passage_over_max_length_is_refused_by_its_unit_before_the_gpu_sees_its_shard(gpu, monkeypatch):
    config = dict(model.load_config(), max_length=20)
    monkeypatch.setattr(model, "load_config", lambda path=None: config)
    manifest([("u1", ["finance"], [], "short"), ("u2", ["finance"], [], "a passage far longer than twenty tokens")])
    with pytest.raises(model.EmbedError, match="the first unit u2 at 41: refused rather than cut"):
        cache.fill()
    assert gpu.calls == [] and stored() == set()


def test_a_cache_file_made_for_another_model_is_refused(gpu):
    config = model.load_config()
    with cache.opened(config):
        pass
    other = dict(config, revision="0" * 40)
    with pytest.raises(cache.CacheError, match="not BAAI/bge-m3@0000"):
        with cache.opened(other, path=cache.cache_path(config)):
            pass


def test_a_stored_state_of_the_wrong_size_is_refused(gpu):
    with cache.opened() as db:
        db.execute("INSERT INTO vectors (key, state) VALUES (?, ?)", (b"k" * 16, b"\x00" * 100))
        with pytest.raises(cache.CacheError, match="1 stored states are not 2048 bytes"):
            cache.read(db, [b"k" * 16])
        with pytest.raises(cache.CacheError, match="1 passages are not in the cache"):
            cache.read(db, [b"x" * 16])


def test_ntk_embed_prints_its_run_and_exits_1_on_a_refusal(gpu, monkeypatch, capsys):
    from needtoknow.cli import main
    manifest([("u1", ["finance"], ["Policy"], "Text A"), ("u2", ["finance"], [], "x" * 600)])
    config = dict(model.load_config(), max_length=100)
    monkeypatch.setattr(model, "load_config", lambda path=None: config)
    assert main(["embed"]) == 1
    assert "embed: 1 passages are longer than max_length 100 tokens, the first unit u2 at 602" in capsys.readouterr().out
    config["max_length"] = 1000
    assert main(["embed"]) == 0
    assert "units 2; passages cached before 0, missing 2, embedded 2:" in capsys.readouterr().out
