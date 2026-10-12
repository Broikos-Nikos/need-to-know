"""The six committed samples: chosen to show every format and language, and readable as their sidecars say."""

import json
import random

from needtoknow.corpus.assemble import sample_ids
from needtoknow.corpus.mutate import corpus_files, read
from needtoknow.corpus.plan import SEED, make_catalogue, make_facts, world
from needtoknow.settings import repo_root


def test_six_samples_cover_each_format_and_language_and_the_legacy_encoding():
    rng = random.Random(SEED)
    docs = {d["id"]: d for d in make_catalogue(rng, make_facts(rng, world()))}
    ids = sample_ids(list(docs.values()))
    assert len(ids) == len(set(ids)) == 6
    shapes = {(docs[i]["language"], docs[i]["format"]) for i in ids}
    assert shapes == {("en", "txt"), ("el", "txt"), ("en", "md"), ("el", "md"), ("en", "html"), ("el", "html")}
    assert len({docs[i]["owner"] for i in ids}) >= 5


def test_every_committed_sample_reads_as_its_sidecar_declares_and_names_its_audiences():
    found = corpus_files(repo_root() / "samples")
    assert len(found) == 6
    encodings = set()
    for doc_id, (path, _, meta) in found.items():
        text = read(path, meta)
        encodings.add(meta["encoding"])
        assert ("Audience: " in text) or ("Κοινό: " in text), doc_id
        assert path.stat().st_size < 64 * 1024, doc_id
    assert encodings == {"utf-8", "windows-1253"}


def test_the_samples_carry_no_sidecar_without_its_file():
    folder = repo_root() / "samples"
    sidecars = sorted(folder.glob("*.meta.json"))
    for sidecar in sidecars:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert (folder / f"{meta['doc_id']}.{meta['format']}").is_file(), sidecar.name
    assert len(sidecars) == 6
