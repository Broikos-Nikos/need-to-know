"""KB4-K9: a role's results are a function of the text it may read and of one average length frozen at build (N50).

A small profile: the committed narrative of every eighth catalogue document, cut into units as `ntk ingest` cuts a
file, every 25th unit under a second label too, synced in Qdrant's local mode with the stand in encoder of
test_embed_cache. Each role's second build freezes the same lexical statistics and holds every unit the role may
read byte for byte, and changes only units it may not read: once writing the questions into all of them, so the
counts a weight over the whole collection would read move as far as they can, and once dropping every other one. The
questions are every fifteenth dev probe's, in both languages, and the attack's, a word the role sees one to three
times that text it may not read holds often beside one that such text never holds (`bench/idf_attack.py`). The
weights the search sends, the role's fused ids, order and scores, and its order under a stand in reranker must be
the same in all three builds.

Local mode scores every point, those the filter leaves out too, and orders them with numpy's quicksort, so units of
exactly equal score come out in an order the hidden points' scores move. The server's filtered exact search scores
only the points the filter passes, in the order they sit in storage; the test sorts stably to stand for that.
"""

import json
import random
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest
from qdrant_client import QdrantClient
from test_embed_cache import Gpu, fake_tokenize

from needtoknow import lexical
from needtoknow.corpus.plan import build
from needtoknow.embed import model
from needtoknow.eval.probes import load as load_probes
from needtoknow.index import frequencies, rerank
from needtoknow.index import sync as loader
from needtoknow.index.collection import COLLECTION, SPARSE, ensure
from needtoknow.index.search import passages_of, search
from needtoknow.ingest.manifest import opened, replace_units
from needtoknow.ingest.pipeline import document_units
from needtoknow.ingest.units import passage
from needtoknow.policy import load_policy, readable
from needtoknow.settings import data_dir, repo_root

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")


def cut() -> list:
    """Every eighth catalogue document's committed narrative as units, as its sidecar would describe it."""
    build()
    docs = json.loads((data_dir() / "corpus" / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    units = []
    for doc in docs[::8]:
        meta = {"doc_id": doc["id"], "format": "txt", "encoding": "utf-8", "language": doc["language"],
                "default_audience": doc["owner"]}
        units += document_units(repo_root() / "narrative" / f"{doc['id']}.txt", meta)[1]
    # The narrative's sections name one audience each; label sets of two are counted as well.
    return [replace(u, labels=tuple(sorted({*u.labels, "legal" if "legal" not in u.labels else "sales"})))
            if n % 25 == 0 else u for n, u in enumerate(units)]


def built(monkeypatch, root, units: list, measured: list) -> QdrantClient:
    """A build of these units in its own data folder, its lexical statistics measured over `measured`."""
    monkeypatch.setenv("NTK_DATA_DIR", str(root))
    with opened() as db:
        db.executemany("INSERT INTO files VALUES (?, ?, 1, 1, 1, 1, 'h', 'txt', 'utf-8', ?, 'v', 1)",
                       sorted({(u.doc_id, f"{u.doc_id}.txt", u.language) for u in units}))
        for doc_id in sorted({u.doc_id for u in units}):
            replace_units(db, 1, doc_id, [u for u in units if u.doc_id == doc_id])
    client = QdrantClient(location=":memory:")
    ensure(client, lambda: (passage(u.path, u.text) for u in measured))
    loader.sync(client)
    return client


def attack(units: list, reads: frozenset, pairs: int = 12) -> list[str]:
    """The attack's questions for a role, from this profile's own words (`bench/idf_attack.py`)."""
    seen = defaultdict(lambda: [0, 0])  # term: passages the role reads, passages it may not
    for u in units:
        for term in set(lexical.terms(passage(u.path, u.text))):
            if term.isalpha() and len(term) > 4:
                seen[term][0 if set(u.labels) & reads else 1] += 1
    heavy = sorted(t for t, (v, h) in seen.items() if 1 <= v <= 3 and h >= 5)
    clean = sorted(t for t, (v, h) in seen.items() if 1 <= v <= 3 and h == 0)
    rng = random.Random(60)
    asked = []
    for t in rng.sample(heavy, min(pairs, len(heavy))):
        partners = [u for u in clean if seen[u][0] == seen[t][0]]
        if partners:
            asked.append(f"{t} {rng.choice(partners)}")
    return asked


def overlap(question: str, passages: list[str]) -> list[float]:
    """A stand in for the cross encoder: the question's words a passage holds, its length breaking ties."""
    asked = set(question.lower().split())
    return [len(asked & set(p.lower().split())) + len(p) / 1e7 for p in passages]


def answers(monkeypatch, root, client, role: str, questions: list[str]) -> list[dict]:
    """What the role gets for each question in the build at `root`: the sparse weights sent, the fused hits, and the
    reranked order."""
    monkeypatch.setenv("NTK_DATA_DIR", str(root))
    sent = []
    real = client.query_points

    def spy(*args, **kwargs):
        sent.append([(p.query.indices, p.query.values) for p in kwargs["prefetch"] if p.using == SPARSE])
        return real(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", spy)
    out = []
    for question in questions:
        fused = search(client, question, role, rerank="none")
        order = rerank.rerank(question, fused, passages_of([h["unit_id"] for h in fused]), "cross") if fused else []
        out.append({"weights": sent[-1], "fused": [(h["unit_id"], h["rank"], h["score"]) for h in fused],
                    "reranked": [h["unit_id"] for h in order]})
    monkeypatch.setattr(client, "query_points", real)
    return out


def whole(monkeypatch, root, questions: list[str]) -> list[list[float]]:
    """Each question's weights over every label set in the build at `root`: what a weight over the whole collection
    would read."""
    monkeypatch.setenv("NTK_DATA_DIR", str(root))
    with frequencies.opened(COLLECTION) as db:
        sets = [s for (s,) in db.execute("SELECT label_set FROM sets")]
        return [frequencies.weights(db, lexical.query(q)[0], sets) for q in questions]


def test_a_roles_results_do_not_move_when_only_text_it_may_not_read_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(model, "tokenize", fake_tokenize)
    monkeypatch.setattr(model, "states", Gpu().states)
    config = model.load_config()  # the stand in counts a token a character, so a paragraph passes 512 of them
    monkeypatch.setattr(model, "load_config", lambda: {**config, "max_length": 1 << 20})
    monkeypatch.setattr(rerank, "cross_scores", overlap)
    argsort = np.argsort
    monkeypatch.setattr(np, "argsort", lambda a, axis=-1, kind=None, order=None: argsort(a, axis, "stable", order))
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path / "plan"))
    units = cut()
    probes = [probe["question"] for probe in load_probes("dev")[::15]]
    policy = load_policy()
    first = built(monkeypatch, tmp_path / "a", units, units)
    checked = []
    for role in policy["roles"]:
        reads = readable(policy, role)
        hidden = [u for u in units if not set(u.labels) & reads]
        if not hidden:  # a role that reads everything has no text behind it to change
            continue
        questions = probes + attack(units, reads)
        before = answers(monkeypatch, tmp_path / "a", first, role, questions)
        assert sum(bool(x["fused"]) for x in before) > len(questions) // 2, role
        written = {u.unit_id: replace(u, text=" ".join([questions[n % len(questions)]] * 2))
                   for n, u in enumerate(hidden)}
        dropped = {u.unit_id for u in hidden[::2]}
        for change, second in (("written", [written.get(u.unit_id, u) for u in units]),
                               ("dropped", [u for u in units if u.unit_id not in dropped])):
            root = tmp_path / f"{change}-{role}"
            client = built(monkeypatch, root, second, units)
            # The change is one a weight over the whole collection would read, so this test can fail.
            assert whole(monkeypatch, tmp_path / "a", questions) != whole(monkeypatch, root, questions), (role, change)
            after = answers(monkeypatch, root, client, role, questions)
            for question, x, y in zip(questions, before, after):
                assert x == y, (role, change, question, [part for part in x if x[part] != y[part]])
        checked.append(role)
    assert len(checked) == len(policy["roles"]) - 1, checked
