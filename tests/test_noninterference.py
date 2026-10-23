"""KB4-K9: a role's results are a function of the text it may read and of one average length frozen at build (N50).

A small profile: the committed narrative of every eighth catalogue document, cut into units as `ntk ingest` cuts a
file, every 25th unit under a second label too, synced in Qdrant's local mode with the stand in encoder of
test_embed_cache. Each role's second build freezes the same lexical statistics and holds every unit the role may
read byte for byte, and changes only units it may not read: once writing the questions into all of them, so the
counts a weight over the whole collection would read move as far as they can, and once dropping every other one. The
questions are every fifteenth dev probe's, in both languages, and the attack's, a word the role sees one to three
times that text it may not read holds often beside one that such text never holds (`bench/idf_attack.py`). The
weights the search sends, the role's fused ids, order and scores, its order under a stand in reranker, each search's
own candidates and scores as the served trace shows them, and the count of units the role's filter passes must be the
same in all three builds. The count of units the filter holds back is the one figure of the trace that moves, by
design (N64).

Local mode scores every point, those the filter leaves out too, and orders them with numpy's quicksort, so units of
exactly equal score come out in an order the hidden points' scores move; the server orders them by its index's layout,
which units a role may not read help shape (`bench/ties.py`, AC-F12). The search takes neither order: it fetches a
group of equal scores across the cut whole and orders it by unit id (AC-F20), so the test leaves the engine's sort as
it is, and holds tied units as it holds the rest. Until then it sorted stably in Qdrant's local collection module
(MT-F15), and held only units whose score no other shared.
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import replace

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

pytestmark = [pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant"),
              pytest.mark.filterwarnings("ignore:Local mode performs exact")]  # the trace asks its searches apart


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
    """What the role gets for each question in the build at `root`: the sparse weights sent, the fused hits, the
    reranked order, and from the served trace each search's own hits and the count of units the filter passes."""
    monkeypatch.setenv("NTK_DATA_DIR", str(root))
    sent = []
    real = client.query_batch_points

    def spy(*args, **kwargs):
        sent.append([(r.query.indices, r.query.values) for r in kwargs["requests"] if r.using == SPARSE])
        return real(*args, **kwargs)

    monkeypatch.setattr(client, "query_batch_points", spy)
    out = []
    for question in questions:
        trace = {}
        asked = len(sent)  # the first call asks both searches; a widened window asks one again
        fused = search(client, question, role, rerank="none", trace=trace)
        passages = passages_of([h["unit_id"] for h in fused])
        order = rerank.rerank(question, fused, [passages[h["unit_id"]] for h in fused], "cross") if fused else []
        out.append({"weights": sent[asked], "fused": [(h["unit_id"], h["rank"], h["score"]) for h in fused],
                    "reranked": [h["unit_id"] for h in order], "passes": trace["filter"]["passes"],
                    **{side: [(h["unit_id"], h["rank"], h["score"]) for h in trace[side]] for side in ("dense", "sparse")}})
    monkeypatch.setattr(client, "query_batch_points", real)
    return out


# A score's last bits: local mode scores in float64 over a matrix the hidden units size, and how a machine's BLAS blocks
# that sum moves the last bits with them. Rounded to float32, a score on a rounding's edge fell on one side here and on
# the other on the Linux runner, one float32 step at 0.0198 (the first deploy, bc89a04). The server scores in float32.
SCORE_REL = 1e-6


def differing(x: dict, y: dict) -> list[str]:
    """The parts of two answers that differ: every part exactly, each search's ids and ranks exactly, and its scores
    to within their last bits."""
    parts = [part for part in x if part not in ("dense", "sparse") and x[part] != y[part]]
    for side in ("dense", "sparse"):
        a, b = x[side], y[side]
        if [hit[:2] for hit in a] != [hit[:2] for hit in b] or not all(
                math.isclose(s, t, rel_tol=SCORE_REL, abs_tol=1e-12) for (_, _, s), (_, _, t) in zip(a, b)):
            parts.append(side)
    return parts


def test_the_comparison_takes_a_scores_last_bits_and_no_more():
    # The first deploy's failure, the runner's dense score one float32 step from this machine's, is the same answer;
    # a score a hidden unit moved in its fourth digit is not, nor is a rank.
    x = {"weights": [], "passes": 3, "dense": [("u", 1, -0.01975477859377861)], "sparse": [("v", 1, 0.0763641744852066)]}
    runner = {**x, "dense": [("u", 1, -0.01975478045642376)]}
    moved = {**x, "sparse": [("v", 1, 0.0763241744852066)]}
    swapped = {**x, "dense": [("u", 2, -0.01975477859377861)]}
    assert differing(x, runner) == [] and differing(x, moved) == ["sparse"] and differing(x, swapped) == ["dense"]


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
                assert not differing(x, y), (role, change, question, differing(x, y))
        checked.append(role)
    assert len(checked) == len(policy["roles"]) - 1, checked
