"""The permission suite's check in the build (T5.6e1, P5's exit): every role and three users the policy does not name
ask over a small profile, built in Qdrant's local mode with the stand in encoder and reranker of
test_noninterference, and no unit the case may not read comes back from the dense search, the sparse search, their
fusion or the search as `ask` makes it, nor reaches the excerpts, and no canary of a label it may not read reaches the
prompt they make. `bench/permission_suite.py` runs the same check over the whole index with the real models; this one
runs wherever the suite runs, with no GPU and no server.

The profile is the committed narrative of every eighth catalogue document, a planted canary in some of its sections;
the questions are every fifteenth dev probe's and, for each case, the attack's, words its view sees rarely beside
words only hidden text holds. Before anything is counted, the test shows it can fail: the reader of everything gets
units the narrowest case may not read for the same questions."""

import copy

import pytest
from test_embed_cache import Gpu, fake_tokenize
from test_noninterference import attack, built, cut, overlap

from needtoknow import canaries
from needtoknow.answer.prompt import build, reader_mark
from needtoknow.embed import model
from needtoknow.eval.probes import load as load_probes
from needtoknow.index import rerank
from needtoknow.index.expand import excerpts
from needtoknow.index.search import search
from needtoknow.policy import load_policy, readable

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
SIDES = {"dense": ("dense",), "sparse": ("sparse",), "fused": ("dense", "sparse")}


def cases(policy: dict) -> list[tuple[str, str, dict]]:
    """Every role as the policy defines it, a user holding sales and hr, a user with general alone, and sales with a
    membership removed in memory, as bench/permission_suite.py asks them."""
    views = copy.deepcopy(policy)
    views["roles"]["sales and hr"] = {"reads": sorted(readable(policy, "sales") | readable(policy, "hr"))}
    views["roles"]["general alone"] = {"reads": ["general"]}
    removed = copy.deepcopy(policy)
    removed["roles"]["sales"]["reads"] = [r for r in removed["roles"]["sales"]["reads"] if r != "returns"]
    return [(role, role, views) for role in policy["roles"]] + [
        ("sales and hr", "sales and hr", views), ("general alone", "general alone", views),
        ("sales, returns removed", "sales", removed)]


def test_no_case_gets_a_unit_or_a_canary_it_may_not_read(monkeypatch, tmp_path):
    monkeypatch.setattr(model, "tokenize", fake_tokenize)
    monkeypatch.setattr(model, "states", Gpu().states)
    config = model.load_config()  # the stand in counts a token a character, so a paragraph passes 512 of them
    monkeypatch.setattr(model, "load_config", lambda: {**config, "max_length": 1 << 20})
    monkeypatch.setattr(rerank, "cross_scores", overlap)
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path / "plan"))
    units = cut()
    labels_of = {u.unit_id: frozenset(u.labels) for u in units}
    client = built(monkeypatch, tmp_path / "a", units, units)
    policy = load_policy()
    names = canaries.suffixes()
    probes = [probe["question"] for probe in load_probes("dev")[::15]]
    narrow = readable(cases(policy)[-2][2], "general alone")

    def foreign(hits, reads):
        return [h["unit_id"] for h in hits if not labels_of[h["unit_id"]] & reads]

    # The questions reach text the narrowest case may not read: without its filter they bring it back.
    reached = sum(len(foreign(search(client, q, "admin", policy, limit=100, sides=SIDES["fused"], rerank="none"),
                              narrow)) for q in probes)
    assert reached > 0
    planted = sum(len(canaries.PATTERN.findall(u.text)) for u in units)
    assert planted > 0, "the profile holds no canary, so the prompt check could not fail"

    leaks, searches = [], 0
    for name, role, case_policy in cases(policy):
        reads = readable(case_policy, role)
        for question in probes + attack(units, reads, pairs=6):
            for side, sides in SIDES.items():
                hits = search(client, question, role, case_policy, limit=100, sides=sides, rerank="none")
                leaks += [(name, side, question, unit) for unit in foreign(hits, reads)]
                searches += 1
            asked = search(client, question, role, case_policy)  # as ask searches, reranked
            shown = excerpts(asked)
            leaks += [(name, "ask", question, unit) for unit in foreign(asked, reads)]
            leaks += [(name, "excerpts", question, unit) for e in shown for unit in e["units"]
                      if not labels_of[unit] & reads]
            leaks += [(name, "prompt", question, c) for c in canaries.found(
                build(question, shown, reader=reader_mark(reads))[1], reads, names)]
            searches += 1
    # Every case asked at least every probe each of the four ways, and nothing it may not read anywhere.
    assert searches >= 4 * len(cases(policy)) * len(probes) and not leaks, leaks[:5]
