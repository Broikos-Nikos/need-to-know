"""The fact families' split (T5.1b, protocol version 4, N56): facts that share a section make one family, seats go by
kind and then label, a draw that does not balance gives way to the next seed's and is recorded, and the committed
split keeps every rule."""

import copy
import json
import random

import pytest

from needtoknow.eval import split as S
from needtoknow.eval.protocol import load_protocol
from needtoknow.settings import repo_root


def fact(fid, family, label="sales", kind="single", docs=("D1",)):
    return {"id": fid, "family": family, "label": label, "kind": kind, "placements": list(docs),
            "en": f"Sentence of {fid}.", "el": f"Πρόταση του {fid}."}


def plan(tmp_path, facts, sections):
    """A catalogue whose documents hold the given sections of fact ids, and a narrative holding each sentence."""
    catalogue, narrative = [], tmp_path / "narrative"
    narrative.mkdir()
    for doc_id, (language, size, groups) in sections.items():
        catalogue.append({"id": doc_id, "language": language, "target_mb": size,
                          "sections": [{"label": "sales", "facts": list(ids)} for ids in groups]})
        sentences = {f["id"]: f[language] for f in facts}
        text = "\n\n".join("Opening words of a paragraph. " + " ".join(sentences[i] for i in ids) + " Closing words."
                           for ids in groups)
        (narrative / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    return catalogue, narrative


def test_facts_planted_in_one_section_make_one_family_and_versions_join_their_fact(tmp_path):
    facts = [fact("F01V00", "F01V00"), fact("F01V01", "F01V01"), fact("F02V00", "F02V00", kind="versioned"),
             fact("F02V00OLD", "F02V00", kind="superseded", docs=("D2",)), fact("F03V00", "F03V00", docs=("D2",))]
    catalogue, narrative = plan(tmp_path, facts, {"D1": ("en", 1.0, [["F01V00", "F01V01"], ["F02V00"]]),
                                                  "D2": ("el", 2.0, [["F02V00OLD"], ["F03V00"]])})
    families = [S.describe(m, {d["id"]: d for d in catalogue}, narrative) for m in S.groups(facts, catalogue)]
    assert [f["registry_families"] for f in families] == [["F01V00", "F01V01"], ["F02V00"], ["F03V00"]]
    assert [f["kind"] for f in families] == ["single", "versioned", "single"]
    assert families[1]["language"] == 0.5 and families[1]["document_size"] == 1.5  # one placement in each document
    assert families[0]["placements"] == 1.0  # per fact, so a big family weighs no more than a small one


def test_salience_is_the_planted_sentences_share_of_its_paragraph_and_a_missing_one_is_named():
    assert S.salience("Two words.", "First line here.\n\nThree words, then Two words.\n\nLast.") == pytest.approx(
        len("Two words.") / len("Three words, then Two words."))
    with pytest.raises(S.SplitError, match="not in its document's narrative"):
        S.salience("Absent.", "Something else.")


def test_a_sentence_planted_twice_in_one_document_is_measured_at_each_paragraph():
    # DR-F21: the second placement was measured at the first one's paragraph.
    text = "Short. The fee is 5.\n\nA much longer paragraph that says it again: The fee is 5. And more words after it."
    first, second = S.salience("The fee is 5.", text), S.salience("The fee is 5.", text, 1)
    assert first == pytest.approx(len("The fee is 5.") / len("Short. The fee is 5.")) and second < first
    with pytest.raises(S.SplitError, match="not 3 times in its document's narrative"):
        S.salience("The fee is 5.", text, 2)


def test_a_draw_that_differs_from_the_registered_split_is_not_written_over_it(tmp_path, monkeypatch, capsys):
    # MT-F12: ntk eval split wrote whatever it drew, over the split the items were written against.
    from needtoknow import cli
    registered = {"split": [{"id": "F1", "split": "test"}, {"id": "F2", "split": "dev"}], "draws": [{"smd": {"p": 0.0}}],
                  "families": 2, "registry_families": 2, "test_families": 1, "seed": 1, "limit": 0.1}
    drawn = {**registered, "split": [{"id": "F1", "split": "dev"}, {"id": "F2", "split": "test"}]}
    path = tmp_path / "families.json"
    path.write_text(json.dumps(registered), encoding="utf-8")
    monkeypatch.setattr(S, "split", lambda: drawn)
    monkeypatch.setattr(S, "split_path", lambda: path)
    assert cli.main(["eval", "split"]) == 1
    assert "2 families changing split (F1, F2): not written" in capsys.readouterr().out
    assert json.loads(path.read_text(encoding="utf-8")) == registered
    assert cli.main(["eval", "split", "--rewrite"]) == 0
    assert json.loads(path.read_text(encoding="utf-8")) == drawn


def test_seats_go_to_each_kind_by_its_share_before_any_label():
    families = [{"id": f"S{i}", "kind": "single", "label": f"L{i % 3}"} for i in range(9)] + \
               [{"id": f"V{i}", "kind": "variant", "label": f"L{i}"} for i in range(3)]
    for seed in range(20):
        test = S.draw(families, 4, seed)
        assert len(test) == 4 and sum(fid.startswith("V") for fid in test) == 1, (seed, test)
    assert len({frozenset(S.draw(families, 4, seed)) for seed in range(20)}) > 1  # the seed moves the draw
    # Equal fractions: the seed, not the strata's names, picks which of three single families gets the seat.
    winners = {next(k for k, v in S.quotas({"a": 1, "b": 1, "c": 1}, 1, random.Random(seed)).items() if v)
               for seed in range(30)}
    assert winners == {"a", "b", "c"}


def test_a_draw_that_does_not_balance_gives_way_to_the_next_seeds_and_every_draw_is_recorded(tmp_path, monkeypatch):
    facts = [fact(f"F{i:02}V00", f"F{i:02}V00", label=["sales", "legal"][i % 2], docs=(f"D{i}",)) for i in range(12)]
    catalogue, narrative = plan(tmp_path, facts, {f"D{i}": (["en", "el"][i % 3 == 0], 1.0 + i, [[f"F{i:02}V00"]])
                                                  for i in range(12)})
    protocol = copy.deepcopy(load_protocol())
    protocol["splits"]["test_families"] = 4
    protocol["splits"]["balance"]["max_standardized_mean_difference"] = 0.5
    base, real = protocol["splits"]["seed"], S.draw
    # The registered seed's draw takes the four largest documents, which no limit of 0.5 lets pass.
    monkeypatch.setattr(S, "draw", lambda families, total, seed: {"F08V00", "F09V00", "F10V00", "F11V00"}
                        if seed == base else real(families, total, seed))
    result = S.split(protocol, facts, catalogue, narrative)
    seeds = [d["seed"] for d in result["draws"]]
    assert seeds == list(range(base, base + len(seeds))) and len(seeds) >= 2
    assert not result["draws"][0]["balanced"] and abs(result["draws"][0]["smd"]["document_size"]) > 0.5
    assert result["draws"][-1]["balanced"] and not any(d["balanced"] for d in result["draws"][:-1])
    assert sum(f["split"] == "test" for f in result["split"]) == 4 and result["seed"] == seeds[-1]
    protocol["splits"]["balance"]["max_standardized_mean_difference"] = 1e-9
    with pytest.raises(S.SplitError, match="balanced the split"):
        S.split(protocol, facts, catalogue, narrative, attempts=30)


def test_the_committed_split_is_the_one_the_registered_seed_draws_now(monkeypatch, tmp_path):
    # MA-F34: 54c3732 wrote planted sentences anew and moved their families' salience while eval/families.json kept the
    # draw before, so `ntk eval split --check`, which the README names, exited 1 and no test failed.
    from needtoknow.corpus.plan import build
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    build()
    assert S.split() == json.loads((repo_root() / "eval" / "families.json").read_text(encoding="utf-8"))


def test_the_committed_split_keeps_every_rule():
    result = json.loads((repo_root() / "eval" / "families.json").read_text(encoding="utf-8"))
    protocol = load_protocol()
    assert result["protocol_version"] == protocol["version"] and result["test_families"] == protocol["splits"][
        "test_families"], ("eval/families.json is not the protocol's version's: ntk eval split --rewrite writes it again, "
                           "the same draw, a family moving split only where the protocol moved one (MT-F34)")
    test = [f for f in result["split"] if f["split"] == "test"]
    assert len(test) == result["test_families"] and len(result["split"]) == result["families"]
    registry = [r for f in result["split"] for r in f["registry_families"]]
    assert len(registry) == len(set(registry)) == result["registry_families"]  # no registry family in both splits
    last = result["draws"][-1]
    assert last["balanced"] and all(abs(gap) < result["limit"] for gap in last["smd"].values())
    assert result["seed"] == protocol["splits"]["seed"] + len(result["draws"]) - 1
    assert {f["kind"] for f in test} == {f["kind"] for f in result["split"]}  # every kind reaches the test
