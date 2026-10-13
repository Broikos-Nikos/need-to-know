"""The registered statistics (T5.3a): each checked on a worked example a reader can redo by hand, the unit of every
test the family, and the arms' decision saying "too small to tell" rather than naming a winner."""

import types

import pytest

from needtoknow.eval import score as S
from needtoknow.eval import stats as T


def test_the_wilson_interval_is_the_specs_worked_example():
    # specs/evaluation.md section 4: "At 40 of 50 that is about 67 to 89 percent".
    low, high = T.wilson(40, 50)
    assert (round(low, 3), round(high, 3)) == (0.670, 0.888)
    assert T.wilson(0, 10)[0] == 0.0 and T.wilson(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        T.wilson(0, 0)


def test_the_sign_flip_test_is_exact_over_the_units_that_differ():
    # Four units differ, +1 +1 +1 -1, one does not: 16 sign patterns, of which those summing to 4, 2, -2 or -4,
    # 1 + 4 + 4 + 1 = 10, are at least as far from zero as the observed 2: p = 10 / 16.
    assert T.sign_flip([1, 1, 1, 0, -1]) == pytest.approx(0.625)
    assert T.sign_flip([0, 0, 0]) == 1.0  # nothing differs, nothing to tell
    # Thirty units each half a point up: only the two all same sign patterns reach the observed, so the resampled p is
    # the counted observation alone, 1 in 10,001.
    assert T.sign_flip([0.5] * 30) == pytest.approx(1 / 10001)
    assert T.sign_flip([0.5, -0.25] * 15) == T.sign_flip([0.5, -0.25] * 15)  # seeded: the same twice


def test_the_bootstrap_interval_holds_the_mean_and_is_seeded():
    assert T.bootstrap([0.2] * 12) == pytest.approx((0.2, 0.2))
    diffs = [1, 0, 0, 0.5, -0.5, 1, 0, 0.5]
    low, high = T.bootstrap(diffs)
    assert low < sum(diffs) / len(diffs) < high and (low, high) == T.bootstrap(diffs)


def test_the_difference_of_two_unpaired_sets_of_units_holds_its_mean_and_is_seeded():
    # N33's diagnostic: dev less test over families, the two splits' families drawn apart, never paired.
    gap, low, high = T.difference([1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    assert gap == pytest.approx(0.75 - 1 / 3) and low < gap < high and (low, high) == T.difference(
        [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0])[1:]
    assert T.difference([0.5] * 5, [0.5] * 7) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        T.difference([], [1.0])


def test_the_unit_is_the_family_an_item_pair_the_pair_and_an_own_item_itself():
    items = [{"id": "D1", "families": ["F06V00"]}, {"id": "D2", "families": ["F06V00"]},
             {"id": "D3", "families": ["F40V00", "F26V01"]}, {"id": "D4", "families": []}]
    assert [T.unit_of(i) for i in items] == ["F06V00", "F06V00", "F26V01+F40V00", "D4"]
    scores = {"D1": [1, 1, 0], "D2": [0, 0, 0], "D3": [1, 1, 1], "D4": [0, 1, 1]}  # three repeats each
    assert T.unit_means(items, scores) == pytest.approx({"D4": 2 / 3, "F06V00": 1 / 3, "F26V01+F40V00": 1.0})
    better = {**scores, "D2": [1, 1, 1]}
    assert T.differences(items, scores, better) == pytest.approx([0.0, 0.5, 0.0])


def test_the_arms_adopt_only_what_the_registered_rules_clear_and_say_too_small_to_tell_otherwise():
    rules = {"alpha": 0.05, "margin": 0.05, "smallest_visible": 0.10}
    assert T.decide([1] * 20 + [0] * 30, **rules)["verdict"] == "up"
    unclear = T.decide([1, -1, 1, 0, 0, -1, 1, 0], **rules)
    assert unclear["verdict"] == "too small to tell" and unclear["smallest_visible_effect"] == 0.10
    assert T.decide([0] * 50, cost_cut=0.20, cost_ratio=0.7, **rules)["verdict"] == "cost"
    assert T.decide([0] * 50, cost_cut=0.20, cost_ratio=0.9, **rules)["verdict"] == "too small to tell"
    assert T.decide([-1] * 20 + [0] * 30, cost_cut=0.20, cost_ratio=0.5, **rules)["verdict"] == "worse"
    assert T.decide([0] * 50, probe_diffs=[0.5] * 30, **rules)["verdict"] == "probe"


def test_each_answered_item_says_whether_its_file_holds_a_section_its_role_may_not_read():
    # RS-F22: D's condition, which an A, B or C item can share, as a property beside the category.
    docs = {"MIX": {"sections": [{"label": "sales", "facts": ["F1"]}, {"label": "legal", "facts": []}]},
            "OWN": {"sections": [{"label": "sales", "facts": ["F2"]}]}}
    placed = {"F1": [("MIX", docs["MIX"]["sections"][0])], "F2": [("OWN", docs["OWN"]["sections"][0])]}
    ctx = types.SimpleNamespace(docs=docs, sections=placed.__getitem__,
                                policy={"roles": {"sales": {"reads": ["general", "sales"]}}})
    item = lambda fid: {"id": "D1", "role": "sales", "expect": {"facts": [{"fact_id": fid}]}}  # noqa: E731
    assert (S.mixed(item("F1"), ctx), S.mixed(item("F2"), ctx)) == (True, False)
    assert S.mixed({"id": "D2", "role": "sales", "expect": {"facts": []}}, ctx) is None
