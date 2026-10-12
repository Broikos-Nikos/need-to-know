"""The registered protocol: the test split can never decide, and the file cannot change quietly."""

import copy
import datetime as dt
import re
import tomllib

import pytest

from needtoknow.eval.protocol import ProtocolError, check_use, fingerprint, history_path, load_protocol, validate


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


def test_the_registered_protocol_is_valid(protocol):
    assert validate(protocol) == []


@pytest.mark.parametrize("purpose", ["development", "experiment"])
def test_the_test_split_can_never_decide(protocol, purpose):
    with pytest.raises(ProtocolError, match=f"not for {purpose}"):
        check_use("test", purpose, protocol)
    check_use("dev", purpose, protocol)


@pytest.mark.parametrize("purpose", ["baseline", "release"])
def test_the_test_split_serves_the_baseline_and_each_release(protocol, purpose):
    check_use("test", purpose, protocol)


def test_an_unknown_split_is_named_rather_than_a_key_error(protocol):
    with pytest.raises(ProtocolError, match="no split named 'train'"):
        check_use("train", "experiment", protocol)


def test_the_plan_as_first_written_is_refused(protocol):
    # Until 2026-10-12 the plan decided every experiment on the same 50 it published.
    old = copy.deepcopy(protocol)
    old["splits"]["test"]["purposes"].append("experiment")
    assert any("test split may not serve experiment" in problem for problem in validate(old))


@pytest.mark.parametrize("margin", [None, 5, 0.0, 0.5])
def test_a_missing_or_mistyped_margin_is_refused(protocol, margin):
    broken = copy.deepcopy(protocol)
    if margin is None:
        del broken["decision"]["non_inferiority_margin"]
    else:
        broken["decision"]["non_inferiority_margin"] = margin
    assert any("non_inferiority_margin" in problem for problem in validate(broken))


def test_a_reordered_queue_is_refused(protocol):
    broken = copy.deepcopy(protocol)
    order = broken["experiments"]["order"]
    order[0], order[1] = order[1], order[0]
    assert any("experiments.order" in problem for problem in validate(broken))


def test_the_protocol_matches_its_newest_recorded_version():
    with open(history_path(), "rb") as handle:
        versions = tomllib.load(handle)["version"]
    assert fingerprint() == versions[-1]["sha256"], (
        "eval/protocol.toml changed: record the new version, dated and with its reason, in eval/protocol_history.toml"
    )


def test_every_recorded_version_is_dated_and_explained(protocol):
    with open(history_path(), "rb") as handle:
        versions = tomllib.load(handle)["version"]
    dates = [version["date"] for version in versions]
    assert all(isinstance(date, dt.date) for date in dates) and dates == sorted(dates)
    assert all(re.fullmatch(r"[0-9a-f]{64}", version["sha256"]) for version in versions)
    assert all(version["reason"].strip() for version in versions)
    assert protocol["version"] == len(versions)


def test_a_score_triggered_redraw_is_refused(protocol):
    # Until version 2 the baseline redrew the split when dev and test disagreed:
    # a test outcome choosing, after the dev items had been read.
    old = copy.deepcopy(protocol)
    old["baseline"]["on_disagreement"] = "redraw"
    assert any("on_disagreement" in problem for problem in validate(old))


def test_balance_is_checked_before_any_score(protocol):
    late = copy.deepcopy(protocol)
    late["splits"]["balance"]["when"] = "baseline"
    assert any("splits.balance.when" in problem for problem in validate(late))
