"""Who reads what: the owner's rules, no bypass for the administrator, and a filter bench that measures the filters roles send."""

import copy
import pathlib
import re

import pytest

import needtoknow
from needtoknow.policy import (
    BENCH_ONLY,
    UNLABELLED,
    PolicyError,
    bench_cases,
    load_labels,
    load_policy,
    planned_share,
    readable,
    validate,
)


@pytest.fixture(scope="module")
def labels():
    return load_labels()


@pytest.fixture(scope="module")
def policy(labels):
    return load_policy(labels=labels)


def test_sales_reads_returns_by_the_owners_rule(policy):
    assert readable(policy, "sales") == {"general", "sales", "returns"}


def test_each_department_reads_general_and_its_own_label(policy, labels):
    for label in set(labels) - {"general"}:
        assert {"general", label} <= readable(policy, label)


def test_the_administrator_reads_every_label_and_unlabelled_text(policy, labels):
    assert readable(policy, "admin") == set(labels) | {UNLABELLED}


def test_unlabelled_text_is_read_by_the_administrator_alone(policy):
    assert [role for role in policy["roles"] if UNLABELLED in readable(policy, role)] == ["admin"]


def test_no_code_path_names_the_administrator():
    # A branch on the administrator's name is the bypass nobody tests; the policy file is the only grant.
    source = pathlib.Path(needtoknow.__file__).parent
    offenders = [str(path) for path in source.rglob("*.py") if re.search(r"""["']admin["']""", path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_an_unknown_role_is_named_rather_than_a_key_error(policy):
    with pytest.raises(PolicyError, match="no role named 'auditor'"):
        readable(policy, "auditor")


def test_a_label_nobody_defined_is_refused(policy, labels):
    broken = copy.deepcopy(policy)
    broken["roles"]["legal"]["reads"].append("legl")
    assert any("legl" in problem for problem in validate(broken, labels))


def test_a_new_label_that_no_role_reads_is_refused(policy, labels):
    grown = copy.deepcopy(labels)
    grown["procurement"] = {"el": "Προμήθειες", "target_share": 0.0}
    assert any("no role reads procurement" in problem for problem in validate(policy, grown))


def test_the_filter_bench_measures_every_filter_as_sent(policy):
    cases = bench_cases(policy)
    sent = {(role, readable(policy, role)) for role in policy["roles"]}
    assert {(name, reads) for name, reads, bench_only in cases if not bench_only} == sent
    assert {name for name, _, bench_only in cases if bench_only} == set(BENCH_ONLY)
    assert not set(BENCH_ONLY) & set(policy["roles"])


def test_the_bench_only_views_reach_below_every_filter_sent(policy, labels):
    sent = [planned_share(labels, readable(policy, role)) for role in policy["roles"]]
    views = [planned_share(labels, reads) for reads in BENCH_ONLY.values()]
    assert max(views) < min(sent)
