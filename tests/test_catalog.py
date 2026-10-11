import datetime as dt

import pytest

from needtoknow.providers.catalog import cost_usd, is_peak, load_catalog

UTC = dt.timezone.utc
ATHENS_SUMMER = dt.timezone(dt.timedelta(hours=3))


@pytest.fixture(scope="module")
def deepseek():
    return load_catalog()["deepseek"]


def test_the_configured_model_has_a_dated_price(deepseek):
    price = deepseek["models"][deepseek["model"]]["price"]
    assert price["as_of"] and price["source"].startswith("https://")
    assert set(price["off_peak"]) == set(price["peak"]) == {"input_hit", "input_miss", "output"}


@pytest.mark.parametrize(
    "when, expected",
    [
        (dt.datetime(2026, 10, 12, 1, 0, tzinfo=UTC), True),    # Monday, first window opens
        (dt.datetime(2026, 10, 12, 3, 59, tzinfo=UTC), True),
        (dt.datetime(2026, 10, 12, 4, 0, tzinfo=UTC), False),   # windows end before their end hour
        (dt.datetime(2026, 10, 12, 9, 59, tzinfo=UTC), True),
        (dt.datetime(2026, 10, 12, 10, 0, tzinfo=UTC), False),
        (dt.datetime(2026, 10, 17, 2, 0, tzinfo=UTC), False),   # Saturday
        (dt.datetime(2026, 10, 12, 5, 30, tzinfo=ATHENS_SUMMER), True),  # 02:30 UTC
    ],
)
def test_peak_windows(deepseek, when, expected):
    assert is_peak(when, deepseek["peak"]) is expected


def test_a_naive_time_is_refused(deepseek):
    with pytest.raises(ValueError):
        is_peak(dt.datetime(2026, 10, 12, 2, 0), deepseek["peak"])


def test_peak_costs_exactly_double_off_peak(deepseek):
    price = deepseek["models"]["deepseek-flash"]["price"]
    usage = {"input_hit": 1_000_000, "input_miss": 1_000_000, "output": 1_000_000}
    off = cost_usd(price, usage, dt.datetime(2026, 10, 17, 12, 0, tzinfo=UTC), deepseek["peak"])
    on = cost_usd(price, usage, dt.datetime(2026, 10, 12, 2, 0, tzinfo=UTC), deepseek["peak"])
    assert off == pytest.approx(0.003 + 0.15 + 0.60)
    assert on == pytest.approx(2 * off)


def test_missing_usage_kinds_cost_nothing(deepseek):
    price = deepseek["models"]["deepseek-flash"]["price"]
    when = dt.datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    assert cost_usd(price, {"output": 1000}, when, deepseek["peak"]) == pytest.approx(0.0006)
