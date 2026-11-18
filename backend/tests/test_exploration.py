import numpy as np
import pandas as pd
import pytest

from app.exploration import prepare_series, stationarity_tests, summary_statistics


def make_series(values):
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates, dtype=float)


def test_prepare_series_sorts_and_cleans():
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"],
            "y": [3, 1, None, 2],
        }
    )
    result = prepare_series(df, "date", "y")
    assert list(result) == [1, 2, 3]
    assert result.index.is_monotonic_increasing


def test_prepare_series_missing_column():
    df = pd.DataFrame({"date": ["2024-01-01"], "y": [1]})
    with pytest.raises(ValueError):
        prepare_series(df, "date", "sales")


def test_summary_statistics_known_values():
    stats = summary_statistics(make_series([1, 2, 3, 4, 5]))
    assert stats["count"] == 5
    assert stats["mean"] == 3
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["median"] == 3
    assert stats["std"] == pytest.approx(1.5811, abs=1e-3)


def test_white_noise_is_stationary_by_adf():
    rng = np.random.default_rng(42)
    result = stationarity_tests(make_series(rng.normal(0, 1, 300)))
    assert result["adf_stationary"] is True


def test_trending_series_is_not_stationary():
    rng = np.random.default_rng(42)
    values = 0.5 * np.arange(300) + rng.normal(0, 1, 300)
    result = stationarity_tests(make_series(values))
    assert result["kpss_stationary"] is False
    assert result["conclusion"] != "stationary"


def test_short_series_raises_error():
    with pytest.raises(ValueError):
        stationarity_tests(make_series([1, 2, 3, 4, 5]))
