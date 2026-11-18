import numpy as np
import pandas as pd
import pytest

from app.structure import autocorrelation, decompose, infer_period


def monthly(values):
    index = pd.date_range("2005-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=index, dtype=float)


def ar1(phi, n, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def test_infer_period_daily():
    assert infer_period(pd.date_range("2024-01-01", periods=30, freq="D")) == 7


def test_infer_period_monthly():
    assert infer_period(pd.date_range("2024-01-01", periods=30, freq="MS")) == 12


def test_infer_period_irregular_returns_none():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-09"])
    assert infer_period(index) is None


def test_decompose_detects_trend_and_seasonality():
    rng = np.random.default_rng(1)
    t = np.arange(240)
    values = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 240)
    result = decompose(monthly(values), period=12)
    assert result["trend_strength"] > 0.9
    assert result["seasonal_strength"] > 0.9
    assert len(result["trend"]) == 240
    assert len(result["dates"]) == 240


def test_noise_has_weaker_seasonality_than_seasonal_series():
    rng = np.random.default_rng(2)
    t = np.arange(240)
    seasonal = decompose(monthly(10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 240)), 12)
    noise = decompose(monthly(rng.normal(0, 1, 240)), 12)
    assert noise["seasonal_strength"] < seasonal["seasonal_strength"]
    assert noise["seasonal_strength"] < 0.5


def test_decompose_short_series_raises_error():
    with pytest.raises(ValueError):
        decompose(monthly(np.arange(10)), period=12)


def test_decompose_bad_period_raises_error():
    with pytest.raises(ValueError):
        decompose(monthly(np.arange(100)), period=1)


def test_autocorrelation_of_ar1_process():
    result = autocorrelation(pd.Series(ar1(0.8, 1000)), nlags=10)
    assert result["acf"][0] == pytest.approx(1.0)
    assert result["acf"][1] == pytest.approx(0.8, abs=0.1)
    assert result["pacf"][1] == pytest.approx(0.8, abs=0.1)
    assert abs(result["pacf"][2]) < 0.2


def test_autocorrelation_limits_lags_for_short_series():
    result = autocorrelation(pd.Series(np.arange(20.0) % 5), nlags=40)
    assert result["nlags"] == 9


def test_autocorrelation_too_short_raises_error():
    with pytest.raises(ValueError):
        autocorrelation(pd.Series([1.0, 2.0, 3.0]))
