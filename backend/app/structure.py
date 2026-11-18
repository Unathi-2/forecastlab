import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

PERIODS = {
    "h": 24,
    "H": 24,
    "B": 5,
    "D": 7,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
}


def infer_period(index: pd.DatetimeIndex) -> int | None:
    """Guess the seasonal period from the frequency of the dates."""
    if len(index) < 3:
        return None
    with pd.option_context("future.infer_freq_returns_offset", True):
        inferred = pd.infer_freq(index)
    if inferred is None:
        return None
    base = inferred.freqstr.lstrip("0123456789").split("-")[0]
    return PERIODS.get(base)


def _strength(component: np.ndarray, resid: np.ndarray) -> float:
    """Strength of a component, from 0 (none) to 1 (very strong)."""
    total_variance = np.var(component + resid)
    if total_variance == 0:
        return 0.0
    return float(max(0.0, 1 - np.var(resid) / total_variance))


def decompose(series: pd.Series, period: int) -> dict:
    """Split a series into trend, seasonal, and residual parts using STL."""
    if period < 2:
        raise ValueError("The seasonal period must be at least 2")
    if len(series) < 2 * period:
        raise ValueError(
            f"At least {2 * period} observations are needed for period {period}"
        )

    result = STL(series.to_numpy(dtype=float), period=period, robust=True).fit()

    return {
        "period": period,
        "dates": [d.isoformat() for d in series.index],
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "resid": result.resid.tolist(),
        "trend_strength": _strength(result.trend, result.resid),
        "seasonal_strength": _strength(result.seasonal, result.resid),
    }


def autocorrelation(series: pd.Series, nlags: int = 40) -> dict:
    """Autocorrelation and partial autocorrelation values."""
    n = len(series)
    if n < 10:
        raise ValueError("At least 10 observations are needed for autocorrelation")

    max_lags = min(nlags, n // 2 - 1)
    values = series.to_numpy(dtype=float)

    return {
        "nlags": max_lags,
        "acf": acf(values, nlags=max_lags, fft=True).tolist(),
        "pacf": pacf(values, nlags=max_lags, method="ywm").tolist(),
        "confidence_bound": float(1.96 / np.sqrt(n)),
    }
