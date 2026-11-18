import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

MIN_OBSERVATIONS = 20
SIGNIFICANCE = 0.05


def prepare_series(df: pd.DataFrame, date_col: str, target_col: str) -> pd.Series:
    """Return a clean, date indexed, sorted series with missing values removed."""
    for col in (date_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[target_col], errors="coerce")

    data = pd.DataFrame({"date": dates, "value": values}).dropna()
    data = data.drop_duplicates(subset="date").sort_values("date")
    return data.set_index("date")["value"]


def summary_statistics(series: pd.Series) -> dict:
    """Basic descriptive statistics for the target variable."""
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def stationarity_tests(series: pd.Series) -> dict:
    """Run the ADF and KPSS tests and combine them into one conclusion.

    ADF null hypothesis: the series has a unit root (not stationary).
    KPSS null hypothesis: the series is stationary.
    """
    if len(series) < MIN_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_OBSERVATIONS} observations are needed for stationarity tests"
        )

    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p is outside its table
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")

    adf_stationary = bool(adf_p < SIGNIFICANCE)
    kpss_stationary = bool(kpss_p > SIGNIFICANCE)

    if adf_stationary and kpss_stationary:
        conclusion = "stationary"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "non stationary"
    else:
        conclusion = "inconclusive"

    return {
        "adf_statistic": float(adf_stat),
        "adf_p_value": float(adf_p),
        "adf_stationary": adf_stationary,
        "kpss_statistic": float(kpss_stat),
        "kpss_p_value": float(kpss_p),
        "kpss_stationary": kpss_stationary,
        "conclusion": conclusion,
    }
