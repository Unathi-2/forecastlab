import pandas as pd


def validate_series(df: pd.DataFrame, date_col: str, target_col: str) -> dict:
    """Check a dataset before forecasting and return a simple report."""
    for col in (date_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[target_col], errors="coerce")

    bad_dates = int(dates.isna().sum())
    missing_values = int(df[target_col].isna().sum())
    non_numeric_values = int(values.isna().sum()) - missing_values

    clean = pd.DataFrame({"date": dates, "value": values}).dropna(subset=["date"])
    duplicate_dates = int(clean["date"].duplicated().sum())
    clean = clean.drop_duplicates(subset="date").sort_values("date")

    frequency = None
    if len(clean) >= 3:
        with pd.option_context("future.infer_freq_returns_offset", True):
            inferred = pd.infer_freq(clean["date"])
        frequency = inferred.freqstr if inferred is not None else None

    return {
        "rows": len(df),
        "bad_dates": bad_dates,
        "duplicate_dates": duplicate_dates,
        "missing_values": missing_values,
        "non_numeric_values": non_numeric_values,
        "frequency": frequency,
        "is_regular": frequency is not None,
        "start": clean["date"].min().isoformat() if len(clean) else None,
        "end": clean["date"].max().isoformat() if len(clean) else None,
    }
