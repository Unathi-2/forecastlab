import pandas as pd
import pytest

from app.validation import validate_series


def make_df(dates, values):
    return pd.DataFrame({"date": dates, "sales": values})


def test_clean_daily_series():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    report = validate_series(make_df(dates, range(10)), "date", "sales")
    assert report["frequency"] == "D"
    assert report["is_regular"] is True
    assert report["bad_dates"] == 0
    assert report["duplicate_dates"] == 0
    assert report["missing_values"] == 0


def test_detects_duplicate_dates():
    dates = ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]
    report = validate_series(make_df(dates, [1, 2, 2, 3, 4]), "date", "sales")
    assert report["duplicate_dates"] == 1


def test_detects_missing_values():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    report = validate_series(make_df(dates, [1, None, 3, 4, 5]), "date", "sales")
    assert report["missing_values"] == 1


def test_detects_bad_dates():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "not a date"]
    report = validate_series(make_df(dates, [1, 2, 3, 4]), "date", "sales")
    assert report["bad_dates"] == 1


def test_detects_gap_in_dates():
    dates = ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]
    report = validate_series(make_df(dates, [1, 2, 4, 5]), "date", "sales")
    assert report["is_regular"] is False


def test_missing_column_raises_error():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.raises(ValueError):
        validate_series(make_df(dates, range(5)), "date", "revenue")
