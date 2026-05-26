from __future__ import annotations

import pandas as pd

from data.feature_engineer import FeatureEngineer


def test_trading_value_excluded_from_features() -> None:
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [1_000_000, 1_100_000],
        "TradingValue": [101_000_000_000, 112_200_000_000],
        "Change": [1.0, 1.0],
        "Timestamp": pd.to_datetime(["2024-01-03 09:00:00", "2024-01-04 09:00:00"]),
        "my_alpha": [0.5, 0.6],
    })
    engineer = FeatureEngineer(drop_na=False)
    columns = engineer._infer_feature_columns(df)
    assert "TradingValue" not in columns
    assert "Change" not in columns
    assert "Timestamp" not in columns
    assert "my_alpha" in columns


def test_existing_raw_columns_still_excluded() -> None:
    df = pd.DataFrame({
        "Date": [], "Open": [], "High": [], "Low": [], "Close": [],
        "Adj Close": [], "Volume": [], "my_feature": [],
    })
    engineer = FeatureEngineer(drop_na=False)
    columns = engineer._infer_feature_columns(df)
    assert columns == ["my_feature"]
