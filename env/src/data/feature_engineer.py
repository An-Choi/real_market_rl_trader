"""Feature engineering pipeline for regime, liquidity, and pressure signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.liquidity_features import create_liquidity_features
from data.pressure_features import create_pressure_features
from data.regime_features import create_regime_features


@dataclass
class FeatureEngineer:
    """Combine feature families used by the trading environment."""

    drop_na: bool = True
    feature_columns: list[str] = field(default_factory=list)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all feature groups and return a feature-enriched DataFrame."""
        # TODO: Add configurable feature sets and scaling/normalization.
        output = data.copy()
        output = create_regime_features(output)
        output = create_liquidity_features(output)
        output = create_pressure_features(output)

        if self.drop_na:
            output = output.dropna().reset_index(drop=True)

        self.feature_columns = self._infer_feature_columns(output)
        return output

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit any feature preprocessing state and transform the data."""
        # TODO: Store scaler/statistics here when normalization is introduced.
        return self.transform(data)

    def _infer_feature_columns(self, data: pd.DataFrame) -> list[str]:
        """Infer non-raw columns that should be passed to the RL observation."""
        raw_columns = {
            "Date", "Datetime", "Timestamp",
            "Open", "High", "Low", "Close", "Adj Close",
            "Volume", "TradingValue", "Change",
        }
        return [column for column in data.columns if column not in raw_columns]
