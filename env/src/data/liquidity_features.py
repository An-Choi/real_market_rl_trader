"""Liquidity feature skeletons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_ratio(
    data: pd.DataFrame,
    volume_col: str = "Volume",
    window: int = 20,
) -> pd.DataFrame:
    """Add current volume relative to rolling average volume."""
    # TODO: Add intraday seasonality adjustment for high-frequency data.
    output = data.copy()
    rolling_volume = output[volume_col].rolling(window=window).mean()
    output[f"volume_ratio_{window}"] = output[volume_col] / rolling_volume
    return output


def add_spread_proxy(
    data: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Add a simple high-low spread proxy."""
    # TODO: Replace with bid-ask spread when quote data is available.
    output = data.copy()
    output["spread_proxy"] = (output[high_col] - output[low_col]) / output[close_col]
    return output


def add_liquidity_score(
    data: pd.DataFrame,
    volume_col: str = "Volume",
    spread_col: str = "spread_proxy",
) -> pd.DataFrame:
    """Add a placeholder liquidity score from volume and spread proxy."""
    # TODO: Normalize score by symbol and timeframe before using it in training.
    output = data.copy()
    epsilon = 1e-9
    if spread_col not in output:
        output = add_spread_proxy(output)
    output["liquidity_score"] = output[volume_col] / (output[spread_col].abs() + epsilon)
    output["liquidity_score"] = np.log1p(output["liquidity_score"])
    return output


def create_liquidity_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create a default set of liquidity features."""
    output = add_volume_ratio(data)
    output = add_spread_proxy(output)
    output = add_liquidity_score(output)
    return output
