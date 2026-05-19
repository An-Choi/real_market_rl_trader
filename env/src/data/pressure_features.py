"""Order-flow and market pressure proxy feature skeletons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_volume_relationship(
    data: pd.DataFrame,
    close_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """Add a price-volume relationship proxy."""
    # TODO: Replace with order-flow imbalance when tick or order book data exists.
    output = data.copy()
    returns = output[close_col].pct_change()
    output["price_volume_pressure"] = returns * output[volume_col]
    return output


def add_candle_body_ratio(
    data: pd.DataFrame,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Add candle body size relative to the full candle range."""
    # TODO: Add wick imbalance features.
    output = data.copy()
    candle_range = (output[high_col] - output[low_col]).replace(0, np.nan)
    output["candle_body_ratio"] = (output[close_col] - output[open_col]).abs() / candle_range
    return output


def add_buy_sell_pressure_proxy(
    data: pd.DataFrame,
    open_col: str = "Open",
    close_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """Add a simple signed volume proxy for buy/sell pressure."""
    # TODO: Calibrate this proxy against more granular trade data.
    output = data.copy()
    direction = np.sign(output[close_col] - output[open_col])
    output["buy_sell_pressure_proxy"] = direction * output[volume_col]
    return output


def create_pressure_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create a default set of market pressure features."""
    output = add_price_volume_relationship(data)
    output = add_candle_body_ratio(output)
    output = add_buy_sell_pressure_proxy(output)
    return output
