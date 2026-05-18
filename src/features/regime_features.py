"""Market regime feature skeletons."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_returns(data: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add simple return features."""
    # TODO: Add log returns and multi-horizon returns.
    output = data.copy()
    output["return_1"] = output[price_col].pct_change()
    return output


def add_moving_averages(
    data: pd.DataFrame,
    windows: Iterable[int] = (5, 20),
    price_col: str = "Close",
) -> pd.DataFrame:
    """Add moving average features for regime detection."""
    # TODO: Add EMA and moving average distance features.
    output = data.copy()
    for window in windows:
        output[f"ma_{window}"] = output[price_col].rolling(window=window).mean()
    return output


def add_volatility(
    data: pd.DataFrame,
    window: int = 20,
    return_col: str = "return_1",
) -> pd.DataFrame:
    """Add rolling volatility features."""
    # TODO: Support realized volatility and downside volatility.
    output = data.copy()
    if return_col not in output:
        output = add_returns(output)
    output[f"volatility_{window}"] = output[return_col].rolling(window=window).std()
    return output


def add_rsi(data: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.DataFrame:
    """Add a basic RSI feature."""
    # TODO: Confirm RSI convention and handle warm-up periods consistently.
    output = data.copy()
    delta = output[price_col].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    output[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return output


def create_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create a default set of regime features."""
    output = add_returns(data)
    output = add_moving_averages(output)
    output = add_volatility(output)
    output = add_rsi(output)
    return output
