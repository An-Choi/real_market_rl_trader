"""Data collection skeletons for OHLCV market data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataCollector:
    """Collect raw OHLCV data from external sources or local files."""

    raw_data_dir: Path

    def fetch_from_yfinance(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance.

        Args:
            symbol: Market symbol such as "AAPL" or "BTC-USD".
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            interval: yfinance interval such as "1d", "1h", or "5m".

        Returns:
            A DataFrame containing OHLCV columns.
        """
        # TODO: Add retry, validation, timezone handling, and symbol metadata.
        import yfinance as yf

        data = yf.download(symbol, start=start, end=end, interval=interval)
        return data.reset_index()

    def fetch_from_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Load OHLCV data from a CSV file."""
        # TODO: Standardize column names and enforce required OHLCV schema.
        return pd.read_csv(file_path)

    def save_raw(self, data: pd.DataFrame, file_name: str) -> Path:
        """Save raw market data under the configured raw data directory."""
        # TODO: Add partitioning by symbol/timeframe if multiple assets are used.
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.raw_data_dir / file_name
        data.to_csv(output_path, index=False)
        return output_path
