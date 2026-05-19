"""Data loading helpers for raw and processed market datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataLoader:
    """Load raw data and prepare basic processed datasets."""

    raw_data_dir: Path
    processed_data_dir: Path

    def load_raw_csv(self, file_name: str, parse_dates: bool = True) -> pd.DataFrame:
        """Load a CSV file from the raw data directory."""
        # TODO: Support parquet and database-backed data sources.
        file_path = self.raw_data_dir / file_name
        parse_cols = ["Date"] if parse_dates else None
        return pd.read_csv(file_path, parse_dates=parse_cols)

    def prepare_processed_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw OHLCV data for feature engineering."""
        # TODO: Add missing data handling, split adjustment checks, and resampling.
        processed = raw_data.copy()
        if "Date" in processed.columns:
            processed = processed.sort_values("Date")
        return processed.dropna().reset_index(drop=True)

    def save_processed(self, data: pd.DataFrame, file_name: str) -> Path:
        """Save processed data for later reuse."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_data_dir / file_name
        data.to_csv(output_path, index=False)
        return output_path

    def load_processed_csv(self, file_name: str) -> pd.DataFrame:
        """Load a processed CSV file."""
        # TODO: Validate that required feature columns are present.
        return pd.read_csv(self.processed_data_dir / file_name)
