"""Data loading helpers for raw and processed market datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataLoader:
    """Load raw data (CSV / Parquet) and prepare basic processed datasets."""

    raw_data_dir: Path
    processed_data_dir: Path

    def load_raw_csv(self, file_name: str, parse_dates: bool = True) -> pd.DataFrame:
        file_path = self.raw_data_dir / file_name
        parse_cols = ["Date"] if parse_dates else None
        return pd.read_csv(file_path, parse_dates=parse_cols)

    def load_raw_parquet(
        self, symbol: str, interval: str, partition: str
    ) -> pd.DataFrame:
        path = self.raw_data_dir / symbol / interval / f"{partition}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No Parquet partition at {path}")
        return pd.read_parquet(path, engine="pyarrow")

    def load_raw_parquet_all(self, symbol: str, interval: str) -> pd.DataFrame:
        directory = self.raw_data_dir / symbol / interval
        if not directory.exists():
            raise FileNotFoundError(f"No Parquet directory at {directory}")
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {directory}")
        frames = [pd.read_parquet(f, engine="pyarrow") for f in files]
        combined = pd.concat(frames, ignore_index=True)
        sort_key = "Date" if "Date" in combined.columns else "Timestamp"
        return combined.sort_values(sort_key).reset_index(drop=True)

    def prepare_processed_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        processed = raw_data.copy()
        if "Date" in processed.columns:
            processed = processed.sort_values("Date")
        return processed.dropna().reset_index(drop=True)
