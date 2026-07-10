"""Shared helpers for experiment entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.data_loader import DataLoader
from data.feature_builder import build_features


def resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else project_root / path


def make_data_loader(
    *,
    project_root: Path,
    config: dict[str, Any],
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> DataLoader:
    """Create the standard experiment data loader."""
    return DataLoader(
        raw_data_dir=(
            resolve_project_path(project_root, raw_dir)
            if raw_dir
            else project_root / config["data"]["raw_dir"]
        ),
        processed_data_dir=(
            resolve_project_path(project_root, processed_dir)
            if processed_dir
            else project_root / config["data"]["processed_dir"]
        ),
    )


def load_feature_data(
    *,
    symbol: str,
    data_loader: DataLoader,
    force_rebuild: bool = False,
) -> Any:
    """Load or build feature data with a friendly missing-data message."""
    try:
        return build_features(symbol, data_loader, force_rebuild=force_rebuild)
    except FileNotFoundError as exc:
        raise SystemExit(
            "No raw minute data found yet. Fill .env with KIS credentials, then run:\n"
            "  conda activate rl-trader-py310\n"
            f"  python scripts/backfill.py --symbols {symbol} --skip-daily"
        ) from exc
