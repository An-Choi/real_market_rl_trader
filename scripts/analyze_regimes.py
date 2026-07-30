"""Report market regimes for the fixed split and expanding walk-forward folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"

for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.common import load_feature_data, make_data_loader
from models.walk_forward import (
    generate_expanding_walk_forward_folds,
    split_by_trading_day,
)
from utils.config_loader import load_config


def summarize_regime(data: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize total and daily market direction without future inputs."""
    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Timestamp"]).dt.date
    daily = frame.groupby("Date", sort=True)["Close"].agg(["first", "last"])
    daily_returns = daily["last"] / daily["first"] - 1.0
    return {
        "start": str(frame["Date"].min()),
        "end": str(frame["Date"].max()),
        "days": int(len(daily)),
        "market_total_return": float(
            frame["Close"].iloc[-1] / frame["Close"].iloc[0] - 1.0
        ),
        "up_day_rate": float((daily_returns > 0).mean()),
        "median_intraday_return": float(daily_returns.median()),
        "intraday_return_volatility": float(daily_returns.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe train/evaluation regimes")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    symbol = args.symbol or config["data"]["symbol"]
    data = load_feature_data(
        symbol=symbol,
        data_loader=make_data_loader(project_root=PROJECT_ROOT, config=config),
    )
    fixed = {
        name: summarize_regime(split_by_trading_day(data, split=name))
        for name in ("train", "validation", "test")
    }
    folds = generate_expanding_walk_forward_folds(data, n_folds=args.folds)
    rolling = [
        {
            "fold": fold.index,
            **{
                name: summarize_regime(getattr(fold, name))
                for name in ("train", "validation", "test")
            },
        }
        for fold in folds
    ]
    print(json.dumps(
        {"symbol": symbol, "fixed_split": fixed, "walk_forward": rolling},
        indent=2,
    ))


if __name__ == "__main__":
    main()
