"""Read-only daily-calendar proxy / raw-minute / feature quality report."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env/src"))
from data.defect_days import classify_trading_day
from data.data_loader import DataLoader
from data.feature_builder import build_features


def audit_symbol(loader, symbol):
    # Read partitions directly so duplicate evidence is not sanitized away.
    paths = sorted((loader.raw_data_dir / symbol / "1m").glob("*.parquet"))
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    dates = pd.to_datetime(raw.Timestamp).dt.date
    daily = loader.load_raw_parquet_all(symbol, "1d")
    calendar = sorted(set(pd.to_datetime(daily["Date" if "Date" in daily else "Timestamp"]).dt.date))
    calendar = [d for d in calendar if dates.min() <= d <= dates.max()]
    features = build_features(symbol, loader)
    feature_counts = pd.to_datetime(features.Timestamp).dt.date.value_counts()
    groups = dict(tuple(raw.groupby(dates)))
    rows = []
    for day in calendar:
        if day not in groups:
            rows.append({"date": str(day), "raw_valid": False,
                         "reasons": ["missing_raw_day"], "feature_rows": 0})
            continue
        quality = classify_trading_day(groups[day], require_complete_session=True).to_dict()
        rows.append({"date": str(day), "raw_valid": quality["is_valid"],
                     "feature_rows": int(feature_counts.get(day, 0)), **quality})
    return {"symbol": symbol, "calendar_source": "per-symbol daily parquet proxy, not official KRX calendar",
            "rows": rows, "summary": {"reference_days": len(rows),
                "raw_valid_days": sum(r["raw_valid"] for r in rows),
                "usable_feature_days": sum(r["feature_rows"] > 0 for r in rows),
                "reasons": dict(Counter(x for r in rows for x in r["reasons"]))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="005930,000660,034220,066570,009150")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/data_quality/research-audit.json")
    args = parser.parse_args()
    loader = DataLoader(ROOT / "data/raw", ROOT / "data/processed")
    reports = [audit_symbol(loader, s) for s in args.symbols.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mode": "read-only", "symbols": reports}, indent=2))
    print(json.dumps({r["symbol"]: r["summary"] for r in reports}, indent=2))


if __name__ == "__main__":
    main()
