"""Audit and optionally repair duplicate timestamps in raw minute Parquet files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
if str(ENV_SRC) not in sys.path:
    sys.path.insert(0, str(ENV_SRC))

from data.defect_days import sanitize_minute_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and repair files")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_dir = PROJECT_ROOT / "runs" / "data_quality"
    backup_root = PROJECT_ROOT / "data" / "quality_backups" / stamp
    records: list[dict] = []

    for directory in sorted(args.raw_dir.glob("*/1m")):
        files = sorted(directory.glob("*.parquet"))
        frames = []
        for path in files:
            frame = pd.read_parquet(path, engine="pyarrow")
            frame["__source_file"] = path.name
            frames.append(frame)
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        if "Timestamp" not in frame:
            records.append({"directory": str(directory), "error": "missing Timestamp"})
            continue
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
        duplicate_mask = frame.duplicated("Timestamp", keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_minutes = int(frame.loc[duplicate_mask, "Timestamp"].nunique())
        if duplicate_rows == 0:
            continue
        affected_names = sorted(frame.loc[duplicate_mask, "__source_file"].unique())
        clean = sanitize_minute_rows(frame.drop(columns="__source_file"))
        record = {
            "symbol": directory.parent.name,
            "files": affected_names,
            "rows_before": len(frame),
            "rows_after": len(clean),
            "duplicate_rows": duplicate_rows,
            "duplicate_minutes": duplicate_minutes,
            "repaired": bool(args.apply),
        }
        if args.apply:
            backups = []
            row_month = clean["Timestamp"].dt.strftime("%Y-%m")
            for name in affected_names:
                path = directory / name
                backup = backup_root / path.relative_to(args.raw_dir)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
                partition = clean.loc[row_month == path.stem].reset_index(drop=True)
                tmp = path.with_suffix(".parquet.repairing")
                partition.to_parquet(
                    tmp, engine="pyarrow", compression="snappy", index=False
                )
                tmp.replace(path)
                backups.append(str(backup.relative_to(PROJECT_ROOT)))
            record["backups"] = backups
        records.append(record)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "audit",
        "affected_files": len(records),
        "duplicate_rows": sum(item.get("duplicate_rows", 0) for item in records),
        "records": records,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"minute-duplicate-{stamp}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
