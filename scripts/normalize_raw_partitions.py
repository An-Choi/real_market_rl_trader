"""Canonicalize overlapping monthly minute files with reversible backups."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env/src"))

import pandas as pd
from data.defect_days import sanitize_minute_rows
from pipeline.data_collector import DataCollector


def normalize(output, symbols):
    if output.exists():
        raise ValueError("backup revision already exists")
    output.mkdir(parents=True)
    collector = DataCollector(ROOT / "data/raw")
    changes, duplicates = [], 0
    for symbol in symbols:
        paths = sorted((ROOT / "data/raw" / symbol / "1m").glob("*.parquet"))
        sources = {path: pd.read_parquet(path) for path in paths}
        original = pd.concat(sources.values(), ignore_index=True)
        canonical = sanitize_minute_rows(original)
        duplicates += len(original) - len(canonical)
        months = canonical.Timestamp.dt.strftime("%Y-%m")
        for path, old in sources.items():
            replacement = canonical.loc[months == path.stem].reset_index(drop=True)
            if len(old) == len(replacement) and sanitize_minute_rows(old).equals(replacement):
                continue
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            backup = output / "backups" / path.relative_to(ROOT / "data")
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            collector.save_raw_parquet(replacement, symbol, "1m", path.stem)
            changes.append({"path": str(path.relative_to(ROOT / "data")), "before_sha256": before,
                "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "old_rows": len(old), "new_rows": len(replacement)})
        after = sanitize_minute_rows(pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True))
        pd.testing.assert_frame_equal(canonical, after)
    payload = {"completed": True, "duplicate_rows_removed": duplicates,
        "changed_partitions": changes, "canonical_observations_unchanged": True,
        "synthetic_rows_added": 0, "backups_preserved": True}
    (output / "completion.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"changed_partitions": len(changes), "duplicate_rows_removed": duplicates,
                      "canonical_observations_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbols", default="005930,000660,034220,066570,009150")
    args = parser.parse_args()
    normalize(args.output, args.symbols.split(","))
