"""Freeze coverage-qualified historical folds and a future holdout reservation."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(directory))

import pandas as pd
from experiments.run_ablation import load_market, data_fingerprints, write_json
from models.research_protocol import reference_calendar, quality_only_folds, fold_coverage
from models.walk_forward import build_expanding_walk_forward_folds


def run(output):
    if output.exists():
        raise ValueError("plan exists; preserve original protocol")
    data, _, _ = load_market(["005930", "000660", "034220", "066570", "009150"])
    start = max(pd.to_datetime(frame.Timestamp).dt.date.min() for frame in data.values())
    end = min(pd.to_datetime(frame.Timestamp).dt.date.max() for frame in data.values())
    calendar, provenance = reference_calendar(start, end)
    selected, candidates = quality_only_folds(data, calendar)
    standard = build_expanding_walk_forward_folds(data, n_folds=3, validation_days=60,
        test_days=20, purge_days=5, reference_days=calendar)
    latest = [{**fold.to_dict(), "coverage": fold_coverage(data, calendar, fold)} for fold in standard]
    future, future_provenance = reference_calendar(date(2026,11,10), date(2026,12,31))
    payload = {"selection": "latest three fixed 60-day validation/20-day test candidates passing 95% coverage; no returns used",
        "historical_status": "retrospective, reused data; not a new independent holdout",
        "minimum_coverage": .95, "calendar": [str(day) for day in calendar], "calendar_provenance": provenance,
        "source_fingerprints": data_fingerprints(data), "folds": [fold.to_dict() for fold in selected],
        "candidate_audit": candidates, "original_latest_three": latest,
        "future_holdout": {"start": str(future[0]), "end": str(future[19]), "sessions": [str(day) for day in future[:20]],
            "calendar": future_provenance, "evaluated": False, "status": "waiting for future observed data",
            "minimum_coverage": .95, "partial_evaluation_allowed": False}}
    write_json(output, payload)
    print({"output": str(output), "eligible_selected_folds": len(selected), "candidates": len(candidates)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
