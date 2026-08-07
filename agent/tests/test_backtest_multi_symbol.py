"""backtest.py 다종목 헬퍼 단위 테스트."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.backtest import (
    ensure_oos_artifact,
    resolve_backtest_symbols,
    resolve_boundaries,
)
from models.walk_forward import SplitBoundaries


class _Meta:
    """ArtifactMetadata 대역 (필요 필드만)."""

    def __init__(self, version: int, train_data: dict):
        self.artifact_format_version = version
        self.train_data = train_data
        self.artifact_id = "test-artifact"


V4_TRAIN_DATA = {
    "symbols": ["AAA", "BBB"],
    "trained_split": "train",
    "split_boundaries": {
        "train_end_date": "2026-01-14", "validation_end_date": "2026-01-17", "purge_days": 2,
    },
}


def test_cli_symbols_override_artifact_metadata():
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    config = {"data": {"symbols": ["CCC"]}}
    out = resolve_backtest_symbols(config=config, meta=meta, cli_symbol=None, cli_symbols="DDD")
    assert out == ["DDD"]


def test_artifact_symbols_beat_config_default():
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    config = {"data": {"symbols": ["CCC", "DDD", "EEE"]}}
    out = resolve_backtest_symbols(config=config, meta=meta, cli_symbol=None, cli_symbols=None)
    assert out == ["AAA", "BBB"]


def test_single_symbol_v3_artifact_defaults_to_its_own_symbol():
    meta = _Meta(3, {"symbols": ["005930"], "start": "2025-05-23", "end": "2026-03-12"})
    config = {"data": {"symbols": ["005930", "000660", "034220", "066570", "009150"]}}
    out = resolve_backtest_symbols(config=config, meta=meta, cli_symbol=None, cli_symbols=None)
    assert out == ["005930"]


def test_no_artifact_falls_back_to_config():
    config = {"data": {"symbols": ["AAA", "BBB"]}}
    out = resolve_backtest_symbols(config=config, meta=None, cli_symbol=None, cli_symbols=None)
    assert out == ["AAA", "BBB"]


def test_v4_boundaries_come_from_metadata_not_recomputed():
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    boundaries = resolve_boundaries(meta=meta, data_by_symbol={}, config={"data": {}})
    assert boundaries == SplitBoundaries.from_metadata(V4_TRAIN_DATA["split_boundaries"])


def test_v3_artifact_gets_none_for_ratio_fallback():
    meta = _Meta(3, {"symbols": ["005930"]})
    assert resolve_boundaries(meta=meta, data_by_symbol={}, config={"data": {}}) is None


def _frame(start: str, days: int) -> pd.DataFrame:
    ts = [pd.Timestamp(start) + pd.Timedelta(days=d) for d in range(days)]
    return pd.DataFrame({"Timestamp": ts, "Close": [100.0] * days})


def test_baseline_only_multi_symbol_computes_shared_boundaries():
    data = {"AAA": _frame("2026-01-01", 20), "BBB": _frame("2026-01-05", 16)}
    config = {"data": {"split": {"purge_days": 1}}}
    boundaries = resolve_boundaries(meta=None, data_by_symbol=data, config=config)
    assert boundaries is not None
    assert boundaries.purge_days == 1


def test_baseline_only_single_symbol_keeps_legacy_ratio_path():
    data = {"AAA": _frame("2026-01-01", 20)}
    assert resolve_boundaries(meta=None, data_by_symbol=data, config={"data": {}}) is None


def test_non_train_artifact_rejected():
    train_data = dict(V4_TRAIN_DATA)
    train_data["trained_split"] = "all"
    with pytest.raises(SystemExit):
        ensure_oos_artifact(_Meta(4, train_data))


def test_train_artifact_accepted_and_v3_passes():
    ensure_oos_artifact(_Meta(4, dict(V4_TRAIN_DATA)))
    ensure_oos_artifact(_Meta(3, {"symbols": ["005930"]}))
