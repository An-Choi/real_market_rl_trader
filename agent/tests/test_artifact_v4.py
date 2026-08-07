"""Artifact format v4 검증 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from models.artifact import ArtifactError, ArtifactMetadata, make_artifact_id, make_training_metadata


def _v4_metadata_dict() -> dict:
    """검증을 통과하는 최소 v4 metadata."""
    return {
        "artifact_format_version": 4,
        "artifact_id": "maskableppo-fs3-s2-20260807-000000",
        "created_at": "2026-08-07T00:00:00+00:00",
        "algo": "MaskablePPO",
        "policy": "MlpPolicy",
        "feature_schema_version": 3,
        "feature_columns": ["f1", "f2"],
        "portfolio_state_fields": [
            "units_held_frac", "unrealized_pnl_norm", "holding_duration_norm", "tod_frac",
        ],
        "observation_dim": 6,
        "action_space": {"type": "discrete", "n": 3, "labels": ["hold", "add_unit", "clear"]},
        "normalization": None,
        "train_git_sha": "unknown",
        "train_data": {
            "symbols": ["AAA", "BBB"],
            "start": "2026-01-01",
            "end": "2026-01-14",
            "trained_split": "train",
            "split_boundaries": {
                "train_end_date": "2026-01-14",
                "validation_end_date": "2026-01-17",
                "purge_days": 2,
            },
            "per_symbol": {
                "AAA": {"start": "2026-01-01", "end": "2026-01-14", "trading_days": 14},
                "BBB": {"start": "2026-01-05", "end": "2026-01-14", "trading_days": 10},
            },
        },
        "env_params": {
            "unit_fraction": 0.199,
            "max_units": 5,
            "initial_cash": 10000.0,
            "episode_days": 1,
            "duration_horizon_bars": 8,
            "nominal_bars_per_day": 8,
        },
        "friction_params": {
            "fee_rate": 0.00018,
            "spread_rate": 0.001,
            "slippage_rate": 0.0,
            "execution_uncertainty_rate": 0.0,
            "sell_tax_rate": 0.002,
            "dynamic_spread": True,
            "date_based_sell_tax": True,
        },
        "training_params": {},
    }


def test_valid_v4_metadata_passes():
    ArtifactMetadata.from_dict(_v4_metadata_dict())


def test_v4_requires_split_boundaries():
    payload = _v4_metadata_dict()
    del payload["train_data"]["split_boundaries"]
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_rejects_non_iso_dates():
    payload = _v4_metadata_dict()
    payload["train_data"]["split_boundaries"]["train_end_date"] = "01/14/2026"
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_rejects_inverted_boundaries():
    payload = _v4_metadata_dict()
    payload["train_data"]["split_boundaries"]["train_end_date"] = "2026-01-18"
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_rejects_negative_purge():
    payload = _v4_metadata_dict()
    payload["train_data"]["split_boundaries"]["purge_days"] = -1
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_rejects_duplicate_symbols():
    payload = _v4_metadata_dict()
    payload["train_data"]["symbols"] = ["AAA", "AAA"]
    payload["train_data"]["per_symbol"] = {
        "AAA": {"start": "2026-01-01", "end": "2026-01-14", "trading_days": 14}
    }
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_rejects_per_symbol_key_mismatch():
    payload = _v4_metadata_dict()
    del payload["train_data"]["per_symbol"]["BBB"]
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_rejects_global_range_mismatch():
    payload = _v4_metadata_dict()
    payload["train_data"]["start"] = "2025-12-31"
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_cross_check_train_split_end_after_boundary():
    payload = _v4_metadata_dict()
    payload["train_data"]["per_symbol"]["AAA"]["end"] = "2026-01-20"  # > train_end_date
    payload["train_data"]["end"] = "2026-01-20"
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(payload)


def test_v4_trained_split_all_has_no_range_constraint():
    payload = _v4_metadata_dict()
    payload["train_data"]["trained_split"] = "all"
    payload["train_data"]["per_symbol"]["AAA"]["end"] = "2026-01-20"
    payload["train_data"]["end"] = "2026-01-20"
    ArtifactMetadata.from_dict(payload)


def test_v3_metadata_still_passes_without_new_fields():
    payload = _v4_metadata_dict()
    payload["artifact_format_version"] = 3
    payload["train_data"] = {"symbols": ["AAA"], "start": "2026-01-01", "end": "2026-01-14"}
    ArtifactMetadata.from_dict(payload)


def test_artifact_id_includes_symbol_count():
    artifact_id = make_artifact_id("MaskablePPO", 3, 5)
    assert artifact_id.startswith("maskableppo-fs3-s5-")


def _frame(start: str, days: int) -> pd.DataFrame:
    ts = [pd.Timestamp(start) + pd.Timedelta(days=d) for d in range(days)]
    return pd.DataFrame({"Timestamp": ts, "Close": [100.0] * days})


class _FakeAgent:
    model_name = "MaskablePPO"
    policy = "MlpPolicy"


def _metadata_kwargs(**overrides):
    kwargs = dict(
        agent=_FakeAgent(),
        symbols=["AAA", "BBB"],
        featured_data_by_symbol={"AAA": _frame("2026-01-01", 14), "BBB": _frame("2026-01-05", 10)},
        trained_split="train",
        split_boundaries={
            "train_end_date": "2026-01-14",
            "validation_end_date": "2026-01-17",
            "purge_days": 2,
        },
        feature_schema_version=3,
        feature_columns=["f1", "f2"],
        env_params={
            "unit_fraction": 0.199, "max_units": 5, "initial_cash": 10000.0,
            "episode_days": 1, "duration_horizon_bars": 8, "nominal_bars_per_day": 8,
        },
        friction_params={
            "fee_rate": 0.00018, "spread_rate": 0.001, "slippage_rate": 0.0,
            "execution_uncertainty_rate": 0.0, "sell_tax_rate": 0.002,
            "dynamic_spread": True, "date_based_sell_tax": True,
        },
    )
    kwargs.update(overrides)
    return kwargs


def test_make_training_metadata_measures_per_symbol_from_data():
    meta = make_training_metadata(**_metadata_kwargs())
    assert meta.artifact_format_version == 4
    assert meta.train_data["per_symbol"]["BBB"] == {
        "start": "2026-01-05", "end": "2026-01-14", "trading_days": 10,
    }
    assert meta.train_data["start"] == "2026-01-01"
    assert meta.train_data["end"] == "2026-01-14"


def test_make_training_metadata_rejects_rows_past_train_boundary():
    bad = _metadata_kwargs(
        featured_data_by_symbol={
            "AAA": _frame("2026-01-01", 20),  # 01-20까지 — train_end 01-14 초과
            "BBB": _frame("2026-01-05", 10),
        },
    )
    with pytest.raises(ArtifactError):
        make_training_metadata(**bad)
