"""Model artifact 저장/로드 계약 테스트 (spec: 2026-07-09-model-artifact-contract)."""

from __future__ import annotations

import pytest

from data.feature_engineer import FeatureEngineer
from models.artifact import ArtifactError, ArtifactMetadata

PORTFOLIO_STATE_FIELDS = [
    "units_held_frac", "unrealized_pnl_norm",
    "holding_duration_norm", "tod_frac",
]


def make_metadata(**overrides) -> ArtifactMetadata:
    base = dict(
        artifact_format_version=1,
        artifact_id="ppo-fs2-test-0001",
        created_at="2026-07-09T00:00:00Z",
        algo="PPO",
        policy="MlpPolicy",
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        portfolio_state_fields=list(PORTFOLIO_STATE_FIELDS),
        observation_dim=len(FeatureEngineer.FEATURE_COLUMNS) + len(PORTFOLIO_STATE_FIELDS),
        action_space={"type": "discrete", "n": 3, "labels": ["hold", "add_unit", "clear"]},
        normalization=None,
        train_git_sha="testsha",
        train_data={"symbols": ["005930"], "start": "2025-05-22", "end": "2026-06-30"},
        env_params={"unit_fraction": 0.2, "max_units": 5, "initial_cash": 100_000_000},
    )
    base.update(overrides)
    return ArtifactMetadata(**base)


class TestArtifactMetadata:
    def test_dict_roundtrip(self):
        meta = make_metadata()
        restored = ArtifactMetadata.from_dict(meta.to_dict())
        assert restored == meta

    def test_from_dict_missing_field_raises(self):
        data = make_metadata().to_dict()
        del data["feature_columns"]
        with pytest.raises(ArtifactError, match="feature_columns"):
            ArtifactMetadata.from_dict(data)

    def test_from_dict_unsupported_format_version_raises(self):
        data = make_metadata().to_dict()
        data["artifact_format_version"] = 999
        with pytest.raises(ArtifactError, match="format_version"):
            ArtifactMetadata.from_dict(data)

    def test_observation_dim_mismatch_raises(self):
        data = make_metadata().to_dict()
        data["observation_dim"] = 99
        with pytest.raises(ArtifactError, match="observation_dim"):
            ArtifactMetadata.from_dict(data)

    def test_action_space_n_labels_mismatch_raises(self):
        data = make_metadata().to_dict()
        data["action_space"] = {"type": "discrete", "n": 4, "labels": ["hold", "add_unit", "clear"]}
        with pytest.raises(ArtifactError, match="action_space"):
            ArtifactMetadata.from_dict(data)

    def test_unknown_action_type_raises(self):
        data = make_metadata().to_dict()
        data["action_space"] = {"type": "continuous", "n": 3, "labels": ["a", "b", "c"]}
        with pytest.raises(ArtifactError, match="action_space"):
            ArtifactMetadata.from_dict(data)

    def test_bad_normalization_block_raises(self):
        data = make_metadata().to_dict()
        data["normalization"] = {"type": "minmax", "file": "x.pkl"}
        with pytest.raises(ArtifactError, match="normalization"):
            ArtifactMetadata.from_dict(data)
