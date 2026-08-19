"""Artifact 기동 검증 + inference — spec §3 기동 거부 목록의 실행부.

검증 순서: load_metadata(내부 일관성) → 서버 기대치 정확 비교 → load_artifact
(모델 로드는 검증 통과 후에만). 정규화는 load_artifact가 복원한 normalizer가
agent.predict() 내부에서 적용한다 — 여기서 미리 하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.feature_engineer import FeatureEngineer
from friction.friction_model import FrictionModel
from models.artifact import (
    DEFAULT_PORTFOLIO_STATE_FIELDS,
    EXPECTED_ACTION_LABELS,
    SERVING_FORMAT_VERSIONS,
    ArtifactError,
    ArtifactMetadata,
    load_artifact,
    load_metadata,
)


def _validate_serving_expectations(meta: ArtifactMetadata) -> None:
    if meta.artifact_format_version not in SERVING_FORMAT_VERSIONS:
        raise ArtifactError(
            f"serving accepts artifact format v3/v4/v5 only, "
            f"got v{meta.artifact_format_version} — re-export the artifact"
        )
    if meta.feature_schema_version != FeatureEngineer.FEATURE_SCHEMA_VERSION:
        raise ArtifactError(
            f"artifact feature_schema_version={meta.feature_schema_version} != "
            f"server FeatureEngineer version {FeatureEngineer.FEATURE_SCHEMA_VERSION}"
        )
    canonical = list(FeatureEngineer.FEATURE_COLUMNS)
    selected = list(meta.feature_columns)
    expected_order = [column for column in canonical if column in selected]
    if not selected or selected != expected_order or len(selected) != len(set(selected)):
        raise ArtifactError(
            f"artifact feature_columns {meta.feature_columns!r} must be a non-empty "
            f"canonical-order subset of server columns {canonical!r}"
        )
    if list(meta.portfolio_state_fields) != list(DEFAULT_PORTFOLIO_STATE_FIELDS):
        raise ArtifactError(
            f"artifact portfolio_state_fields {meta.portfolio_state_fields!r} != "
            f"{DEFAULT_PORTFOLIO_STATE_FIELDS!r}"
        )
    if meta.action_space.get("labels") != EXPECTED_ACTION_LABELS:
        raise ArtifactError(
            f"artifact action labels {meta.action_space.get('labels')!r} != "
            f"{EXPECTED_ACTION_LABELS!r}"
        )
    if meta.artifact_format_version >= 5:
        validation = meta.training_params.get("validation", {})
        qualified = (validation.get("best") or {}).get("qualified", False)
        if meta.deployment_status != "approved" or not qualified:
            raise ArtifactError(
                "format v5 artifact is not an approved qualified checkpoint; "
                f"deployment_status={meta.deployment_status!r}"
            )
        if meta.train_data.get("trained_split") != "train":
            raise ArtifactError("serving requires a v5 artifact trained on the train split")
    # friction_params 존재·타입은 format v3/v4의 metadata.validate()가 보장한다.


class Predictor:
    def __init__(self, agent, meta: ArtifactMetadata) -> None:
        self._agent = agent
        self.meta = meta
        self.env_params = dict(meta.env_params)
        self.friction_model = FrictionModel(**meta.friction_params)

    @classmethod
    def load(cls, artifact_dir: "str | Path") -> "Predictor":
        meta = load_metadata(artifact_dir)
        _validate_serving_expectations(meta)
        agent, meta = load_artifact(artifact_dir)
        model = agent.model
        observation_space = getattr(model, "observation_space", None)
        action_space = getattr(model, "action_space", None)
        if observation_space is None or observation_space.shape[0] != meta.observation_dim:
            raise ArtifactError(
                "loaded model observation space does not match artifact metadata"
            )
        if action_space is None or getattr(action_space, "n", None) != meta.action_space["n"]:
            raise ArtifactError("loaded model action space does not match artifact metadata")
        return cls(agent, meta)

    def predict(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # mask는 항상 명시적으로 전달 — RLAgent의 obs 기반 fallback mask는
        # cash 게이트가 없어 학습 env mask보다 약하다 (spec §1).
        action, _ = self._agent.predict(
            observation, deterministic=True, action_masks=action_mask
        )
        return int(action)
