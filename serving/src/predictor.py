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
    SERVING_FORMAT_VERSION,
    ArtifactError,
    ArtifactMetadata,
    load_artifact,
    load_metadata,
)


def _validate_serving_expectations(meta: ArtifactMetadata) -> None:
    if meta.artifact_format_version != SERVING_FORMAT_VERSION:
        raise ArtifactError(
            f"serving accepts artifact format v{SERVING_FORMAT_VERSION} only, "
            f"got v{meta.artifact_format_version} — re-export the artifact"
        )
    if meta.feature_schema_version != FeatureEngineer.FEATURE_SCHEMA_VERSION:
        raise ArtifactError(
            f"artifact feature_schema_version={meta.feature_schema_version} != "
            f"server FeatureEngineer version {FeatureEngineer.FEATURE_SCHEMA_VERSION}"
        )
    if list(meta.feature_columns) != list(FeatureEngineer.FEATURE_COLUMNS):
        raise ArtifactError(
            f"artifact feature_columns {meta.feature_columns!r} != server "
            f"{list(FeatureEngineer.FEATURE_COLUMNS)!r} (이름+순서 정확 일치 필요)"
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
    # friction_params 존재·타입은 format v3의 metadata.validate()가 보장한다.


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
        return cls(agent, meta)

    def predict(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # mask는 항상 명시적으로 전달 — RLAgent의 obs 기반 fallback mask는
        # cash 게이트가 없어 학습 env mask보다 약하다 (spec §1).
        action, _ = self._agent.predict(
            observation, deterministic=True, action_masks=action_mask
        )
        return int(action)
