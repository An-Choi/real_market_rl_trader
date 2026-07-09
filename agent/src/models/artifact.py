"""Versioned model artifact 저장/로드 — 학습 산출물 ↔ 서빙 계약.

load_metadata()는 SB3 없이 동작해야 한다(서버가 모델 로드 전에 검증하는 경로).
이 모듈 top-level에서 stable_baselines3를 import하지 않는다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

SUPPORTED_FORMAT_VERSIONS = (1,)
KNOWN_ACTION_TYPES = ("discrete",)
KNOWN_NORMALIZATION_TYPES = ("sb3_vecnormalize",)
METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.zip"


class ArtifactError(Exception):
    """Artifact 저장/로드/검증 실패."""


@dataclass
class ArtifactMetadata:
    artifact_format_version: int
    artifact_id: str
    created_at: str
    algo: str
    policy: str
    feature_schema_version: int
    feature_columns: list[str]
    portfolio_state_fields: list[str]
    observation_dim: int
    action_space: dict[str, Any]
    normalization: dict[str, Any] | None
    train_git_sha: str
    train_data: dict[str, Any]
    env_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        names = [f.name for f in dataclasses.fields(cls)]
        missing = [n for n in names if n not in data]
        if missing:
            raise ArtifactError(f"metadata missing fields: {missing}")
        meta = cls(**{n: data[n] for n in names})
        meta.validate()
        return meta

    def validate(self) -> None:
        """내부 일관성만 검증한다. 서버 기대치(schema version 등) 비교는 호출자 책임."""
        if self.artifact_format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ArtifactError(
                f"unsupported artifact_format_version: {self.artifact_format_version}"
            )
        expected_dim = len(self.feature_columns) + len(self.portfolio_state_fields)
        if self.observation_dim != expected_dim:
            raise ArtifactError(
                f"observation_dim {self.observation_dim} != "
                f"feature+portfolio 합계 {expected_dim}"
            )
        action = self.action_space
        if not isinstance(action, dict) or action.get("type") not in KNOWN_ACTION_TYPES:
            raise ArtifactError(f"invalid action_space: {action!r}")
        labels = action.get("labels")
        if not isinstance(labels, list) or action.get("n") != len(labels):
            raise ArtifactError(f"invalid action_space: n != len(labels): {action!r}")
        norm = self.normalization
        if norm is not None:
            if not isinstance(norm, dict) or norm.get("type") not in KNOWN_NORMALIZATION_TYPES:
                raise ArtifactError(f"invalid normalization block: {norm!r}")
            if not norm.get("file"):
                raise ArtifactError(f"invalid normalization block, missing file: {norm!r}")
