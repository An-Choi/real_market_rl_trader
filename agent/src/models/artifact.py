"""Versioned model artifact 저장/로드 — 학습 산출물 ↔ 서빙 계약.

load_metadata()는 SB3 없이 동작해야 한다(서버가 모델 로드 전에 검증하는 경로).
이 모듈 top-level에서 stable_baselines3를 import하지 않는다.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_FORMAT_VERSIONS = (1,)
KNOWN_ACTION_TYPES = ("discrete",)
KNOWN_NORMALIZATION_TYPES = ("sb3_vecnormalize",)
REQUIRED_ENV_PARAMS = ("unit_fraction", "max_units", "initial_cash")
METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.zip"

# artifact.py: agent/src/models/ → repo root는 3단계 위
_REPO_ROOT = Path(__file__).resolve().parents[3]


def current_git_sha() -> str:
    """저장 시점 repo 상태 추적용. 실패해도 저장을 막지 않는다("unknown")."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


class ArtifactError(Exception):
    """Artifact 저장/로드/검증 실패."""


def _require_simple_name(value: Any, field_name: str) -> None:
    """경로로 쓰이는 필드는 단순 이름만 허용 — 절대경로/구분자/..로 artifact 밖 탈출 금지."""
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{field_name} must be a non-empty string: {value!r}")
    p = Path(value)
    if p.is_absolute() or p.name != value or value in (".", ".."):
        raise ArtifactError(f"{field_name} must be a bare name (no path parts): {value!r}")


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
        if not isinstance(data, dict):
            raise ArtifactError(f"metadata must be a JSON object, got: {type(data).__name__}")
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
        _require_simple_name(self.artifact_id, "artifact_id")
        for list_field in ("feature_columns", "portfolio_state_fields"):
            value = getattr(self, list_field)
            if not isinstance(value, list) or not all(
                isinstance(col, str) and col for col in value
            ):
                raise ArtifactError(
                    f"{list_field} must be a list of non-empty strings: {value!r}"
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
        if not all(isinstance(label, str) and label for label in labels):
            raise ArtifactError(f"invalid action_space: labels must be strings: {action!r}")
        norm = self.normalization
        if norm is not None:
            if not isinstance(norm, dict) or norm.get("type") not in KNOWN_NORMALIZATION_TYPES:
                raise ArtifactError(f"invalid normalization block: {norm!r}")
            if not norm.get("file"):
                raise ArtifactError(f"invalid normalization block, missing file: {norm!r}")
            _require_simple_name(norm["file"], "normalization file")
        env = self.env_params
        if not isinstance(env, dict):
            raise ArtifactError(f"env_params must be a dict: {env!r}")
        missing_params = [k for k in REQUIRED_ENV_PARAMS if k not in env]
        if missing_params:
            raise ArtifactError(f"env_params missing keys: {missing_params}")
        for key in REQUIRED_ENV_PARAMS:
            value = env[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArtifactError(f"env_params[{key!r}] must be numeric: {value!r}")


def make_artifact_id(algo: str, feature_schema_version: int) -> str:
    """생성 시점 포함한 artifact ID: "{algo소문자}-fs{v}-{YYYYMMDD-HHMMSS}" (UTC)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{algo.lower()}-fs{feature_schema_version}-{ts}"


def load_metadata(artifact_dir: "str | Path") -> ArtifactMetadata:
    """metadata.json만 파싱·검증. SB3 불필요 — 서버가 모델 로드 전에 쓰는 경로."""
    artifact_dir = Path(artifact_dir)
    meta_path = artifact_dir / METADATA_FILENAME
    if not meta_path.is_file():
        raise ArtifactError(f"metadata.json not found in: {artifact_dir}")
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"failed to parse {meta_path}: {exc}") from exc
    meta = ArtifactMetadata.from_dict(data)
    if not (artifact_dir / MODEL_FILENAME).is_file():
        raise ArtifactError(f"model.zip not found in: {artifact_dir}")
    if meta.normalization is not None:
        norm_file = artifact_dir / meta.normalization["file"]
        if not norm_file.is_file():
            raise ArtifactError(f"normalization file not found: {norm_file}")
    return meta


def load_artifact(
    artifact_dir: "str | Path", env: Any = None
) -> "tuple[Any, ArtifactMetadata]":
    """metadata 검증 통과 후에만 SB3 모델을 로드한다."""
    artifact_dir = Path(artifact_dir)
    meta = load_metadata(artifact_dir)
    if meta.algo != "PPO":
        raise ArtifactError(f"unsupported algo: {meta.algo}")
    if meta.normalization is not None:
        # VecNormalize stats 적용은 미구현 — 조용히 스케일 어긋난 predict를 내느니 거부.
        raise ArtifactError(
            "normalization artifact is not supported by load_artifact yet; "
            "apply the stats per docs/observation-contract.md §5"
        )

    from models.rl_agent import RLAgent  # SB3 의존 경로 — 함수 내부 import 유지

    agent = RLAgent(model_name=meta.algo, policy=meta.policy)
    agent.load(str(artifact_dir / MODEL_FILENAME), env=env)
    return agent, meta


def save_artifact(
    agent: Any,
    metadata: ArtifactMetadata,
    artifacts_dir: "str | Path",
    *,
    vecnormalize_path: "str | Path | None" = None,
) -> Path:
    """검증 → temp 디렉토리에 전부 기록 → 성공 시에만 최종 경로로 rename.

    실패 시 최종 artifact 경로에 partial 상태가 남지 않는다.
    """
    metadata.validate()
    if getattr(agent, "model", None) is None:
        raise ArtifactError("agent has no built model to save")

    norm = metadata.normalization
    if norm is None and vecnormalize_path is not None:
        raise ArtifactError("vecnormalize_path given but metadata.normalization is null")
    if norm is not None:
        if vecnormalize_path is None:
            raise ArtifactError("metadata.normalization set but vecnormalize_path missing")
        if not Path(vecnormalize_path).is_file():
            raise ArtifactError(f"vecnormalize file not found: {vecnormalize_path}")

    artifacts_dir = Path(artifacts_dir)
    final_dir = artifacts_dir / metadata.artifact_id
    if final_dir.exists():
        raise ArtifactError(f"artifact already exists: {final_dir}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = artifacts_dir / f".tmp-{metadata.artifact_id}-{uuid.uuid4().hex[:8]}"
    try:
        tmp_dir.mkdir()
        agent.model.save(str(tmp_dir / MODEL_FILENAME))
        if norm is not None:
            shutil.copy2(vecnormalize_path, tmp_dir / norm["file"])
        payload = json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False)
        (tmp_dir / METADATA_FILENAME).write_text(payload, encoding="utf-8")
        tmp_dir.rename(final_dir)
    except ArtifactError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ArtifactError(f"artifact save failed: {exc}") from exc
    return final_dir
