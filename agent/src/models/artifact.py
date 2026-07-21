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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_FORMAT_VERSIONS = (1, 2)
KNOWN_ACTION_TYPES = ("discrete",)
KNOWN_NORMALIZATION_TYPES = ("sb3_vecnormalize", "feature_standardization")
REQUIRED_ENV_PARAMS_BY_VERSION = {
    1: ("unit_fraction", "max_units", "initial_cash"),
    2: (
        "unit_fraction",
        "max_units",
        "initial_cash",
        "episode_days",
        "duration_horizon_bars",
        "nominal_bars_per_day",
    ),
}
# v1 alias — 기존 테스트/외부 참조 호환
REQUIRED_ENV_PARAMS = REQUIRED_ENV_PARAMS_BY_VERSION[1]
POSITIVE_INT_ENV_PARAMS = ("episode_days", "duration_horizon_bars", "nominal_bars_per_day")
LEGACY_SEMANTICS_MAX_VERSION = 1  # v1 = 당일 기준 holding_duration_norm으로 학습됨
# trading_env의 0/20/40/60/80/100% 목표 비중과 순서 고정 계약
EXPECTED_ACTION_LABELS = [
    "target_0pct",
    "target_20pct",
    "target_40pct",
    "target_60pct",
    "target_80pct",
    "target_100pct",
]
METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.zip"
DEFAULT_PORTFOLIO_STATE_FIELDS = [
    "units_held_frac",
    "unrealized_pnl_norm",
    "holding_duration_norm",
    "tod_frac",
]

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
    training_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        if not isinstance(data, dict):
            raise ArtifactError(f"metadata must be a JSON object, got: {type(data).__name__}")
        fields = list(dataclasses.fields(cls))
        missing = [
            item.name
            for item in fields
            if item.name not in data
            and item.default is dataclasses.MISSING
            and item.default_factory is dataclasses.MISSING
        ]
        if missing:
            raise ArtifactError(f"metadata missing fields: {missing}")
        values = {item.name: data[item.name] for item in fields if item.name in data}
        meta = cls(**values)
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
            if norm["file"] in (MODEL_FILENAME, METADATA_FILENAME):
                # model.zip이면 모델이 stats로 덮어써지고, metadata.json이면 stats가 유실된다.
                raise ArtifactError(f"normalization file uses a reserved name: {norm['file']!r}")
        env = self.env_params
        if not isinstance(env, dict):
            raise ArtifactError(f"env_params must be a dict: {env!r}")
        required = REQUIRED_ENV_PARAMS_BY_VERSION[self.artifact_format_version]
        missing_params = [k for k in required if k not in env]
        if missing_params:
            raise ArtifactError(f"env_params missing keys: {missing_params}")
        for key in required:
            value = env[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArtifactError(f"env_params[{key!r}] must be numeric: {value!r}")
        for key in POSITIVE_INT_ENV_PARAMS:
            if key in required:
                value = env[key]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ArtifactError(
                        f"env_params[{key!r}] must be a positive int: {value!r}"
                    )
        if not isinstance(self.training_params, dict):
            raise ArtifactError("training_params must be a dict")


def make_artifact_id(algo: str, feature_schema_version: int) -> str:
    """생성 시점 포함한 artifact ID: "{algo소문자}-fs{v}-{YYYYMMDD-HHMMSS}" (UTC)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{algo.lower()}-fs{feature_schema_version}-{ts}"


def make_training_metadata(
    *,
    agent: Any,
    symbol: str,
    featured_data: Any,
    feature_schema_version: int,
    feature_columns: list[str],
    env_params: dict[str, Any],
    portfolio_state_fields: list[str] | None = None,
    normalization: dict[str, Any] | None = None,
    training_params: dict[str, Any] | None = None,
) -> ArtifactMetadata:
    """Build versioned artifact metadata from a completed training run."""
    portfolio_fields = portfolio_state_fields or list(DEFAULT_PORTFOLIO_STATE_FIELDS)
    timestamps = featured_data["Timestamp"]
    train_start = str(timestamps.min().date()) if not timestamps.empty else "unknown"
    train_end = str(timestamps.max().date()) if not timestamps.empty else "unknown"

    return ArtifactMetadata(
        artifact_format_version=2,
        artifact_id=make_artifact_id(agent.model_name, feature_schema_version),
        created_at=datetime.now(timezone.utc).isoformat(),
        algo=agent.model_name,
        policy=agent.policy,
        feature_schema_version=feature_schema_version,
        feature_columns=list(feature_columns),
        portfolio_state_fields=portfolio_fields,
        observation_dim=len(feature_columns) + len(portfolio_fields),
        action_space={
            "type": "discrete",
            "n": len(EXPECTED_ACTION_LABELS),
            "labels": list(EXPECTED_ACTION_LABELS),
        },
        normalization=normalization,
        train_git_sha=current_git_sha(),
        train_data={"symbols": [symbol], "start": train_start, "end": train_end},
        env_params=env_params,
        training_params=dict(training_params or {}),
    )


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


# v2 artifact를 실행 env에 연결할 때 observation 의미를 좌우하는 파라미터.
# config가 바뀌어도 학습 당시 값과 다른 env에 조용히 연결되는 것을 막는다
# (v1을 거부하는 이유와 동일한 입력 계약).
SEMANTIC_ENV_PARAMS = (
    "duration_horizon_bars",
    "unit_fraction",
    "max_units",
    "initial_cash",
    "nominal_bars_per_day",
)


def _check_env_compatibility(meta: ArtifactMetadata, env: Any) -> None:
    """v2 artifact와 실행 env의 observation 의미론 파라미터 일치 검증 (fail-closed).

    wrapper(gym.Wrapper 계열)는 .unwrapped로 실제 env를 찾고, 그래도 semantic
    파라미터를 노출하지 않으면 검증 불가로 거부한다 — 조용한 불일치 방지가 목적.
    """
    if env is None:
        return
    target = getattr(env, "unwrapped", env)

    env_schema_version = getattr(target, "feature_schema_version", None)
    if env_schema_version is None:
        raise ArtifactError(
            "env does not declare 'feature_schema_version'; cannot verify observation semantics"
        )
    if int(env_schema_version) != int(meta.feature_schema_version):
        raise ArtifactError(
            f"env feature_schema_version={env_schema_version!r} != artifact "
            f"feature_schema_version={meta.feature_schema_version!r}; "
            "feature 계산식이 다른 schema — observation semantics would silently differ"
        )

    for key in SEMANTIC_ENV_PARAMS:
        env_value = getattr(target, key, None)
        if env_value is None:
            raise ArtifactError(
                f"env does not expose {key!r}; cannot verify observation semantics"
            )
        meta_value = meta.env_params.get(key)
        if float(env_value) != float(meta_value):
            raise ArtifactError(
                f"env {key}={env_value!r} != artifact env_params {key}={meta_value!r}; "
                "observation semantics would silently differ"
            )

    env_feature_columns = getattr(target, "feature_columns", None)
    if env_feature_columns is None:
        raise ArtifactError(
            "env does not expose 'feature_columns'; cannot verify observation semantics"
        )
    if list(env_feature_columns) != list(meta.feature_columns):
        raise ArtifactError(
            f"env feature_columns {list(env_feature_columns)!r} != artifact "
            f"feature_columns {meta.feature_columns!r}; observation semantics would silently differ"
        )
    if meta.portfolio_state_fields != DEFAULT_PORTFOLIO_STATE_FIELDS:
        raise ArtifactError(
            f"artifact portfolio_state_fields {meta.portfolio_state_fields!r} != "
            f"environment layout {DEFAULT_PORTFOLIO_STATE_FIELDS!r}"
        )
    observation_space = getattr(target, "observation_space", None)
    if observation_space is not None and observation_space.shape[0] != meta.observation_dim:
        raise ArtifactError(
            f"env observation dim {observation_space.shape[0]} != "
            f"artifact observation_dim {meta.observation_dim}"
        )
    action_space = getattr(target, "action_space", None)
    if action_space is not None and getattr(action_space, "n", None) != meta.action_space.get("n"):
        raise ArtifactError(
            f"env action space n={getattr(action_space, 'n', None)!r} != "
            f"artifact action_space n={meta.action_space.get('n')!r}"
        )


def load_artifact(
    artifact_dir: "str | Path",
    env: Any = None,
    *,
    allow_legacy_semantics: bool = False,
) -> "tuple[Any, ArtifactMetadata]":
    """metadata 검증 통과 후에만 SB3 모델을 로드한다.

    v1 artifact는 기본 거부: 당일 기준 holding_duration_norm으로 학습된 모델이
    고정 horizon 기준 observation(v2 env)을 받으면 차원이 같아도 입력 계약
    위반이다. allow_legacy_semantics=True는 임시 진단 전용.
    """
    artifact_dir = Path(artifact_dir)
    meta = load_metadata(artifact_dir)
    meta_labels = meta.action_space.get("labels")
    if meta_labels != EXPECTED_ACTION_LABELS:
        raise ArtifactError(
            f"artifact action labels {meta_labels!r} != environment contract "
            f"{EXPECTED_ACTION_LABELS!r}; action semantics would silently differ"
        )
    if meta.artifact_format_version <= LEGACY_SEMANTICS_MAX_VERSION:
        if not allow_legacy_semantics:
            raise ArtifactError(
                "legacy artifact (format v1) uses day-based holding_duration_norm; "
                "observation semantics differ from the v2 environment. "
                "Pass allow_legacy_semantics=True to run anyway (diagnostics only)."
            )
        import warnings

        warnings.warn(
            "running legacy (v1) artifact against v2 observation semantics",
            UserWarning,
            stacklevel=2,
        )
    else:
        _check_env_compatibility(meta, env)
    if meta.algo not in {"PPO", "MaskablePPO"}:
        raise ArtifactError(f"unsupported algo: {meta.algo}")
    if (
        meta.normalization is not None
        and meta.normalization["type"] == "sb3_vecnormalize"
    ):
        # VecNormalize stats 적용은 미구현 — 조용히 스케일 어긋난 predict를 내느니 거부.
        raise ArtifactError(
            "normalization artifact is not supported by load_artifact yet; "
            "apply the stats per docs/observation-contract.md §5"
        )

    from models.rl_agent import RLAgent  # SB3 의존 경로 — 함수 내부 import 유지

    normalizer = None
    if meta.normalization is not None:
        from models.normalization import FeatureNormalizer

        try:
            normalizer = FeatureNormalizer.load(
                artifact_dir / meta.normalization["file"]
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"failed to load normalization stats: {exc}") from exc
        if list(normalizer.feature_columns) != meta.feature_columns:
            raise ArtifactError("normalization feature columns do not match metadata")

    agent = RLAgent(
        model_name=meta.algo,
        policy=meta.policy,
        observation_normalizer=normalizer,
    )
    agent.load(str(artifact_dir / MODEL_FILENAME), env=env)
    return agent, meta


def save_artifact(
    agent: Any,
    metadata: ArtifactMetadata,
    artifacts_dir: "str | Path",
    *,
    vecnormalize_path: "str | Path | None" = None,
    normalization_path: "str | Path | None" = None,
    normalization_payload: str | bytes | None = None,
) -> Path:
    """검증 → temp 디렉토리에 전부 기록 → 성공 시에만 최종 경로로 rename.

    실패 시 최종 artifact 경로에 partial 상태가 남지 않는다.
    """
    metadata.validate()
    if getattr(agent, "model", None) is None:
        raise ArtifactError("agent has no built model to save")

    norm = metadata.normalization
    if sum(
        value is not None
        for value in (vecnormalize_path, normalization_path, normalization_payload)
    ) > 1:
        raise ArtifactError("provide only one normalization stats source")
    stats_path = normalization_path or vecnormalize_path
    if norm is None and (stats_path is not None or normalization_payload is not None):
        raise ArtifactError("normalization stats given but metadata.normalization is null")
    if norm is not None:
        if stats_path is None and normalization_payload is None:
            raise ArtifactError(
                "metadata.normalization set but vecnormalize/normalization stats path missing"
            )
        if stats_path is not None and not Path(stats_path).is_file():
            raise ArtifactError(f"normalization file not found: {stats_path}")

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
            normalization_file = tmp_dir / norm["file"]
            if isinstance(normalization_payload, bytes):
                normalization_file.write_bytes(normalization_payload)
            elif isinstance(normalization_payload, str):
                normalization_file.write_text(normalization_payload, encoding="utf-8")
            else:
                shutil.copy2(stats_path, normalization_file)
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
