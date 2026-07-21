"""Model artifact 저장/로드 계약 테스트 (spec: 2026-07-09-model-artifact-contract)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from data.feature_engineer import FeatureEngineer
from models.artifact import (
    EXPECTED_ACTION_LABELS,
    ArtifactError,
    ArtifactMetadata,
    current_git_sha,
    load_artifact,
    load_metadata,
    make_artifact_id,
    save_artifact,
)
from models.rl_agent import RLAgent

OBS_DIM = 13


class DummyTradingEnv(gym.Env):
    """PPO build용 초소형 env. 학습하지 않으므로 dynamics는 무의미."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self._t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, self._t >= 5, False, {}


@pytest.fixture(scope="module")
def built_agent():
    agent = RLAgent(model_kwargs={"seed": 0})
    agent.build(DummyTradingEnv())
    return agent


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
        action_space={
            "type": "discrete",
            "n": len(EXPECTED_ACTION_LABELS),
            "labels": list(EXPECTED_ACTION_LABELS),
        },
        normalization=None,
        train_git_sha="testsha",
        train_data={"symbols": ["005930"], "start": "2025-05-22", "end": "2026-06-30"},
        env_params={"unit_fraction": 0.2, "max_units": 5, "initial_cash": 100_000_000},
    )
    base.update(overrides)
    return ArtifactMetadata(**base)


@pytest.fixture
def valid_metadata_dict() -> dict:
    """v1 valid metadata dict (normalization=None) — v2/legacy-gate 테스트가 공유."""
    return make_metadata().to_dict()


def _write_fake_artifact(root, meta_dict, with_model=True):
    """SB3 없이 만드는 가짜 artifact (load_metadata 단독 테스트용)."""
    art = root / meta_dict["artifact_id"]
    art.mkdir(parents=True)
    (art / "metadata.json").write_text(
        json.dumps(meta_dict, ensure_ascii=False), encoding="utf-8"
    )
    if with_model:
        (art / "model.zip").write_bytes(b"fake")
    return art


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
        data["action_space"] = {
            "type": "discrete",
            "n": 7,
            "labels": list(EXPECTED_ACTION_LABELS),
        }
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


class TestCurrentGitSha:
    def test_returns_nonempty(self):
        sha = current_git_sha()
        assert isinstance(sha, str) and sha

    def test_cwd_independent(self, tmp_path, monkeypatch):
        # workspace root 등 git repo 밖에서 실행돼도 모듈 위치 기준으로 repo를 찾는다.
        monkeypatch.chdir(tmp_path)
        assert current_git_sha() != "unknown"


class TestSaveArtifact:
    def test_save_creates_versioned_dir(self, built_agent, tmp_path):
        meta = make_metadata()
        out = save_artifact(built_agent, meta, tmp_path)
        assert out == tmp_path / meta.artifact_id
        assert (out / "model.zip").is_file()
        saved = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert saved == meta.to_dict()

    def test_duplicate_id_rejected(self, built_agent, tmp_path):
        meta = make_metadata()
        save_artifact(built_agent, meta, tmp_path)
        with pytest.raises(ArtifactError, match="exists"):
            save_artifact(built_agent, meta, tmp_path)

    def test_unbuilt_agent_rejected(self, tmp_path):
        with pytest.raises(ArtifactError, match="model"):
            save_artifact(RLAgent(), make_metadata(), tmp_path)

    def test_normalization_block_without_file_rejected(self, built_agent, tmp_path):
        meta = make_metadata(
            normalization={"type": "sb3_vecnormalize", "file": "vecnormalize.pkl"}
        )
        with pytest.raises(ArtifactError, match="vecnormalize"):
            save_artifact(built_agent, meta, tmp_path)

    def test_vecnormalize_file_copied(self, built_agent, tmp_path):
        pkl = tmp_path / "stats.pkl"
        pkl.write_bytes(b"fake-stats")
        meta = make_metadata(
            normalization={"type": "sb3_vecnormalize", "file": "vecnormalize.pkl"}
        )
        out = save_artifact(built_agent, meta, tmp_path / "artifacts", vecnormalize_path=pkl)
        assert (out / "vecnormalize.pkl").read_bytes() == b"fake-stats"

    def test_vecnormalize_path_without_block_rejected(self, built_agent, tmp_path):
        pkl = tmp_path / "stats.pkl"
        pkl.write_bytes(b"fake-stats")
        with pytest.raises(ArtifactError, match="null"):
            save_artifact(built_agent, make_metadata(), tmp_path, vecnormalize_path=pkl)

    def test_midwrite_failure_leaves_nothing(self, built_agent, tmp_path):
        # json 직렬화 불가 객체로 metadata.json 기록 단계 실패 유도
        # (validate()는 train_data 내용을 보지 않으므로 통과함)
        meta = make_metadata(train_data={"symbols": object()})
        with pytest.raises(ArtifactError):
            save_artifact(built_agent, meta, tmp_path)
        assert not (tmp_path / meta.artifact_id).exists()
        assert list(tmp_path.glob(".tmp-*")) == []

    def test_make_artifact_id_format(self):
        aid = make_artifact_id("PPO", 2)
        assert aid.startswith("ppo-fs2-")


class TestLoadMetadata:
    def test_load_valid(self, tmp_path):
        art = _write_fake_artifact(tmp_path, make_metadata().to_dict())
        meta = load_metadata(art)
        assert meta == make_metadata()

    def test_missing_metadata_json(self, tmp_path):
        (tmp_path / "empty-art").mkdir()
        with pytest.raises(ArtifactError, match="metadata.json"):
            load_metadata(tmp_path / "empty-art")

    def test_corrupt_json(self, tmp_path):
        art = tmp_path / "bad"
        art.mkdir()
        (art / "metadata.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="parse"):
            load_metadata(art)

    def test_missing_model_zip(self, tmp_path):
        art = _write_fake_artifact(tmp_path, make_metadata().to_dict(), with_model=False)
        with pytest.raises(ArtifactError, match="model.zip"):
            load_metadata(art)

    def test_missing_normalization_file(self, tmp_path):
        meta = make_metadata(
            normalization={"type": "sb3_vecnormalize", "file": "vecnormalize.pkl"}
        )
        art = _write_fake_artifact(tmp_path, meta.to_dict())
        with pytest.raises(ArtifactError, match="vecnormalize.pkl"):
            load_metadata(art)

    def test_no_sb3_in_fresh_subprocess(self, tmp_path):
        # 같은 프로세스 검사는 앞선 PPO 테스트가 SB3를 이미 로딩해 순서 의존적으로
        # 거짓 실패한다. fresh subprocess에서 load_metadata만 실행해 확인한다.
        art = _write_fake_artifact(tmp_path, make_metadata().to_dict())
        root = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(root / 'agent' / 'src')!r})\n"
            f"from models.artifact import load_metadata\n"
            f"meta = load_metadata({str(art)!r})\n"
            "assert meta.algo == 'PPO'\n"
            "assert 'stable_baselines3' not in sys.modules, 'SB3 imported!'\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout


class TestLoadArtifact:
    def test_roundtrip_predict_identical(self, built_agent, tmp_path):
        meta = make_metadata()
        out = save_artifact(built_agent, meta, tmp_path)

        with pytest.warns(UserWarning, match="legacy"):
            loaded_agent, loaded_meta = load_artifact(out, allow_legacy_semantics=True)
        assert loaded_meta == meta

        obs = np.arange(OBS_DIM, dtype=np.float32) / OBS_DIM
        original_action, _ = built_agent.predict(obs, deterministic=True)
        loaded_action, _ = loaded_agent.predict(obs, deterministic=True)
        assert loaded_action == original_action

    def test_non_ppo_algo_rejected(self, tmp_path):
        art = _write_fake_artifact(tmp_path, make_metadata(algo="DQN").to_dict())
        with pytest.warns(UserWarning, match="legacy"), pytest.raises(ArtifactError, match="DQN"):
            load_artifact(art, allow_legacy_semantics=True)

    def test_invalid_artifact_rejected_before_sb3_load(self, tmp_path):
        with pytest.raises(ArtifactError, match="metadata.json"):
            load_artifact(tmp_path / "nonexistent")


class TestMetadataHardening:
    """P1/P2 방어: 경로 탈출 및 타입 위조 metadata 거부."""

    def test_normalization_file_path_traversal_rejected(self):
        data = make_metadata().to_dict()
        data["normalization"] = {"type": "sb3_vecnormalize", "file": "../escaped.pkl"}
        with pytest.raises(ArtifactError, match="normalization"):
            ArtifactMetadata.from_dict(data)

    def test_normalization_file_absolute_path_rejected(self):
        data = make_metadata().to_dict()
        data["normalization"] = {"type": "sb3_vecnormalize", "file": "/tmp/escaped.pkl"}
        with pytest.raises(ArtifactError, match="normalization"):
            ArtifactMetadata.from_dict(data)

    def test_artifact_id_with_path_separator_rejected(self):
        data = make_metadata(artifact_id="../evil").to_dict()
        with pytest.raises(ArtifactError, match="artifact_id"):
            ArtifactMetadata.from_dict(data)

    def test_string_feature_columns_rejected(self):
        # len("abcdefghi") + len("abcd") == 13이라 dim 검사만으로는 통과하는 위조
        data = make_metadata().to_dict()
        data["feature_columns"] = "abcdefghi"
        data["portfolio_state_fields"] = "abcd"
        with pytest.raises(ArtifactError, match="feature_columns"):
            ArtifactMetadata.from_dict(data)

    def test_non_string_column_element_rejected(self):
        data = make_metadata().to_dict()
        data["feature_columns"] = list(range(9))
        with pytest.raises(ArtifactError, match="feature_columns"):
            ArtifactMetadata.from_dict(data)

    def test_non_string_action_labels_rejected(self):
        data = make_metadata().to_dict()
        data["action_space"] = {"type": "discrete", "n": 3, "labels": [0, 1, 2]}
        with pytest.raises(ArtifactError, match="action_space"):
            ArtifactMetadata.from_dict(data)

    def test_save_rejects_normalization_escape(self, built_agent, tmp_path):
        pkl = tmp_path / "stats.pkl"
        pkl.write_bytes(b"fake-stats")
        meta = make_metadata(
            normalization={"type": "sb3_vecnormalize", "file": "../escaped.pkl"}
        )
        with pytest.raises(ArtifactError):
            save_artifact(built_agent, meta, tmp_path / "artifacts", vecnormalize_path=pkl)
        assert not (tmp_path / "escaped.pkl").exists()

    def test_from_dict_non_mapping_rejected(self):
        with pytest.raises(ArtifactError, match="JSON object"):
            ArtifactMetadata.from_dict(42)

    def test_load_metadata_non_object_json_rejected(self, tmp_path):
        art = tmp_path / "scalar"
        art.mkdir()
        (art / "metadata.json").write_text("42", encoding="utf-8")
        (art / "model.zip").write_bytes(b"fake")
        with pytest.raises(ArtifactError, match="JSON object"):
            load_metadata(art)

    def test_env_params_non_dict_rejected(self):
        data = make_metadata().to_dict()
        data["env_params"] = "not-a-dict"
        with pytest.raises(ArtifactError, match="env_params"):
            ArtifactMetadata.from_dict(data)

    def test_env_params_missing_keys_rejected(self):
        data = make_metadata().to_dict()
        data["env_params"] = {"unit_fraction": 0.2}
        with pytest.raises(ArtifactError, match="env_params"):
            ArtifactMetadata.from_dict(data)

    def test_env_params_non_numeric_rejected(self):
        data = make_metadata().to_dict()
        data["env_params"] = {"unit_fraction": "0.2", "max_units": 5, "initial_cash": 100_000_000}
        with pytest.raises(ArtifactError, match="env_params"):
            ArtifactMetadata.from_dict(data)

    def test_load_artifact_rejects_normalization_artifact(self, tmp_path):
        # VecNormalize 적용은 미지원 — 조용히 스케일 어긋난 predict를 내느니 로드를 거부한다.
        meta = make_metadata(
            normalization={"type": "sb3_vecnormalize", "file": "vecnormalize.pkl"}
        )
        art = _write_fake_artifact(tmp_path, meta.to_dict())
        (art / "vecnormalize.pkl").write_bytes(b"fake-stats")
        with pytest.warns(UserWarning, match="legacy"), pytest.raises(ArtifactError, match="normalization"):
            load_artifact(art, allow_legacy_semantics=True)

    def test_normalization_file_reserved_name_rejected(self):
        # "model.zip"이면 모델이 stats로 덮어써지고, "metadata.json"이면 stats가 유실된다.
        for reserved in ("model.zip", "metadata.json"):
            data = make_metadata().to_dict()
            data["normalization"] = {"type": "sb3_vecnormalize", "file": reserved}
            with pytest.raises(ArtifactError, match="reserved"):
                ArtifactMetadata.from_dict(data)


def _v2_env_params() -> dict:
    return {
        "unit_fraction": 0.2,
        "max_units": 5,
        "initial_cash": 10_000.0,
        "episode_days": 20,
        "duration_horizon_bars": 1280,
        "nominal_bars_per_day": 64,
    }


def test_v2_metadata_requires_new_env_params(valid_metadata_dict) -> None:
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    with pytest.raises(ArtifactError, match="env_params"):
        ArtifactMetadata.from_dict(data)  # v1 키만 있음 → 거부
    data["env_params"] = _v2_env_params()
    ArtifactMetadata.from_dict(data)  # 통과


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_v2_positive_int_validation(valid_metadata_dict, bad) -> None:
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    params = _v2_env_params()
    params["episode_days"] = bad
    data["env_params"] = params
    with pytest.raises(ArtifactError):
        ArtifactMetadata.from_dict(data)


def test_v1_metadata_still_loads_without_new_keys(valid_metadata_dict) -> None:
    ArtifactMetadata.from_dict(dict(valid_metadata_dict))  # v1 회귀 금지


def test_load_artifact_rejects_v1_by_default(tmp_path, valid_metadata_dict) -> None:
    artifact_dir = tmp_path / "legacy-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(valid_metadata_dict), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="legacy"):
        load_artifact(artifact_dir)


def test_load_artifact_v1_opt_in_passes_gate(tmp_path, valid_metadata_dict, monkeypatch) -> None:
    artifact_dir = tmp_path / "legacy-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(valid_metadata_dict), encoding="utf-8"
    )

    class StubAgent:
        def __init__(self, **kwargs):
            self.loaded = None

        def load(self, path, env=None):
            self.loaded = path

    import models.rl_agent as rl_agent_module
    monkeypatch.setattr(rl_agent_module, "RLAgent", StubAgent)
    with pytest.warns(UserWarning, match="legacy"):
        agent, meta = load_artifact(artifact_dir, allow_legacy_semantics=True)
    assert meta.artifact_format_version == 1


def test_load_artifact_v2_rejects_mismatched_env_semantics(
    tmp_path, valid_metadata_dict
) -> None:
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    data["env_params"] = _v2_env_params()  # duration_horizon_bars=1280
    artifact_dir = tmp_path / "v2-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")

    class MismatchedEnv:
        duration_horizon_bars = 640  # 학습 당시 1280과 불일치
        unit_fraction = 0.2
        max_units = 5
        initial_cash = 10_000.0
        feature_columns = list(FeatureEngineer.FEATURE_COLUMNS)
        nominal_bars_per_day = 64
        feature_schema_version = FeatureEngineer.FEATURE_SCHEMA_VERSION

    with pytest.raises(ArtifactError, match="duration_horizon_bars"):
        load_artifact(artifact_dir, env=MismatchedEnv())


def _matching_v2_artifact(tmp_path, valid_metadata_dict) -> Path:
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    data["env_params"] = _v2_env_params()
    artifact_dir = tmp_path / "v2-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    return artifact_dir


def test_load_artifact_v2_rejects_mismatched_feature_columns(
    tmp_path, valid_metadata_dict
) -> None:
    """feature_columns 이름/순서가 다르면 차원(개수)이 같아도 거부해야 한다 —
    observation semantics가 조용히 달라지는 것을 막는 fail-closed 계약."""
    artifact_dir = _matching_v2_artifact(tmp_path, valid_metadata_dict)
    mismatched_columns = list(FeatureEngineer.FEATURE_COLUMNS)
    mismatched_columns[0], mismatched_columns[1] = mismatched_columns[1], mismatched_columns[0]

    class MismatchedFeaturesEnv:
        unit_fraction = 0.2
        max_units = 5
        initial_cash = 10_000.0
        episode_days = 20
        duration_horizon_bars = 1280
        nominal_bars_per_day = 64
        feature_columns = mismatched_columns
        feature_schema_version = FeatureEngineer.FEATURE_SCHEMA_VERSION

    with pytest.raises(ArtifactError, match="feature_columns"):
        load_artifact(artifact_dir, env=MismatchedFeaturesEnv())


def test_load_artifact_v2_rejects_env_without_feature_columns(
    tmp_path, valid_metadata_dict
) -> None:
    """env가 feature_columns를 노출하지 않으면 검증 불가 → 거부(fail-closed)."""
    artifact_dir = _matching_v2_artifact(tmp_path, valid_metadata_dict)

    class NoFeatureColumnsEnv:
        unit_fraction = 0.2
        max_units = 5
        initial_cash = 10_000.0
        episode_days = 20
        duration_horizon_bars = 1280
        nominal_bars_per_day = 64
        feature_schema_version = FeatureEngineer.FEATURE_SCHEMA_VERSION

    with pytest.raises(ArtifactError, match="feature_columns"):
        load_artifact(artifact_dir, env=NoFeatureColumnsEnv())


def test_load_artifact_v2_rejects_mismatched_feature_schema_version(
    tmp_path, valid_metadata_dict
) -> None:
    """feature_columns 이름/순서가 같아도 계산식이 바뀐 schema(version 999)면 거부해야 한다."""
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    data["env_params"] = _v2_env_params()
    data["feature_schema_version"] = 999
    artifact_dir = tmp_path / "v2-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")

    class SchemaMismatchEnv:
        unit_fraction = 0.2
        max_units = 5
        initial_cash = 10_000.0
        episode_days = 20
        duration_horizon_bars = 1280
        nominal_bars_per_day = 64
        feature_columns = list(FeatureEngineer.FEATURE_COLUMNS)
        feature_schema_version = FeatureEngineer.FEATURE_SCHEMA_VERSION

    with pytest.raises(ArtifactError, match="feature_schema_version"):
        load_artifact(artifact_dir, env=SchemaMismatchEnv())


def test_load_artifact_v2_rejects_env_without_feature_schema_version(
    tmp_path, valid_metadata_dict
) -> None:
    """env가 feature_schema_version을 노출하지 않으면 검증 불가 → 거부(fail-closed)."""
    artifact_dir = _matching_v2_artifact(tmp_path, valid_metadata_dict)

    class NoSchemaVersionEnv:
        unit_fraction = 0.2
        max_units = 5
        initial_cash = 10_000.0
        episode_days = 20
        duration_horizon_bars = 1280
        nominal_bars_per_day = 64
        feature_columns = list(FeatureEngineer.FEATURE_COLUMNS)

    with pytest.raises(ArtifactError, match="feature_schema_version"):
        load_artifact(artifact_dir, env=NoSchemaVersionEnv())


def test_load_artifact_v2_rejects_reversed_action_labels(
    tmp_path, valid_metadata_dict
) -> None:
    """labels 순서가 뒤집히면(n은 동일) trading_env의 action 계약과 어긋난다 —
    조용한 오거래(model action 0 != env Hold)를 막는 fail-closed 계약."""
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    data["env_params"] = _v2_env_params()
    data["action_space"] = {
        "type": "discrete",
        "n": 6,
        "labels": list(reversed(EXPECTED_ACTION_LABELS)),
    }
    artifact_dir = tmp_path / "v2-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")

    class MatchingEnv:
        unit_fraction = 0.2
        max_units = 5
        initial_cash = 10_000.0
        episode_days = 20
        duration_horizon_bars = 1280
        nominal_bars_per_day = 64
        feature_columns = list(FeatureEngineer.FEATURE_COLUMNS)
        feature_schema_version = FeatureEngineer.FEATURE_SCHEMA_VERSION

        class action_space:
            n = 6

    with pytest.raises(ArtifactError, match="labels"):
        load_artifact(artifact_dir, env=MatchingEnv())


def test_load_artifact_rejects_reversed_action_labels_without_env(
    tmp_path, valid_metadata_dict
) -> None:
    """env=None로 로드해도 labels 계약은 env 속성과 무관하게 검증돼야 한다 —
    _check_env_compatibility()는 env is None이면 즉시 return하므로 이 검사를
    우회해선 안 된다(labels는 고정 artifact 계약이지 env 속성 비교가 아님)."""
    data = dict(valid_metadata_dict)
    data["artifact_format_version"] = 2
    data["env_params"] = _v2_env_params()
    data["action_space"] = {
        "type": "discrete",
        "n": 6,
        "labels": list(reversed(EXPECTED_ACTION_LABELS)),
    }
    artifact_dir = tmp_path / "v2-artifact-no-env"
    artifact_dir.mkdir()
    (artifact_dir / "model.zip").write_bytes(b"")
    (artifact_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ArtifactError, match="labels"):
        load_artifact(artifact_dir)
