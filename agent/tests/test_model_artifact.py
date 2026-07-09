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
from models.artifact import ArtifactError, ArtifactMetadata, current_git_sha, load_artifact, load_metadata, make_artifact_id, save_artifact
from models.rl_agent import RLAgent

OBS_DIM = 13


class DummyTradingEnv(gym.Env):
    """PPO build용 초소형 env. 학습하지 않으므로 dynamics는 무의미."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
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
        action_space={"type": "discrete", "n": 3, "labels": ["hold", "add_unit", "clear"]},
        normalization=None,
        train_git_sha="testsha",
        train_data={"symbols": ["005930"], "start": "2025-05-22", "end": "2026-06-30"},
        env_params={"unit_fraction": 0.2, "max_units": 5, "initial_cash": 100_000_000},
    )
    base.update(overrides)
    return ArtifactMetadata(**base)


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

        loaded_agent, loaded_meta = load_artifact(out)
        assert loaded_meta == meta

        obs = np.arange(OBS_DIM, dtype=np.float32) / OBS_DIM
        original_action, _ = built_agent.predict(obs, deterministic=True)
        loaded_action, _ = loaded_agent.predict(obs, deterministic=True)
        assert loaded_action == original_action

    def test_non_ppo_algo_rejected(self, tmp_path):
        art = _write_fake_artifact(tmp_path, make_metadata(algo="DQN").to_dict())
        with pytest.raises(ArtifactError, match="DQN"):
            load_artifact(art)

    def test_invalid_artifact_rejected_before_sb3_load(self, tmp_path):
        with pytest.raises(ArtifactError, match="metadata.json"):
            load_artifact(tmp_path / "nonexistent")
