from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.trading_env import TradingEnvironment


def _make_env() -> TradingEnvironment:
    frames = []
    for day in ("2025-06-02", "2025-06-03"):
        ts = pd.date_range(f"{day} 09:00", periods=3, freq="5min", tz="Asia/Seoul")
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Close": np.full(3, 100.0),
            "ma_5": np.full(3, 100.0),
        }))
    return TradingEnvironment(
        market_data=pd.concat(frames, ignore_index=True),
        feature_columns=["ma_5"],
        episode_days=2,
    )


def test_dummy_vec_env_reports_truncation_for_bootstrap() -> None:
    vec = DummyVecEnv([_make_env])
    vec.reset()
    done = np.array([False])
    infos = [{}]
    for _ in range(5):  # 6 bars → 5 steps
        _, _, done, infos = vec.step(np.array([0]))
    assert done[0]
    assert infos[0].get("TimeLimit.truncated") is True
    assert "terminal_observation" in infos[0]


def test_ppo_smoke_collects_rollout_across_truncation() -> None:
    model = PPO(
        "MlpPolicy",
        DummyVecEnv([_make_env]),
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        seed=0,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=32)  # truncation 경계 여러 번 포함, 예외 없이 완료
