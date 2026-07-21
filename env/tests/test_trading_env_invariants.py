from __future__ import annotations

import numpy as np
import pandas as pd

from env.trading_env import TradingEnvironment


def _day(date_str: str, closes: list[float]) -> pd.DataFrame:
    ts = pd.date_range(f"{date_str} 09:00", periods=len(closes), freq="1min", tz="Asia/Seoul")
    return pd.DataFrame({"Timestamp": ts, "Close": np.array(closes),
                         "ExecPrice": np.array(closes, dtype=float), "ma_5": np.array(closes)})


def make_env(data: pd.DataFrame) -> TradingEnvironment:
    return TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
    )


def test_same_seed_date_same_trajectory() -> None:
    data = _day("2025-06-02", [100, 101, 102, 103, 104, 105])
    actions = [1, 0, 1, 0, 2, 0]

    def run() -> list[float]:
        env = make_env(data)
        env.reset(seed=7, options={"date": "2025-06-02"})
        out = []
        for a in actions:
            _, r, terminated, truncated, _ = env.step(a)
            out.append(r)
            if terminated or truncated:
                break
        return out

    assert run() == run()


def test_observation_does_not_leak_future() -> None:
    # current_step 이후의 Close를 바꿔도 현재 observation은 불변이어야 한다.
    base = _day("2025-06-02", [100, 200, 300, 400, 500])
    env_a = make_env(base.copy())
    obs_a, _ = env_a.reset(seed=0, options={"date": "2025-06-02"})

    altered = base.copy()
    altered.loc[2:, "Close"] = -999.0  # 미래 봉만 변조
    altered.loc[2:, "ma_5"] = base.loc[2:, "ma_5"].to_numpy()  # feature는 동일
    env_b = make_env(altered)
    obs_b, _ = env_b.reset(seed=0, options={"date": "2025-06-02"})

    assert np.array_equal(obs_a, obs_b)
