"""spec §4.3: 같은 데이터·artifact에서 env와 서빙 경로의 obs·mask·action 완전 일치.

as_of = 5분 bar 라벨(label=right → bucket 종료 시각) 그 자체:
builder의 완료 분봉 cutoff(1분봉 라벨+1분 <= as_of)가 정확히 그 bar까지의
분봉만 남기므로, 전체(미래 포함) minute_data를 그대로 넘겨도 서빙의
'마지막 완료 bar' == env의 현재 bar가 된다.
"""
import numpy as np
import pandas as pd
import pytest

from data.feature_engineer import FeatureEngineer
from env.trading_env import TradingEnvironment
from friction.friction_model import FrictionModel
from observation_builder import build_decision_inputs
from predictor import Predictor


@pytest.fixture(scope="module")
def predictor(tiny_artifact_dir):
    return Predictor.load(tiny_artifact_dir)


def test_replay_parity_full_day(minute_data, predictor):
    meta = predictor.meta
    fe = FeatureEngineer()
    featured = fe.transform(minute_data)

    env = TradingEnvironment(
        market_data=featured,
        feature_columns=list(meta.feature_columns),
        initial_cash=meta.env_params["initial_cash"],
        unit_fraction=meta.env_params["unit_fraction"],
        max_units=meta.env_params["max_units"],
        friction_model=FrictionModel(**meta.friction_params),
        episode_days=1,
        duration_horizon_bars=meta.env_params["duration_horizon_bars"],
        nominal_bars_per_day=meta.env_params["nominal_bars_per_day"],
        feature_schema_version=meta.feature_schema_version,
    )
    target_date = env.available_dates[-1]
    env_obs, _ = env.reset(options={"start_date": str(target_date), "episode_days": 1})

    steps = 0
    while True:
        row = env.get_current_market_row()
        bar_ts = pd.Timestamp(row["Timestamp"])
        serving = build_decision_inputs(
            bars_1m=minute_data,
            as_of=bar_ts,  # bar 라벨 = 완료 시각
            units_held=env.units_held,
            shares_held=env.shares_held,
            bars_since_entry=(
                env.current_step - env.entry_step
                if env.units_held > 0 and env.entry_step is not None else 0
            ),
            available_cash=env.cash,
            env_params=meta.env_params,
            friction_model=predictor.friction_model,
            max_bar_age=pd.Timedelta(minutes=10),
            feature_engineer=fe,
        )
        assert serving.bar_ts == bar_ts
        np.testing.assert_array_equal(serving.observation, env_obs)
        np.testing.assert_array_equal(serving.action_mask, env.action_masks())

        action = predictor.predict(serving.observation, serving.action_mask)
        env_action, _ = predictor._agent.predict(
            env_obs, deterministic=True, action_masks=env.action_masks())
        assert action == int(env_action)

        env_obs, _, _, truncated, _ = env.step(action)
        steps += 1
        if truncated:
            break

    assert steps >= 10  # 하루치가 실제로 재생됐는지 방어
