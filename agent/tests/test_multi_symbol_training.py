"""VecEnv 기반 multi-symbol 학습 테스트 (2종목 초소형 fixture)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.feature_engineer import FeatureEngineer
from models.normalization import FeatureNormalizer
from models.training import build_vec_training_environment, train_ppo_artifact

ENV_CFG = {
    "initial_cash": 10_000.0, "unit_fraction": 0.199, "max_units": 5,
    "risk_penalty_rate": 0.0, "episode_days": 1, "nominal_bars_per_day": 8,
}
FRICTION_CFG = {
    "fee_rate": 0.00018, "spread_rate": 0.001, "slippage_rate": 0.0,
    "execution_uncertainty_rate": 0.0, "sell_tax_rate": 0.002,
    "dynamic_spread": True, "date_based_sell_tax": True,
}


def _features(seed: int, days: int = 6, bars_per_day: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0 + seed
    day = pd.Timestamp("2026-01-05 09:00", tz="Asia/Seoul")
    for _ in range(days):
        for bar in range(bars_per_day):
            price *= float(np.exp(rng.normal(0, 0.001)))
            row = {"Timestamp": day + pd.Timedelta(minutes=5 * bar), "Close": price, "ExecPrice": price}
            for col in FeatureEngineer.FEATURE_COLUMNS:
                row[col] = float(rng.normal(0, 1))
            rows.append(row)
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "environment": dict(ENV_CFG),
        "friction": dict(FRICTION_CFG),
        "agent": {
            "rl_model_name": "MaskablePPO",
            "normalization": {"enabled": True, "clip": 5.0},
            "validation": {"enabled": False},
            "tensorboard": {"enabled": False},
            "ppo": {"n_steps": 16, "batch_size": 16, "n_epochs": 1, "verbose": 0},
        },
    }


BOUNDARIES = {"train_end_date": "2026-01-10", "validation_end_date": "2026-01-11", "purge_days": 0}


def test_vec_env_has_one_sub_env_per_symbol_and_forwards_masks():
    data = {"AAA": _features(1), "BBB": _features(2)}
    normalizer = FeatureNormalizer.fit(
        pd.concat(data.values(), ignore_index=True),
        list(FeatureEngineer.FEATURE_COLUMNS),
        clip=5.0,
    )
    vec_env = build_vec_training_environment(
        data, environment_config=ENV_CFG, friction_config=FRICTION_CFG, normalizer=normalizer,
    )
    assert vec_env.num_envs == 2
    vec_env.reset()
    masks = vec_env.env_method("action_masks")
    assert len(masks) == 2
    assert all(np.asarray(m).shape == (3,) for m in masks)


def test_normalizer_fit_uses_pooled_statistics():
    a, b = _features(1), _features(2)
    cols = list(FeatureEngineer.FEATURE_COLUMNS)
    combined = pd.concat([a, b], ignore_index=True)
    pooled = FeatureNormalizer.fit(combined, cols, clip=5.0)
    single = FeatureNormalizer.fit(a, cols, clip=5.0)
    # to_dict = {"feature_columns", "means", "scales", "clip"} (normalization.py 계약)
    assert pooled.to_dict()["means"] != single.to_dict()["means"]  # 한 종목 fit과 달라야 함
    assert pooled.to_dict()["means"][0] == pytest.approx(float(combined[cols[0]].mean()))


def test_train_ppo_artifact_multi_symbol_smoke(tmp_path):
    data = {"AAA": _features(1), "BBB": _features(2)}
    artifact_dir = train_ppo_artifact(
        featured_data=data,
        validation_data=None,
        config=_config(),
        total_timesteps=32,
        seed=7,
        artifacts_dir=tmp_path,
        trained_split="train",
        split_boundaries=dict(BOUNDARIES),
    )
    assert (artifact_dir / "metadata.json").is_file()
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    assert meta["artifact_format_version"] == 4
    assert meta["train_data"]["symbols"] == ["AAA", "BBB"]
    assert meta["train_data"]["split_boundaries"] == BOUNDARIES
    assert "-s2-" in meta["artifact_id"]


def test_training_with_validation_enabled_smoke(tmp_path):
    data = {"AAA": _features(1), "BBB": _features(2)}
    config = _config()
    config["agent"]["validation"] = {
        "enabled": True, "eval_freq": 16, "deterministic": True, "seed": 7, "verbose": 0,
    }
    artifact_dir = train_ppo_artifact(
        featured_data=data,
        validation_data={"AAA": _features(3), "BBB": _features(4)},
        config=config,
        total_timesteps=32,
        seed=7,
        artifacts_dir=tmp_path,
        trained_split="train",
        split_boundaries=dict(BOUNDARIES),
    )
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    validation_summary = meta["training_params"]["validation"]
    assert validation_summary["metric"] == "mean_total_return"
    assert set(validation_summary["best"]["per_symbol"]) == {"AAA", "BBB"}


def test_training_is_deterministic_for_same_seed(tmp_path):
    def _run(subdir: str):
        data = {"AAA": _features(1), "BBB": _features(2)}
        return train_ppo_artifact(
            featured_data=data,
            validation_data=None,
            config=_config(),
            total_timesteps=32,
            seed=7,
            artifacts_dir=tmp_path / subdir,
            trained_split="train",
            split_boundaries=dict(BOUNDARIES),
        )

    from sb3_contrib import MaskablePPO
    first = MaskablePPO.load(_run("a") / "model.zip")
    second = MaskablePPO.load(_run("b") / "model.zip")
    for (k1, p1), (k2, p2) in zip(
        first.policy.state_dict().items(), second.policy.state_dict().items()
    ):
        assert k1 == k2
        assert np.allclose(p1.cpu().numpy(), p2.cpu().numpy()), f"parameter mismatch: {k1}"
