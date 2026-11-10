from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.run_ablation import (synthetic_data, make_env, save_research_bundle,
                                      load_research_bundle, evaluate)
from models.normalization import FeatureNormalizer
from data.feature_engineer import FeatureEngineer
from utils.config_loader import load_config
from models.artifact import load_metadata, ArtifactError
from sb3_contrib import MaskablePPO


@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_research_save_load_and_training_contract(tmp_path, condition):
    torch.set_num_threads(1)
    data = synthetic_data(days=2)[0]["SYNTHETIC"]
    config = load_config(ROOT / "env/configs/config.yaml")
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    env = make_env(data, config, condition, stats)
    model = MaskablePPO("MlpPolicy", env, n_steps=16, batch_size=16,
                        n_epochs=1, seed=1, verbose=0, policy_kwargs={"net_arch": [16,16]})
    model.learn(32)
    expected, _ = evaluate(model, env)
    save_research_bundle(model, stats, env, config, tmp_path,
                         {"condition": condition})
    restored, restored_env, _ = load_research_bundle(tmp_path, data)
    actual, _ = evaluate(restored, restored_env)
    assert actual == expected
    # Research contracts cannot accidentally enter production serving.
    with pytest.raises(ArtifactError):
        load_metadata(tmp_path)
    with (tmp_path / "research_model.zip").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_research_bundle(tmp_path, data)


def test_normalization_fit_does_not_depend_on_validation_values():
    data = synthetic_data(days=2)[0]["SYNTHETIC"]
    train = data.iloc[:32].copy()
    stats = FeatureNormalizer.fit(train, list(FeatureEngineer.FEATURE_COLUMNS))
    original = stats.to_dict()
    data.loc[32:, list(FeatureEngineer.FEATURE_COLUMNS)] = 1e9
    assert FeatureNormalizer.fit(train, list(FeatureEngineer.FEATURE_COLUMNS)).to_dict() == original


def test_future_execution_price_is_not_part_of_observation_or_mask():
    data = synthetic_data(days=2)[0]["SYNTHETIC"]
    config = load_config(ROOT / "env/configs/config.yaml")
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    for condition in ("A", "B", "C", "D"):
        first = make_env(data, config, condition, stats)
        altered = data.copy()
        altered["ExecPrice"] *= 10
        second = make_env(altered, config, condition, stats)
        obs, _ = first.reset(seed=1)
        changed, _ = second.reset(seed=1)
        np.testing.assert_array_equal(obs, changed)
        np.testing.assert_array_equal(first.action_masks(), second.action_masks())


def test_runner_rejects_low_coverage_before_training(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import experiments.run_ablation as runner
    frames, calendar, provenance = synthetic_data()
    bad_days = set(calendar[55:60])  # first fold's validation segment
    frame = frames["SYNTHETIC"]
    frames["SYNTHETIC"] = frame.loc[~frame.Timestamp.dt.date.isin(bad_days)]
    args = SimpleNamespace(output=tmp_path / "study", symbols="SYNTHETIC", conditions="A",
        seeds="42", folds=1, validation_days=20, test_days=20, timesteps=32,
        n_steps=16, eval_freq=32, parameter_budget=40000, device="cpu",
        synthetic=False, plan_only=False, evaluate_test=False)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "load_market", lambda symbols: (frames, calendar, provenance))
    def unexpected_training(*args, **kwargs):
        pytest.fail("training must not start with failed coverage")
    monkeypatch.setattr(runner, "MaskablePPO", unexpected_training)
    with pytest.raises(SystemExit, match="coverage below"):
        runner.main()
    assert not list(args.output.rglob("research_model.zip"))
