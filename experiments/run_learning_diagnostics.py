"""Train on easy causal rising/falling/flat markets; emit raw diagnostics only."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(source))

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.logger import configure

from experiments.run_ablation import (write_json, code_snapshot, make_env, evaluate,
    save_research_bundle, load_research_bundle, evaluation_config, archive_source)
from models.constrained_ppo import ResearchPPO, RiskBudgetCallback, DurationAwareRolloutBuffer, initialize_cash_prior
from models.normalization import FeatureNormalizer
from data.feature_engineer import FeatureEngineer
from utils.config_loader import load_config


def easy_market(regime, seed, *, start="2025-01-02", days=20):
    if regime not in ("rising", "falling", "flat"):
        raise ValueError("unknown diagnostic regime")
    rng = np.random.default_rng(seed)
    timestamps = [t for day in pd.bdate_range(start, periods=days, tz="Asia/Seoul")
        for t in pd.date_range(day + pd.Timedelta(hours=10, minutes=5), periods=32, freq="5min")]
    drift = {"rising": .002, "falling": -.002, "flat": 0.}[regime]
    changes = (drift + rng.normal(0, .00005, len(timestamps) - 1)
               if regime != "flat" else np.zeros(len(timestamps) - 1))
    prices = 10000 * np.exp(np.r_[0, np.cumsum(changes)])
    frame = pd.DataFrame({"Timestamp": timestamps, "Close": prices,
                          "ExecPrice": prices.copy(), "Adv20": 1e11})
    for column in FeatureEngineer.FEATURE_COLUMNS:
        frame[column] = 0.
    # Only a backward difference of observed prices, never the next return or
    # a regime label. Each regime is learned separately as a capability check.
    frame[FeatureEngineer.FEATURE_COLUMNS[0]] = np.r_[0, changes]
    return frame


def run(output, seeds, bar_budget, *, gamma=.99, gae_lambda=.95, ent_coef=.001, episode_days=20,
        regimes=("rising", "falling", "flat"), cash_logit_bias=0.):
    output = Path(output)
    if output.exists():
        raise ValueError("diagnostic output already exists")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    config = load_config(ROOT / "env/configs/config.yaml")
    config["agent"]["ppo"].update(gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef)
    config["environment"]["episode_days"] = episode_days
    config["research"] = {"cash_logit_bias": cash_logit_bias}
    write_json(output / "manifest.json", {"code": code_snapshot(), "seeds": seeds,
        "source_archive_sha256": archive_source(output),
        "bar_budget": bar_budget, "condition": "R", "config": config,
        "regimes": list(regimes),
        "purpose": "easy market capability checks; no market performance evidence",
        "selection": "last trained parameters, no validation checkpoint selection",
        "checks": {"rising": "net_return > 0 and mean_exposure >= .5",
                   "falling": "net_return >= -.01 and mean_exposure <= .2",
                   "flat": "net_return >= -.01 and mean_exposure <= .2"}})
    results = []
    for regime in regimes:
        for seed in seeds:
            directory = output / regime / f"seed_{seed}"
            directory.mkdir(parents=True)
            train = easy_market(regime, seed)
            test = easy_market(regime, seed + 10000, start="2025-04-01")
            normalizer = FeatureNormalizer.fit(train, list(FeatureEngineer.FEATURE_COLUMNS))
            env = make_env(train, config, "R", normalizer)
            ppo = copy.deepcopy(config["agent"]["ppo"])
            ppo.update(n_steps=256, batch_size=256, verbose=0,
                       policy_kwargs={"net_arch": [64, 64]}, rollout_buffer_class=DurationAwareRolloutBuffer)
            model = ResearchPPO("MlpPolicy", env, seed=seed, device="cpu", **ppo)
            initialize_cash_prior(model, cash_logit_bias)
            model.research_bar_budget = bar_budget
            model.set_logger(configure(str(directory), ["csv"]))
            initial, _ = evaluate(model, make_env(test, config, "R", normalizer))
            callback = RiskBudgetCallback()
            model.learn(bar_budget, callback=callback)
            frozen = evaluation_config(config, model)
            evaluation_env = make_env(test, frozen, "R", normalizer)
            metrics, trace = evaluate(model, evaluation_env)
            trace.to_csv(directory / "diagnostic_trace.csv", index=False)
            save_research_bundle(model, normalizer, evaluation_env, frozen, directory,
                {"condition": "R", "regime": regime, "seed": seed, "synthetic": True})
            restored, restored_env, _ = load_research_bundle(directory, test)
            if evaluate(restored, restored_env)[0] != metrics:
                raise RuntimeError("diagnostic save/load mismatch")
            passed = (metrics["net_return"] > 0 and metrics["mean_exposure"] >= .5
                if regime == "rising" else metrics["net_return"] >= -.01 and metrics["mean_exposure"] <= .2)
            result = {"regime": regime, "seed": seed, "passed": bool(passed),
                "training_bars": model.research_bar_transitions, "initial": initial, "trained": metrics,
                "roundtrip_verified": True}
            results.append(result)
            write_json(directory / "result.json", result)
            write_json(output / "results.json", results)
            print(__import__("json").dumps({"diagnostic": regime, "seed": seed,
                "passed": bool(passed), "return": metrics["net_return"],
                "exposure": metrics["mean_exposure"]}), flush=True)
            env.close()
    write_json(output / "completion.json", {"completed": True, "runs": len(results),
        "all_capability_checks_passed": all(r["passed"] for r in results)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bar-budget", type=int, default=25000)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--ent-coef", type=float, default=.001)
    parser.add_argument("--episode-days", type=int, default=20)
    parser.add_argument("--regimes", default="rising,falling,flat")
    parser.add_argument("--cash-logit-bias", type=float, default=0.)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    regimes = args.regimes.split(",")
    if not set(regimes) <= {"rising", "falling", "flat"} or len(set(regimes)) != len(regimes):
        raise SystemExit("unique regimes must be rising,falling,flat")
    if args.bar_budget < 1 or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise SystemExit("positive budget and unique non-negative seeds required")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or not np.isfinite(args.ent_coef) or args.ent_coef < 0 or args.episode_days < 1:
        raise SystemExit("invalid learning profile")
    run(args.output, seeds, args.bar_budget, gamma=args.gamma, gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef, episode_days=args.episode_days, regimes=regimes,
        cash_logit_bias=args.cash_logit_bias)
