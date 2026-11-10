"""Reproducible offline A/B/C/D research runner (not a serving artifact exporter)."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(source))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from data.data_loader import DataLoader
from data.defect_days import classify_trading_day
from data.feature_builder import build_features
from data.feature_engineer import FeatureEngineer
from env.research_env import ResearchTradingEnvironment, TargetExposureEnvironment
from env.controlled_exposure_env import ControlledExposureEnvironment
from friction.friction_model import FrictionModel
from friction.research_friction import ScaledFrictionModel
from models.constrained_ppo import DurationAwareRolloutBuffer, ResearchPPO, RiskBudgetCallback, initialize_cash_prior
from models.normalization import FeatureNormalizer
from models.walk_forward import build_expanding_walk_forward_folds, slice_walk_forward_fold
from utils.config_loader import load_config
from evaluation.metrics import calculate_max_drawdown, calculate_sharpe_ratio

CONDITIONS = {"A": ("incremental", 1), "B": ("target", 1),
              "C": ("incremental", 12), "D": ("target", 12)}
EXTENSIONS = {
    "I": {"tolerance": .05, "interval": 6, "risk_mode": "off", "settle_daily": True},
    "J": {"tolerance": .05, "interval": 6, "risk_mode": "fixed", "settle_daily": True},
    "K": {"tolerance": .05, "interval": 6, "risk_mode": "dual", "settle_daily": True},
    "Q": {"tolerance": .01, "interval": 1, "risk_mode": "off", "reserve_exit_cost": False},
    "R": {"tolerance": .01, "interval": 1, "risk_mode": "off"},
    "E": {"tolerance": .05, "interval": 1, "risk_mode": "off"},
    "F": {"tolerance": .05, "interval": 6, "risk_mode": "off"},
    "G": {"tolerance": .05, "interval": 6, "risk_mode": "fixed"},
    "H": {"tolerance": .05, "interval": 6, "risk_mode": "dual"},
}
CONDITIONS.update({name: ("controlled", 12) for name in EXTENSIONS})


def evaluation_config(config, model):
    result = copy.deepcopy(config)
    if hasattr(model, "research_risk_multiplier"):
        result.setdefault("research", {})["risk_multiplier"] = model.research_risk_multiplier
    return result


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")


def code_snapshot():
    digest = hashlib.sha256()
    for area in ("agent/src", "env/src", "evaluation", "experiments"):
        for path in sorted((ROOT / area).rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        # Exported source has no Git history. Keep the content hash authoritative
        # and never claim that an unverifiable working tree is clean.
        commit, dirty = "unknown", True
    return {"git_commit": commit, "dirty": dirty,
            "python_source_sha256": digest.hexdigest()}


def data_fingerprints(data):
    # Include ADV/liquidity and every other input column, not just policy inputs.
    result = {}
    for symbol, frame in data.items():
        digest = hashlib.sha256(json.dumps([(str(c), str(frame[c].dtype)) for c in frame]).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        result[symbol] = digest.hexdigest()
    return result


def archive_source(directory):
    path = Path(directory) / "source_snapshot.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for area in ("agent/src", "env/src", "evaluation", "experiments"):
            for source in sorted((ROOT / area).rglob("*.py")):
                archive.add(source, arcname=str(source.relative_to(ROOT)))
        archive.add(ROOT / "env/configs/config.yaml", arcname="env/configs/config.yaml")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_data(days=100):
    """Known causal signal for pipeline diagnostics only, never market evidence."""
    rng = np.random.default_rng(19)
    timestamps = [t for day in pd.bdate_range("2025-01-02", periods=days, tz="Asia/Seoul")
                  for t in pd.date_range(day + pd.Timedelta(hours=10, minutes=5), periods=32, freq="5min")]
    signal = np.sin(np.arange(len(timestamps)) / 10)
    increments = .001 * np.sign(signal) + rng.normal(0, .0002, len(signal))
    closes = 10_000 * np.exp(np.r_[0, np.cumsum(increments[:-1])])
    result = pd.DataFrame({"Timestamp": timestamps, "Close": closes,
                           "ExecPrice": closes.copy(), "Adv20": 1e11})
    for i, column in enumerate(FeatureEngineer.FEATURE_COLUMNS):
        result[column] = signal if i == 0 else rng.normal(0, .1, len(result))
    return {"SYNTHETIC": result}, sorted(set(result.Timestamp.dt.date)), {"source": "synthetic-diagnostic-only"}


def load_market(symbols):
    loader = DataLoader(ROOT / "data/raw", ROOT / "data/processed")
    data, calendars, provenance = {}, [], {}
    for symbol in symbols:
        raw = loader.load_raw_parquet_all(symbol, "1m")
        quality = [classify_trading_day(group, require_complete_session=True)
                   for _, group in raw.groupby(pd.to_datetime(raw.Timestamp).dt.date)]
        valid = {q.trading_date for q in quality if q.is_valid}
        features = build_features(symbol, loader)
        features = features.loc[pd.to_datetime(features.Timestamp).dt.date.isin(valid)].reset_index(drop=True)
        data[symbol] = features
        daily = loader.load_raw_parquet_all(symbol, "1d")
        calendars.append(set(pd.to_datetime(daily["Date" if "Date" in daily else "Timestamp"]).dt.date))
        manifest = ROOT / "data/processed" / symbol / f"features_v{FeatureEngineer.FEATURE_SCHEMA_VERSION}.manifest.json"
        provenance[symbol] = {"feature_manifest": json.loads(manifest.read_text()),
                              "excluded_raw_days": [q.to_dict() for q in quality if not q.is_valid]}
    start = max(pd.to_datetime(f.Timestamp).dt.date.min() for f in data.values())
    end = min(pd.to_datetime(f.Timestamp).dt.date.max() for f in data.values())
    calendar = sorted(d for d in set.intersection(*calendars) if start <= d <= end)
    provenance["calendar_source"] = "intersection of symbol daily parquets; not official KRX calendar"
    return data, calendar, provenance


def make_env(frame, config, condition, normalizer):
    action, window = CONDITIONS[condition]
    cls = (ControlledExposureEnvironment if action == "controlled" else
           TargetExposureEnvironment if action == "target" else ResearchTradingEnvironment)
    env = config["environment"]
    research = config.get("research", {})
    extra = {}
    if condition in EXTENSIONS:
        settings = EXTENSIONS[condition]
        extra = dict(rebalance_tolerance=settings["tolerance"],
            reserve_exit_cost=settings.get("reserve_exit_cost", True),
            settle_daily=settings.get("settle_daily", False),
            decision_interval=settings["interval"], bar_gamma=config["agent"]["ppo"]["gamma"],
            drawdown_threshold=research.get("drawdown_threshold", .02),
            risk_budget=research.get("risk_budget", .001),
            risk_multiplier=research.get("risk_multiplier",
                .1 if settings["risk_mode"] != "off" else 0.))
    return cls(market_data=frame, feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
               initial_cash=env["initial_cash"], unit_fraction=env["unit_fraction"], max_units=env["max_units"],
               friction_model=ScaledFrictionModel(**config["friction"],
                    multiplier=research.get("friction_multiplier", 1.)),
               episode_days=env["episode_days"], duration_horizon_bars=env["episode_days"] * env["nominal_bars_per_day"],
               nominal_bars_per_day=env["nominal_bars_per_day"],
               reward_scale=env["reward_scale"], reward_return_mode=env["reward_return_mode"],
               risk_penalty_rate=env.get("risk_penalty_rate", 0),
               turnover_penalty_rate=env.get("turnover_penalty_rate", 0),
               drawdown_penalty_rate=env.get("drawdown_penalty_rate", 0),
               downside_penalty_rate=env.get("downside_penalty_rate", 0),
               benchmark_relative_rate=env.get("benchmark_relative_rate", 0),
               feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
               window_length=window, normalizer=normalizer, **extra)


def evaluate(model, env, baseline=None):
    if isinstance(env, ControlledExposureEnvironment):
        from evaluation.research_evaluator import evaluate_controlled
        return evaluate_controlled(model, env, baseline)
    observation, _ = env.reset(seed=0, options={"start_date": env.available_dates[0],
                                              "episode_days": len(env.available_dates)})
    records = []
    action_counts = [0] * env.action_space.n
    probability_sum = np.zeros(env.action_space.n)
    mask_sum = np.zeros(env.action_space.n)
    entropies = []
    while True:
        mask = env.action_masks()
        if baseline is None:
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(observation)
                distribution = model.policy.get_distribution(tensor, action_masks=mask)
                probabilities = distribution.distribution.probs.cpu().numpy().ravel()
            action = int(np.argmax(probabilities))
            probability_sum += probabilities
            entropies.append(float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()))
        else:
            if baseline == "cash":
                target = 0
            elif baseline == "buy_and_hold":
                target = 5 if env.shares_held == 0 else None
            elif baseline == "mean_reversion":
                # Predeclared rule, no tuning: buy deep negative standardized return,
                # clear on positive return; otherwise retain current position.
                z = env.normalizer.transform_observation(
                    env.get_current_market_row()[env.feature_columns].to_numpy(float))[0]
                target = 5 if z < -1 else (0 if z > 0 else None)
            else:
                raise ValueError(baseline)
            if env.action_schema == "target-weight-6":
                if target is None:
                    # A true hold baseline must not rebalance to a rounded target.
                    action = 0
                else:
                    action = target
            else:
                action = 3 if target == 0 else (1 if target == 5 and mask[1] else 0)
        if baseline is not None and env.action_schema == "target-weight-6" and target is None:
            # Baselines use incremental env below, so reaching here is a misuse.
            raise ValueError("baseline evaluation requires incremental environment")
        mask_sum += mask
        action_counts[action] += 1
        observation, _, terminated, truncated, info = env.step(action)
        records.extend(info.get("bar_records", [info]))
        if terminated or truncated:
            break
    trace = pd.DataFrame(records)
    settlement = env.estimate_liquidation_cost()
    equity = [env.initial_cash] + trace.portfolio_value.tolist()
    equity[-1] -= settlement
    daily = trace.groupby(trace.valuation_date).portfolio_value.last().astype(float)
    daily.iloc[-1] -= settlement
    daily_returns = pd.Series([env.initial_cash] + daily.tolist()).pct_change().dropna()
    count = sum(action_counts)
    absolute_values = trace["absolute_trade_value"] if "absolute_trade_value" in trace else trace.trade_value.abs()
    terminal_value = env.shares_held * float(env.get_current_market_row()[env.price_col])
    trades = trace["executed_trades"].sum() if "executed_trades" in trace else trace.executed.sum()
    result = {"net_return": equity[-1] / env.initial_cash - 1,
              "bar_max_drawdown": calculate_max_drawdown(pd.Series(equity)),
              "daily_sharpe": calculate_sharpe_ratio(daily_returns),
              "flat_fraction": float(trace.flat.mean()), "mean_exposure": float(trace.exposure.mean()),
              "trade_count": int(trades + int(terminal_value > 0)),
              "unavailable_fill_bars": int(trace.fill_unavailable.sum()),
              "turnover": float((absolute_values.sum() + terminal_value) / trace.portfolio_value.mean()),
              "friction_cost": float(trace.friction_cost.sum() + settlement),
              "terminal_liquidation_cost": settlement + float(trace.get("terminal_settlement_cost", pd.Series([0.])).sum()),
              "pre_cost_return_addback": float((equity[-1] + trace.friction_cost.sum() + settlement) / env.initial_cash - 1),
              "mean_drawdown_excess": float(trace.risk_cost.mean()) if "risk_cost" in trace else None,
              "drawdown_exceedance_fraction": float((trace.risk_cost > 0).mean()) if "risk_cost" in trace else None,
              "action_counts": action_counts,
              "valid_action_fraction": (mask_sum / count).tolist(),
              "mean_action_probability": (probability_sum / count).tolist() if model is not None else None,
              "mean_policy_entropy": float(np.mean(entropies)) if entropies else None,
              "decisions": count, "bars": len(trace), "days": len(env.available_dates)}
    return result, trace


class ValidationCallback(BaseCallback):
    def __init__(self, frames, config, condition, normalizer, output, frequency):
        super().__init__()
        self.frames, self.config, self.condition = frames, config, condition
        self.normalizer, self.output, self.frequency = normalizer, output, frequency
        self.history, self.best_parameters, self.best_score = [], None, -math.inf
        self.best_step, self.last_step = 0, -1
        self.training_bars, self.last_evaluation_bar = 0, 0

    def _evaluate(self):
        config = evaluation_config(self.config, self.model)
        metrics = {s: evaluate(self.model, make_env(f, config, self.condition, self.normalizer))[0]
                   for s, f in self.frames.items()}
        returns = [m["net_return"] for m in metrics.values()]
        drawdowns = [m["bar_max_drawdown"] for m in metrics.values()]
        # Same fixed criterion for all conditions; research selection only.
        score = float(np.median(returns) + .5 * min(min(returns), 0) + .5 * np.mean(drawdowns))
        record = {"timesteps": self.num_timesteps, "training_bars": self.training_bars,
                  "score": score, "per_symbol": metrics,
                  "risk_multiplier": getattr(self.model, "research_risk_multiplier", None)}
        self.history.append(record)
        if score > self.best_score or (score == self.best_score and self.best_step == 0 and self.num_timesteps > 0):
            self.best_parameters = copy.deepcopy(self.model.get_parameters())
            self.best_score, self.best_step = score, self.num_timesteps
            self.best_risk_multiplier = getattr(self.model, "research_risk_multiplier", None)
        self.last_step = self.num_timesteps
        self.last_evaluation_bar = self.training_bars
        write_json(self.output / "validation.json", self.history)

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self):
        self.training_bars += sum(i.get("elapsed_bars", 1) for i in self.locals["infos"])
        if self.training_bars - self.last_evaluation_bar >= self.frequency:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.last_step != self.num_timesteps:
            self._evaluate()
        self.model.set_parameters(self.best_parameters, exact_match=True)
        if self.best_risk_multiplier is not None:
            self.model.research_risk_multiplier = self.best_risk_multiplier


def save_research_bundle(model, normalizer, env, config, output, extra):
    if isinstance(env, ControlledExposureEnvironment):
        if not isinstance(model.rollout_buffer, DurationAwareRolloutBuffer) or not np.isclose(model.gamma, env.bar_gamma, rtol=0, atol=1e-12):
            raise ValueError("controlled artifact requires duration-aware PPO with matching gamma")
    model.save(output / "research_model")
    normalizer.save(output / "normalization.json")
    payload = {"research_format": "ablation-v1", "deployment_status": "research-only",
               "contract": env.contract(), "config": config, **extra}
    payload["sha256"] = {f: hashlib.sha256((output / f).read_bytes()).hexdigest()
                         for f in ("research_model.zip", "normalization.json")}
    write_json(output / "research_manifest.json", payload)


def load_research_bundle(directory, frame):
    """Offline trusted-local artifacts only; not compatible with serving v5.

    Checksums detect accidental file changes, not untrusted-source authenticity.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "research_manifest.json").read_text())
    if manifest.get("research_format") != "ablation-v1" or manifest.get("deployment_status") != "research-only":
        raise ValueError("unknown research artifact")
    for name in ("research_model.zip", "normalization.json"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"research artifact checksum mismatch: {name}")
    normalizer = FeatureNormalizer.load(directory / "normalization.json")
    env = make_env(frame, manifest["config"], manifest["condition"], normalizer)
    if env.contract() != manifest["contract"]:
        raise ValueError("research environment contract mismatch")
    model = ResearchPPO.load(directory / "research_model.zip", env=env, device="cpu")
    return model, env, manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--symbols", default="005930,000660,034220,066570,009150")
    p.add_argument("--conditions", default="A,B,C,D")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--folds", type=int, default=1)
    p.add_argument("--validation-days", type=int, default=20)
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--parameter-budget", type=int, default=40_000,
                   help="Match total actor+critic parameters approximately across conditions")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--evaluate-test", action="store_true", help="Opt in only after freezing study settings")
    p.add_argument("--bar-budget", type=int, default=0,
                   help="Controlled conditions only: stop after a full rollout reaches this many underlying bars")
    p.add_argument("--friction-multiplier", type=float, default=1.)
    p.add_argument("--gamma", type=float)
    p.add_argument("--gae-lambda", type=float)
    p.add_argument("--ent-coef", type=float)
    p.add_argument("--episode-days", type=int)
    p.add_argument("--cash-logit-bias", type=float, default=0.)
    p.add_argument("--fold-plan", type=Path, help="Coverage-only predeclared dates and independent calendar")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--drawdown-threshold", type=float, default=.02)
    p.add_argument("--risk-budget", type=float, default=.001)
    p.add_argument("--dual-learning-rate", type=float, default=10.)
    return p.parse_args()


def main():
    args = parse_args()
    if any(x < 1 for x in (args.folds, args.validation_days, args.test_days,
                           args.timesteps, args.n_steps, args.eval_freq, args.parameter_budget)):
        raise SystemExit("counts must be positive")
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    if not set(conditions) <= CONDITIONS.keys() or len(set(conditions)) != len(conditions):
        raise SystemExit("unique conditions must be drawn from " + ",".join(CONDITIONS))
    if getattr(args, "bar_budget", 0) < 0:
        raise SystemExit("bar budget must be non-negative")
    if getattr(args, "bar_budget", 0) and not set(conditions) <= EXTENSIONS.keys():
        raise SystemExit("bar budget requires a controlled research condition")
    if not np.isfinite(getattr(args, "friction_multiplier", 1.)) or getattr(args, "friction_multiplier", 1.) < 0:
        raise SystemExit("friction multiplier must be finite and non-negative")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise SystemExit("seeds must be unique non-negative integers")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use CPU or repair GPU driver")
    if args.output.exists():
        raise SystemExit("output already exists; use a new directory to preserve earlier experiments")
    args.output.mkdir(parents=True)
    if getattr(args, "num_threads", 2) < 1:
        raise SystemExit("num threads must be positive")
    torch.set_num_threads(getattr(args, "num_threads", 2))
    config = load_config(ROOT / "env/configs/config.yaml")
    for name in ("gamma", "gae_lambda", "ent_coef"):
        value = getattr(args, name, None)
        if value is not None:
            if not np.isfinite(value) or value < 0 or (name in ("gamma", "gae_lambda") and value > 1) or (name == "gamma" and value == 0):
                raise SystemExit("invalid learning profile")
            config["agent"]["ppo"][name] = value
    if getattr(args, "episode_days", None) is not None:
        if args.episode_days < 1:
            raise SystemExit("episode days must be positive")
        config["environment"]["episode_days"] = args.episode_days
    config["research"] = {"friction_multiplier": getattr(args, "friction_multiplier", 1.),
        "drawdown_threshold": getattr(args, "drawdown_threshold", .02),
        "risk_budget": getattr(args, "risk_budget", .001),
        "dual_learning_rate": getattr(args, "dual_learning_rate", 10.),
        "cash_logit_bias": getattr(args, "cash_logit_bias", 0.)}
    if not all(np.isfinite(config["research"][key]) and config["research"][key] >= 0
               for key in ("drawdown_threshold", "risk_budget", "dual_learning_rate")):
        raise SystemExit("invalid risk definition")
    if config["research"]["risk_budget"] >= 1 or config["research"]["drawdown_threshold"] >= 1:
        raise SystemExit("risk threshold and budget must be below one")
    if not np.isfinite(config["research"]["cash_logit_bias"]) or not 0 <= config["research"]["cash_logit_bias"] <= 5:
        raise SystemExit("cash logit bias must be in [0, 5]")
    if config["research"]["cash_logit_bias"] and not set(conditions) <= EXTENSIONS.keys():
        raise SystemExit("cash prior is only defined for target-exposure research conditions")
    data, calendar, provenance = synthetic_data() if args.synthetic else load_market(args.symbols.split(","))
    folds = build_expanding_walk_forward_folds(data, n_folds=args.folds,
        validation_days=args.validation_days, test_days=args.test_days, purge_days=5,
        reference_days=calendar)
    fold_plan = None
    if getattr(args, "fold_plan", None) is not None:
        from datetime import date
        from models.walk_forward import ExpandingWalkForwardFold
        fold_plan = json.loads(args.fold_plan.read_text())
        if fold_plan["source_fingerprints"] != data_fingerprints(data):
            raise SystemExit("data changed after fold planning")
        calendar = [date.fromisoformat(day) for day in fold_plan["calendar"]]
        folds = [ExpandingWalkForwardFold(**{key: date.fromisoformat(value) if key.endswith(("start", "end")) else value
            for key, value in item.items()}) for item in fold_plan["folds"]]
        if len(folds) != args.folds:
            raise SystemExit("fold plan does not contain the requested number of eligible folds")
        for fold in folds:
            if not (fold.train_start <= fold.train_end < fold.validation_start <= fold.validation_end < fold.test_start <= fold.test_end):
                raise SystemExit("fold plan is not chronological")
        provenance["calendar_source"] = fold_plan["calendar_provenance"]
    plans = []
    for fold in folds:
        coverage = {}
        for segment in ("train", "validation", "test"):
            start, end = getattr(fold, f"{segment}_start"), getattr(fold, f"{segment}_end")
            expected = {d for d in calendar if start <= d <= end}
            coverage[segment] = {}
            for s, frame in data.items():
                observed = set(pd.to_datetime(frame.Timestamp).dt.date)
                missing = sorted(expected - observed)
                coverage[segment][s] = {"ratio": 1 - len(missing) / len(expected),
                                         "missing_days": [str(d) for d in missing]}
        passed = all(m["ratio"] >= .95 for d in coverage.values() for m in d.values())
        plans.append({**fold.to_dict(), "coverage": coverage, "passed": passed})
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
        "study_type": "synthetic-diagnostic" if args.synthetic else "retrospective-research-screening",
        "holdout_status": "synthetic" if args.synthetic else "research_reused_not_pristine",
        "arguments": vars(args), "config": config, "code": code_snapshot(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "stable-baselines3", "sb3-contrib")},
        "selection": "median_return + .5*min(worst_return,0) + .5*mean_bar_drawdown; validation only",
        "data": provenance, "folds": plans, "minimum_coverage": .95}
    manifest["data_fingerprints"] = data_fingerprints(data)
    if fold_plan is not None:
        manifest["fold_plan"] = fold_plan
    manifest["source_archive_sha256"] = archive_source(args.output)
    write_json(args.output / "study_manifest.json", manifest)
    if args.plan_only:
        print(json.dumps({"output": str(args.output), "coverage_passed": all(p["passed"] for p in plans)}))
        return
    if not all(p["passed"] for p in plans):
        raise SystemExit("coverage below 95%; see study_manifest.json; no training performed")
    results = []
    for fold in folds:
        parts = {segment: {s: slice_walk_forward_fold(f, fold, segment=segment) for s, f in data.items()}
                 for segment in ("train", "validation", "test")}
        normalizer = FeatureNormalizer.fit(pd.concat(parts["train"].values()), list(FeatureEngineer.FEATURE_COLUMNS))
        # Predeclared baselines use a single unit then hold (initial unit budget
        # is overridden to .995 below so B&H is near full exposure after costs).
        baseline_config = copy.deepcopy(config)
        baseline_config["environment"]["unit_fraction"] = .995
        baselines = {}
        for split in (["validation", "test"] if args.evaluate_test else ["validation"]):
            baselines[split] = {s: {name: evaluate(None, make_env(f, baseline_config, "A", normalizer), baseline=name)[0]
                                    for name in ("cash", "buy_and_hold", "mean_reversion")}
                                for s, f in parts[split].items()}
            intraday_config = copy.deepcopy(config)
            intraday_config["environment"]["episode_days"] = 1
            for symbol, frame in parts[split].items():
                for name in ("intraday_buy_and_hold", "intraday_mean_reversion"):
                    baselines[split][symbol][name] = evaluate(None, make_env(frame, intraday_config, "R", normalizer), baseline=name)[0]
        write_json(args.output / f"fold_{fold.index:02d}_baselines.json", baselines)
        for condition in conditions:
            for seed in seeds:
                output = args.output / f"fold_{fold.index:02d}" / condition / f"seed_{seed}"
                output.mkdir(parents=True)
                start = time.monotonic()
                vec = DummyVecEnv([lambda f=f: make_env(f, config, condition, normalizer)
                                   for f in parts["train"].values()])
                ppo = dict(config["agent"]["ppo"])
                interval = EXTENSIONS.get(condition, {}).get("interval", 1)
                n_steps = max(2, args.n_steps // interval)
                ppo.update(n_steps=n_steps, batch_size=min(256, n_steps * len(data)), verbose=0)
                if condition in EXTENSIONS:
                    ppo["rollout_buffer_class"] = DurationAwareRolloutBuffer
                dimension, actions = vec.observation_space.shape[0], vec.action_space.n
                def parameter_count(width):
                    return 2 * ((dimension + 1) * width + (width + 1) * width) + (width + 1) * (actions + 1)
                width = min(range(16, 513), key=lambda w: abs(parameter_count(w) - args.parameter_budget))
                ppo["policy_kwargs"] = {"net_arch": [width, width]}
                model = ResearchPPO("MlpPolicy", vec, seed=seed, device=args.device, **ppo)
                initialize_cash_prior(model, config["research"]["cash_logit_bias"])
                model.research_bar_budget = getattr(args, "bar_budget", 0)
                model.set_logger(configure(str(output), ["csv"]))
                callback = ValidationCallback(parts["validation"], config, condition,
                                                normalizer, output, args.eval_freq)
                risk_callback = None
                callbacks = callback
                if condition in EXTENSIONS:
                    mode = EXTENSIONS[condition]["risk_mode"]
                    risk_callback = RiskBudgetCallback(mode=mode,
                        initial_multiplier=.1 if mode != "off" else 0.,
                        budget=config["research"]["risk_budget"],
                        learning_rate=config["research"]["dual_learning_rate"])
                    callbacks = [risk_callback, callback]
                model.learn(total_timesteps=max(args.timesteps, model.research_bar_budget), callback=callbacks)
                frozen_config = evaluation_config(config, model)
                training_seconds = time.monotonic() - start
                result = {"fold": fold.index, "condition": condition, "seed": seed,
                    "requested_timesteps": args.timesteps, "actual_timesteps": model.num_timesteps,
                    "actual_training_bars": getattr(model, "research_bar_transitions", model.num_timesteps),
                    "risk_multiplier": getattr(model, "research_risk_multiplier", None),
                    "training_and_validation_seconds": training_seconds,
                    "best_validation_step": callback.best_step,
                    "parameter_count": sum(p.numel() for p in model.policy.parameters()),
                    "hidden_width": width,
                    "splits": {}}
                example_env = make_env(next(iter(parts["validation"].values())), frozen_config, condition, normalizer)
                save_research_bundle(model, normalizer, example_env, frozen_config, output,
                                     {"condition": condition, "seed": seed, "fold": fold.to_dict(),
                                      "code": manifest["code"], "best_validation_step": callback.best_step})
                for split in (["validation", "test"] if args.evaluate_test else ["validation"]):
                    result["splits"][split] = {}
                    for symbol, frame in parts[split].items():
                        env = make_env(frame, frozen_config, condition, normalizer)
                        metrics, trace = evaluate(model, env)
                        trace.to_csv(output / f"{split}_{symbol}_trace.csv", index=False)
                        result["splits"][split][symbol] = metrics
                # Save/load must reproduce the frozen validation policy.
                symbol, frame = next(iter(parts["validation"].items()))
                reloaded, restored_env, _ = load_research_bundle(output, frame)
                restored, _ = evaluate(reloaded, restored_env)
                if restored != result["splits"]["validation"][symbol]:
                    raise RuntimeError("saved research policy did not reproduce validation metrics")
                result["roundtrip_verified"] = True
                if risk_callback is not None:
                    write_json(output / "risk_training.json", risk_callback.history)
                write_json(output / "result.json", result)
                results.append(result)
                write_json(args.output / "results.json", results)
                print(json.dumps({"fold": fold.index, "condition": condition, "seed": seed,
                                  "seconds": round(training_seconds, 2),
                                  "validation_return": float(np.mean([m["net_return"] for m in result["splits"]["validation"].values()]))}), flush=True)
                vec.close()
    write_json(args.output / "completion.json", {"completed": True, "runs": len(results),
        "research_only": True, "test_evaluated": args.evaluate_test})


if __name__ == "__main__":
    main()
