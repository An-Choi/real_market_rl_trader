"""Train-only linear/nonlinear signal diagnostics; emit models, tables and plots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(path))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.run_ablation import load_market, data_fingerprints, code_snapshot, write_json, archive_source
from experiments.finalize_research import fold_from_dict
from data.feature_engineer import FeatureEngineer
from models.normalization import FeatureNormalizer
from models.walk_forward import slice_walk_forward_fold
from friction.friction_model import FrictionModel
from env.observation import adv_value_from_row, liquidity_score_from_adv
from utils.config_loader import load_config


def causal_inputs(frame, normalizer, window):
    features = normalizer.transform_observation(frame[list(normalizer.feature_columns)].to_numpy(np.float32))
    stamps = pd.to_datetime(frame.Timestamp)
    n, width = features.shape
    values = np.zeros((n, window, width), np.float32)
    valid = np.zeros((n, window), np.float32)
    for lag in range(window):
        if lag >= n:
            break
        indices = np.arange(lag, n)
        continuous = ((stamps.iloc[indices].reset_index(drop=True) - stamps.iloc[indices-lag].reset_index(drop=True))
                      == pd.Timedelta(minutes=5*lag)).to_numpy()
        same_day = (stamps.iloc[indices].dt.date.to_numpy() == stamps.iloc[indices-lag].dt.date.to_numpy())
        # Check every intermediate gap, not merely elapsed time at endpoints.
        breaks = (stamps.diff() != pd.Timedelta(minutes=5)).cumsum().to_numpy()
        keep = continuous & same_day & (breaks[indices] == breaks[indices-lag])
        indices = indices[keep]
        values[indices, window-lag-1] = features[indices-lag]
        valid[indices, window-lag-1] = 1.
    return np.concatenate((values.reshape(n, -1), valid), axis=1)


def forward_targets(frame, horizon):
    stamps = pd.to_datetime(frame.Timestamp)
    future = frame.Close.shift(-horizon)
    later = stamps.shift(-horizon)
    valid = (later.dt.date == stamps.dt.date) & (later-stamps == pd.Timedelta(minutes=5*horizon))
    segments = (stamps.diff() != pd.Timedelta(minutes=5)).cumsum()
    valid &= segments.shift(-horizon) == segments
    entry = pd.to_numeric(frame.ExecPrice, errors="coerce")
    valid &= np.isfinite(entry) & (entry > 0) & np.isfinite(future) & (future > 0)
    result = np.log(future / entry)
    return result.where(valid).to_numpy(float)


def scores(predicted, actual, dates):
    mse = float(np.mean((predicted-actual)**2))
    zero_mse = float(np.mean(actual**2))
    predicted_ranks, actual_ranks = pd.Series(predicted).rank(), pd.Series(actual).rank()
    correlation = predicted_ranks.corr(actual_ranks) if predicted_ranks.nunique()>1 and actual_ranks.nunique()>1 else 0.
    daily = []
    for day in sorted(set(dates)):
        mask = dates == day
        if mask.sum() >= 10:
            p, a = pd.Series(predicted[mask]).rank(), pd.Series(actual[mask]).rank()
            if p.nunique()>1 and a.nunique()>1:
                daily.append(float(p.corr(a)))
    rng = np.random.default_rng(42)
    bootstrap = np.mean(rng.choice(daily, (1000, len(daily)), replace=True), axis=1) if daily else np.array([0.])
    return {"observations": len(actual), "mse": mse, "mse_skill_vs_zero": 1-mse/max(zero_mse,1e-20),
        "directional_accuracy": float(np.mean((predicted>0)==(actual>0))),
        "spearman_ic": float(correlation), "daily_mean_ic": float(np.mean(daily)) if daily else 0.,
        "daily_ic_ci_low": float(np.quantile(bootstrap,.025)), "daily_ic_ci_high": float(np.quantile(bootstrap,.975))}


def cost_events(frame, predictions, horizon, config):
    friction = FrictionModel(**config["friction"])
    budget = config["environment"]["initial_cash"] * config["environment"]["unit_fraction"]
    labels = forward_targets(frame, horizon)
    records, next_index = [], 0
    for index, prediction in enumerate(predictions):
        if index < next_index or not np.isfinite(labels[index]):
            continue
        row, exit_row = frame.iloc[index], frame.iloc[index+horizon]
        price = float(row.Close)
        qty = int(budget // price)
        if qty < 1:
            continue
        value = qty * price
        liquidity = liquidity_score_from_adv(adv_value_from_row(row), value)
        when = pd.Timestamp(row.Timestamp).date()
        estimated = sum(friction.calculate_total_friction(value, side, liquidity, price, when)
                        for side in ("buy", "sell")) / value
        # This gate uses observable price/ADV/date and model output only.
        if prediction <= np.log1p(estimated + .0005):
            continue
        entry, exit_price = float(row.ExecPrice), float(exit_row.Close)
        buy = friction.calculate_total_friction(qty*entry, "buy", liquidity, entry, when)
        sell_liquidity = liquidity_score_from_adv(adv_value_from_row(exit_row), qty*exit_price)
        sell = friction.calculate_total_friction(qty*exit_price, "sell", sell_liquidity, exit_price,
                                               pd.Timestamp(exit_row.Timestamp).date())
        records.append({"timestamp": str(row.Timestamp), "prediction": float(prediction),
            "gross_return": exit_price/entry-1,
            "net_return": (qty*(exit_price-entry)-buy-sell)/(qty*entry+buy),
            "cost": buy+sell, "shares": qty})
        next_index = index+horizon
    return {"events": len(records), "mean_event_gross_return": float(np.mean([r["gross_return"] for r in records])) if records else 0.,
        "mean_event_net_return": float(np.mean([r["net_return"] for r in records])) if records else 0.,
        "total_event_cost": sum(r["cost"] for r in records)}, records


def run(args):
    if args.output.exists():
        raise ValueError("diagnostics output exists")
    plan = json.loads(args.plan.read_text())
    data, _, _ = load_market(["005930","000660","034220","066570","009150"])
    if data_fingerprints(data) != plan["source_fingerprints"]:
        raise ValueError("data changed after planning")
    args.output.mkdir(parents=True)
    config = load_config(ROOT / "env/configs/config.yaml")
    write_json(args.output / "manifest.json", {"code": code_snapshot(), "source_archive_sha256": archive_source(args.output),
        "fold_plan": plan, "windows": [1,12], "horizons": [1,3,12],
        "fit": "train-only normalization and model fitting; no random temporal validation split",
        "target": "log(future observed Close / next-minute execution price), contiguous intraday rows only",
        "event_gate": "predicted log return > observable modeled round-trip cost + 5bp; fixed before evaluation",
        "test_evaluated": False, "event_accounting": "non-overlapping fixed-notional diagnostics, not compounded portfolio returns",
        "label_missingness": "metrics conditional on available observed entry/exit labels; excluded rows counted"})
    rows, fitted = [], 0
    for payload in plan["folds"]:
        fold = fold_from_dict(payload)
        parts = {split: {symbol: slice_walk_forward_fold(frame, fold, segment=split)
                         for symbol, frame in data.items()} for split in ("train","validation")}
        normalizer = FeatureNormalizer.fit(pd.concat(parts["train"].values()), list(FeatureEngineer.FEATURE_COLUMNS))
        for window in (1,12):
            inputs = {split: {symbol: causal_inputs(frame, normalizer, window) for symbol,frame in frames.items()}
                      for split,frames in parts.items()}
            for horizon in (1,3,12):
                x, y = [], []
                for symbol,frame in parts["train"].items():
                    target = forward_targets(frame,horizon); valid = np.isfinite(target)
                    x.append(inputs["train"][symbol][valid]); y.append(target[valid])
                x, y = np.concatenate(x), np.concatenate(y)
                models = {"ridge":Ridge(alpha=10.,solver="lsqr"),
                    "hist_gradient_boosting":HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,
                        min_samples_leaf=100,l2_regularization=10.,learning_rate=.05,early_stopping=False,random_state=42)}
                for name,model in models.items():
                    model.fit(x,y)
                    fitted += 1
                    directory=args.output/f"fold_{fold.index:02d}"/f"window_{window}"/f"horizon_{horizon}"/name
                    directory.mkdir(parents=True)
                    joblib.dump(model,directory/"model.joblib")
                    normalizer.save(directory/"normalization.json")
                    for symbol,frame in parts["validation"].items():
                        predicted=model.predict(inputs["validation"][symbol]); actual=forward_targets(frame,horizon)
                        valid=np.isfinite(actual)
                        dates=pd.to_datetime(frame.Timestamp).dt.date.to_numpy()[valid]
                        events,records=cost_events(frame,predicted,horizon,config)
                        result={"fold":fold.index,"window":window,"horizon":horizon,"model":name,"symbol":symbol,
                            "train_rows":len(y),"excluded_label_rows":int((~valid).sum()),
                            "zero_prediction_mse":float(np.mean(actual[valid]**2)),
                            "train_mean_prediction_mse":float(np.mean((actual[valid]-y.mean())**2)),
                            **scores(predicted[valid],actual[valid],dates),**events}
                        rows.append(result)
                        pd.DataFrame(records,columns=["timestamp","prediction","gross_return","net_return","cost","shares"]).to_csv(directory/f"validation_{symbol}_events.csv",index=False)
                        pd.DataFrame({"Timestamp":frame.Timestamp,"prediction":predicted,"observed_target":actual,
                            "label_available":valid}).to_csv(directory/f"validation_{symbol}_predictions.csv",index=False)
                    pd.DataFrame(rows).to_csv(args.output/"validation_metrics.csv",index=False)
                    print(json.dumps({"fold":fold.index,"window":window,"horizon":horizon,"model":name}),flush=True)
    table=pd.DataFrame(rows)
    table.groupby(["window","horizon","model"])[["mse_skill_vs_zero","spearman_ic","daily_mean_ic","mean_event_net_return"]].mean().to_csv(args.output/"summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for (window,name),group in table.groupby(["window","model"]):
        means=group.groupby("horizon")[["spearman_ic","mse_skill_vs_zero"]].mean()
        for axis,metric in zip(axes,("spearman_ic","mse_skill_vs_zero")):
            axis.plot(means.index,means[metric],marker="o",label=f"{name}/w{window}")
            axis.set_ylabel(metric);axis.set_xlabel("Forward horizon (bars)");axis.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(args.output/"predictive_skill.png",dpi=160);plt.close(fig)
    write_json(args.output/"completion.json",{"completed":True,"fitted_models":fitted,"validation_symbol_metrics":len(rows),
        "test_evaluated":False,"report_generated":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
