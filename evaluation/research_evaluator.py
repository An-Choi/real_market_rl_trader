"""Evaluate finite research episodes with the training horizon and compound cash."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import calculate_max_drawdown, calculate_sharpe_ratio


def evaluate_controlled(model, env, baseline=None):
    if baseline not in (None, "cash", "intraday_buy_and_hold", "intraday_mean_reversion"):
        raise ValueError("unknown finite-horizon control")
    initial_cash = env.initial_cash
    capital = initial_cash
    records, entropies = [], []
    counts = np.zeros(env.action_space.n, np.int64)
    probabilities_sum = np.zeros(env.action_space.n)
    masks_sum = np.zeros(env.action_space.n)
    starts = env.available_dates[::env.episode_days]
    try:
        for start in starts:
            env.initial_cash = capital
            observation, _ = env.reset(seed=0, options={"start_date": start, "episode_days": env.episode_days})
            while True:
                masks = env.action_masks()
                rebalance = True
                if baseline is None:
                    with torch.no_grad():
                        tensor, _ = model.policy.obs_to_tensor(observation)
                        distribution = model.policy.get_distribution(tensor, action_masks=masks)
                        probabilities = distribution.distribution.probs.cpu().numpy().ravel()
                    action = int(np.argmax(probabilities))
                    probabilities_sum += probabilities
                    entropies.append(float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()))
                elif baseline == "cash":
                    action = 0
                elif baseline == "intraday_buy_and_hold":
                    action, rebalance = 5, env.current_step == 0
                else:
                    z = env.normalizer.transform_observation(
                        env.get_current_market_row()[env.feature_columns].to_numpy(float))[0]
                    action = 5 if z < -1 else 0
                    rebalance = z < -1 or z > 0
                masks_sum += masks
                counts[action] += 1
                observation, _, terminated, truncated, info = env.step(action, rebalance=rebalance)
                records.extend(info["bar_records"])
                if terminated or truncated:
                    if env.shares_held != 0 or truncated:
                        raise RuntimeError("controlled evaluation must end in an absorbing cash settlement")
                    capital = env.cash
                    break
    finally:
        env.initial_cash = initial_cash
    trace = pd.DataFrame(records)
    daily = trace.groupby("valuation_date").portfolio_value.last()
    daily_returns = pd.Series([initial_cash, *daily.tolist()]).pct_change().dropna()
    decisions = int(counts.sum())
    costs = float(trace.friction_cost.sum())
    risk_cost = float(trace.risk_cost.mean())
    weighted_risk_cost = float((trace.risk_cost * trace.risk_discount_weight).sum() / trace.risk_discount_weight.sum())
    result = {"net_return": capital / initial_cash - 1,
        "bar_max_drawdown": calculate_max_drawdown(pd.Series([initial_cash, *trace.portfolio_value.tolist()])),
        "liquidation_max_drawdown": calculate_max_drawdown(pd.Series([initial_cash, *trace.liquidation_value.tolist()])),
        "daily_sharpe": calculate_sharpe_ratio(daily_returns),
        "flat_fraction": float(trace.flat.mean()), "mean_exposure": float(trace.exposure.mean()),
        "trade_count": int(trace.executed_trades.sum()),
        "unavailable_fill_bars": int(trace.fill_unavailable.sum()),
        "turnover": float(trace.absolute_trade_value.sum() / trace.portfolio_value.mean()),
        "friction_cost": costs,
        "terminal_liquidation_cost": float(trace.terminal_settlement_cost.sum()),
        "pre_cost_return_addback": (capital + costs) / initial_cash - 1,
        "mean_drawdown_excess": risk_cost,
        "discounted_drawdown_excess": weighted_risk_cost,
        "drawdown_exceedance_fraction": float((trace.risk_cost > 0).mean()),
        "risk_budget_satisfied": bool(weighted_risk_cost <= env.risk_budget),
        "action_counts": counts.tolist(), "valid_action_fraction": (masks_sum / decisions).tolist(),
        "mean_action_probability": (probabilities_sum / decisions).tolist() if model is not None else None,
        "mean_policy_entropy": float(np.mean(entropies)) if entropies else None,
        "decisions": decisions, "bars": len(trace), "days": len(env.available_dates),
        "evaluation_episodes": len(starts)}
    return result, trace
