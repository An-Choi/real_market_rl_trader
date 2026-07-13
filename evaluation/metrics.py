"""Backtest metric skeletons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Calculate total return from an equity curve."""
    # TODO: Add annualized return based on timeframe.
    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate a simple annualized Sharpe ratio."""
    # TODO: Add risk-free rate and robust handling for intraday periods.
    volatility = float(returns.std()) if not returns.empty else 0.0
    if returns.empty or not np.isfinite(volatility) or volatility == 0:
        return 0.0
    return float((returns.mean() / volatility) * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from an equity curve."""
    # TODO: Return drawdown start/end dates when timestamps are available.
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def calculate_win_rate(trade_returns: pd.Series) -> float:
    """Calculate fraction of positive trade returns."""
    # TODO: Define trade return extraction from environment trade history.
    if trade_returns.empty:
        return 0.0
    return float((trade_returns > 0).mean())


def calculate_number_of_trades(actions: pd.Series) -> int:
    """Count non-hold actions as trades."""
    if actions.empty:
        return 0
    return int((actions != 0).sum())


def calculate_trade_count(results: pd.DataFrame) -> int:
    """Count executed trades when trade values are available."""
    if results.empty:
        return 0
    if "trade_value" in results:
        return int((results["trade_value"].abs() > 0).sum())
    return calculate_number_of_trades(results["action"])


def calculate_turnover(results: pd.DataFrame) -> float:
    """Calculate traded notional divided by average portfolio value."""
    if results.empty or "portfolio_value" not in results:
        return 0.0
    portfolio_values = results["portfolio_value"]
    average_value = float(portfolio_values.mean())
    if average_value == 0:
        return 0.0
    if "trade_value" in results:
        traded_notional = float(results["trade_value"].abs().sum())
        return traded_notional / max(abs(average_value), 1e-9)
    trade_count = calculate_number_of_trades(results["action"])
    return float(trade_count / max(abs(average_value), 1e-9))


def calculate_forced_clear_count(results: pd.DataFrame) -> int:
    """Count forced end-of-day clear events."""
    if results.empty or "forced_clear" not in results:
        return 0
    return int(results["forced_clear"].fillna(False).astype(bool).sum())


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    """Create a compact metric summary from backtest results."""
    # TODO: Add benchmark-relative metrics and transaction cost breakdowns.
    if results.empty:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "final_portfolio_value": 0.0,
            "win_rate": 0.0,
            "trade_count": 0.0,
            "number_of_trades": 0.0,
            "turnover": 0.0,
            "forced_clear_count": 0.0,
            "evaluated_days": 0.0,
            "hold_action_rate": 0.0,
            "add_action_rate": 0.0,
            "clear_action_rate": 0.0,
        }

    has_episode_boundaries = {
        "episode",
        "initial_portfolio_value",
    }.issubset(results.columns)
    if has_episode_boundaries:
        equity_parts: list[pd.Series] = []
        episode_returns: list[float] = []
        capital_factor = 1.0
        initial_value = float(results["initial_portfolio_value"].iloc[0])
        for _, episode in results.groupby("episode", sort=False):
            episode_initial = float(episode["initial_portfolio_value"].iloc[0])
            normalized_equity = episode["portfolio_value"].astype(float) / episode_initial
            equity_parts.append(normalized_equity * initial_value * capital_factor)
            episode_return = float(normalized_equity.iloc[-1] - 1.0)
            episode_returns.append(episode_return)
            capital_factor *= 1.0 + episode_return
        equity_curve = pd.concat(equity_parts, ignore_index=True)
        returns = pd.Series(episode_returns, dtype="float64")
        total_return = capital_factor - 1.0
        final_portfolio_value = initial_value * capital_factor
        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
        capital_base = sum(
            float(episode["initial_portfolio_value"].iloc[0])
            for _, episode in results.groupby("episode", sort=False)
        )
        turnover = (
            float(results["trade_value"].abs().sum()) / max(capital_base, 1e-9)
            if "trade_value" in results
            else 0.0
        )
        if "timestamp" in results:
            evaluated_days = float(
                pd.to_datetime(results["timestamp"]).dt.date.nunique()
            )
        else:
            evaluated_days = float(results["episode"].nunique())
    else:
        equity_curve = results["portfolio_value"]
        returns = equity_curve.pct_change().dropna()
        total_return = calculate_total_return(equity_curve)
        final_portfolio_value = float(equity_curve.iloc[-1])
        win_rate = calculate_win_rate(results["reward"])
        turnover = calculate_turnover(results)
        evaluated_days = 1.0

    trade_count = float(calculate_trade_count(results))
    action_rates = results["action"].value_counts(normalize=True)
    return {
        "total_return": float(total_return),
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "final_portfolio_value": final_portfolio_value,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "number_of_trades": trade_count,
        "turnover": turnover,
        "forced_clear_count": float(calculate_forced_clear_count(results)),
        "evaluated_days": evaluated_days,
        "hold_action_rate": float(action_rates.get(0, 0.0)),
        "add_action_rate": float(action_rates.get(1, 0.0)),
        "clear_action_rate": float(action_rates.get(2, 0.0)),
    }
