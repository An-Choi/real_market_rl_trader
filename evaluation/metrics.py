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
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


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
    # TODO: Count actual fills instead of submitted actions.
    if actions.empty:
        return 0
    return int((actions != 0).sum())


def calculate_turnover(actions: pd.Series, portfolio_values: pd.Series) -> float:
    """Calculate placeholder turnover from action activity."""
    # TODO: Replace with traded notional divided by average portfolio value.
    if actions.empty or portfolio_values.empty:
        return 0.0
    trade_count = calculate_number_of_trades(actions)
    average_value = portfolio_values.mean()
    return float(trade_count / max(average_value, 1e-9))


def calculate_total_friction_cost(results: pd.DataFrame) -> float:
    """Calculate total friction paid during a backtest."""
    if "friction_cost" not in results or results["friction_cost"].empty:
        return 0.0
    return float(results["friction_cost"].sum())


def calculate_average_friction_cost(results: pd.DataFrame) -> float:
    """Calculate average friction cost per step."""
    if "friction_cost" not in results or results["friction_cost"].empty:
        return 0.0
    return float(results["friction_cost"].mean())


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    """Create a compact metric summary from backtest results."""
    # TODO: Add benchmark-relative metrics and transaction cost breakdowns.
    if results.empty:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "number_of_trades": 0.0,
            "turnover": 0.0,
            "total_friction_cost": 0.0,
            "average_friction_cost": 0.0,
            "friction_to_initial_value": 0.0,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "final_portfolio_value": 0.0,
        }

    equity_curve = results["portfolio_value"]
    returns = equity_curve.pct_change().dropna()
    total_friction_cost = calculate_total_friction_cost(results)
    initial_value = float(equity_curve.iloc[0])
    return {
        "total_return": calculate_total_return(equity_curve),
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "win_rate": calculate_win_rate(results["reward"]),
        "number_of_trades": float(calculate_number_of_trades(results["action"])),
        "turnover": calculate_turnover(results["action"], equity_curve),
        "total_friction_cost": total_friction_cost,
        "average_friction_cost": calculate_average_friction_cost(results),
        "friction_to_initial_value": total_friction_cost / max(initial_value, 1e-9),
        "total_reward": float(results["reward"].sum()),
        "average_reward": float(results["reward"].mean()),
        "final_portfolio_value": float(equity_curve.iloc[-1]),
    }
