"""Backtest metrics on daily settled equity (valuation-date attribution)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Calculate total return from an equity curve."""
    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe from daily returns."""
    volatility = float(returns.std()) if not returns.empty else 0.0
    if returns.empty or not np.isfinite(volatility) or volatility == 0:
        return 0.0
    return float((returns.mean() / volatility) * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown from a daily equity curve."""
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def calculate_win_rate(trade_returns: pd.Series) -> float:
    """Calculate fraction of positive trade returns."""
    if trade_returns.empty:
        return 0.0
    return float((trade_returns > 0).mean())


def calculate_number_of_trades(actions: pd.Series) -> int:
    """Count non-hold actions as trades."""
    if actions.empty:
        return 0
    return int((actions != 0).sum())


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


def calculate_trade_count(results: pd.DataFrame) -> int:
    """Count executed trades (rows with nonzero trade_value)."""
    if results.empty:
        return 0
    if "trade_value" in results:
        return int((results["trade_value"].abs() > 0).sum())
    return int((results["action"] != 0).sum())


def build_daily_frame(
    results: pd.DataFrame,
    *,
    initial_value: float,
    terminal_liquidation_cost: float = 0.0,
) -> pd.DataFrame:
    """valuation date 기준 일별 equity와 수익률 분해 (spec r5 §5.2).

    모든 metric 계산 전에 마지막 close equity E_D를 E_D_settled로 교체한다.
    (1+r_d) = (1+r_transition)(1+r_intraday)(1+r_settlement).
    """
    val_dates = pd.to_datetime(results["valuation_timestamp"]).dt.date.to_numpy()
    grouped = results.groupby(val_dates, sort=True)["portfolio_value"]
    close_raw = grouped.last().astype(float)
    open_equity = grouped.first().astype(float)

    settled_close = close_raw.copy()
    settled_close.iloc[-1] = close_raw.iloc[-1] - terminal_liquidation_cost

    prev_close = settled_close.shift(1)
    prev_close.iloc[0] = float(initial_value)

    daily_return = settled_close / prev_close - 1
    r_transition = open_equity / prev_close - 1
    r_transition.iloc[0] = 0.0  # 첫 valuation 날짜는 경계 없음
    r_intraday = close_raw / open_equity - 1
    r_intraday.iloc[0] = close_raw.iloc[0] / float(initial_value) - 1
    r_settlement = pd.Series(0.0, index=close_raw.index)
    r_settlement.iloc[-1] = settled_close.iloc[-1] / close_raw.iloc[-1] - 1

    return pd.DataFrame({
        "close_equity": settled_close,
        "open_equity": open_equity,
        "daily_return": daily_return,
        "r_transition": r_transition,
        "r_intraday": r_intraday,
        "r_settlement": r_settlement,
    })


_EMPTY_SUMMARY = {
    "total_return": 0.0,
    "final_portfolio_value": 0.0,
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "win_rate": 0.0,
    "trade_count": 0.0,
    "number_of_trades": 0.0,
    "turnover": 0.0,
    "evaluated_days": 0.0,
    "overnight_hold_rate": 0.0,
    "open_at_end": 0.0,
    "terminal_liquidation_cost": 0.0,
    "market_return": 0.0,
    "hold_action_rate": 0.0,
    "add_action_rate": 0.0,
    "clear_action_rate": 0.0,
    "cum_transition_return": 0.0,
    "cum_intraday_return": 0.0,
    "cum_settlement_return": 0.0,
}


def summarize_backtest(
    results: pd.DataFrame,
    *,
    initial_value: float,
    terminal_liquidation_cost: float = 0.0,
    initial_market_price: float | None = None,
) -> dict[str, float]:
    """단일 연속 episode 결과를 일별 settled 시리즈로 요약한다."""
    if results.empty:
        return dict(_EMPTY_SUMMARY)

    daily = build_daily_frame(
        results,
        initial_value=initial_value,
        terminal_liquidation_cost=terminal_liquidation_cost,
    )
    returns = daily["daily_return"]
    equity_curve = pd.concat(
        [pd.Series([float(initial_value)]), daily["close_equity"]],
        ignore_index=True,
    )
    total_return = float((1.0 + returns).prod() - 1.0)

    exec_dates = pd.to_datetime(results["timestamp"]).dt.date.to_numpy()
    val_dates = pd.to_datetime(results["valuation_timestamp"]).dt.date.to_numpy()
    boundary_rows = results[val_dates != exec_dates]
    overnight_hold_rate = (
        float((boundary_rows["units_held"] > 0).mean()) if len(boundary_rows) else 0.0
    )

    market_return = 0.0
    if initial_market_price and "valuation_price" in results:
        market_return = float(
            results["valuation_price"].iloc[-1] / initial_market_price - 1.0
        )

    trade_count = float(calculate_trade_count(results))
    turnover = (
        float(results["trade_value"].abs().sum()) / max(float(initial_value), 1e-9)
        if "trade_value" in results
        else 0.0
    )
    action_rates = results["action"].value_counts(normalize=True)
    cum_transition_return = float((1 + daily["r_transition"]).prod() - 1)
    cum_intraday_return = float((1 + daily["r_intraday"]).prod() - 1)
    cum_settlement_return = float((1 + daily["r_settlement"]).prod() - 1)
    return {
        "total_return": total_return,
        "final_portfolio_value": float(initial_value) * (1.0 + total_return),
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "win_rate": float((returns > 0).mean()),
        "trade_count": trade_count,
        "number_of_trades": trade_count,
        "turnover": turnover,
        "evaluated_days": float(len(daily)),
        "overnight_hold_rate": overnight_hold_rate,
        "open_at_end": float(results["units_held"].iloc[-1] > 0),
        "terminal_liquidation_cost": float(terminal_liquidation_cost),
        "market_return": market_return,
        "hold_action_rate": float(action_rates.get(0, 0.0)),
        "add_action_rate": float(action_rates.get(1, 0.0)),
        "clear_action_rate": float(action_rates.get(2, 0.0)),
        "cum_transition_return": cum_transition_return,
        "cum_intraday_return": cum_intraday_return,
        "cum_settlement_return": cum_settlement_return,
    }
