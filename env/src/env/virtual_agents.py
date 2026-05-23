from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class VirtualAgent(Protocol):
    """Phase 1 가상 참여자 인터페이스.

    `history`는 `market_data.iloc[: step + 1]`. 미래 데이터에 접근 불가.
    `compute_impact`는 price impact ratio를 반환 (예: 0.001 = +0.1%).
    """

    def compute_impact(self, history: pd.DataFrame, step: int) -> float: ...

    def reset(self, seed: int | None = None) -> None: ...


class NoiseTrader:
    """랜덤하게 시장을 흔드는 가상 참여자."""

    def __init__(self, noise_strength: float = 0.001, seed: int | None = None) -> None:
        self.noise_strength = noise_strength
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def compute_impact(self, history: pd.DataFrame, step: int) -> float:
        direction = int(self.rng.choice([-1, 0, 1]))
        return float(direction * self.noise_strength)


class TrendFollower:
    """N-period 단순 추세 추종."""

    def __init__(
        self,
        trend_strength: float = 0.002,
        window: int = 5,
        price_col: str = "Close",
    ) -> None:
        self.trend_strength = trend_strength
        self.window = window
        self.price_col = price_col

    def reset(self, seed: int | None = None) -> None:
        pass

    def compute_impact(self, history: pd.DataFrame, step: int) -> float:
        if step < self.window:
            return 0.0
        past = float(history[self.price_col].iloc[step - self.window])
        curr = float(history[self.price_col].iloc[step])
        if past <= 0:
            return 0.0
        recent_return = (curr - past) / past
        if recent_return > 0:
            return self.trend_strength
        if recent_return < 0:
            return -self.trend_strength
        return 0.0


class MeanReversionTrader:
    """가격이 이동평균에서 벗어나면 되돌리려는 참여자."""

    def __init__(
        self,
        mean_reversion_strength: float = 0.001,
        ma_col: str = "ma_20",
        price_col: str = "Close",
    ) -> None:
        self.mean_reversion_strength = mean_reversion_strength
        self.ma_col = ma_col
        self.price_col = price_col

    def reset(self, seed: int | None = None) -> None:
        pass

    def compute_impact(self, history: pd.DataFrame, step: int) -> float:
        if self.ma_col not in history.columns:
            return 0.0
        row = history.iloc[step]
        price = float(row[self.price_col])
        ma = float(row[self.ma_col])
        if not np.isfinite(ma) or ma <= 0:
            return 0.0
        if price > ma:
            return -self.mean_reversion_strength
        if price < ma:
            return self.mean_reversion_strength
        return 0.0
