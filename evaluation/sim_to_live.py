"""Utilities for comparing simulation behavior with live-market behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimToLiveGap:
    """Compact report for simulation-to-live differences."""

    metric: str
    simulated_value: float
    live_value: float

    @property
    def absolute_gap(self) -> float:
        """Return the absolute difference between live and simulated values."""
        return abs(self.live_value - self.simulated_value)
