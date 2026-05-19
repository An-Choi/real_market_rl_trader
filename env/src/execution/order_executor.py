"""Order execution abstractions for simulated and paper trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderExecution:
    """Result of a simulated order execution."""

    action: int
    fill_price: float
    quantity: float
    friction_cost: float


class OrderExecutor:
    """Minimal immediate-fill executor used before order-book simulation."""

    def execute(
        self,
        action: int,
        price: float,
        quantity: float,
        friction_cost: float = 0.0,
    ) -> OrderExecution:
        """Return an immediate fill at the supplied price."""
        return OrderExecution(
            action=action,
            fill_price=price,
            quantity=quantity,
            friction_cost=friction_cost,
        )
