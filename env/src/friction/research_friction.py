"""Explicit all-in cost stress multiplier, including dynamic spread and tax."""
from dataclasses import dataclass
import math

from friction.friction_model import FrictionModel


@dataclass
class ScaledFrictionModel(FrictionModel):
    multiplier: float = 1.

    def __post_init__(self):
        if not math.isfinite(self.multiplier) or self.multiplier < 0:
            raise ValueError("friction multiplier must be finite and non-negative")

    def calculate_total_friction(self, *args, **kwargs):
        return self.multiplier * super().calculate_total_friction(*args, **kwargs)
