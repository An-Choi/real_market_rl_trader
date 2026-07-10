"""Training-split feature normalization shared by training and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureNormalizer:
    """Standardize feature values while leaving portfolio state untouched."""

    feature_columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    clip: float = 5.0

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        feature_columns: list[str],
        *,
        clip: float = 5.0,
        epsilon: float = 1e-8,
    ) -> "FeatureNormalizer":
        if clip <= 0:
            raise ValueError("normalization clip must be positive")
        values = data[feature_columns].apply(pd.to_numeric, errors="coerce")
        means = values.mean().fillna(0.0)
        scales = values.std(ddof=0).fillna(0.0).clip(lower=epsilon)
        return cls(
            feature_columns=tuple(feature_columns),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
            clip=float(clip),
        )

    def transform_observation(self, observation: Any) -> np.ndarray:
        """Normalize the feature prefix of one observation or a batch."""
        result = np.asarray(observation, dtype=np.float32).copy()
        feature_count = len(self.feature_columns)
        if result.shape[-1] < feature_count:
            raise ValueError(
                f"observation has {result.shape[-1]} values, expected at least {feature_count}"
            )
        means = np.asarray(self.means, dtype=np.float32)
        scales = np.asarray(self.scales, dtype=np.float32)
        normalized = (result[..., :feature_count] - means) / scales
        result[..., :feature_count] = np.clip(normalized, -self.clip, self.clip)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "means": list(self.means),
            "scales": list(self.scales),
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureNormalizer":
        required = {"feature_columns", "means", "scales", "clip"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"normalization stats missing fields: {sorted(missing)}")
        columns = tuple(payload["feature_columns"])
        means = tuple(float(value) for value in payload["means"])
        scales = tuple(float(value) for value in payload["scales"])
        if not columns or len(columns) != len(means) or len(columns) != len(scales):
            raise ValueError("normalization stats have inconsistent dimensions")
        if any(scale <= 0 or not np.isfinite(scale) for scale in scales):
            raise ValueError("normalization scales must be finite and positive")
        clip = float(payload["clip"])
        if clip <= 0 or not np.isfinite(clip):
            raise ValueError("normalization clip must be finite and positive")
        return cls(columns, means, scales, clip)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("normalization stats must be a JSON object")
        return cls.from_dict(payload)


class NormalizedObservationEnv(gym.ObservationWrapper):
    """Apply fixed training-split statistics to environment observations."""

    def __init__(self, env: gym.Env, normalizer: FeatureNormalizer) -> None:
        super().__init__(env)
        self.normalizer = normalizer

    def observation(self, observation: Any) -> np.ndarray:
        return self.normalizer.transform_observation(observation)
