"""RL agent wrapper skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "PPO": {
        "verbose": 0,
        "n_steps": 64,
        "batch_size": 32,
    },
}


class RLAgent:
    """Wrapper around future Stable-Baselines3 agents."""

    def __init__(
        self,
        model_name: str = "PPO",
        policy: str = "MlpPolicy",
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Store model configuration without forcing immediate model creation."""
        self.model_name = model_name
        self.policy = policy
        self.model_kwargs = self._merge_model_kwargs(model_kwargs)
        self.model: Any | None = None

    def _merge_model_kwargs(
        self,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge local defaults with caller-provided model options."""
        defaults = DEFAULT_MODEL_KWARGS.get(self.model_name, {})
        return {**defaults, **(model_kwargs or {})}

    def build(self, env: Any) -> None:
        """Build the underlying Stable-Baselines3 model."""
        # TODO: Add a registry for PPO, A2C, DQN, and custom policies.
        if self.model_name != "PPO":
            raise NotImplementedError(f"Model is not wired yet: {self.model_name}")

        from stable_baselines3 import PPO

        self.model = PPO(self.policy, env, **self.model_kwargs)

    def train(self, env: Any, total_timesteps: int) -> None:
        """Train the RL model."""
        # TODO: Add callbacks, evaluation environments, and checkpointing.
        if self.model is None:
            self.build(env)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, Any]:
        """Predict an action from an observation."""
        # TODO: Add action masking or risk controls before live use.
        if self.model is None:
            raise RuntimeError("RL model has not been built or loaded.")
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: str | Path) -> None:
        """Save the underlying model."""
        # TODO: Save feature schema and config alongside model weights.
        if self.model is None:
            raise RuntimeError("No model is available to save.")
        self.model.save(path)

    def load(self, path: str | Path, env: Any | None = None) -> None:
        """Load a model from disk."""
        # TODO: Restore the correct model class from metadata.
        if self.model_name != "PPO":
            raise NotImplementedError(f"Model is not wired yet: {self.model_name}")

        from stable_baselines3 import PPO

        self.model = PPO.load(path, env=env)
