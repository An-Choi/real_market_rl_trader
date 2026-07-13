"""RL agent wrapper skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models.normalization import FeatureNormalizer


class RLAgent:
    """Wrapper around future Stable-Baselines3 agents."""

    def __init__(
        self,
        model_name: str = "PPO",
        policy: str = "MlpPolicy",
        model_kwargs: dict[str, Any] | None = None,
        observation_normalizer: FeatureNormalizer | None = None,
    ) -> None:
        """Store model configuration without forcing immediate model creation."""
        self.model_name = model_name
        self.policy = policy
        self.model_kwargs = model_kwargs or {}
        self.observation_normalizer = observation_normalizer
        self.model: Any | None = None

    def build(self, env: Any) -> None:
        """Build the underlying Stable-Baselines3 model."""
        # TODO: Add model dispatch for A2C, DQN, and custom policies.
        if self.model_name != "PPO":
            raise NotImplementedError(f"Model is not wired yet: {self.model_name}")

        from stable_baselines3 import PPO

        self.model = PPO(self.policy, env, **self.model_kwargs)

    def train(
        self,
        env: Any,
        total_timesteps: int,
        *,
        tb_log_name: str | None = None,
        callback: Any | None = None,
    ) -> None:
        """Train the RL model."""
        # TODO: Add callbacks, evaluation environments, and checkpointing.
        if self.model is None:
            self.build(env)
        learn_kwargs = {"total_timesteps": total_timesteps}
        if tb_log_name is not None:
            learn_kwargs["tb_log_name"] = tb_log_name
        if callback is not None:
            learn_kwargs["callback"] = callback
        self.model.learn(**learn_kwargs)

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, Any]:
        """Predict an action from an observation."""
        # TODO: Add action masking or risk controls before live use.
        if self.model is None:
            raise RuntimeError("RL model has not been built or loaded.")
        model_observation = (
            self.observation_normalizer.transform_observation(observation)
            if self.observation_normalizer is not None
            else observation
        )
        action, state = self.model.predict(model_observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: str | Path) -> None:
        """Save the underlying model."""
        # 서빙용 저장은 models.artifact.save_artifact 사용 (schema/metadata 포함).
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


def make_rl_agent(
    *,
    model_name: str = "PPO",
    policy: str = "MlpPolicy",
    seed: int | None = None,
    device: str | None = "cpu",
    model_kwargs: dict[str, Any] | None = None,
) -> RLAgent:
    """Create a learned RL agent wrapper with common reproducibility defaults."""
    kwargs = dict(model_kwargs or {})
    if seed is not None:
        kwargs.setdefault("seed", seed)
    if device is not None:
        kwargs.setdefault("device", device)
    return RLAgent(model_name=model_name, policy=policy, model_kwargs=kwargs)
