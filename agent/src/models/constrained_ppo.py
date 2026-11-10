"""Duration-aware GAE and rollout-wise projected dual ascent for MaskablePPO.

This is a PPO-Lagrangian research implementation with one penalized-return
critic, not CPO and not a guarantee of satisfying a hard risk constraint.
"""
from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback


class DurationAwareRolloutBuffer(MaskableRolloutBuffer):
    def reset(self):
        super().reset()
        self.durations = np.ones((self.buffer_size, self.n_envs), np.int32)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.detach().cpu().numpy().flatten()
        last_gae = np.zeros(self.n_envs, np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                alive, next_values = 1. - dones.astype(np.float32), last_values
            else:
                alive, next_values = 1. - self.episode_starts[step + 1], self.values[step + 1]
            duration = self.durations[step]
            discount = self.gamma ** duration
            delta = self.rewards[step] + discount * next_values * alive - self.values[step]
            # Lambda is defined per underlying bar, just like gamma.
            last_gae = delta + (self.gamma * self.gae_lambda) ** duration * alive * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values


class ResearchPPO(MaskablePPO):
    """Stop after a completed rollout reaches the underlying-bar budget."""

    def collect_rollouts(self, *args, **kwargs):
        budget = getattr(self, "research_bar_budget", 0)
        if budget and getattr(self, "research_bar_transitions", 0) >= budget:
            return False
        return super().collect_rollouts(*args, **kwargs)


def initialize_cash_prior(model, logit_bias):
    """A disclosed weak initial cash preference; every action remains sampled.

    Apply exactly once before training. Save initial and trained diagnostics so
    retaining a prior is not mistaken for learning a cash policy from scratch.
    """
    if not np.isfinite(logit_bias) or not 0 <= logit_bias <= 5:
        raise ValueError("cash logit bias must be in [0, 5]")
    with torch.no_grad():
        model.policy.action_net.bias[0] += logit_bias


def projected_dual_update(multiplier, mean_cost, budget, learning_rate, maximum=10.):
    values = (multiplier, mean_cost, budget, learning_rate, maximum)
    if not np.isfinite(values).all() or min(values) < 0 or maximum == 0:
        raise ValueError("dual update requires finite non-negative inputs and positive maximum")
    return float(np.clip(multiplier + learning_rate * (mean_cost - budget), 0., maximum))


class RiskBudgetCallback(BaseCallback):
    """Update lambda from training bars only; keep it fixed within each rollout."""

    def __init__(self, *, mode="off", initial_multiplier=0., budget=.001,
                 learning_rate=10., maximum=10.):
        super().__init__()
        if mode not in ("off", "fixed", "dual"):
            raise ValueError("unknown risk mode")
        values = (initial_multiplier, budget, learning_rate, maximum)
        if not np.isfinite(values).all() or min(values) < 0 or not 0 < maximum <= 10 or initial_multiplier > maximum or budget >= 1:
            raise ValueError("invalid risk budget or dual update settings")
        if mode == "off" and initial_multiplier != 0:
            raise ValueError("off mode requires a zero multiplier")
        self.mode, self.multiplier, self.budget = mode, initial_multiplier, budget
        self.learning_rate, self.maximum = learning_rate, maximum
        self.history = []

    def _on_training_start(self):
        if not isinstance(self.model.rollout_buffer, DurationAwareRolloutBuffer):
            raise TypeError("controlled exposure requires DurationAwareRolloutBuffer")
        if not np.allclose(self.training_env.get_attr("bar_gamma"), self.model.gamma, rtol=0, atol=1e-12):
            raise ValueError("environment and PPO must use the same per-bar gamma")
        self.model.research_bar_transitions = 0
        self._publish()

    def _publish(self):
        self.model.research_risk_multiplier = self.multiplier
        self.training_env.env_method("set_risk_multiplier", self.multiplier)
        # Changing lambda changes the observation. Refresh the cached current
        # observation before the next rollout (without resetting the portfolio).
        self.model._last_obs = np.asarray(self.training_env.env_method("_get_observation"))

    def _on_rollout_start(self):
        self.cost_sum, self.bar_count = 0., 0
        self.weighted_cost_sum, self.weight_sum = 0., 0.

    def _on_step(self):
        buffer = self.model.rollout_buffer
        infos = self.locals["infos"]
        if not isinstance(buffer, DurationAwareRolloutBuffer):
            raise TypeError("controlled exposure requires DurationAwareRolloutBuffer")
        buffer.durations[buffer.pos] = [info["elapsed_bars"] for info in infos]
        self.cost_sum += sum(info["risk_cost_sum"] for info in infos)
        self.weighted_cost_sum += sum(info["weighted_risk_cost_sum"] for info in infos)
        self.weight_sum += sum(info["risk_discount_weight_sum"] for info in infos)
        self.bar_count += sum(info["elapsed_bars"] for info in infos)
        return True

    def _on_rollout_end(self):
        self.model.research_bar_transitions += self.bar_count
        mean = self.weighted_cost_sum / max(self.weight_sum, 1e-9)
        previous = self.multiplier
        if self.mode == "dual":
            self.multiplier = projected_dual_update(previous, mean, self.budget,
                                                    self.learning_rate, self.maximum)
        self.history.append({"timesteps": self.num_timesteps, "bars": self.bar_count,
            "mean_risk_cost": mean, "budget": self.budget,
            "unweighted_mean_risk_cost": self.cost_sum / max(self.bar_count, 1),
            "multiplier_before": previous, "multiplier_after": self.multiplier})
        self.logger.record("risk/discounted_drawdown_excess", mean)
        self.logger.record("risk/unweighted_drawdown_excess", self.cost_sum / max(self.bar_count, 1))
        self.logger.record("risk/multiplier", self.multiplier)
        self.logger.record("risk/budget", self.budget)
        self._publish()
