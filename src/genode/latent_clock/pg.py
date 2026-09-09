from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from genode.latent_clock.clocks import Clock, clock_from_interval_logits


class ClockPolicy(nn.Module):
    def __init__(self, context_dim: int, target_nfe: int):
        super().__init__()
        if context_dim <= 0 or target_nfe < 2:
            raise ValueError("Clock policy dimensions must be positive.")
        self.target_nfe = int(target_nfe)
        self.network = nn.Sequential(
            nn.Linear(int(context_dim), 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self.target_nfe - 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.log_std = nn.Parameter(torch.full((self.target_nfe - 1,), math.log(0.5)))

    def distribution(self, context: torch.Tensor) -> torch.distributions.Independent:
        mean = self.network(context)
        std = torch.exp(self.log_std.clamp(-5.0, 1.0)).expand_as(mean)
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)


@dataclass(frozen=True)
class PGRollout:
    round_index: int
    prompt_id: str
    interval_logits: tuple[float, ...]
    mean_utility: float


def fit_pg_clock_policy(
    contexts: Mapping[str, np.ndarray],
    target_nfe: int,
    evaluate: Callable[[str, Clock, int], float],
    *,
    seed: int,
    rounds: int = 24,
    learning_rate: float = 3e-4,
    clip_ratio: float = 0.2,
    update_epochs: int = 4,
    kl_coefficient: float = 0.01,
    entropy_coefficient: float = 0.001,
) -> tuple[ClockPolicy, tuple[PGRollout, ...]]:
    """Fit the preregistered DDPO-inspired policy over complete clocks."""
    if not contexts or int(rounds) != 24:
        raise ValueError("PG requires contexts and exactly 24 post-anchor rounds.")
    prompt_ids = sorted(contexts)
    matrix = np.stack([np.asarray(contexts[key], dtype=np.float32) for key in prompt_ids])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("PG contexts must form a finite matrix.")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    policy = ClockPolicy(matrix.shape[1], int(target_nfe))
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    base_mean = torch.zeros(int(target_nfe) - 1)
    base_std = torch.full((int(target_nfe) - 1,), 0.5)
    rollouts: list[PGRollout] = []
    context_tensor = torch.tensor(matrix, dtype=torch.float32)
    for round_index in range(int(rounds)):
        permutation = rng.permutation(len(prompt_ids))
        with torch.no_grad():
            old_distribution = policy.distribution(context_tensor)
            actions = old_distribution.sample()
            old_log_prob = old_distribution.log_prob(actions)
        rewards = []
        for position in range(len(prompt_ids)):
            prompt_id = prompt_ids[position]
            values = actions[position].cpu().numpy()
            clock = clock_from_interval_logits(f"pg_r{round_index:02d}_{prompt_id}", values, pg_precision=True)
            reward = float(evaluate(prompt_id, clock, int(seed) + round_index))
            if not math.isfinite(reward):
                raise ValueError("PG evaluator returned a non-finite reward.")
            rewards.append(reward)
            rollouts.append(PGRollout(round_index, prompt_id, tuple(float(x) for x in values), reward))
        advantage = torch.tensor(rewards, dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        for _ in range(int(update_epochs)):
            for start in range(0, len(prompt_ids), 16):
                index = torch.tensor(permutation[start : start + 16], dtype=torch.long)
                distribution = policy.distribution(context_tensor.index_select(0, index))
                log_prob = distribution.log_prob(actions.index_select(0, index))
                ratio = torch.exp(log_prob - old_log_prob.index_select(0, index))
                local_advantage = advantage.index_select(0, index)
                clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * local_advantage
                policy_loss = -torch.minimum(ratio * local_advantage, clipped).mean()
                normal = distribution.base_dist
                base = torch.distributions.Normal(base_mean, base_std)
                kl = torch.distributions.kl_divergence(normal, base).sum(-1).mean()
                loss = (
                    policy_loss
                    + float(kl_coefficient) * kl
                    - float(entropy_coefficient) * distribution.entropy().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    return policy.eval(), tuple(rollouts)


def sample_policy_clock(policy: ClockPolicy, context: np.ndarray, *, seed: int, key: str = "pg") -> Clock:
    torch.manual_seed(int(seed))
    value = torch.tensor(np.asarray(context, dtype=np.float32)).reshape(1, -1)
    with torch.no_grad():
        logits = policy.distribution(value).sample()[0].cpu().numpy()
    return clock_from_interval_logits(key, logits, pg_precision=True)
