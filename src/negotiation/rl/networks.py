from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(in_dim: int, hidden: int, out_dim: int, out_gain: float) -> nn.Sequential:
    layers = [
        _init(nn.Linear(in_dim, hidden)), nn.Tanh(),
        _init(nn.Linear(hidden, hidden)), nn.Tanh(),
        _init(nn.Linear(hidden, out_dim), gain=out_gain),
    ]
    return nn.Sequential(*layers)


def _init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Actor(nn.Module):
    """Shared across both agents, with no agent-identity input.

    Withholding the index is the point: if the actor could condition on "am I agent 0",
    it could learn a fixed tie-break and the intersection would be solved by convention
    rather than by inferring the other car's intent. The only thing that can break the
    symmetry is the observation, which includes this agent's own filter posterior.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, hidden, n_actions, out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> Categorical:
        # The matmuls run in whatever dtype autocast picked, but the logits come back to
        # fp32 before the softmax. Everything downstream -- the log-ratio, the 0.2 clip
        # test, the entropy bonus -- compares small differences between numbers near 1,
        # and bf16 carries about three decimal digits, which is not enough for that.
        return Categorical(logits=self.net(obs).float())

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self(obs)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self(obs)
        return dist.log_prob(actions), dist.entropy()


class CentralisedCritic(nn.Module):
    """Sees both agents' observations. Input is ordered self-first, other-second, so the
    same weights serve both agents without needing an identity feature."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(state_dim, hidden, 1, out_gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1).float()


class ValueNormalizer:
    """Running mean/std on the value targets.

    Straight out of the MAPPO paper's recipe: returns here drift a lot over training
    (early policies deadlock and collect the time penalty forever, later ones clear in a
    couple of seconds), and an unnormalised critic spends most of its capacity chasing
    that shift instead of the relative ordering of states.
    """

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5, device="cpu"):
        self.mean = torch.zeros((), device=device)
        self.mean_sq = torch.zeros((), device=device)
        # Both moments are accumulated with the same decay and divided by the same
        # accumulated weight, otherwise the early estimates are biased towards zero by
        # different amounts and the variance can come out negative.
        self.debias = torch.zeros((), device=device)
        self.beta = beta
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        weight = self.beta ** x.numel()
        self.mean = self.mean * weight + x.mean() * (1 - weight)
        self.mean_sq = self.mean_sq * weight + (x**2).mean() * (1 - weight)
        self.debias = self.debias * weight + (1 - weight)

    def _moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.debias.clamp(min=self.epsilon)
        mean = self.mean / d
        var = (self.mean_sq / d - mean**2).clamp(min=1e-2)
        return mean, var

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._moments()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._moments()
        return x * torch.sqrt(var) + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "mean_sq": self.mean_sq, "debias": self.debias}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.mean_sq, self.debias = d["mean"], d["mean_sq"], d["debias"]
