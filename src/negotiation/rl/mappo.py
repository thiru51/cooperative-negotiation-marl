from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .networks import Actor, CentralisedCritic, ValueNormalizer


@dataclass
class MAPPOConfig:
    horizon: int = 128
    num_envs: int = 16
    hidden: int = 128
    lr: float = 3e-4
    eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    epochs: int = 8
    num_minibatches: int = 4
    max_grad_norm: float = 0.5
    use_value_norm: bool = True
    anneal_lr: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class MAPPO:
    """Multi-agent PPO with a centralised critic and decentralised, parameter-shared actors.

    Written out rather than imported so every choice below is inspectable: the clipped
    surrogate, the clipped value loss, per-batch advantage normalisation, and the value
    normaliser are the four tricks Yu et al. found actually mattered.
    """

    def __init__(self, obs_dim: int, state_dim: int, n_actions: int,
                 config: MAPPOConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = config or MAPPOConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)

        self.actor = Actor(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.critic = CentralisedCritic(state_dim, self.cfg.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            [{"params": self.actor.parameters()}, {"params": self.critic.parameters()}],
            lr=self.cfg.lr, eps=self.cfg.eps,
        )
        self.value_normalizer = ValueNormalizer(device=self.device) if self.cfg.use_value_norm else None
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim

    def make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(self.cfg.horizon, self.cfg.num_envs, 2,
                             self.obs_dim, self.state_dim, self.device)

    @torch.no_grad()
    def step_policy(self, obs: np.ndarray, states: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, device=self.device).reshape(-1, self.obs_dim)
        states_t = torch.as_tensor(states, device=self.device).reshape(-1, self.state_dim)
        actions, log_probs = self.actor.act(obs_t, deterministic)
        values = self.critic(states_t)
        shape = obs.shape[:-1]
        return (actions.reshape(shape).cpu().numpy(),
                log_probs.reshape(shape).cpu().numpy(),
                values.reshape(shape).cpu().numpy())

    @torch.no_grad()
    def value(self, states: np.ndarray) -> np.ndarray:
        states_t = torch.as_tensor(states, device=self.device).reshape(-1, self.state_dim)
        return self.critic(states_t).reshape(states.shape[:-1]).cpu().numpy()

    def update(self, buffer: RolloutBuffer, lr_frac: float = 1.0) -> dict:
        if self.cfg.anneal_lr:
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.lr * lr_frac

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
        n_updates = 0

        for _ in range(self.cfg.epochs):
            for batch in buffer.minibatches(self.cfg.num_minibatches, self.generator):
                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_probs, entropy = self.actor.evaluate(batch["obs"], batch["actions"])
                ratio = torch.exp(log_probs - batch["log_probs"])
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = self._value_loss(batch)
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - batch["log_probs"]
                    # Schulman's low-variance KL estimator; the naive -mean(log_ratio) is
                    # unbiased but noisy enough to be useless as a stopping signal.
                    approx_kl = torch.mean((ratio - 1) - log_ratio)
                    clip_frac = torch.mean((torch.abs(ratio - 1) > self.cfg.clip_ratio).float())

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy.mean())
                stats["approx_kl"] += float(approx_kl)
                stats["clip_frac"] += float(clip_frac)
                stats["grad_norm"] += float(grad_norm)
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    def _value_loss(self, batch) -> torch.Tensor:
        values = self.critic(batch["states"])
        returns = batch["returns"]
        if self.value_normalizer is not None:
            self.value_normalizer.update(returns)
            returns = self.value_normalizer.normalize(returns)

        clipped = batch["values"] + torch.clamp(
            values - batch["values"], -self.cfg.value_clip, self.cfg.value_clip
        )
        loss_unclipped = (values - returns) ** 2
        loss_clipped = (clipped - returns) ** 2
        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

    def save(self, path) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict() if self.value_normalizer else None,
            "config": self.cfg.as_dict(),
            "dims": {"obs": self.obs_dim, "state": self.state_dim, "actions": self.n_actions},
        }, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if self.value_normalizer is not None and ckpt.get("value_normalizer"):
            self.value_normalizer.load_state_dict(ckpt["value_normalizer"])
