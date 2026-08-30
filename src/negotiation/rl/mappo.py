from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ..device import DeviceSetup, setup_device
from .buffer import RolloutBuffer
from .networks import Actor, CentralisedCritic, ValueNormalizer


@dataclass
class MAPPOConfig:
    horizon: int = 128
    num_envs: int = 64
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
    # If set, overrides num_minibatches: the rollout is cut into ceil(rows / batch_size)
    # pieces instead. Easier to reason about when sizing a run to a particular GPU, since
    # it is the number that decides how much VRAM one backward pass needs.
    batch_size: int | None = None
    max_grad_norm: float = 0.5
    use_value_norm: bool = True
    anneal_lr: bool = True
    amp: bool = True
    compile: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class MAPPO:
    """Multi-agent PPO with a centralised critic and decentralised, parameter-shared actors.

    Written out rather than imported so every choice below is inspectable: the clipped
    surrogate, the clipped value loss, per-batch advantage normalisation, and the value
    normaliser are the four tricks Yu et al. found actually mattered.
    """

    def __init__(self, obs_dim: int, state_dim: int, n_actions: int,
                 config: MAPPOConfig | None = None,
                 device: str | DeviceSetup = "cpu", seed: int = 0):
        self.cfg = config or MAPPOConfig()
        self.dev = device if isinstance(device, DeviceSetup) else setup_device(device, self.cfg.amp)
        self.device = self.dev.device
        torch.manual_seed(seed)

        self.actor = Actor(obs_dim, n_actions, self.cfg.hidden).to(self.device)
        self.critic = CentralisedCritic(state_dim, self.cfg.hidden).to(self.device)
        if self.cfg.compile:
            # Only the MLP bodies. Compiling the modules themselves would leave `act` and
            # `evaluate` calling the eager forward, so nothing would actually be compiled.
            self.actor.net = torch.compile(self.actor.net)
            self.critic.net = torch.compile(self.critic.net)

        self.optimizer = torch.optim.Adam(
            [{"params": self.actor.parameters()}, {"params": self.critic.parameters()}],
            lr=self.cfg.lr, eps=self.cfg.eps,
        )
        self.scaler = self.dev.scaler()
        self.value_normalizer = ValueNormalizer(device=self.device) if self.cfg.use_value_norm else None
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim

    def make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(self.cfg.horizon, self.cfg.num_envs, 2,
                             self.obs_dim, self.state_dim, self.device)

    def _to_device(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(
            self.device, non_blocking=True)

    @torch.no_grad()
    def step_policy(self, obs, states, deterministic: bool = False):
        """One batched forward covering every environment and both agents at once.

        obs is (envs, 2, obs_dim), flattened to (2*envs, obs_dim), so a vector env of any
        width costs exactly one actor call and one critic call per step. Returns device
        tensors; the caller pulls back only the actions, which is all the CPU physics needs.
        """
        shape = tuple(obs.shape[:-1])
        obs_t = self._to_device(obs).reshape(-1, self.obs_dim)
        states_t = self._to_device(states).reshape(-1, self.state_dim)
        actions, log_probs = self.actor.act(obs_t, deterministic)
        values = self.critic(states_t)
        return actions.reshape(shape), log_probs.reshape(shape), values.reshape(shape)

    @torch.no_grad()
    def value(self, states) -> torch.Tensor:
        shape = tuple(states.shape[:-1])
        states_t = self._to_device(states).reshape(-1, self.state_dim)
        return self.critic(states_t).reshape(shape)

    def num_minibatches(self, buffer: RolloutBuffer) -> int:
        if self.cfg.batch_size:
            return max(1, math.ceil(buffer.num_samples / self.cfg.batch_size))
        return self.cfg.num_minibatches

    def update(self, buffer: RolloutBuffer, lr_frac: float = 1.0) -> dict:
        if self.cfg.anneal_lr:
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.lr * lr_frac

        n_minibatches = self.num_minibatches(buffer)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")
        running = {k: torch.zeros((), device=self.device) for k in keys}
        n_updates = 0

        for _ in range(self.cfg.epochs):
            for batch in buffer.minibatches(n_minibatches, self.generator):
                # Advantages stay fp32 throughout: they come out of a length-T recursion
                # and are then standardised, so their scale is not something to hand to a
                # low-precision type.
                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                with self.dev.autocast():
                    log_probs, entropy = self.actor.evaluate(batch["obs"], batch["actions"])
                    values = self.critic(batch["states"])

                ratio = torch.exp(log_probs - batch["log_probs"])
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = self._value_loss(values, batch)
                loss = (policy_loss + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy.mean())

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                # Unscale before clipping, or the clip threshold gets applied to gradients
                # that are still multiplied by the loss scale. A no-op under bf16.
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    log_ratio = log_probs - batch["log_probs"]
                    # Schulman's low-variance KL estimator; the naive -mean(log_ratio) is
                    # unbiased but noisy enough to be useless as a stopping signal.
                    running["approx_kl"] += torch.mean((ratio - 1) - log_ratio)
                    running["clip_frac"] += torch.mean(
                        (torch.abs(ratio - 1) > self.cfg.clip_ratio).float())
                    running["policy_loss"] += policy_loss
                    running["value_loss"] += value_loss
                    running["entropy"] += entropy.mean()
                    running["grad_norm"] += grad_norm
                n_updates += 1

        # One device-to-host sync for the whole update instead of six per minibatch.
        return {k: float(v) / max(n_updates, 1) for k, v in running.items()}

    def _value_loss(self, values: torch.Tensor, batch) -> torch.Tensor:
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
            "actor": _uncompiled_state_dict(self.actor),
            "critic": _uncompiled_state_dict(self.critic),
            "value_normalizer": self.value_normalizer.state_dict() if self.value_normalizer else None,
            "config": self.cfg.as_dict(),
            "dims": {"obs": self.obs_dim, "state": self.state_dim, "actions": self.n_actions},
        }, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(_restore_keys(self.actor, ckpt["actor"]))
        self.critic.load_state_dict(_restore_keys(self.critic, ckpt["critic"]))
        if self.value_normalizer is not None and ckpt.get("value_normalizer"):
            self.value_normalizer.load_state_dict(ckpt["value_normalizer"])


def _uncompiled_state_dict(module: nn.Module) -> dict:
    # torch.compile wraps the body in an OptimizedModule, which inserts `_orig_mod.` into
    # every key. Strip it so a checkpoint loads with or without --compile.
    return {k.replace("_orig_mod.", ""): v for k, v in module.state_dict().items()}


def _restore_keys(module: nn.Module, state: dict) -> dict:
    if any("_orig_mod." in k for k in module.state_dict()):
        return {k.replace("net.", "net._orig_mod."): v for k, v in state.items()}
    return state
