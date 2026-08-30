from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size on-policy storage, laid out (T, envs, agents, ...).

    Everything lives on the training device for the whole run. The alternative -- numpy
    storage plus a host-to-device copy inside every epoch -- moves the entire rollout
    across PCIe `epochs` times per update for no reason; here the only traffic is one
    small copy per environment step, which is unavoidable because the physics is on the
    CPU.

    The two agents are treated as separate samples that happen to share a policy, so all
    the PPO maths below is the single-agent version applied to T*envs*agents rows.
    """

    def __init__(self, horizon: int, num_envs: int, num_agents: int,
                 obs_dim: int, state_dim: int, device: torch.device):
        self.horizon = horizon
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        shape = (horizon, num_envs, num_agents)
        f32 = dict(dtype=torch.float32, device=device)
        self.obs = torch.zeros((*shape, obs_dim), **f32)
        self.states = torch.zeros((*shape, state_dim), **f32)
        self.actions = torch.zeros(shape, dtype=torch.int64, device=device)
        self.log_probs = torch.zeros(shape, **f32)
        self.values = torch.zeros(shape, **f32)
        self.rewards = torch.zeros(shape, **f32)
        # `dones` marks a genuine terminal state; `truncateds` marks the time limit, where
        # the value of the final state still has to be bootstrapped.
        self.dones = torch.zeros((horizon, num_envs), **f32)
        self.truncateds = torch.zeros((horizon, num_envs), **f32)
        self.advantages = torch.zeros(shape, **f32)
        self.returns = torch.zeros(shape, **f32)
        self.step = 0

    def _stage(self, x, dtype=torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype, non_blocking=True)
        return torch.from_numpy(np.ascontiguousarray(x)).to(
            self.device, dtype=dtype, non_blocking=True)

    def add(self, obs, states, actions, log_probs, values, rewards, dones, truncateds) -> None:
        t = self.step
        self.obs[t] = self._stage(obs)
        self.states[t] = self._stage(states)
        self.actions[t] = self._stage(actions, dtype=torch.int64)
        self.log_probs[t] = self._stage(log_probs)
        self.values[t] = self._stage(values)
        self.rewards[t] = self._stage(rewards)
        self.dones[t] = self._stage(dones)
        self.truncateds[t] = self._stage(truncateds)
        self.step += 1

    def reset(self) -> None:
        self.step = 0

    @torch.no_grad()
    def compute_gae(self, last_values, gamma: float, lam: float,
                    value_normalizer=None) -> None:
        # Deliberately fp32 whatever AMP is doing in the update: this is a length-T
        # recursion, so a rounding error made at t=T is still sitting in the estimate at
        # t=0.
        values = self.values.float()
        last_values = self._stage(last_values)
        if value_normalizer is not None:
            values = value_normalizer.denormalize(values)
            last_values = value_normalizer.denormalize(last_values)

        adv = torch.zeros_like(self.advantages)
        last_gae = torch.zeros(self.num_envs, self.num_agents,
                               dtype=torch.float32, device=self.device)
        for t in reversed(range(self.horizon)):
            # Whenever an episode ended, values[t+1] belongs to the *next* episode, since
            # the vector env auto-resets. Cutting the bootstrap and the GAE recursion on
            # both termination and truncation is therefore mandatory; the value that a
            # time-limit truncation should have bootstrapped from was already folded into
            # rewards[t] by the collector.
            episode_end = torch.maximum(self.dones[t], self.truncateds[t]).unsqueeze(-1)
            next_values = last_values if t == self.horizon - 1 else values[t + 1]
            delta = self.rewards[t] + gamma * next_values * (1.0 - episode_end) - values[t]
            last_gae = delta + gamma * lam * (1.0 - episode_end) * last_gae
            adv[t] = last_gae

        self.advantages = adv
        self.returns = adv + values

    @property
    def num_samples(self) -> int:
        return self.horizon * self.num_envs * self.num_agents

    def minibatches(self, num_minibatches: int, generator: torch.Generator):
        n = self.num_samples
        flat = {
            "obs": self.obs.reshape(n, -1),
            "states": self.states.reshape(n, -1),
            "actions": self.actions.reshape(n),
            "log_probs": self.log_probs.reshape(n),
            "values": self.values.reshape(n),
            "advantages": self.advantages.reshape(n),
            "returns": self.returns.reshape(n),
        }
        perm = torch.randperm(n, generator=generator, device=self.device)
        for chunk in perm.chunk(num_minibatches):
            yield {k: v[chunk] for k, v in flat.items()}
