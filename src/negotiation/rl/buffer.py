from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size on-policy storage, laid out (T, envs, agents, ...).

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
        self.obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.states = np.zeros((*shape, state_dim), dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        # `dones` marks a genuine terminal state; `truncateds` marks the time limit, where
        # the value of the final state still has to be bootstrapped.
        self.dones = np.zeros((horizon, num_envs), dtype=np.float32)
        self.truncateds = np.zeros((horizon, num_envs), dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.step = 0

    def add(self, obs, states, actions, log_probs, values, rewards, dones, truncateds) -> None:
        t = self.step
        self.obs[t] = obs
        self.states[t] = states
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.truncateds[t] = truncateds
        self.step += 1

    def reset(self) -> None:
        self.step = 0

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float,
                    value_normalizer=None) -> None:
        values = self.values
        if value_normalizer is not None:
            values = value_normalizer.denormalize(torch.as_tensor(values)).numpy()
            last_values = value_normalizer.denormalize(torch.as_tensor(last_values)).numpy()

        adv = np.zeros_like(self.advantages)
        last_gae = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for t in reversed(range(self.horizon)):
            # Whenever an episode ended, values[t+1] belongs to the *next* episode, since
            # the vector env auto-resets. Cutting the bootstrap and the GAE recursion on
            # both termination and truncation is therefore mandatory; the value that a
            # time-limit truncation should have bootstrapped from was already folded into
            # rewards[t] by the collector.
            episode_end = np.maximum(self.dones[t], self.truncateds[t])[:, None]
            next_values = last_values if t == self.horizon - 1 else values[t + 1]
            delta = self.rewards[t] + gamma * next_values * (1.0 - episode_end) - values[t]
            last_gae = delta + gamma * lam * (1.0 - episode_end) * last_gae
            adv[t] = last_gae

        self.advantages = adv
        self.returns = adv + values

    def minibatches(self, num_minibatches: int, generator: torch.Generator):
        n = self.horizon * self.num_envs * self.num_agents
        flat = {
            "obs": self.obs.reshape(n, -1),
            "states": self.states.reshape(n, -1),
            "actions": self.actions.reshape(n),
            "log_probs": self.log_probs.reshape(n),
            "values": self.values.reshape(n),
            "advantages": self.advantages.reshape(n),
            "returns": self.returns.reshape(n),
        }
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in flat.items()}
        perm = torch.randperm(n, generator=generator, device=self.device)
        for chunk in perm.chunk(num_minibatches):
            yield {k: v[chunk] for k, v in tensors.items()}
