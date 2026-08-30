from __future__ import annotations

import numpy as np

from ..inference.particle_filter import FilterConfig
from ..metrics import EpisodeTracker
from .belief_wrapper import BeliefWrapper
from .intersection import NegotiationIntersectionEnv, Scenario
from .scenarios import sample_training_scenario


def make_env(scenario: Scenario | None = None,
             filter_config: FilterConfig | None = None,
             seed: int | None = None,
             config_overrides: dict | None = None) -> BeliefWrapper:
    config = dict(config_overrides or {})
    if scenario is not None:
        config["scenario"] = scenario.as_dict()
    env = NegotiationIntersectionEnv(config=config)
    return BeliefWrapper(env, filter_config=filter_config, seed=seed)


class SyncVecEnv:
    """A plain synchronous stack of environments.

    Subprocess workers are not worth it here: a two-vehicle highway-env step plus two
    256-particle filter updates is dominated by many small numpy calls, and the pickling
    round-trip per step costs more than it saves at this scale.
    """

    def __init__(self, num_envs: int, seed: int = 0,
                 filter_config: FilterConfig | None = None,
                 scenario_sampler=sample_training_scenario,
                 fixed_scenarios: list[Scenario] | None = None,
                 config_overrides: dict | None = None):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.scenario_sampler = scenario_sampler
        self.fixed_scenarios = fixed_scenarios
        self.envs = [make_env(filter_config=filter_config, seed=seed + i,
                              config_overrides=config_overrides)
                     for i in range(num_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.state_dim = self.envs[0].state_dim
        self.dt = self.envs[0].dt
        wait_speed = self.envs[0].env.config["wait_speed"]
        self.trackers = [EpisodeTracker(self.dt, wait_speed) for _ in range(num_envs)]
        self._seeds = [seed * 1000 + i for i in range(num_envs)]

    def _reset_one(self, i: int) -> np.ndarray:
        if self.fixed_scenarios is not None:
            scenario = self.fixed_scenarios[i % len(self.fixed_scenarios)]
        else:
            scenario = self.scenario_sampler(self.rng)
        self.envs[i].set_scenario(scenario)
        self._seeds[i] += self.num_envs
        obs = self.envs[i].reset(seed=self._seeds[i])
        self.trackers[i].reset(scenario.name)
        return obs

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack([self._reset_one(i) for i in range(self.num_envs)])
        return obs, self._states(obs)

    def _states(self, obs: np.ndarray) -> np.ndarray:
        return np.stack([self.envs[i].states(obs[i]) for i in range(self.num_envs)])

    def step(self, actions: np.ndarray, reward_model):
        """actions: (num_envs, 2). Returns per-agent rewards and any finished episodes."""
        obs_out = np.zeros((self.num_envs, 2, self.obs_dim), dtype=np.float32)
        # Kept separately because a time-limit truncation still has to be bootstrapped
        # from the state the episode actually ended in, not from the reset state.
        final_states = np.zeros((self.num_envs, 2, self.state_dim), dtype=np.float32)
        rewards = np.zeros((self.num_envs, 2), dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        finished = []

        for i, env in enumerate(self.envs):
            obs, terms, terminated, truncated, info = env.step(actions[i])
            posteriors = env.posterior_assertive()
            rewards[i] = reward_model(terms, posteriors)

            positions = np.stack([v.position for v in env.env.controlled_vehicles])
            self.trackers[i].update(terms, positions, reward_model.leader(posteriors))

            dones[i] = terminated
            truncateds[i] = truncated
            if terminated or truncated:
                final_states[i] = env.states(obs)
                finished.append(self.trackers[i].finish(info["outcome"]))
                obs_out[i] = self._reset_one(i)
            else:
                obs_out[i] = obs

        return obs_out, self._states(obs_out), rewards, dones, truncateds, final_states, finished
