from __future__ import annotations

import numpy as np

from ..inference.particle_filter import FilterConfig, IntentionParticleFilter
from .intents import N_INTENTS, intent_one_hot
from .intersection import (
    OBS_DIM,
    NegotiationIntersectionEnv,
    Scenario,
    kinematic_features,
)


class BeliefWrapper:
    """Runs one intention filter per agent and folds the posterior into that agent's
    observation.

    Agent i's filter tracks agent 1-i. Nothing is shared between the two filters, so what
    each actor sees at execution time is reachable from its own sensors plus the map.
    """

    def __init__(self, env: NegotiationIntersectionEnv,
                 filter_config: FilterConfig | None = None,
                 assertive_ema: float = 0.25,
                 seed: int | None = None):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.filter_config = filter_config or FilterConfig()
        self.filters = [IntentionParticleFilter(self.filter_config, self.rng) for _ in range(2)]
        self.assertive_ema = assertive_ema
        self._last_intent = [0, 0]
        self._assertiveness = [0.0, 0.0]
        self.obs_dim = OBS_DIM
        self.state_dim = 2 * OBS_DIM

    @property
    def dt(self) -> float:
        return 1.0 / self.env.config["policy_frequency"]

    def set_scenario(self, scenario: Scenario) -> None:
        self.env.set_scenario(scenario)

    def reset(self, seed: int | None = None):
        self.env.reset(seed=seed)
        for f in self.filters:
            f.reset()
        self._last_intent = [0, 0]
        self._assertiveness = [0.0, 0.0]
        self._update_filters()
        return self.observations()

    def step(self, actions):
        _, terms, terminated, truncated, info = self.env.step(actions)
        for i, a in enumerate(actions):
            self._last_intent[i] = int(a)
            is_assertive = float(int(a) == N_INTENTS - 1)
            self._assertiveness[i] += self.assertive_ema * (is_assertive - self._assertiveness[i])
        self._update_filters()
        return self.observations(), terms, terminated, truncated, info

    def _measure(self, agent: int):
        """Noisy measurement of another vehicle, sampled with the same sigmas the filter
        assumes. Feeding the filter a noiseless state would make its measurement model
        wrong in the optimistic direction and flatter the posterior."""
        v = self.env.controlled_vehicles[agent]
        c = self.filter_config
        pos = v.position + self.rng.normal(0.0, c.sigma_pos, 2)
        heading = float(v.heading + self.rng.normal(0.0, c.sigma_heading))
        speed = float(max(0.0, v.speed + self.rng.normal(0.0, c.sigma_speed)))
        return pos, heading, speed

    def _update_filters(self) -> None:
        for i in range(2):
            tracked = 1 - i
            pos, heading, speed = self._measure(tracked)
            self.filters[i].update(
                pos, heading, speed, self.dt,
                lambda p, a=tracked: self.env.distance_to_conflict_batch(a, p),
            )

    def posterior_assertive(self) -> list[float]:
        """posteriors[j] = P(agent j is assertive), estimated by the other agent's filter."""
        return [self.filters[1].posterior_assertive(), self.filters[0].posterior_assertive()]

    def observations(self) -> np.ndarray:
        obs = []
        for i in range(2):
            f = self.filters[i]
            obs.append(np.concatenate([
                kinematic_features(self.env, i),
                np.array([f.posterior_assertive(), f.entropy()], dtype=np.float32),
                intent_one_hot(self._last_intent[i]),
                np.array([self._assertiveness[i]], dtype=np.float32),
            ]).astype(np.float32))
        return np.stack(obs)

    def states(self, obs: np.ndarray) -> np.ndarray:
        """Centralised critic input: each agent's own observation first, then the other's,
        so one shared critic can be used for both agents without an identity feature."""
        return np.stack([
            np.concatenate([obs[0], obs[1]]),
            np.concatenate([obs[1], obs[0]]),
        ])
