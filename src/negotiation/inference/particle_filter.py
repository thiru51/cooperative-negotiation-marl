from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .motion import ASSERTIVE, YIELDING, ctrv_step, intent_target_speed


@dataclass
class FilterConfig:
    n_particles: int = 256
    # Measurement noise: what a perception stack would give you for another road user at
    # this range. Deliberately not tiny -- with a near-perfect sensor model every particle
    # gets a vanishing weight and the filter collapses on the first update.
    sigma_pos: float = 0.50
    sigma_heading: float = 0.05
    sigma_speed: float = 0.35
    # Process noise
    sigma_accel: float = 0.60
    sigma_omega: float = 0.05
    # Prior over the latent driver parameters
    v_desired_range: tuple[float, float] = (5.0, 10.0)
    a_comfort_range: tuple[float, float] = (1.2, 3.5)
    param_jitter: float = 0.05
    # Per-step probability that the tracked driver changes its mind. Without this the
    # posterior saturates and never recovers when the other car actually switches
    # behaviour, which is the case we care most about.
    switch_prob: float = 0.02
    speed_gain: float = 1.2
    stop_margin: float = 6.0
    ess_threshold: float = 0.5
    prior_assertive: float = 0.5


@dataclass
class FilterState:
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hypothesis: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    v_desired: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a_comfort: np.ndarray = field(default_factory=lambda: np.zeros(0))
    log_w: np.ndarray = field(default_factory=lambda: np.zeros(0))


class IntentionParticleFilter:
    """Online leader/follower belief about another road user, from motion alone.

    The particle state is mixed discrete/continuous: a binary behaviour hypothesis plus the
    nuisance parameters (desired speed, comfortable deceleration) that decide what that
    hypothesis actually predicts. That coupling is why this is a particle filter and not a
    two-state Bayes filter -- you cannot score "is it yielding" without simultaneously
    estimating how hard this particular driver brakes.

    No V2V, no access to the other agent's policy or reward: the only input is a noisy
    measurement of its pose and speed plus the map geometry that both agents can see.
    """

    def __init__(self, config: FilterConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = config or FilterConfig()
        self.rng = rng or np.random.default_rng()
        self.state = FilterState()
        self._initialised = False

    def reset(self) -> None:
        self.state = FilterState()
        self._initialised = False

    @property
    def initialised(self) -> bool:
        return self._initialised

    def initialise(self, position, heading: float, speed: float) -> None:
        n = self.cfg.n_particles
        c = self.cfg
        s = FilterState(
            x=position[0] + self.rng.normal(0.0, c.sigma_pos, n),
            y=position[1] + self.rng.normal(0.0, c.sigma_pos, n),
            psi=heading + self.rng.normal(0.0, c.sigma_heading, n),
            v=np.clip(speed + self.rng.normal(0.0, c.sigma_speed, n), 0.0, None),
            omega=self.rng.normal(0.0, c.sigma_omega, n),
            hypothesis=(self.rng.random(n) < c.prior_assertive).astype(np.int64),
            v_desired=self.rng.uniform(*c.v_desired_range, n),
            a_comfort=self.rng.uniform(*c.a_comfort_range, n),
            log_w=np.full(n, -np.log(n)),
        )
        self.state = s
        self._initialised = True

    def update(self, position, heading: float, speed: float, dt: float,
               distance_to_conflict) -> float:
        if not self._initialised:
            self.initialise(position, heading, speed)
            return self.posterior_assertive()

        self._predict(dt, distance_to_conflict)
        self._reweight(position, heading, speed)
        self._maybe_resample()
        return self.posterior_assertive()

    def _predict(self, dt: float, distance_to_conflict) -> None:
        s, c, rng = self.state, self.cfg, self.rng
        n = c.n_particles

        flip = rng.random(n) < c.switch_prob
        s.hypothesis = np.where(flip, 1 - s.hypothesis, s.hypothesis)

        d = np.asarray(distance_to_conflict(np.stack([s.x, s.y], axis=1)), dtype=np.float64)
        v_target = intent_target_speed(s.hypothesis, d, s.v_desired, s.a_comfort, c.stop_margin)
        accel = np.clip(c.speed_gain * (v_target - s.v), -s.a_comfort, s.a_comfort)
        accel += rng.normal(0.0, c.sigma_accel, n)

        s.v = np.clip(s.v + accel * dt, 0.0, None)
        s.omega = s.omega + rng.normal(0.0, c.sigma_omega, n)
        s.x, s.y, s.psi = ctrv_step(s.x, s.y, s.psi, s.v, s.omega, dt)

        # Roughening: the latent parameters have no dynamics of their own, so without a
        # random walk they can only ever lose diversity through resampling.
        s.v_desired = np.clip(s.v_desired + rng.normal(0.0, c.param_jitter, n), *c.v_desired_range)
        s.a_comfort = np.clip(s.a_comfort + rng.normal(0.0, c.param_jitter, n), *c.a_comfort_range)

    def _reweight(self, position, heading: float, speed: float) -> None:
        s, c = self.state, self.cfg
        dx = (s.x - position[0]) / c.sigma_pos
        dy = (s.y - position[1]) / c.sigma_pos
        dpsi = np.arctan2(np.sin(s.psi - heading), np.cos(s.psi - heading)) / c.sigma_heading
        dv = (s.v - speed) / c.sigma_speed

        log_lik = -0.5 * (dx**2 + dy**2 + dpsi**2 + dv**2)
        log_w = s.log_w + log_lik
        s.log_w = log_w - _logsumexp(log_w)

    def _maybe_resample(self) -> None:
        s, c = self.state, self.cfg
        w = np.exp(s.log_w)
        ess = 1.0 / np.sum(w**2)
        if ess >= c.ess_threshold * c.n_particles:
            return

        idx = _systematic_resample(w, self.rng)
        for name in ("x", "y", "psi", "v", "omega", "hypothesis", "v_desired", "a_comfort"):
            setattr(s, name, getattr(s, name)[idx])
        s.log_w = np.full(c.n_particles, -np.log(c.n_particles))

    def weights(self) -> np.ndarray:
        return np.exp(self.state.log_w)

    def posterior_assertive(self) -> float:
        if not self._initialised:
            return float(self.cfg.prior_assertive)
        p = float(np.sum(self.weights()[self.state.hypothesis == ASSERTIVE]))
        # The weights are normalised in log space, so the sum can land a few ulps outside
        # [0, 1]. This number goes straight into the observation and into the leader rule
        # as a probability, so clamp it rather than let a 1.0000000000000002 through.
        return min(max(p, 0.0), 1.0)

    def entropy(self) -> float:
        p = self.posterior_assertive()
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0))

    def mean_state(self) -> np.ndarray:
        w = self.weights()
        s = self.state
        return np.array([w @ s.x, w @ s.y, w @ s.v], dtype=np.float64)


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions).clip(0, n - 1)


__all__ = ["IntentionParticleFilter", "FilterConfig", "ASSERTIVE", "YIELDING"]
