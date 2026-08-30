from __future__ import annotations

import numpy as np
import pytest

from negotiation.inference.motion import ASSERTIVE, YIELDING, ctrv_step, intent_target_speed
from negotiation.inference.particle_filter import (
    FilterConfig,
    IntentionParticleFilter,
    _systematic_resample,
)

DT = 0.2


def _run(kind: str, steps: int = 45, seed: int = 0, cfg: FilterConfig | None = None) -> float:
    """Drive a synthetic car straight down the x axis toward a conflict at x=0 and report
    the filter's final P(assertive). The ground truth is generated outside the filter, so
    this is a real inference test and not a self-consistency check."""
    cfg = cfg or FilterConfig(n_particles=512)
    pf = IntentionParticleFilter(cfg, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)

    x, v = -35.0, 8.0
    a_comfort, v_desired = 2.5, 8.0

    def distance_to_conflict(points):
        return -points[:, 0]

    posterior = pf.posterior_assertive()
    for _ in range(steps):
        d = -x
        if kind == "assertive":
            target = v_desired
        else:
            target = min(v_desired, np.sqrt(2.0 * a_comfort * max(d - cfg.stop_margin, 0.0)))
        accel = float(np.clip(cfg.speed_gain * (target - v), -a_comfort, a_comfort))
        v = max(0.0, v + accel * DT)
        x += v * DT

        measured = np.array([x + rng.normal(0, cfg.sigma_pos), rng.normal(0, cfg.sigma_pos)])
        posterior = pf.update(measured, float(rng.normal(0, cfg.sigma_heading)),
                              max(0.0, v + rng.normal(0, cfg.sigma_speed)), DT,
                              distance_to_conflict)
    return posterior


def test_ctrv_straight_line_matches_the_linear_case():
    z = np.zeros(3)
    x, y, psi = ctrv_step(z, z, z, np.full(3, 5.0), np.zeros(3), 0.2)
    assert np.allclose(x, 1.0)
    assert np.allclose(y, 0.0)
    assert np.allclose(psi, 0.0)


def test_ctrv_arc_is_continuous_as_the_turn_rate_vanishes():
    """The closed-form arc divides by omega; the near-zero branch must agree with it in
    the limit or the filter gets a discontinuity in its motion model."""
    args = (np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, 6.0))
    tiny = ctrv_step(*args, np.full(1, 1e-5), 0.2)
    small = ctrv_step(*args, np.full(1, 1e-3), 0.2)
    assert abs(float(tiny[0]) - float(small[0])) < 1e-3
    assert abs(float(tiny[1]) - float(small[1])) < 1e-3


def test_yielding_target_speed_never_exceeds_the_assertive_one():
    d = np.linspace(0.0, 60.0, 50)
    v_des = np.full_like(d, 8.0)
    a = np.full_like(d, 2.5)
    yielding = intent_target_speed(np.zeros_like(d, dtype=np.int64) + YIELDING, d, v_des, a)
    assertive = intent_target_speed(np.zeros_like(d, dtype=np.int64) + ASSERTIVE, d, v_des, a)
    assert np.all(yielding <= assertive + 1e-9)
    assert yielding[0] == pytest.approx(0.0)


def test_uninitialised_filter_returns_the_prior():
    pf = IntentionParticleFilter(FilterConfig(prior_assertive=0.5))
    assert pf.posterior_assertive() == pytest.approx(0.5)
    assert pf.entropy() == pytest.approx(1.0, abs=1e-6)


def test_filter_separates_assertive_from_yielding_behaviour():
    assertive = np.mean([_run("assertive", seed=s) for s in range(4)])
    yielding = np.mean([_run("yielding", seed=s) for s in range(4)])
    assert assertive > 0.65, assertive
    assert yielding < 0.35, yielding
    assert assertive - yielding > 0.4


def test_weights_stay_normalised_and_finite():
    pf = IntentionParticleFilter(FilterConfig(n_particles=128), np.random.default_rng(3))
    rng = np.random.default_rng(4)
    x = -30.0
    for _ in range(60):
        x += 6.0 * DT
        pf.update(np.array([x, 0.0]), 0.0, 6.0, DT, lambda p: -p[:, 0])
        w = pf.weights()
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= pf.posterior_assertive() <= 1.0
    _ = rng


def test_systematic_resample_favours_heavy_particles():
    w = np.array([0.0, 0.9, 0.05, 0.05])
    idx = _systematic_resample(w, np.random.default_rng(0))
    assert len(idx) == 4
    assert np.sum(idx == 1) >= 3
    assert 0 not in idx
