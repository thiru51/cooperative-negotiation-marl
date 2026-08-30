from __future__ import annotations

import numpy as np
import pytest

from negotiation.envs.intents import Intent, N_INTENTS
from negotiation.envs.intersection import OBS_DIM
from negotiation.envs.scenarios import EVAL_SUITE, sample_training_scenario
from negotiation.envs.vec_env import SyncVecEnv, make_env
from negotiation.metrics import EpisodeTracker, aggregate
from negotiation.rewards import make_reward


@pytest.fixture(scope="module")
def env():
    return make_env(scenario=EVAL_SUITE[0], seed=0)


def test_reset_gives_two_finite_observations(env):
    obs = env.reset(seed=0)
    assert obs.shape == (2, OBS_DIM)
    assert np.isfinite(obs).all()


def test_centralised_state_is_the_pair_in_self_first_order(env):
    obs = env.reset(seed=1)
    states = env.states(obs)
    assert states.shape == (2, 2 * OBS_DIM)
    assert np.allclose(states[0], np.concatenate([obs[0], obs[1]]))
    assert np.allclose(states[1], np.concatenate([obs[1], obs[0]]))


def test_both_cars_start_before_the_conflict_point(env):
    env.reset(seed=2)
    assert env.env.distance_to_conflict(0) > 5.0
    assert env.env.distance_to_conflict(1) > 5.0


def test_assertive_advance_closes_on_the_conflict_faster_than_yielding(env):
    """The three intents have to be behaviourally distinct or there is nothing to infer."""
    def travel(intent):
        env.reset(seed=3)
        start = env.env.distance_to_conflict(0)
        for _ in range(12):
            _, _, terminated, truncated, _ = env.step([int(intent), int(Intent.YIELD_NUDGE)])
            if terminated or truncated:
                break
        return start - env.env.distance_to_conflict(0)

    assert travel(Intent.ASSERTIVE_ADVANCE) > travel(Intent.CREEP) > travel(Intent.YIELD_NUDGE)


def test_mutual_yield_never_resolves(env):
    """The deadlock the project is about: with both cars permanently signalling 'after
    you', the episode has to time out with neither through."""
    env.reset(seed=4)
    outcome = "running"
    for _ in range(400):
        _, _, terminated, truncated, info = env.step([int(Intent.YIELD_NUDGE)] * 2)
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome == "timeout"
    assert not any(info["cleared"])


def test_posteriors_are_probabilities(env):
    env.reset(seed=5)
    for _ in range(15):
        env.step([int(Intent.CREEP), int(Intent.ASSERTIVE_ADVANCE)])
    assert all(0.0 <= p <= 1.0 for p in env.posterior_assertive())


def test_each_agents_filter_tracks_the_other_one(env):
    """Agent i's filter must be fed agent 1-i's motion; a swap here would silently make the
    whole belief channel self-referential."""
    env.reset(seed=6)
    for _ in range(30):
        env.step([int(Intent.YIELD_NUDGE), int(Intent.ASSERTIVE_ADVANCE)])
    p0, p1 = env.posterior_assertive()
    assert p1 > p0


def test_scenario_sampler_stays_inside_its_declared_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = sample_training_scenario(rng)
        assert len(set(s.arms)) == 2
        assert all(24.0 <= d <= 36.0 for d in s.approach_distance)
        assert all(4.0 <= v <= 8.0 for v in s.approach_speed)


def test_vec_env_autoresets_and_reports_finished_episodes():
    vec = SyncVecEnv(3, seed=7)
    reward = make_reward("stackelberg")
    obs, states = vec.reset()
    assert obs.shape == (3, 2, vec.obs_dim)

    rng = np.random.default_rng(0)
    finished_total = []
    for _ in range(250):
        actions = rng.integers(0, N_INTENTS, size=(3, 2))
        obs, states, rewards, dones, truncateds, final_states, finished = vec.step(actions, reward)
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        finished_total.extend(finished)
        if (dones | truncateds).any():
            assert np.isfinite(final_states).all()
    assert finished_total, "no episode finished in 250 steps"
    stats = aggregate(finished_total)
    assert 0.0 <= stats["resolve_rate"] <= 1.0
    assert stats["episodes"] == len(finished_total)


def test_tracker_only_calls_a_stalled_timeout_a_deadlock():
    dt = 0.2

    class T:
        def __init__(self, speed):
            self.speed = speed
            self.jerk = 0.0
            self.cleared = 0.0

    tracker = EpisodeTracker(dt, wait_speed=0.5, stall_window=2.0)
    tracker.reset("stalled")
    for _ in range(40):
        tracker.update([T(0.0), T(0.0)], np.array([[0.0, 0.0], [8.0, 0.0]]), None)
    assert tracker.finish("timeout").deadlock

    tracker.reset("crawling")
    for _ in range(40):
        tracker.update([T(3.0), T(3.0)], np.array([[0.0, 0.0], [8.0, 0.0]]), None)
    assert not tracker.finish("timeout").deadlock
