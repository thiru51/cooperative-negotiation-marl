from __future__ import annotations

import numpy as np

from negotiation.envs.intents import N_INTENTS
from negotiation.envs.vec_env import AsyncVecEnv, SyncVecEnv, make_vec_env, states_from_obs
from negotiation.rewards import make_reward


def test_states_from_obs_matches_the_per_env_wrapper():
    obs = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    states = states_from_obs(obs)
    assert states.shape == (2, 2, 6)
    for i in range(2):
        assert np.array_equal(states[i, 0], np.concatenate([obs[i, 0], obs[i, 1]]))
        assert np.array_equal(states[i, 1], np.concatenate([obs[i, 1], obs[i, 0]]))


def test_worker_processes_reproduce_the_single_process_rollout():
    """The whole point of splitting environments across processes is that it is only a
    speed change. Per-env RNGs keyed on the global env id are what make that true."""
    reward = make_reward("stackelberg")
    sync = SyncVecEnv(4, seed=3, reward_model=reward)
    fast = AsyncVecEnv(4, 2, reward, seed=3)
    try:
        obs_a, states_a = sync.reset()
        obs_b, states_b = fast.reset()
        assert np.allclose(obs_a, obs_b)
        assert np.allclose(states_a, states_b)

        rng = np.random.default_rng(0)
        for _ in range(30):
            actions = rng.integers(0, N_INTENTS, size=(4, 2))
            a = sync.step(actions)
            b = fast.step(actions)
            for x, y in zip(a[:6], b[:6]):
                assert np.allclose(np.asarray(x, dtype=np.float64),
                                   np.asarray(y, dtype=np.float64), atol=1e-6)
            assert [e.as_dict() for e in a[6]] == [e.as_dict() for e in b[6]]
    finally:
        fast.close()


def test_make_vec_env_stays_in_process_for_one_worker():
    vec = make_vec_env(2, reward_model=make_reward("symmetric"), num_workers=1, seed=0)
    assert isinstance(vec, SyncVecEnv)
    vec.close()
