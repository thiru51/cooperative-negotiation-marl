from __future__ import annotations

import numpy as np
import pytest
import torch

from negotiation.rl.buffer import RolloutBuffer
from negotiation.rl.networks import ValueNormalizer


def _buffer(horizon=6, num_envs=1, num_agents=1) -> RolloutBuffer:
    return RolloutBuffer(horizon, num_envs, num_agents, obs_dim=3, state_dim=6,
                         device=torch.device("cpu"))


def _fill(buf: RolloutBuffer, rewards, values, dones=None, truncateds=None) -> None:
    shape = (buf.num_envs, buf.num_agents)
    for t in range(buf.horizon):
        buf.add(
            obs=np.zeros((*shape, 3), dtype=np.float32),
            states=np.zeros((*shape, 6), dtype=np.float32),
            actions=np.zeros(shape, dtype=np.int64),
            log_probs=np.zeros(shape, dtype=np.float32),
            values=np.full(shape, values[t], dtype=np.float32),
            rewards=np.full(shape, rewards[t], dtype=np.float32),
            dones=np.array([0.0 if dones is None else dones[t]] * buf.num_envs),
            truncateds=np.array([0.0 if truncateds is None else truncateds[t]] * buf.num_envs),
        )


def test_gae_with_zero_values_is_the_discounted_return():
    horizon, gamma, lam = 5, 0.9, 1.0
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    buf = _buffer(horizon)
    _fill(buf, rewards, [0.0] * horizon)
    buf.compute_gae(np.zeros((1, 1), dtype=np.float32), gamma, lam)

    expected = 0.0
    for t in reversed(range(horizon)):
        expected = rewards[t] + gamma * expected
        assert buf.advantages[t, 0, 0] == pytest.approx(expected, rel=1e-5)


def test_gae_is_zero_when_the_critic_is_exact():
    """If values already satisfy the Bellman equation, every TD error is zero, so the
    advantage must be zero for any lambda. Catches sign and off-by-one errors."""
    gamma, horizon = 0.9, 5
    rewards = [1.0] * horizon
    values = [sum(gamma**k for k in range(horizon - t + 1)) for t in range(horizon)]
    last_value = 1.0 / (1 - gamma) * 0 + 1.0
    buf = _buffer(horizon)
    _fill(buf, rewards, values)
    buf.compute_gae(np.full((1, 1), last_value, dtype=np.float32), gamma, 0.95)
    assert np.allclose(buf.advantages, 0.0, atol=1e-4)


def test_episode_boundary_cuts_the_bootstrap():
    horizon, gamma = 4, 0.99
    buf = _buffer(horizon)
    _fill(buf, [1.0] * horizon, [0.0] * horizon, dones=[0, 0, 1, 0])
    buf.compute_gae(np.zeros((1, 1), dtype=np.float32), gamma, 1.0)
    # Step 2 terminates, so its advantage is just its own reward and nothing after it.
    assert buf.advantages[2, 0, 0] == pytest.approx(1.0)
    assert buf.advantages[1, 0, 0] == pytest.approx(1.0 + gamma * 1.0)


def test_truncation_cuts_the_recursion_like_termination():
    horizon, gamma = 4, 0.99
    buf = _buffer(horizon)
    _fill(buf, [1.0] * horizon, [0.0] * horizon, truncateds=[0, 0, 1, 0])
    buf.compute_gae(np.zeros((1, 1), dtype=np.float32), gamma, 1.0)
    assert buf.advantages[2, 0, 0] == pytest.approx(1.0)


def test_returns_equal_advantages_plus_values():
    buf = _buffer(4)
    _fill(buf, [0.5, -0.2, 1.0, 0.3], [0.1, 0.2, 0.3, 0.4])
    buf.compute_gae(np.zeros((1, 1), dtype=np.float32), 0.99, 0.95)
    assert np.allclose(buf.returns, buf.advantages + buf.values, atol=1e-6)


def test_minibatches_cover_every_sample_exactly_once():
    buf = _buffer(horizon=4, num_envs=3, num_agents=2)
    _fill(buf, [1.0] * 4, [0.0] * 4)
    buf.compute_gae(np.zeros((3, 2), dtype=np.float32), 0.99, 0.95)
    gen = torch.Generator().manual_seed(0)
    sizes = [len(b["actions"]) for b in buf.minibatches(4, gen)]
    assert sum(sizes) == 4 * 3 * 2


def test_value_normalizer_roundtrips():
    vn = ValueNormalizer()
    x = torch.randn(4096) * 7.0 + 3.0
    for _ in range(20):
        vn.update(x)
    z = vn.normalize(x)
    assert abs(float(z.mean())) < 0.2
    assert abs(float(z.std()) - 1.0) < 0.2
    assert torch.allclose(vn.denormalize(z), x, atol=1e-3)


def test_value_normalizer_variance_is_never_negative():
    """The debias term is shared by both moments precisely so that E[x^2] - E[x]^2 cannot
    come out negative in the first few updates."""
    vn = ValueNormalizer()
    vn.update(torch.full((8,), 100.0))
    mean, var = vn._moments()
    assert float(var) > 0.0
    assert float(mean) == pytest.approx(100.0, rel=1e-3)
