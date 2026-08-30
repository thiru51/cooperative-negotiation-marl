"""Quick end-to-end sanity check: build the env, roll a few steps under a random policy,
run one tiny MAPPO update. Fast enough to run before every training job."""
from __future__ import annotations

import sys

import numpy as np
import torch

from negotiation.envs.intents import N_INTENTS
from negotiation.envs.scenarios import EVAL_SUITE
from negotiation.envs.vec_env import SyncVecEnv, make_env
from negotiation.metrics import aggregate
from negotiation.rewards import make_reward
from negotiation.rl.mappo import MAPPO, MAPPOConfig


def check_single_env() -> None:
    env = make_env(scenario=EVAL_SUITE[0], seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (2, env.obs_dim), obs.shape
    assert np.isfinite(obs).all()

    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, terms, terminated, truncated, info = env.step(rng.integers(0, N_INTENTS, size=2))
        assert np.isfinite(obs).all()
        assert len(terms) == 2
        if terminated or truncated:
            break
    post = env.posterior_assertive()
    assert all(0.0 <= p <= 1.0 for p in post), post
    print(f"single env ok  obs_dim={env.obs_dim} state_dim={env.state_dim} outcome={info['outcome']} "
          f"posteriors={[round(p, 3) for p in post]}")


def check_vec_and_update() -> None:
    cfg = MAPPOConfig(horizon=16, num_envs=4, num_minibatches=2, epochs=2)
    vec = SyncVecEnv(cfg.num_envs, seed=1)
    reward = make_reward("stackelberg")
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg, device="cpu", seed=1)
    buffer = agent.make_buffer()

    obs, states = vec.reset()
    finished_all = []
    for _ in range(cfg.horizon):
        actions, log_probs, values = agent.step_policy(obs, states)
        obs, states, rewards, dones, truncateds, final_states, finished = vec.step(actions, reward)
        buffer.add(obs, states, actions, log_probs, values, rewards, dones, truncateds)
        finished_all.extend(finished)
        assert np.isfinite(rewards).all()

    buffer.compute_gae(agent.value(states), cfg.gamma, cfg.gae_lambda, agent.value_normalizer)
    assert np.isfinite(buffer.advantages).all()
    assert np.isfinite(buffer.returns).all()

    before = [p.detach().clone() for p in agent.actor.parameters()]
    stats = agent.update(buffer)
    after = list(agent.actor.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "actor did not move"
    assert all(np.isfinite(v) for v in stats.values()), stats
    print(f"vec+update ok  {  {k: round(v, 4) for k, v in stats.items()} }")
    if finished_all:
        print(f"episodes finished during smoke: {aggregate(finished_all)}")


def main() -> int:
    check_single_env()
    check_vec_and_update()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
