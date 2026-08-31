"""Quick end-to-end sanity check: build the env, roll a few steps under a random policy,
run one tiny MAPPO update on whatever device is available, and confirm that splitting the
vector env across worker processes reproduces the single-process trajectories exactly.

Fast enough to run before every training job."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from negotiation.device import peak_memory, setup_device
from negotiation.envs.intents import N_INTENTS
from negotiation.envs.scenarios import EVAL_SUITE
from negotiation.envs.vec_env import AsyncVecEnv, SyncVecEnv, make_env
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


def check_vec_and_update(dev) -> None:
    cfg = MAPPOConfig(horizon=16, num_envs=8, num_minibatches=2, epochs=2)
    reward = make_reward("stackelberg")
    vec = SyncVecEnv(cfg.num_envs, seed=1, reward_model=reward)
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg, device=dev, seed=1)
    buffer = agent.make_buffer()

    obs, states = vec.reset()
    finished_all = []
    started = time.time()
    for _ in range(cfg.horizon):
        actions, log_probs, values = agent.step_policy(obs, states)
        next_obs, next_states, rewards, dones, truncateds, _, finished = \
            vec.step(actions.cpu().numpy())
        # A buffer row has to hold the observation its action was chosen from, so store
        # before advancing, exactly as training.py does.
        buffer.add(obs, states, actions, log_probs, values, rewards, dones, truncateds)
        obs, states = next_obs, next_states
        finished_all.extend(finished)
        assert np.isfinite(rewards).all()
    sps = cfg.horizon * cfg.num_envs / (time.time() - started)

    buffer.compute_gae(agent.value(states), cfg.gamma, cfg.gae_lambda, agent.value_normalizer)
    assert torch.isfinite(buffer.advantages).all()
    assert torch.isfinite(buffer.returns).all()

    before = [p.detach().clone() for p in agent.actor.parameters()]
    stats = agent.update(buffer)
    after = list(agent.actor.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "actor did not move"
    assert all(np.isfinite(v) for v in stats.values()), stats

    print(f"vec+update ok  {  {k: round(v, 4) for k, v in stats.items()} }")
    print(f"  rollout {sps:.0f} env-steps/s (8 envs, single process), "
          f"peak VRAM {peak_memory(dev)['peak_allocated_gb']} GB")
    if finished_all:
        print(f"  episodes finished during smoke: {aggregate(finished_all)}")


def check_workers_match_single_process() -> None:
    """Same seeds, same actions, envs split two ways: the trajectories must be identical.

    If this fails, every worker-process number in the repo is measuring a different
    experiment from the single-process one.
    """
    reward = make_reward("stackelberg")
    kwargs = dict(seed=3)
    sync = SyncVecEnv(4, reward_model=reward, **kwargs)
    fast = AsyncVecEnv(4, 2, reward, **kwargs)

    obs_a, _ = sync.reset()
    obs_b, _ = fast.reset()
    assert np.allclose(obs_a, obs_b), "reset diverged"

    rng = np.random.default_rng(0)
    for t in range(25):
        actions = rng.integers(0, N_INTENTS, size=(4, 2))
        out_a = sync.step(actions)
        out_b = fast.step(actions)
        for name, a, b in zip(("obs", "states", "rewards"), out_a, out_b):
            assert np.allclose(a, b, atol=1e-6), f"{name} diverged at step {t}"
    fast.close()
    print("worker split ok  4 envs in 2 processes reproduce the single-process rollout")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-workers", action="store_true")
    args = p.parse_args(argv)

    dev = setup_device(args.device)
    print(dev.describe())
    check_single_env()
    check_vec_and_update(dev)
    if not args.skip_workers:
        check_workers_match_single_process()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
