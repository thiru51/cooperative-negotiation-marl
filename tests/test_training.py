from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from negotiation.envs.intents import N_INTENTS
from negotiation.envs.vec_env import SyncVecEnv
from negotiation.evaluation import evaluate, greedy_policy, random_policy
from negotiation.rewards import make_reward
from negotiation.rl.mappo import MAPPO, MAPPOConfig
from negotiation.training import TrainConfig, train


def _tiny_mappo_config() -> MAPPOConfig:
    return MAPPOConfig(horizon=8, num_envs=2, hidden=32, epochs=2, num_minibatches=2)


def test_actor_has_no_agent_identity_input():
    """Parameter sharing with no index feature is what forces the tie-break to come from
    the belief rather than from a learned convention, so the input width must be exactly
    one agent's observation."""
    vec = SyncVecEnv(1, seed=0)
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, _tiny_mappo_config(), seed=0)
    assert agent.actor.net[0].in_features == vec.obs_dim
    assert agent.critic.net[0].in_features == 2 * vec.obs_dim


def test_one_update_changes_the_policy_and_stays_finite():
    cfg = _tiny_mappo_config()
    vec = SyncVecEnv(cfg.num_envs, seed=1)
    reward = make_reward("stackelberg")
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg, seed=1)
    buffer = agent.make_buffer()

    obs, states = vec.reset()
    for _ in range(cfg.horizon):
        actions, log_probs, values = agent.step_policy(obs, states)
        obs, states, rewards, dones, truncateds, _, _ = vec.step(actions.cpu().numpy(), reward)
        buffer.add(obs, states, actions, log_probs, values, rewards, dones, truncateds)

    buffer.compute_gae(agent.value(states), cfg.gamma, cfg.gae_lambda, agent.value_normalizer)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    stats = agent.update(buffer)

    assert all(np.isfinite(v) for v in stats.values()), stats
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert stats["entropy"] > 0.0


def test_checkpoint_roundtrip_preserves_the_policy(tmp_path: Path):
    cfg = _tiny_mappo_config()
    vec = SyncVecEnv(1, seed=2)
    a = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg, seed=2)
    b = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg, seed=99)

    path = tmp_path / "ckpt.pt"
    a.save(path)
    b.load(path)

    obs, states = vec.reset()
    assert torch.equal(a.step_policy(obs, states, deterministic=True)[0],
                       b.step_policy(obs, states, deterministic=True)[0])


def test_evaluate_returns_one_entry_per_scenario():
    result = evaluate(random_policy(N_INTENTS, seed=0), make_reward("symmetric"),
                      episodes_per_scenario=1, seed=3)
    from negotiation.envs.scenarios import EVAL_SUITE

    assert set(result["per_scenario"]) == {s.name for s in EVAL_SUITE}
    o = result["overall"]
    assert o["episodes"] >= len(EVAL_SUITE)
    assert 0.0 <= o["resolve_rate"] <= 1.0


def test_deadlock_and_resolve_rates_partition_the_outcomes():
    result = evaluate(random_policy(N_INTENTS, seed=1), make_reward("symmetric"),
                      episodes_per_scenario=2, seed=4)
    o = result["overall"]
    # timeout_moving_rate counts every timeout, deadlocks included, so the four categories
    # sum to one only after removing that overlap.
    total = o["resolve_rate"] + o["collision_rate"] + o["timeout_moving_rate"]
    assert abs(total - 1.0) < 1e-6, o


def test_short_training_run_writes_its_artifacts(tmp_path: Path):
    cfg = TrainConfig(
        variant="stackelberg",
        total_steps=64,
        seed=0,
        device="cpu",
        out_dir=str(tmp_path),
        run_name="tiny",
        final_eval_episodes=1,
        mappo=_tiny_mappo_config(),
    )
    result = train(cfg)

    run_dir = tmp_path / "tiny"
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "config.json").exists()
    lines = (run_dir / "train_log.jsonl").read_text().strip().splitlines()
    assert lines
    record = json.loads(lines[-1])
    assert record["env_steps"] > 0
    assert np.isfinite(record["mean_step_reward"])
    assert json.loads((run_dir / "final_eval.json").read_text())["variant"] == "stackelberg"
    assert 0.0 <= result["overall"]["resolve_rate"] <= 1.0


def test_greedy_policy_is_deterministic():
    vec = SyncVecEnv(2, seed=5)
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, _tiny_mappo_config(), seed=5)
    policy = greedy_policy(agent)
    obs, states = vec.reset()
    assert np.array_equal(policy(obs, states), policy(obs, states))
