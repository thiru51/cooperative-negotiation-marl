from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .envs.intents import N_INTENTS
from .envs.vec_env import SyncVecEnv
from .evaluation import evaluate, greedy_policy
from .metrics import aggregate
from .rewards import RewardConfig, make_reward
from .rl.mappo import MAPPO, MAPPOConfig


@dataclass
class TrainConfig:
    variant: str = "stackelberg"
    total_steps: int = 300_000
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs"
    run_name: str = ""
    log_every: int = 1
    eval_every_updates: int = 0
    eval_episodes: int = 10
    final_eval_episodes: int = 30
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: TrainConfig) -> dict:
    device = resolve_device(cfg.device)
    run_name = cfg.run_name or f"{cfg.variant}_seed{cfg.seed}"
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    reward_model = make_reward(cfg.variant, cfg.reward)
    vec = SyncVecEnv(cfg.mappo.num_envs, seed=cfg.seed)
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg.mappo, device=device, seed=cfg.seed)
    buffer = agent.make_buffer()

    steps_per_update = cfg.mappo.horizon * cfg.mappo.num_envs
    num_updates = max(1, cfg.total_steps // steps_per_update)

    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.as_dict(), "device": device, "obs_dim": vec.obs_dim, "state_dim": vec.state_dim}, indent=2))
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")

    obs, states = vec.reset()
    recent = deque(maxlen=200)
    started = time.time()
    env_steps = 0

    for update in range(1, num_updates + 1):
        buffer.reset()
        reward_sum = 0.0

        for _ in range(cfg.mappo.horizon):
            actions, log_probs, values = agent.step_policy(obs, states)
            next_obs, next_states, rewards, dones, truncateds, final_states, finished = vec.step(
                actions, reward_model)

            if truncateds.any():
                # Fold the missing bootstrap for time-limit truncations into the reward so
                # the GAE recursion can treat truncation and termination identically.
                bootstrap = agent.value(final_states[truncateds])
                if agent.value_normalizer is not None:
                    bootstrap = agent.value_normalizer.denormalize(
                        torch.as_tensor(bootstrap, device=agent.device)).cpu().numpy()
                rewards[truncateds] += cfg.mappo.gamma * bootstrap

            buffer.add(obs, states, actions, log_probs, values, rewards, dones, truncateds)
            obs, states = next_obs, next_states
            reward_sum += float(rewards.mean())
            recent.extend(finished)
            env_steps += cfg.mappo.num_envs

        last_values = agent.value(states)
        buffer.compute_gae(last_values, cfg.mappo.gamma, cfg.mappo.gae_lambda, agent.value_normalizer)
        stats = agent.update(buffer, lr_frac=1.0 - (update - 1) / num_updates)

        if update % cfg.log_every == 0 or update == num_updates:
            record = {
                "update": update,
                "env_steps": env_steps,
                "wall_time": round(time.time() - started, 1),
                "sps": round(env_steps / max(time.time() - started, 1e-6)),
                "mean_step_reward": round(reward_sum / cfg.mappo.horizon, 4),
                **{k: round(v, 5) for k, v in stats.items()},
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in aggregate(list(recent)).items() if v is not None},
            }
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(_format_log(record), flush=True)

        if cfg.eval_every_updates and update % cfg.eval_every_updates == 0:
            res = evaluate(greedy_policy(agent), reward_model, cfg.eval_episodes,
                           seed=cfg.seed + 9000)
            with (out_dir / "eval_during_training.jsonl").open("a") as f:
                f.write(json.dumps({"update": update, "env_steps": env_steps, **res["overall"]}) + "\n")

    agent.save(out_dir / "checkpoint.pt")

    final = evaluate(greedy_policy(agent), reward_model, cfg.final_eval_episodes, seed=cfg.seed + 9000)
    final["variant"] = cfg.variant
    final["seed"] = cfg.seed
    final["env_steps"] = env_steps
    final["wall_time_s"] = round(time.time() - started, 1)
    (out_dir / "final_eval.json").write_text(json.dumps(final, indent=2))
    return final


def _format_log(r: dict) -> str:
    return (f"u{r['update']:>4} steps={r['env_steps']:>8} sps={r['sps']:>5} "
            f"r={r['mean_step_reward']:>8.4f} ent={r['entropy']:.3f} kl={r['approx_kl']:.4f} "
            f"resolve={r.get('resolve_rate', float('nan')):.2f} "
            f"deadlock={r.get('deadlock_rate', float('nan')):.2f} "
            f"coll={r.get('collision_rate', float('nan')):.2f} "
            f"ttr={r.get('time_to_resolve_mean', float('nan'))}")
