from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .device import peak_memory, reset_peak_memory, setup_device
from .envs.intents import N_INTENTS
from .envs.vec_env import make_vec_env
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
    # 0 = pick from the core count, 1 = step the environments in this process.
    num_workers: int = 0
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


def train(cfg: TrainConfig) -> dict:
    dev = setup_device(cfg.device, amp=cfg.mappo.amp)
    run_name = cfg.run_name or f"{cfg.variant}_seed{cfg.seed}"
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    reset_peak_memory(dev)

    reward_model = make_reward(cfg.variant, cfg.reward)
    vec = make_vec_env(cfg.mappo.num_envs, reward_model=reward_model,
                       num_workers=cfg.num_workers, seed=cfg.seed)
    agent = MAPPO(vec.obs_dim, vec.state_dim, N_INTENTS, cfg.mappo, device=dev, seed=cfg.seed)
    buffer = agent.make_buffer()

    steps_per_update = cfg.mappo.horizon * cfg.mappo.num_envs
    num_updates = max(1, cfg.total_steps // steps_per_update)

    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.as_dict(), "device": str(dev.device), "device_name": dev.name,
         "amp_dtype": str(dev.amp_dtype), "workers": getattr(vec, "num_workers", 1),
         "obs_dim": vec.obs_dim, "state_dim": vec.state_dim}, indent=2))
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    print(dev.describe() + f"  envs={cfg.mappo.num_envs} workers={getattr(vec, 'num_workers', 1)}",
          flush=True)

    obs, states = vec.reset()
    recent = deque(maxlen=200)
    started = time.time()
    rollout_time = 0.0
    update_time = 0.0
    env_steps = 0

    try:
        for update in range(1, num_updates + 1):
            buffer.reset()
            reward_total = torch.zeros((), device=dev.device)
            t0 = time.time()

            for _ in range(cfg.mappo.horizon):
                actions, log_probs, values = agent.step_policy(obs, states)
                next_obs, next_states, rewards, dones, truncateds, final_states, finished = \
                    vec.step(actions.cpu().numpy())

                rewards_t = torch.from_numpy(rewards).to(dev.device, non_blocking=True)
                if truncateds.any():
                    # Fold the missing bootstrap for time-limit truncations into the reward
                    # so the GAE recursion can treat truncation and termination identically.
                    bootstrap = agent.value(final_states[truncateds])
                    if agent.value_normalizer is not None:
                        bootstrap = agent.value_normalizer.denormalize(bootstrap)
                    idx = torch.from_numpy(np.nonzero(truncateds)[0]).to(dev.device)
                    rewards_t[idx] += cfg.mappo.gamma * bootstrap

                buffer.add(obs, states, actions, log_probs, values, rewards_t, dones, truncateds)
                obs, states = next_obs, next_states
                # Accumulated on the device; pulling a scalar back every step would sync
                # the GPU 128 times per update for a logging number.
                reward_total += rewards_t.mean()
                recent.extend(finished)
                env_steps += cfg.mappo.num_envs

            rollout_time += time.time() - t0
            t0 = time.time()
            last_values = agent.value(states)
            buffer.compute_gae(last_values, cfg.mappo.gamma, cfg.mappo.gae_lambda,
                               agent.value_normalizer)
            stats = agent.update(buffer, lr_frac=1.0 - (update - 1) / num_updates)
            update_time += time.time() - t0

            if update % cfg.log_every == 0 or update == num_updates:
                elapsed = time.time() - started
                record = {
                    "update": update,
                    "env_steps": env_steps,
                    "wall_time": round(elapsed, 1),
                    "sps": round(env_steps / max(elapsed, 1e-6)),
                    "mean_step_reward": round(float(reward_total) / cfg.mappo.horizon, 4),
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
                    f.write(json.dumps({"update": update, "env_steps": env_steps,
                                        **res["overall"]}) + "\n")
    finally:
        vec.close()

    agent.save(out_dir / "checkpoint.pt")
    train_seconds = time.time() - started

    perf = {
        "device": str(dev.device),
        "device_name": dev.name,
        "amp_dtype": str(dev.amp_dtype),
        "num_envs": cfg.mappo.num_envs,
        "num_workers": getattr(vec, "num_workers", 1),
        "horizon": cfg.mappo.horizon,
        "hidden": cfg.mappo.hidden,
        "env_steps": env_steps,
        "train_seconds": round(train_seconds, 1),
        "rollout_seconds": round(rollout_time, 1),
        "update_seconds": round(update_time, 1),
        "steps_per_second": round(env_steps / max(train_seconds, 1e-6), 1),
        "rollout_fraction": round(rollout_time / max(train_seconds, 1e-6), 3),
        **peak_memory(dev),
    }
    (out_dir / "perf.json").write_text(json.dumps(perf, indent=2))
    print(_format_perf(perf), flush=True)

    final = evaluate(greedy_policy(agent), reward_model, cfg.final_eval_episodes,
                     seed=cfg.seed + 9000)
    final["variant"] = cfg.variant
    final["seed"] = cfg.seed
    final["env_steps"] = env_steps
    final["wall_time_s"] = round(train_seconds, 1)
    final["perf"] = perf
    (out_dir / "final_eval.json").write_text(json.dumps(final, indent=2))
    return final


def _format_log(r: dict) -> str:
    return (f"u{r['update']:>4} steps={r['env_steps']:>8} sps={r['sps']:>5} "
            f"r={r['mean_step_reward']:>8.4f} ent={r['entropy']:.3f} kl={r['approx_kl']:.4f} "
            f"resolve={r.get('resolve_rate', float('nan')):.2f} "
            f"deadlock={r.get('deadlock_rate', float('nan')):.2f} "
            f"coll={r.get('collision_rate', float('nan')):.2f} "
            f"ttr={r.get('time_to_resolve_mean', float('nan'))}")


def _format_perf(p: dict) -> str:
    vram = ("n/a (cpu)" if p["peak_allocated_gb"] is None else
            f"{p['peak_allocated_gb']:.3f} GB allocated / {p['peak_reserved_gb']:.3f} GB reserved")
    return (
        "\n" + "-" * 62 + "\n"
        f"device        {p['device']} ({p['device_name']}), amp {p['amp_dtype']}\n"
        f"env steps     {p['env_steps']} in {p['train_seconds']} s\n"
        f"throughput    {p['steps_per_second']} env-steps/s  "
        f"(rollout {p['rollout_fraction'] * 100:.0f}%, update {(1 - p['rollout_fraction']) * 100:.0f}%)\n"
        f"peak VRAM     {vram}\n"
        + "-" * 62
    )
