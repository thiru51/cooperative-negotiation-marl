"""Score a trained checkpoint (or a scripted baseline) on the fixed evaluation suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from negotiation.envs.intents import N_INTENTS, Intent
from negotiation.envs.vec_env import SyncVecEnv
from negotiation.evaluation import evaluate, format_table, greedy_policy, random_policy
from negotiation.rewards import make_reward
from negotiation.rl.mappo import MAPPO, MAPPOConfig


def always_yield_policy():
    """Both cars permanently signal 'after you'. The pathological standoff, kept as a
    reference point so the deadlock metric has a known-1.0 anchor."""
    def policy(obs, states):
        return np.full(obs.shape[:2], int(Intent.YIELD_NUDGE), dtype=np.int64)
    return policy


def always_go_policy():
    def policy(obs, states):
        return np.full(obs.shape[:2], int(Intent.ASSERTIVE_ADVANCE), dtype=np.int64)
    return policy


def load_checkpoint(path: Path, device: str) -> MAPPO:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    dims, cfg = ckpt["dims"], MAPPOConfig(**ckpt["config"])
    agent = MAPPO(dims["obs"], dims["state"], dims["actions"], cfg, device=device)
    agent.load(path)
    return agent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--policy", choices=["checkpoint", "random", "always-yield", "always-go"],
                   default="checkpoint")
    p.add_argument("--variant", choices=["stackelberg", "symmetric"], default="stackelberg",
                   help="only affects the logged leader diagnostic and mean_return")
    p.add_argument("--episodes", type=int, default=20, help="episodes per scenario")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--out", type=str, default=None, help="write the full result JSON here")
    args = p.parse_args(argv)

    if args.policy == "checkpoint":
        if not args.checkpoint:
            p.error("--checkpoint is required unless --policy is a scripted baseline")
        agent = load_checkpoint(Path(args.checkpoint), args.device)
        policy = greedy_policy(agent, deterministic=not args.stochastic)
        label = f"checkpoint:{args.checkpoint}"
    elif args.policy == "random":
        policy, label = random_policy(N_INTENTS, seed=args.seed), "random"
    elif args.policy == "always-yield":
        policy, label = always_yield_policy(), "always-yield"
    else:
        policy, label = always_go_policy(), "always-go"

    result = evaluate(policy, make_reward(args.variant), args.episodes, seed=args.seed)
    result["policy"] = label
    result["variant"] = args.variant
    result["episodes_per_scenario"] = args.episodes
    result["seed"] = args.seed

    print(f"policy={label}")
    print(format_table(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
