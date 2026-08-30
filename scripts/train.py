"""Train one reward variant. Everything the run needs lands in <out-dir>/<run-name>/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from negotiation.evaluation import format_table
from negotiation.rewards import RewardConfig
from negotiation.rl.mappo import MAPPOConfig
from negotiation.training import TrainConfig, train


def build_config(args: argparse.Namespace) -> TrainConfig:
    raw: dict = {}
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text()) or {}

    mappo = MAPPOConfig(**raw.pop("mappo", {}))
    reward = RewardConfig(**raw.pop("reward", {}))
    cfg = TrainConfig(mappo=mappo, reward=reward, **raw)

    for name in ("variant", "total_steps", "seed", "device", "out_dir", "run_name",
                 "eval_every_updates", "final_eval_episodes"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.num_envs is not None:
        cfg.mappo.num_envs = args.num_envs
    if args.horizon is not None:
        cfg.mappo.horizon = args.horizon
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML config; CLI flags override it")
    p.add_argument("--variant", choices=["stackelberg", "symmetric"], default=None)
    p.add_argument("--total-steps", dest="total_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    p.add_argument("--out-dir", dest="out_dir", type=str, default=None)
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--num-envs", dest="num_envs", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--eval-every-updates", dest="eval_every_updates", type=int, default=None)
    p.add_argument("--final-eval-episodes", dest="final_eval_episodes", type=int, default=None)
    args = p.parse_args(argv)

    cfg = build_config(args)
    print(f"variant={cfg.variant} seed={cfg.seed} total_steps={cfg.total_steps} "
          f"envs={cfg.mappo.num_envs} horizon={cfg.mappo.horizon}", flush=True)

    result = train(cfg)
    print()
    print(format_table(result))
    print()
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
