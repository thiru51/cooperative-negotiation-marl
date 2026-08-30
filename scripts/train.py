"""Train one reward variant. Everything the run needs lands in <out-dir>/<run-name>/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from negotiation.cli import add_runtime_flags, apply_flags
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
    return apply_flags(TrainConfig(mappo=mappo, reward=reward, **raw), args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML config; CLI flags override it")
    p.add_argument("--variant", choices=["stackelberg", "symmetric"], default=None)
    p.add_argument("--total-steps", dest="total_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", dest="out_dir", type=str, default=None)
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--eval-every-updates", dest="eval_every_updates", type=int, default=None)
    p.add_argument("--final-eval-episodes", dest="final_eval_episodes", type=int, default=None)
    add_runtime_flags(p)
    args = p.parse_args(argv)

    cfg = build_config(args)
    print(f"variant={cfg.variant} seed={cfg.seed} total_steps={cfg.total_steps} "
          f"envs={cfg.mappo.num_envs} horizon={cfg.mappo.horizon} hidden={cfg.mappo.hidden}",
          flush=True)

    result = train(cfg)
    print()
    print(format_table(result))
    print()
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
