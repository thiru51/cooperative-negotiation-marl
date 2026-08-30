from __future__ import annotations

import argparse

MAPPO_FIELDS = ("num_envs", "horizon", "hidden", "batch_size", "num_minibatches",
                "epochs", "lr", "amp", "compile")
TRAIN_FIELDS = ("variant", "total_steps", "seed", "device", "num_workers", "out_dir",
                "run_name", "eval_every_updates", "final_eval_episodes")


def add_runtime_flags(p: argparse.ArgumentParser) -> None:
    """The flags that decide how hard the machine gets worked.

    Shared by train.py and run_comparison.py so the two entry points cannot drift apart.
    Everything defaults to None, which means "leave whatever the config file said".
    """
    p.add_argument("--device", type=str, default=None, help="auto | cpu | cuda | cuda:1")
    p.add_argument("--num-envs", dest="num_envs", type=int, default=None,
                   help="parallel environments; one batched policy forward serves all of them")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None,
                   help="processes stepping the environments. 0 = half the cores, 1 = in-process")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                   help="minibatch rows in the update; overrides --num-minibatches")
    p.add_argument("--num-minibatches", dest="num_minibatches", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None, help="MLP width for actor and critic")
    p.add_argument("--horizon", type=int, default=None, help="rollout length per update")
    p.add_argument("--epochs", type=int, default=None, help="PPO epochs per update")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="mixed precision in the update (bf16 where supported). On by default")
    p.add_argument("--compile", action="store_true", default=None,
                   help="torch.compile the MLP bodies. Off by default: warm-up costs tens "
                        "of seconds and these networks are small")


def apply_flags(cfg, args: argparse.Namespace):
    """Overlay whichever CLI flags were actually given onto a TrainConfig."""
    for name in TRAIN_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    for name in MAPPO_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.mappo, name, value)
    return cfg
