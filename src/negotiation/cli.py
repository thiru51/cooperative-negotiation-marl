from __future__ import annotations

import argparse

MAPPO_FIELDS = ("num_envs", "horizon", "hidden", "batch_size", "num_minibatches",
                "epochs", "lr", "entropy_coef", "amp", "compile")
TRAIN_FIELDS = ("variant", "total_steps", "seed", "device", "num_workers", "out_dir",
                "run_name", "eval_every_updates", "final_eval_episodes")
# Reward coefficients reachable from the command line. These live on cfg.reward, not
# cfg.mappo, and run_comparison.py builds its config from flags rather than from
# configs/*.yaml -- so a coefficient missing from this tuple is silently unchangeable,
# which is exactly how the first entropy_coef sweep turned into a no-op.
REWARD_FIELDS = ("collision", "cleared", "yield_bonus", "assertive")


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
    # Reachable from the command line because the collapse it guards against is the
    # first thing a run goes wrong by, and run_comparison.py does not read the YAML.
    p.add_argument("--entropy-coef", dest="entropy_coef", type=float, default=None,
                   help="entropy bonus weight; raise it if policy entropy collapses "
                        "in the first few updates")
    p.add_argument("--collision-penalty", dest="collision", type=float, default=None,
                   help="reward weight on a collision (default 12.0); raising it trades "
                        "resolve rate for safety")
    p.add_argument("--cleared-bonus", dest="cleared", type=float, default=None,
                   help="reward for clearing the junction (default 5.0)")
    p.add_argument("--yield-bonus", dest="yield_bonus", type=float, default=None)
    p.add_argument("--assertive-coef", dest="assertive", type=float, default=None)
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
    for name in REWARD_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.reward, name, value)
    return cfg
