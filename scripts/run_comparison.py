"""The actual experiment: train both reward variants on matched seeds and step budgets,
then print the paired comparison.

Everything except the reward function is held fixed -- same env, same scenario sampler,
same network init seed, same number of environment steps -- so a difference in the outcome
metrics is attributable to the reward and nothing else."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from negotiation.evaluation import format_table
from negotiation.rl.mappo import MAPPOConfig
from negotiation.training import TrainConfig, train

HEADLINE = ("deadlock_rate", "resolve_rate", "collision_rate",
            "time_to_resolve_mean", "mean_speed", "mean_jerk")


def summarise(runs: list[dict]) -> dict:
    """Mean over seeds of each headline metric, skipping seeds where it is undefined
    (time_to_resolve is None when a variant never resolved a single episode)."""
    out = {}
    for key in HEADLINE:
        vals = [r["overall"][key] for r in runs if r["overall"].get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
        out[key + "_per_seed"] = [r["overall"].get(key) for r in runs]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", dest="total_steps", type=int, default=200_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", dest="out_dir", type=str, default="runs")
    p.add_argument("--results-dir", dest="results_dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="comparison")
    p.add_argument("--num-envs", dest="num_envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=25)
    p.add_argument("--eval-every-updates", dest="eval_every_updates", type=int, default=0)
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_runs: dict[str, list[dict]] = {"stackelberg": [], "symmetric": []}
    for seed in args.seeds:
        for variant in ("stackelberg", "symmetric"):
            print(f"\n=== {variant}  seed {seed}  {args.total_steps} steps ===", flush=True)
            cfg = TrainConfig(
                variant=variant,
                total_steps=args.total_steps,
                seed=seed,
                device=args.device,
                out_dir=args.out_dir,
                run_name=f"{args.tag}_{variant}_seed{seed}",
                eval_every_updates=args.eval_every_updates,
                final_eval_episodes=args.eval_episodes,
                mappo=MAPPOConfig(num_envs=args.num_envs, horizon=args.horizon),
            )
            result = train(cfg)
            all_runs[variant].append(result)
            print()
            print(format_table(result))

    comparison = {
        "tag": args.tag,
        "total_steps_per_run": args.total_steps,
        "seeds": args.seeds,
        "eval_episodes_per_scenario": args.eval_episodes,
        "mappo": {"num_envs": args.num_envs, "horizon": args.horizon},
        "summary": {v: summarise(runs) for v, runs in all_runs.items()},
        "runs": all_runs,
    }
    out_path = results_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(comparison, indent=2))

    print("\n" + "=" * 68)
    print(f"{'metric':<26}{'stackelberg':>14}{'symmetric':>14}{'delta':>14}")
    print("-" * 68)
    for key in HEADLINE:
        a = comparison["summary"]["stackelberg"][key]
        b = comparison["summary"]["symmetric"][key]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        delta = "n/a" if (a is None or b is None) else f"{a - b:+.3f}"
        print(f"{key:<26}{fmt(a):>14}{fmt(b):>14}{delta:>14}")
    print("=" * 68)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
