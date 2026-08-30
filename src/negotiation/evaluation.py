from __future__ import annotations

from collections import defaultdict

import numpy as np

from .envs.scenarios import EVAL_SUITE
from .envs.vec_env import SyncVecEnv
from .metrics import aggregate
from .rewards import RewardModel


def evaluate(policy, reward_model: RewardModel, episodes_per_scenario: int = 20,
             scenarios=EVAL_SUITE, seed: int = 12345, max_steps: int = 20000) -> dict:
    """Run the fixed scenario suite and report reward-independent outcome metrics.

    One environment is pinned to each scenario and they are stepped in lockstep so the
    policy can be queried in a single batch. `reward_model` is only used here for its
    leader rule (a logged diagnostic) and to keep the returns comparable; none of the
    reported metrics depend on it.
    """
    scenarios = list(scenarios)
    vec = SyncVecEnv(len(scenarios), seed=seed, fixed_scenarios=scenarios)
    obs, states = vec.reset()

    per_scenario = defaultdict(list)
    returns = np.zeros((vec.num_envs, 2))
    return_log = defaultdict(list)
    done_counts = np.zeros(vec.num_envs, dtype=int)
    steps = 0

    while done_counts.min() < episodes_per_scenario and steps < max_steps:
        actions = policy(obs, states)
        obs, states, rewards, dones, truncateds, _, finished = vec.step(actions, reward_model)
        returns += rewards
        steps += 1

        idx = 0
        for i in range(vec.num_envs):
            if dones[i] or truncateds[i]:
                if done_counts[i] < episodes_per_scenario:
                    per_scenario[scenarios[i].name].append(finished[idx])
                    return_log[scenarios[i].name].append(float(returns[i].mean()))
                done_counts[i] += 1
                returns[i] = 0.0
                idx += 1

    out = {"per_scenario": {}, "steps": steps}
    everything = []
    for name, episodes in per_scenario.items():
        stats = aggregate(episodes)
        stats["mean_return"] = float(np.mean(return_log[name]))
        out["per_scenario"][name] = stats
        everything.extend(episodes)
    out["overall"] = aggregate(everything)
    out["overall"]["mean_return"] = float(np.mean([v for vs in return_log.values() for v in vs]))
    return out


def greedy_policy(mappo, deterministic: bool = True):
    def policy(obs, states):
        actions, _, _ = mappo.step_policy(obs, states, deterministic=deterministic)
        return actions.cpu().numpy()
    return policy


def random_policy(n_actions: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    def policy(obs, states):
        return rng.integers(0, n_actions, size=obs.shape[:2])
    return policy


def format_table(results: dict) -> str:
    header = f"{'scenario':<18}{'resolve':>9}{'deadlock':>10}{'collide':>9}{'ttr_mean':>10}{'jerk':>8}"
    lines = [header, "-" * len(header)]
    for name, s in results["per_scenario"].items():
        ttr = f"{s['time_to_resolve_mean']:.2f}" if s["time_to_resolve_mean"] is not None else "  n/a"
        lines.append(
            f"{name:<18}{s['resolve_rate']:>9.2f}{s['deadlock_rate']:>10.2f}"
            f"{s['collision_rate']:>9.2f}{ttr:>10}{s['mean_jerk']:>8.2f}"
        )
    s = results["overall"]
    ttr = f"{s['time_to_resolve_mean']:.2f}" if s["time_to_resolve_mean"] is not None else "  n/a"
    lines.append("-" * len(header))
    lines.append(
        f"{'OVERALL':<18}{s['resolve_rate']:>9.2f}{s['deadlock_rate']:>10.2f}"
        f"{s['collision_rate']:>9.2f}{ttr:>10}{s['mean_jerk']:>8.2f}"
    )
    return "\n".join(lines)
