from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeStats:
    scenario: str = ""
    outcome: str = "running"
    steps: int = 0
    time_to_resolve: float | None = None
    collision: bool = False
    deadlock: bool = False
    mean_jerk: float = 0.0
    min_separation: float = float("inf")
    mean_speed: float = 0.0
    leader_switches: int = 0

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "outcome": self.outcome,
            "steps": self.steps,
            "time_to_resolve": self.time_to_resolve,
            "collision": self.collision,
            "deadlock": self.deadlock,
            "mean_jerk": self.mean_jerk,
            "min_separation": self.min_separation,
            "mean_speed": self.mean_speed,
            "leader_switches": self.leader_switches,
        }


class EpisodeTracker:
    """Accumulates the reward-independent outcome measures for one encounter.

    A deadlock here means: the clock ran out, at least one agent never got through, and
    both cars were essentially stationary for the last `stall_window` seconds. Requiring
    the stall makes it distinguishable from "slow but still making progress", which is a
    different failure and should not be counted as a standoff.
    """

    def __init__(self, dt: float, wait_speed: float = 0.5, stall_window: float = 2.0):
        self.dt = dt
        self.wait_speed = wait_speed
        self.window = max(1, int(round(stall_window / dt)))
        self.reset("")

    def reset(self, scenario: str) -> None:
        self.stats = EpisodeStats(scenario=scenario)
        self._speeds: deque[float] = deque(maxlen=self.window)
        self._jerks: list[float] = []
        self._all_speeds: list[float] = []
        self._resolved_at: int | None = None
        self._last_leader: int | None = None

    def update(self, terms, positions, leader: int | None) -> None:
        self.stats.steps += 1
        speeds = [t.speed for t in terms]
        self._speeds.append(float(np.mean(speeds)))
        self._all_speeds.extend(speeds)
        self._jerks.extend(t.jerk for t in terms)

        sep = float(np.linalg.norm(positions[0] - positions[1]))
        self.stats.min_separation = min(self.stats.min_separation, sep)

        if leader is not None and leader != self._last_leader:
            if self._last_leader is not None:
                self.stats.leader_switches += 1
            self._last_leader = leader

        if self._resolved_at is None and all(t.cleared for t in terms):
            self._resolved_at = self.stats.steps

    def finish(self, outcome: str) -> EpisodeStats:
        s = self.stats
        s.outcome = outcome
        s.collision = outcome == "collision"
        s.mean_jerk = float(np.mean(self._jerks)) if self._jerks else 0.0
        s.mean_speed = float(np.mean(self._all_speeds)) if self._all_speeds else 0.0

        if outcome == "resolved" and self._resolved_at is not None:
            s.time_to_resolve = self._resolved_at * self.dt

        stalled = len(self._speeds) == self.window and max(self._speeds) < self.wait_speed
        s.deadlock = outcome == "timeout" and stalled
        if s.deadlock:
            s.outcome = "deadlock"
        return s


def aggregate(episodes) -> dict:
    if not episodes:
        return {}
    n = len(episodes)
    resolved = [e for e in episodes if e.outcome == "resolved"]
    ttr = [e.time_to_resolve for e in resolved if e.time_to_resolve is not None]
    return {
        "episodes": n,
        "resolve_rate": len(resolved) / n,
        "deadlock_rate": sum(e.deadlock for e in episodes) / n,
        "collision_rate": sum(e.collision for e in episodes) / n,
        "timeout_moving_rate": sum(e.outcome == "timeout" for e in episodes) / n,
        "time_to_resolve_mean": float(np.mean(ttr)) if ttr else None,
        "time_to_resolve_p90": float(np.percentile(ttr, 90)) if ttr else None,
        "mean_jerk": float(np.mean([e.mean_jerk for e in episodes])),
        "mean_speed": float(np.mean([e.mean_speed for e in episodes])),
        "min_separation_mean": float(np.mean([e.min_separation for e in episodes])),
        "leader_switches_mean": float(np.mean([e.leader_switches for e in episodes])),
    }
