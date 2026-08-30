from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.road.road import Road

from .geometry import RouteFrame, conflict_point
from .intents import Intent, N_INTENTS
from .vehicle import IntentVehicle

V_MAX = 10.0
D_NORM = 40.0
TTC_NORM = 8.0


@dataclass
class Scenario:
    """One encounter geometry. `arm` indexes the four approach roads of the intersection."""
    name: str = "default"
    arms: tuple[int, int] = (0, 1)
    approach_distance: tuple[float, float] = (34.0, 34.0)
    approach_speed: tuple[float, float] = (6.0, 6.0)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "arms": list(self.arms),
            "approach_distance": list(self.approach_distance),
            "approach_speed": list(self.approach_speed),
        }


@dataclass
class AgentTerms:
    """Raw, reward-function-agnostic outcome of one decision step for one agent.

    The two reward variants under comparison are pure functions of these terms, which is
    what makes the comparison controlled: the environment dynamics and the measured
    quantities are byte-identical, only the mapping to a scalar differs.
    """
    progress: float = 0.0
    speed: float = 0.0
    waiting: float = 0.0
    collision: float = 0.0
    cleared: float = 0.0
    just_cleared: float = 0.0
    jerk: float = 0.0
    intent: int = int(Intent.CREEP)
    distance_to_conflict: float = 0.0
    time_to_conflict: float = 0.0


class NegotiationIntersectionEnv(IntersectionEnv):
    """Two controlled vehicles on crossing straight routes through highway-env's
    unsignalized intersection, each commanded by a discrete intention signal.

    Two things are deliberately removed relative to stock highway-env:

    1. The background IDM traffic. We want a clean two-body negotiation; adding traffic
       adds variance without adding anything to the question being asked.
    2. The RegulatedRoad. highway-env's road object contains an explicit right-of-way rule
       that freezes the lower-priority vehicle at a conflict. That is precisely the
       hand-coded priority rule this project is trying to do without, so the road is
       rebuilt as a plain Road and neither agent gets any scripted deference.
    """

    ACTION_IDLE = 1

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 2,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": True,
                    "normalize": False,
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0.0, 2.0, 8.0],
                },
            },
            "controlled_vehicles": 2,
            "initial_vehicle_count": 0,
            "spawn_probability": 0.0,
            "duration": 20,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "destination": None,
            "normalize_reward": False,
            "offroad_terminal": False,
            "clearance": 6.0,
            "wait_speed": 0.5,
            "scenario": Scenario().as_dict(),
        })
        return config

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        super().__init__(config, render_mode)
        self.frames: list[RouteFrame] = []
        self.conflict_s: list[float] = []
        self.conflict_xy = np.zeros(2)
        self._cleared = [False, False]
        self._prev_s = [0.0, 0.0]

    @property
    def scenario(self) -> Scenario:
        s = self.config["scenario"]
        return Scenario(
            name=s["name"],
            arms=tuple(s["arms"]),
            approach_distance=tuple(s["approach_distance"]),
            approach_speed=tuple(s["approach_speed"]),
        )

    def set_scenario(self, scenario: Scenario) -> None:
        self.config["scenario"] = scenario.as_dict()

    def _make_road(self) -> None:
        super()._make_road()
        self.road = Road(
            network=self.road.network,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),
        )

    def _make_vehicles(self, n_vehicles: int = 0) -> None:
        scenario = self.scenario
        self.controlled_vehicles = []
        for i, arm in enumerate(scenario.arms):
            entry = (f"o{arm}", f"ir{arm}", 0)
            lane = self.road.network.get_lane(entry)
            longitudinal = max(0.0, lane.length - scenario.approach_distance[i])
            vehicle = IntentVehicle.make_on_lane(
                self.road, entry,
                longitudinal=longitudinal,
                speed=scenario.approach_speed[i],
            )
            # Straight through: the destination arm is the one opposite the entry, so the
            # two routes cross at the middle of the junction. Turning routes merge instead
            # of crossing and the right-of-way question mostly disappears.
            vehicle.plan_route_to(f"o{(arm + 2) % 4}")
            vehicle.set_intent(Intent.CREEP)
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        self._build_conflict_geometry()

    def _build_conflict_geometry(self) -> None:
        self.frames = [RouteFrame(self.road, v.route) for v in self.controlled_vehicles]
        self.conflict_xy, s0, s1 = conflict_point(self.frames[0], self.frames[1])
        self.conflict_s = [s0, s1]
        self._cleared = [False, False]
        self._prev_s = [self.frames[i].project(v.position)
                        for i, v in enumerate(self.controlled_vehicles)]

    def distance_to_conflict(self, agent: int, position=None) -> float:
        pos = self.controlled_vehicles[agent].position if position is None else position
        return self.conflict_s[agent] - self.frames[agent].project(pos)

    def distance_to_conflict_batch(self, agent: int, positions: np.ndarray) -> np.ndarray:
        frame, s_c = self.frames[agent], self.conflict_s[agent]
        d = np.linalg.norm(positions[:, None, :] - frame.points[None, :, :], axis=2)
        return s_c - frame.s[np.argmin(d, axis=1)]

    def step(self, action):
        for vehicle, a in zip(self.controlled_vehicles, action):
            vehicle.set_intent(a)
        # highway-env's own meta-action channel is pinned to IDLE: the intent already set
        # target_speed, and IDLE is the one meta-action that leaves target_speed alone
        # while still running the lateral route follower every simulation substep.
        obs, _, _, _, info = super().step(tuple(self.ACTION_IDLE for _ in action))
        terms = self._compute_terms()
        terminated, truncated, outcome = self._episode_status()
        info = dict(info)
        info.update({"terms": terms, "outcome": outcome, "cleared": list(self._cleared)})
        return obs, terms, terminated, truncated, info

    def _compute_terms(self) -> list[AgentTerms]:
        dt = 1.0 / self.config["policy_frequency"]
        clearance = self.config["clearance"]
        wait_speed = self.config["wait_speed"]

        terms = []
        for i, vehicle in enumerate(self.controlled_vehicles):
            s = self.frames[i].project(vehicle.position)
            progress = s - self._prev_s[i]
            self._prev_s[i] = s

            d = self.conflict_s[i] - s
            was_cleared = self._cleared[i]
            self._cleared[i] = bool(d < -clearance)

            speed = float(vehicle.speed)
            terms.append(AgentTerms(
                progress=float(progress),
                speed=speed,
                waiting=float(speed < wait_speed and not self._cleared[i]),
                collision=float(vehicle.crashed),
                cleared=float(self._cleared[i]),
                just_cleared=float(self._cleared[i] and not was_cleared),
                jerk=vehicle.pop_mean_jerk(),
                intent=int(vehicle.intent),
                distance_to_conflict=float(d),
                time_to_conflict=float(d / max(speed, 0.1)) if d > 0 else 0.0,
            ))
        _ = dt
        return terms

    def _episode_status(self) -> tuple[bool, bool, str]:
        crashed = any(v.crashed for v in self.controlled_vehicles)
        if crashed:
            return True, False, "collision"
        if all(self._cleared):
            return True, False, "resolved"
        if super()._is_truncated():
            return False, True, "timeout"
        return False, False, "running"

    def _reward(self, action) -> float:
        return 0.0

    def _rewards(self, action) -> dict:
        return {}

    def _is_terminated(self) -> bool:
        return any(v.crashed for v in self.controlled_vehicles) or all(self._cleared)


def kinematic_features(env: NegotiationIntersectionEnv, agent: int) -> np.ndarray:
    """Ego-centric kinematics plus the conflict-frame quantities the negotiation is about."""
    ego = env.controlled_vehicles[agent]
    other = env.controlled_vehicles[1 - agent]

    d_ego = env.distance_to_conflict(agent)
    d_other = env.distance_to_conflict(1 - agent)
    ttc_ego = d_ego / max(ego.speed, 0.1)
    ttc_other = d_other / max(other.speed, 0.1)

    c, s = np.cos(ego.heading), np.sin(ego.heading)
    rot = np.array([[c, s], [-s, c]])
    rel_pos = rot @ (other.position - ego.position)
    rel_vel = rot @ (other.velocity - ego.velocity)

    return np.array([
        ego.speed / V_MAX,
        np.clip(d_ego / D_NORM, -1.5, 1.5),
        np.clip(ttc_ego / TTC_NORM, -2.0, 2.0),
        float(env._cleared[agent]),
        other.speed / V_MAX,
        np.clip(d_other / D_NORM, -1.5, 1.5),
        np.clip(ttc_other / TTC_NORM, -2.0, 2.0),
        float(env._cleared[1 - agent]),
        np.clip((ttc_ego - ttc_other) / TTC_NORM, -2.0, 2.0),
        np.clip(rel_pos[0] / D_NORM, -1.5, 1.5),
        np.clip(rel_pos[1] / D_NORM, -1.5, 1.5),
        np.clip(rel_vel[0] / V_MAX, -2.0, 2.0),
        np.clip(rel_vel[1] / V_MAX, -2.0, 2.0),
    ], dtype=np.float32)


KINEMATIC_DIM = 13
BELIEF_DIM = 2
HISTORY_DIM = N_INTENTS + 1
OBS_DIM = KINEMATIC_DIM + BELIEF_DIM + HISTORY_DIM
