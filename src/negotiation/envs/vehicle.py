from __future__ import annotations

import numpy as np
from highway_env.vehicle.controller import MDPVehicle

from .intents import Intent, target_speed


class IntentVehicle(MDPVehicle):
    """A route-following vehicle whose longitudinal command is a negotiation intent.

    highway-env keeps doing the lateral work (steering along the planned route); we only
    take over the speed target. Acceleration is clipped to comfort limits so that the three
    intents differ in *how they arrive* at a speed, not just in the setpoint -- otherwise
    ASSERTIVE_ADVANCE would be an instantaneous teleport in speed and the intention filter
    would have nothing to infer from.
    """

    MAX_ACCEL = 3.0
    MAX_DECEL = 5.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent = Intent.CREEP
        self.target_speed = target_speed(self.intent)
        self.prev_accel = 0.0
        self.jerk_sum = 0.0
        self.jerk_steps = 0

    def set_intent(self, intent: int) -> None:
        self.intent = Intent(int(intent))
        self.target_speed = target_speed(self.intent)

    def act(self, action=None) -> None:
        super().act(action)
        a = float(np.clip(self.action["acceleration"], -self.MAX_DECEL, self.MAX_ACCEL))
        self.action["acceleration"] = a

    def step(self, dt: float) -> None:
        a = float(self.action["acceleration"])
        self.jerk_sum += abs(a - self.prev_accel) / max(dt, 1e-6)
        self.jerk_steps += 1
        self.prev_accel = a
        super().step(dt)

    def pop_mean_jerk(self) -> float:
        if self.jerk_steps == 0:
            return 0.0
        j = self.jerk_sum / self.jerk_steps
        self.jerk_sum = 0.0
        self.jerk_steps = 0
        return float(j)
