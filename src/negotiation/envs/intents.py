from __future__ import annotations

from enum import IntEnum

import numpy as np


class Intent(IntEnum):
    CREEP = 0
    YIELD_NUDGE = 1
    ASSERTIVE_ADVANCE = 2


# Target speeds in m/s. These are the whole action space: an intent is a longitudinal
# target-speed profile that the vehicle's low-level controller tracks, so a policy that
# flips between them produces the kind of speed signature a human driver would read as
# "I'm going" or "after you". Raw throttle would not be legible as a signal.
INTENT_TARGET_SPEED = {
    Intent.CREEP: 2.0,
    Intent.YIELD_NUDGE: 0.0,
    Intent.ASSERTIVE_ADVANCE: 8.0,
}

N_INTENTS = len(Intent)


def target_speed(intent: int) -> float:
    return INTENT_TARGET_SPEED[Intent(int(intent))]


def intent_one_hot(intent: int) -> np.ndarray:
    v = np.zeros(N_INTENTS, dtype=np.float32)
    v[int(intent)] = 1.0
    return v
