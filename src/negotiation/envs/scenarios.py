from __future__ import annotations

import numpy as np

from .intersection import Scenario

# Fixed suite used for every evaluation, so numbers from different runs are comparable.
# The first two are the interesting ones: identical approach distance and speed means
# neither agent has any geometric excuse to go first, which is exactly the situation a
# symmetric reward turns into a standoff.
EVAL_SUITE: tuple[Scenario, ...] = (
    Scenario("sym_slow", (0, 1), (34.0, 34.0), (5.0, 5.0)),
    Scenario("sym_fast", (0, 1), (34.0, 34.0), (8.0, 8.0)),
    Scenario("offset_small", (0, 1), (34.0, 30.0), (6.0, 6.0)),
    Scenario("offset_large", (0, 1), (34.0, 24.0), (6.0, 6.0)),
    Scenario("speed_mismatch", (0, 1), (34.0, 34.0), (8.0, 4.0)),
    Scenario("sym_mid_arms12", (1, 2), (34.0, 34.0), (6.0, 6.0)),
    Scenario("sym_mid_arms23", (2, 3), (34.0, 34.0), (6.0, 6.0)),
    Scenario("offset_reversed", (0, 1), (26.0, 34.0), (6.0, 6.0)),
)

ARM_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))


def sample_training_scenario(rng: np.random.Generator) -> Scenario:
    arms = ARM_PAIRS[rng.integers(len(ARM_PAIRS))]
    return Scenario(
        name="train",
        arms=arms,
        approach_distance=(float(rng.uniform(24.0, 36.0)), float(rng.uniform(24.0, 36.0))),
        approach_speed=(float(rng.uniform(4.0, 8.0)), float(rng.uniform(4.0, 8.0))),
    )
