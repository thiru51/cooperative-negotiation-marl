from __future__ import annotations

import numpy as np

YIELDING = 0
ASSERTIVE = 1

_OMEGA_EPS = 1e-4


def ctrv_step(x: np.ndarray, y: np.ndarray, psi: np.ndarray, v: np.ndarray,
              omega: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant turn rate and velocity pose update, vectorised over particles.

    The closed-form arc solution degenerates as omega -> 0 (division by the turn rate), so
    the near-straight case falls back to the first-order expansion.
    """
    turning = np.abs(omega) > _OMEGA_EPS
    psi_next = psi + omega * dt

    x_arc = x + np.divide(v, omega, out=np.zeros_like(v), where=turning) * (np.sin(psi_next) - np.sin(psi))
    y_arc = y + np.divide(v, omega, out=np.zeros_like(v), where=turning) * (np.cos(psi) - np.cos(psi_next))
    x_lin = x + v * np.cos(psi) * dt
    y_lin = y + v * np.sin(psi) * dt

    return np.where(turning, x_arc, x_lin), np.where(turning, y_arc, y_lin), psi_next


def intent_target_speed(hypothesis: np.ndarray, distance_to_conflict: np.ndarray,
                        v_desired: np.ndarray, a_comfort: np.ndarray,
                        stop_margin: float = 6.0) -> np.ndarray:
    """Speed a driver of each hypothesised type would be aiming for right now.

    A yielding driver bleeds speed so as to stop `stop_margin` metres short of the conflict
    point under comfortable braking: v = sqrt(2*a*d) is exactly the speed from which that
    stop is still comfortable. An assertive driver just holds its desired speed. Both are
    capped by v_desired so the yielding branch never asks for more speed than the assertive
    one, which would invert the hypotheses when the car is far away.
    """
    braking_speed = np.sqrt(2.0 * a_comfort * np.clip(distance_to_conflict - stop_margin, 0.0, None))
    yielding = np.minimum(v_desired, braking_speed)
    return np.where(hypothesis == ASSERTIVE, v_desired, yielding)
