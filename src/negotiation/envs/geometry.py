from __future__ import annotations

import numpy as np


def route_polyline(road, route, resolution: float = 0.5) -> np.ndarray:
    pts = []
    for lane_index in route:
        lane = road.network.get_lane(lane_index)
        n = max(2, int(np.ceil(lane.length / resolution)))
        for s in np.linspace(0.0, lane.length, n):
            pts.append(lane.position(s, 0.0))
    return np.asarray(pts, dtype=np.float64)


def arc_lengths(polyline: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


class RouteFrame:
    """Arc-length parameterisation of one vehicle's planned route.

    Everything the negotiation needs (how far am I from the conflict, how far is the other
    car from it) is a 1-D quantity along the route, so we flatten the route to a polyline
    once at reset and project onto it, instead of reasoning about lane objects every step.
    """

    def __init__(self, road, route, resolution: float = 0.5):
        self.points = route_polyline(road, route, resolution)
        self.s = arc_lengths(self.points)
        self.length = float(self.s[-1])

    def project(self, position) -> float:
        d = np.linalg.norm(self.points - np.asarray(position, dtype=np.float64), axis=1)
        i = int(np.argmin(d))
        # Refine within the neighbouring segment so the projection is smooth in time;
        # a pure nearest-vertex lookup quantises to the polyline resolution and makes
        # the derived time-to-conflict features jitter.
        best_s = self.s[i]
        p = np.asarray(position, dtype=np.float64)
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self.points):
                continue
            a, b = self.points[j], self.points[j + 1]
            ab = b - a
            denom = float(ab @ ab)
            if denom < 1e-12:
                continue
            t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
            proj = a + t * ab
            if np.linalg.norm(p - proj) <= np.linalg.norm(p - self.points[i]) + 1e-9:
                best_s = self.s[j] + t * np.linalg.norm(ab)
        return float(best_s)


def conflict_point(frame_a: RouteFrame, frame_b: RouteFrame) -> tuple[np.ndarray, float, float]:
    """Closest approach between two routes: where the two paths actually contend."""
    d = np.linalg.norm(frame_a.points[:, None, :] - frame_b.points[None, :, :], axis=2)
    ia, ib = np.unravel_index(int(np.argmin(d)), d.shape)
    point = 0.5 * (frame_a.points[ia] + frame_b.points[ib])
    return point, float(frame_a.s[ia]), float(frame_b.s[ib])
