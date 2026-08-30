from __future__ import annotations

import numpy as np
import pytest

from negotiation.envs.geometry import RouteFrame, arc_lengths, conflict_point


class _FakeLane:
    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.length = float(np.linalg.norm(self.end - self.start))

    def position(self, s, lateral):
        direction = (self.end - self.start) / self.length
        normal = np.array([-direction[1], direction[0]])
        return self.start + s * direction + lateral * normal


class _FakeNetwork:
    def __init__(self, lanes):
        self.lanes = lanes

    def get_lane(self, index):
        return self.lanes[index]


class _FakeRoad:
    def __init__(self, lanes):
        self.network = _FakeNetwork(lanes)


def _straight_road():
    return _FakeRoad({
        "ns": _FakeLane((0.0, -40.0), (0.0, 40.0)),
        "ew": _FakeLane((-40.0, 0.0), (40.0, 0.0)),
    })


def test_arc_lengths_are_cumulative_distances():
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert np.allclose(arc_lengths(pts), [0.0, 3.0, 8.0])


def test_projection_recovers_arc_length_on_a_straight_route():
    frame = RouteFrame(_straight_road(), ["ns"], resolution=0.5)
    assert frame.length == pytest.approx(80.0, abs=1e-6)
    for expected in (0.0, 12.3, 55.0, 80.0):
        pos = np.array([0.0, -40.0 + expected])
        assert frame.project(pos) == pytest.approx(expected, abs=0.3)


def test_projection_is_finer_than_the_polyline_resolution():
    """The refinement step exists so time-to-conflict does not jitter at the sample rate;
    a bare nearest-vertex lookup would be off by up to half a resolution step."""
    frame = RouteFrame(_straight_road(), ["ns"], resolution=2.0)
    pos = np.array([0.0, -40.0 + 11.0])
    assert abs(frame.project(pos) - 11.0) < 0.5


def test_crossing_routes_conflict_at_the_middle():
    road = _straight_road()
    a, b = RouteFrame(road, ["ns"]), RouteFrame(road, ["ew"])
    point, s_a, s_b = conflict_point(a, b)
    assert np.allclose(point, [0.0, 0.0], atol=0.5)
    assert s_a == pytest.approx(40.0, abs=0.5)
    assert s_b == pytest.approx(40.0, abs=0.5)
