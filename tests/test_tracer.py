import numpy as np

from rt_core.antenna import AntennaPort
from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths


def _ants():
    tx = AntennaPort(np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    rx = AntennaPort(np.array([5.0, 0.0, 1.0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    return tx, rx


def test_los_only():
    tx, rx = _ants()
    f = np.linspace(3e9, 5e9, 8)
    paths = trace_paths([], tx, rx, f, max_bounce=0)
    assert len(paths) == 1
    assert paths[0].meta["bounce_count"] == 0
    assert paths[0].a_f.shape == (len(f), 2, 2)


def test_one_bounce_plane():
    tx, rx = _ants()
    f = np.linspace(3e9, 5e9, 8)
    plane = Plane(np.array([2.5, -2.0, 0.0]), np.array([0.0, 1.0, 0.0]), Material("PEC"), "wall")
    paths = trace_paths([plane], tx, rx, f, max_bounce=1, los_blocked=True)
    assert any(p.meta["bounce_count"] == 1 for p in paths)


def test_two_bounce_corner():
    tx, rx = _ants()
    rx = AntennaPort(np.array([5.0, 1.0, 1.0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    f = np.linspace(3e9, 5e9, 8)
    p1 = Plane(np.array([2.0, -2.0, 0.0]), np.array([0.0, 1.0, 0.0]), Material("PEC"), "w1")
    p2 = Plane(np.array([3.5, 2.0, 0.0]), np.array([0.0, -1.0, 0.0]), Material("PEC"), "w2")
    paths = trace_paths([p1, p2], tx, rx, f, max_bounce=2, los_blocked=True)
    assert any(p.meta["bounce_count"] == 2 for p in paths)
