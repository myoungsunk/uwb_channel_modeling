import numpy as np

from rt_core.antenna import AntennaPort
from rt_core.geometry import Material
from rt_core.surfaces import RectSurface
from rt_core.tracer import _solve_reflection_points, trace_paths


def _ants():
    tx = AntennaPort(np.array([0.0, 0.0, 1.5]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    rx = AntennaPort(np.array([6.0, 0.0, 1.5]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    return tx, rx


def test_rectsurface_intersection_in_bounds():
    wall = RectSurface(
        center=np.array([2.5, -2.0, 1.5]),
        normal=np.array([0.0, 1.0, 0.0]),
        width=6.0,
        height=3.0,
        u_axis=np.array([1.0, 0.0, 0.0]),
        material=Material("PEC"),
        surface_id="wall",
    )
    pts = _solve_reflection_points(np.array([0.0, 0.0, 1.5]), np.array([6.0, 0.0, 1.5]), [wall])
    assert pts is not None
    assert wall.in_bounds(pts[0])


def test_rectsurface_rejects_out_of_bounds_reflection():
    wall = RectSurface(
        center=np.array([2.5, -2.0, 1.5]),
        normal=np.array([0.0, 1.0, 0.0]),
        # 예상 반사점 x=3.0이 경계 밖이 되도록 폭을 충분히 작게 설정
        # (center x=2.5, |du|=0.5 > width/2)
        width=0.8,
        height=1.0,
        u_axis=np.array([1.0, 0.0, 0.0]),
        material=Material("PEC"),
        surface_id="small-wall",
    )
    pts = _solve_reflection_points(np.array([0.0, 0.0, 1.5]), np.array([6.0, 0.0, 1.5]), [wall])
    assert pts is None

    tx, rx = _ants()
    f = np.linspace(3e9, 5e9, 8)
    paths = trace_paths([wall], tx, rx, f, max_bounce=1, los_blocked=True)
    assert not any(p.meta["bounce_count"] == 1 for p in paths)

def test_multibounce_visibility_receives_non_participating_blockers():
    tx, rx = _ants()
    f = np.linspace(3e9, 5e9, 8)
    wall1 = RectSurface(
        center=np.array([2.0, -2.0, 1.5]),
        normal=np.array([0.0, 1.0, 0.0]),
        width=8.0,
        height=4.0,
        u_axis=np.array([1.0, 0.0, 0.0]),
        material=Material("PEC"),
        surface_id="wall1",
    )
    wall2 = RectSurface(
        center=np.array([4.0, 0.0, 1.5]),
        normal=np.array([-1.0, 0.0, 0.0]),
        width=8.0,
        height=4.0,
        u_axis=np.array([0.0, 1.0, 0.0]),
        material=Material("PEC"),
        surface_id="wall2",
    )
    blocker = RectSurface(
        center=np.array([3.0, 1.5, 1.5]),
        normal=np.array([0.0, -1.0, 0.0]),
        width=8.0,
        height=4.0,
        u_axis=np.array([1.0, 0.0, 0.0]),
        material=Material("PEC"),
        surface_id="blocker",
    )

    blocker_counts = []

    def visibility_fn(p1, p2, blockers):
        blocker_counts.append(len(blockers))
        return True

    trace_paths([wall1, wall2, blocker], tx, rx, f, max_bounce=2, los_blocked=True, visibility_fn=visibility_fn)
    assert any(count > 0 for count in blocker_counts)
