from __future__ import annotations

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas, make_freq


def build_scene(gap: float = 2.0):
    m = Material("PEC")
    return [
        Plane(np.array([2.0, -gap, 0.0]), np.array([0.0, 1.0, 0.0]), m, "w1"),
        Plane(np.array([3.5, gap, 0.0]), np.array([0.0, -1.0, 0.0]), m, "w2"),
    ]


def build_sweep_params():
    return [{"case_id": "a3_even", "gap": 2.0, "max_bounce": 2, "los_blocked": True}]


def run_case(params):
    tx, rx = default_antennas()
    f = make_freq()
    paths = trace_paths(build_scene(params["gap"]), tx, rx, f, max_bounce=params["max_bounce"], los_blocked=params["los_blocked"])
    return f, paths
