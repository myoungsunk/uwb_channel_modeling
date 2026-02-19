from __future__ import annotations

import numpy as np

from rt_core.geometry import Material, Plane
from scenarios.common import default_antennas, make_freq
from rt_core.tracer import trace_paths


def build_scene(offset_y: float = -2.0):
    return [Plane(np.array([2.5, offset_y, 0.0]), np.array([0.0, 1.0, 0.0]), Material("PEC"), "pec1")]


def build_sweep_params():
    return [
        {"case_id": "a2_o-2.0", "offset_y": -2.0, "max_bounce": 1, "los_blocked": True},
        {"case_id": "a2_o-3.0", "offset_y": -3.0, "max_bounce": 1, "los_blocked": True},
    ]


def run_case(params):
    tx, rx = default_antennas()
    f = make_freq()
    paths = trace_paths(build_scene(params["offset_y"]), tx, rx, f, max_bounce=params["max_bounce"], los_blocked=params["los_blocked"])
    return f, paths
