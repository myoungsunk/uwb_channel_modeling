from __future__ import annotations

import numpy as np

from rt_core.polarization import DepolConfig
from scenarios.common import default_antennas, make_freq
from rt_core.tracer import trace_paths


def build_scene():
    return []


def build_sweep_params():
    return [{"case_id": "c0_los", "max_bounce": 0, "los_blocked": False}]


def run_case(params):
    tx, rx = default_antennas()
    f = make_freq()
    paths = trace_paths(build_scene(), tx, rx, f, max_bounce=params["max_bounce"], los_blocked=params["los_blocked"])
    return f, paths
