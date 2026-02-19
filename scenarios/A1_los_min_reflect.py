from __future__ import annotations

from scenarios.C0_free_space import build_scene, run_case as _run


def build_sweep_params():
    return [{"case_id": "a1_los", "max_bounce": 0, "los_blocked": False}]


def run_case(params):
    return _run(params)
