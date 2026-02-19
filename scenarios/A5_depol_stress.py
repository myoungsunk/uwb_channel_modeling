from __future__ import annotations

from rt_core.polarization import DepolConfig
from rt_core.tracer import trace_paths
from scenarios.A3_dihedral import build_scene
from scenarios.common import default_antennas, make_freq


def build_sweep_params():
    return [
        {"case_id": "a5_rho0.2", "rho": 0.2, "max_bounce": 2, "los_blocked": True, "loss_enabled": False, "loss_alpha": 0.0},
        {"case_id": "a5_rho0.5", "rho": 0.5, "max_bounce": 2, "los_blocked": True, "loss_enabled": False, "loss_alpha": 0.0},
        {"case_id": "a5_rho0.2_loss", "rho": 0.2, "max_bounce": 2, "los_blocked": True, "loss_enabled": True, "loss_alpha": 0.8},
        {"case_id": "a5_rho0.5_loss", "rho": 0.5, "max_bounce": 2, "los_blocked": True, "loss_enabled": True, "loss_alpha": 0.8},
    ]


def run_case(params):
    tx, rx = default_antennas()
    f = make_freq()
    dep = DepolConfig(
        enabled=True,
        rho=params["rho"],
        mode="inout",
        seed=11,
        loss_enabled=params.get("loss_enabled", False),
        loss_alpha=params.get("loss_alpha", 0.0),
    )
    paths = trace_paths(build_scene(), tx, rx, f, max_bounce=params["max_bounce"], los_blocked=params["los_blocked"], depol=dep)
    return f, paths
