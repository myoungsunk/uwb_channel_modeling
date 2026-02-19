from __future__ import annotations

import numpy as np

from rt_core.geometry import Material
from rt_core.surfaces import RectSurface
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas, make_freq

MATERIALS = {
    "glass": Material("dielectric", eps_r=6.5, tan_delta=0.01),
    "wood": Material("dielectric", eps_r=2.2, tan_delta=0.03),
    "gypsum": Material("dielectric", eps_r=2.9, tan_delta=0.02),
}


def build_scene(material_name: str = "glass"):
    return [
        RectSurface(
            center=np.array([2.5, -2.0, 1.5]),
            normal=np.array([0.0, 1.0, 0.0]),
            width=6.0,
            height=3.0,
            u_axis=np.array([1.0, 0.0, 0.0]),
            material=MATERIALS[material_name],
            surface_id=material_name,
        )
    ]


def build_sweep_params():
    return [{"case_id": f"a4_{m}", "material": m, "max_bounce": 1, "los_blocked": True} for m in MATERIALS]


def run_case(params):
    tx, rx = default_antennas()
    f = make_freq()
    paths = trace_paths(build_scene(params["material"]), tx, rx, f, max_bounce=params["max_bounce"], los_blocked=params["los_blocked"])
    return f, paths
