"""Scenario A3 setup and sweep execution."""

from __future__ import annotations

from rt_core.polarization import DepolConfig
from scenarios.common import room_planes, run_case


def run():
    planes = room_planes()
    max_bounce = 0 if "A3" == "C0" else (1 if "A3" in ["A1", "A2"] else 2)
    depol = DepolConfig(enabled=True, rho=0.35, mode="inout", seed=1234) if "A3" == "A5" else None
    return run_case(planes, max_bounce=max_bounce, depol=depol)
