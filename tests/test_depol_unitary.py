import numpy as np

from rt_core.antenna import AntennaPort
from rt_core.polarization import DepolConfig
from rt_core.tracer import trace_paths
from scenarios.A3_dihedral import build_scene
from scenarios.common import make_freq


def _ants():
    tx = AntennaPort(np.array([0.0, 0.0, 1.5]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    rx = AntennaPort(np.array([6.0, 0.0, 1.5]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    return tx, rx


def _total_path_power(paths):
    if not paths:
        return 0.0
    a_f = np.array([p.a_f for p in paths], dtype=np.complex128)
    return float(np.sum(np.abs(a_f) ** 2))


def test_unitary_mixing_does_not_increase_total_power_without_loss():
    tx, rx = _ants()
    f = make_freq()
    planes = build_scene()

    p_low = trace_paths(
        planes,
        tx,
        rx,
        f,
        max_bounce=2,
        los_blocked=True,
        depol=DepolConfig(enabled=True, rho=0.2, mode="inout", seed=11, loss_enabled=False),
    )
    p_high = trace_paths(
        planes,
        tx,
        rx,
        f,
        max_bounce=2,
        los_blocked=True,
        depol=DepolConfig(enabled=True, rho=0.5, mode="inout", seed=11, loss_enabled=False),
    )

    assert _total_path_power(p_high) <= _total_path_power(p_low) * (1.0 + 1e-12)


def test_loss_enabled_power_monotonic_decrease_with_rho():
    tx, rx = _ants()
    f = make_freq()
    planes = build_scene()

    p_low = trace_paths(
        planes,
        tx,
        rx,
        f,
        max_bounce=2,
        los_blocked=True,
        depol=DepolConfig(enabled=True, rho=0.2, mode="inout", seed=11, loss_enabled=True, loss_alpha=0.8),
    )
    p_high = trace_paths(
        planes,
        tx,
        rx,
        f,
        max_bounce=2,
        los_blocked=True,
        depol=DepolConfig(enabled=True, rho=0.5, mode="inout", seed=11, loss_enabled=True, loss_alpha=0.8),
    )

    assert _total_path_power(p_high) <= _total_path_power(p_low)
