"""Common scenario helpers."""

from __future__ import annotations

import numpy as np

from rt_core.antenna import AntennaPort


def default_antennas(port_basis: str = "HV") -> tuple[AntennaPort, AntennaPort]:
    tx = AntennaPort(np.array([0.0, 0.0, 1.5]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), port_basis)
    rx = AntennaPort(np.array([6.0, 0.0, 1.5]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), port_basis)
    return tx, rx


def make_freq() -> np.ndarray:
    return np.linspace(3e9, 10e9, 256)
