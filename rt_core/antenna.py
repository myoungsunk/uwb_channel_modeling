"""Antenna port bases and direction-aware polarization projections.

Example:
    >>> import numpy as np
    >>> from rt_core.antenna import AntennaPort
    >>> ant = AntennaPort(position=np.zeros(3), boresight=np.array([1,0,0]), h_axis=np.array([0,1,0]), v_axis=np.array([0,0,1]), port_basis="HV")
    >>> basis = ant.transverse_port_basis(np.array([1.0, 0.1, 0.0]))
    >>> basis[0].shape, basis[1].shape
    ((3,), (3,))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from rt_core.polarization import transverse_basis

Vector = NDArray[np.floating]


def _normalize(v: NDArray) -> NDArray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return v / n


@dataclass(frozen=True)
class AntennaPort:
    position: NDArray[np.float64]
    boresight: NDArray[np.float64]
    h_axis: NDArray[np.float64]
    v_axis: NDArray[np.float64]
    port_basis: str = "HV"  # HV or RL

    def transverse_port_basis(self, k_dir: NDArray[np.float64]) -> Tuple[NDArray, NDArray]:
        """Return (port1, port2) vectors for the current arrival/departure direction."""

        h = _normalize(np.asarray(self.h_axis, dtype=float))
        v = _normalize(np.asarray(self.v_axis, dtype=float))
        k = _normalize(np.asarray(k_dir, dtype=float))
        h_t = h - np.dot(h, k) * k
        v_t = v - np.dot(v, k) * k
        if np.linalg.norm(h_t) < 1e-9 or np.linalg.norm(v_t) < 1e-9:
            h_t, v_t = transverse_basis(k, reference_up=v)
        else:
            h_t = _normalize(h_t)
            v_t = _normalize(v_t)
        if self.port_basis.upper() == "RL":
            r = _normalize(h_t + 1j * v_t)
            l = _normalize(h_t - 1j * v_t)
            return r, l
        return h_t, v_t
