"""Path-gain utilities for physically meaningful distance attenuation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

C0 = 299_792_458.0


def fspl_amplitude(freq_hz: NDArray[np.float64], path_length_m: float) -> NDArray[np.float64]:
    """Free-space amplitude scale: g(f) = lambda(f) / (4*pi*r)."""

    r = float(path_length_m)
    if r <= 0.0:
        raise ValueError("path_length_m must be > 0")
    f = np.asarray(freq_hz, dtype=float)
    lam = C0 / np.maximum(f, 1.0)
    return lam / (4.0 * np.pi * r)
