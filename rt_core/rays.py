"""Ray container and reflection helpers.

Example:
    >>> import numpy as np
    >>> from rt_core.rays import Ray, reflect
    >>> d = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    >>> n = np.array([0.0, 1.0, 0.0])
    >>> np.allclose(reflect(d, n), np.array([1.0, 1.0, 0.0]) / np.sqrt(2))
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass
class Ray:
    origin: Vector
    direction: Vector
    path_points: List[Vector] = field(default_factory=list)
    surface_ids: List[str] = field(default_factory=list)


def reflect(direction: Vector, normal: Vector) -> Vector:
    """Specular reflection direction with unit normal."""

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    r = d - 2.0 * np.dot(d, n) * n
    return r / np.linalg.norm(r)
