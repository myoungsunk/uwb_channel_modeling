"""Geometry primitives and intersection helpers.

Example:
    >>> import numpy as np
    >>> from rt_core.geometry import Plane, ray_plane_intersection
    >>> plane = Plane(point=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
    >>> hit = ray_plane_intersection(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), plane)
    >>> np.allclose(hit, np.array([0.0, 0.0, 0.0]))
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class Material:
    """Surface material model for reflection.

    kind: "PEC" or "dielectric".
    eps_r: Relative permittivity for dielectric model.
    tan_delta: Loss tangent for dielectric model.
    """

    kind: str = "PEC"
    eps_r: float = 4.0
    tan_delta: float = 0.0


@dataclass(frozen=True)
class Plane:
    """Infinite plane with optional id metadata."""

    point: Vector
    normal: Vector
    material: Material = Material()
    surface_id: str = "plane"

    def unit_normal(self) -> Vector:
        n = np.asarray(self.normal, dtype=float)
        return n / np.linalg.norm(n)


def normalize(v: Vector) -> Vector:
    vv = np.asarray(v, dtype=float)
    n = np.linalg.norm(vv)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return vv / n


def ray_plane_intersection(
    origin: Vector,
    direction: Vector,
    plane: Plane,
    eps: float = 1e-9,
) -> Optional[Vector]:
    """Return intersection point for a ray and plane or None.

    Ray equation: x = origin + t direction, t >= 0.
    """

    o = np.asarray(origin, dtype=float)
    d = normalize(np.asarray(direction, dtype=float))
    n = plane.unit_normal()
    denom = float(np.dot(n, d))
    if abs(denom) < eps:
        return None
    t = float(np.dot(n, plane.point - o) / denom)
    if t < eps:
        return None
    return o + t * d


def mirror_point_across_plane(point: Vector, plane: Plane) -> Vector:
    """Reflect a point across an infinite plane."""

    p = np.asarray(point, dtype=float)
    n = plane.unit_normal()
    signed_dist = np.dot(p - plane.point, n)
    return p - 2.0 * signed_dist * n
