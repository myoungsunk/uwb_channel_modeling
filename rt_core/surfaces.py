"""Finite surface primitives for physically valid reflections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from rt_core.geometry import Material, Plane, normalize, ray_plane_intersection

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class RectSurface:
    """Finite rectangular surface embedded in a plane."""

    center: Vector
    normal: Vector
    width: float
    height: float
    u_axis: Vector
    material: Material = Material()
    surface_id: str = "rect"

    def unit_normal(self) -> Vector:
        return normalize(np.asarray(self.normal, dtype=float))

    def unit_u(self) -> Vector:
        u = np.asarray(self.u_axis, dtype=float)
        n = self.unit_normal()
        u = u - np.dot(u, n) * n
        return normalize(u)

    def unit_v(self) -> Vector:
        return normalize(np.cross(self.unit_normal(), self.unit_u()))

    def as_plane(self) -> Plane:
        return Plane(
            point=np.asarray(self.center, dtype=float),
            normal=self.unit_normal(),
            material=self.material,
            surface_id=self.surface_id,
        )

    def in_bounds(self, point: Vector, eps: float = 1e-9) -> bool:
        p = np.asarray(point, dtype=float)
        c = np.asarray(self.center, dtype=float)
        rel = p - c
        du = float(np.dot(rel, self.unit_u()))
        dv = float(np.dot(rel, self.unit_v()))
        return abs(du) <= self.width / 2.0 + eps and abs(dv) <= self.height / 2.0 + eps

    def intersect(self, origin: Vector, direction: Vector, eps: float = 1e-9) -> Optional[tuple[Vector, bool]]:
        hit = ray_plane_intersection(origin, direction, self.as_plane(), eps=eps)
        if hit is None:
            return None
        return hit, self.in_bounds(hit, eps=eps)
