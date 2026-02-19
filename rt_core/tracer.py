"""Specular path enumeration and polarimetric transfer computation.

Example:
    >>> import numpy as np
    >>> from rt_core.geometry import Plane, Material
    >>> from rt_core.antenna import AntennaPort
    >>> from rt_core.tracer import trace_paths
    >>> tx = AntennaPort(np.array([0,0,1.0]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    >>> rx = AntennaPort(np.array([5,0,1.0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    >>> paths = trace_paths([], tx, rx, np.linspace(3e9,5e9,4), max_bounce=0)
    >>> len(paths)
    1
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from rt_core.antenna import AntennaPort
from rt_core.geometry import Plane, mirror_point_across_plane
from rt_core.path_gain import fspl_amplitude
from rt_core.polarization import DepolConfig, fresnel_coefficients, projection_matrix, sp_basis
from rt_core.surfaces import RectSurface

C0 = 299_792_458.0


@dataclass
class Path:
    tau_s: float
    a_f: NDArray[np.complex128]
    meta: Dict[str, object]


Surface = Plane | RectSurface


def _surface_plane(surface: Surface) -> Plane:
    return surface if isinstance(surface, Plane) else surface.as_plane()


def _line_surface_intersection(p1: NDArray[np.float64], p2: NDArray[np.float64], surface: Surface) -> tuple[NDArray[np.float64], bool] | None:
    d = p2 - p1
    plane = _surface_plane(surface)
    n = plane.unit_normal()
    denom = np.dot(n, d)
    if abs(denom) < 1e-9:
        return None
    t = np.dot(n, plane.point - p1) / denom
    if t <= 1e-9 or t >= 1 - 1e-9:
        return None
    hit = p1 + t * d
    if isinstance(surface, RectSurface):
        return hit, surface.in_bounds(hit)
    return hit, True


def _solve_reflection_points(tx: NDArray[np.float64], rx: NDArray[np.float64], planes: Sequence[Surface]) -> List[NDArray[np.float64]] | None:
    images = [rx]
    for pl in reversed(planes):
        images.append(mirror_point_across_plane(images[-1], _surface_plane(pl)))
    images = list(reversed(images))
    points: List[NDArray[np.float64]] = []
    cur = tx
    target = images[0]
    for i, pl in enumerate(planes):
        out = _line_surface_intersection(cur, target, pl)
        if out is None:
            return None
        hit, in_bounds = out
        if not in_bounds:
            return None
        points.append(hit)
        cur = hit
        target = images[i + 1]
    return points


def _segment_clear(p1: NDArray[np.float64], p2: NDArray[np.float64], blockers: Sequence[Surface]) -> bool:
    for pl in blockers:
        out = _line_surface_intersection(p1, p2, pl)
        if out is not None:
            return False
    return True


def _angles_from_dir(d: NDArray[np.float64]) -> Tuple[float, float]:
    dd = d / np.linalg.norm(d)
    az = float(np.arctan2(dd[1], dd[0]))
    el = float(np.arcsin(dd[2]))
    return az, el


def _unit_dir(d: NDArray[np.float64]) -> NDArray[np.float64]:
    dd = np.asarray(d, dtype=float)
    return dd / np.linalg.norm(dd)


def trace_paths(
    planes: Sequence[Surface],
    tx: AntennaPort,
    rx: AntennaPort,
    freq_hz: NDArray[np.float64],
    max_bounce: int = 2,
    los_blocked: bool = False,
    depol: DepolConfig | None = None,
    visibility_fn: Callable[[NDArray[np.float64], NDArray[np.float64], Sequence[Surface]], bool] | None = None,
    normalize_total_power: bool = False,
) -> List[Path]:
    """Enumerate LOS and up-to-2-bounce specular paths."""

    paths: List[Path] = []
    txp = np.asarray(tx.position, dtype=float)
    rxp = np.asarray(rx.position, dtype=float)

    visibility = visibility_fn or _segment_clear

    if max_bounce >= 0 and not los_blocked and visibility(txp, rxp, planes):
        d = rxp - txp
        dist = np.linalg.norm(d)
        a = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], len(freq_hz), axis=0)
        g_los = fspl_amplitude(freq_hz, dist)
        a = g_los[:, None, None] * a
        paths.append(
            Path(
                tau_s=float(dist / C0),
                a_f=a,
                meta={
                    "bounce_count": 0,
                    "interactions": ["LOS"],
                    "surface_ids": [],
                    "incidence_angles": [],
                    "AoD": _angles_from_dir(d),
                    "AoA": _angles_from_dir(-d),
                    "AoD_unit": _unit_dir(d),
                    "AoA_unit": _unit_dir(-d),
                    "u_v_basis": {"tx": tx.transverse_port_basis(d), "rx": rx.transverse_port_basis(-d)},
                },
            )
        )

    for b in range(1, max_bounce + 1):
        for idxs in product(range(len(planes)), repeat=b):
            pl_seq = [planes[i] for i in idxs]
            pts = _solve_reflection_points(txp, rxp, pl_seq)
            if pts is None:
                continue
            way = [txp] + pts + [rxp]
            used_surface_idxs = set(idxs)
            blockers: List[Surface] = [planes[j] for j in range(len(planes)) if j not in used_surface_idxs]
            if any(not visibility(way[i], way[i + 1], blockers) for i in range(len(way) - 1)):
                continue
            seg_dirs = [way[i + 1] - way[i] for i in range(len(way) - 1)]
            seg_u = [d / np.linalg.norm(d) for d in seg_dirs]
            dist = float(sum(np.linalg.norm(d) for d in seg_dirs))
            a_freq = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], len(freq_hz), axis=0)
            incidence: List[float] = []
            for i, pl in enumerate(pl_seq):
                k_in = seg_u[i]
                k_out = seg_u[i + 1]
                n = _surface_plane(pl).unit_normal()
                theta_i = float(np.arccos(np.clip(abs(np.dot(-k_in, n)), 0.0, 1.0)))
                incidence.append(theta_i)
                sp_in = sp_basis(k_in, n)
                sp_out = sp_basis(k_out, n)
                basis_in = tx.transverse_port_basis(seg_u[0]) if i == 0 else sp_basis(seg_u[i], pl_seq[i - 1].unit_normal())
                basis_out = rx.transverse_port_basis(-seg_u[-1]) if i == (b - 1) else sp_basis(seg_u[i + 1], pl_seq[i + 1].unit_normal())
                t_in = projection_matrix(basis_in, sp_in)
                t_out = projection_matrix(sp_out, basis_out)
                mat = _surface_plane(pl).material
                gs, gp = fresnel_coefficients(freq_hz, theta_i, mat.kind, mat.eps_r, mat.tan_delta)
                for fi in range(len(freq_hz)):
                    r = np.diag([gs[fi], gp[fi]])
                    a_freq[fi] = t_out.conj().T @ r @ t_in @ a_freq[fi]
                    if depol and depol.enabled and depol.rho > 0:
                        from rt_core.polarization import depol_loss_scalar, unitary_depol_matrix

                        rng = np.random.default_rng(depol.seed + fi + i if depol.seed is not None else None)
                        dmat = unitary_depol_matrix(depol.rho, rng)
                        a_freq[fi] = dmat @ a_freq[fi] if depol.mode == "single" else dmat @ a_freq[fi] @ dmat
                        if depol.loss_enabled:
                            a_freq[fi] = depol_loss_scalar(depol.rho, depol.loss_alpha) * a_freq[fi]
            g_path = fspl_amplitude(freq_hz, dist)
            a_freq = g_path[:, None, None] * a_freq
            paths.append(
                Path(
                    tau_s=dist / C0,
                    a_f=a_freq,
                    meta={
                        "bounce_count": b,
                        "interactions": ["reflection"] * b,
                        "surface_ids": list(idxs),
                        "surface_labels": [p.surface_id for p in pl_seq],
                        "incidence_angles": incidence,
                        "AoD": _angles_from_dir(seg_u[0]),
                        "AoA": _angles_from_dir(-seg_u[-1]),
                        "AoD_unit": _unit_dir(seg_u[0]),
                        "AoA_unit": _unit_dir(-seg_u[-1]),
                        "u_v_basis": {"tx": tx.transverse_port_basis(seg_u[0]), "rx": rx.transverse_port_basis(-seg_u[-1])},
                    },
                )
            )
    if normalize_total_power and paths:
        total = 0.0
        for p in paths:
            total += float(np.sum(np.abs(p.a_f) ** 2))
        if total > 0.0:
            scale = 1.0 / np.sqrt(total)
            for p in paths:
                p.a_f = p.a_f * scale

    return paths
