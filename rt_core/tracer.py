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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from rt_core.antenna import AntennaPort
from rt_core.geometry import Plane, mirror_point_across_plane
from rt_core.polarization import DepolConfig, fresnel_coefficients, projection_matrix, sp_basis

C0 = 299_792_458.0


@dataclass
class Path:
    tau_s: float
    a_f: NDArray[np.complex128]
    meta: Dict[str, object]


def _line_plane_intersection(p1: NDArray[np.float64], p2: NDArray[np.float64], plane: Plane) -> NDArray[np.float64] | None:
    d = p2 - p1
    n = plane.unit_normal()
    denom = np.dot(n, d)
    if abs(denom) < 1e-9:
        return None
    t = np.dot(n, plane.point - p1) / denom
    if t <= 1e-9 or t >= 1 - 1e-9:
        return None
    return p1 + t * d


def _solve_reflection_points(tx: NDArray[np.float64], rx: NDArray[np.float64], planes: Sequence[Plane]) -> List[NDArray[np.float64]] | None:
    images = [rx]
    for pl in reversed(planes):
        images.append(mirror_point_across_plane(images[-1], pl))
    images = list(reversed(images))
    points: List[NDArray[np.float64]] = []
    cur = tx
    target = images[0]
    for i, pl in enumerate(planes):
        hit = _line_plane_intersection(cur, target, pl)
        if hit is None:
            return None
        points.append(hit)
        cur = hit
        target = images[i + 1]
    return points


def _segment_clear(p1: NDArray[np.float64], p2: NDArray[np.float64], blockers: Sequence[Plane]) -> bool:
    for pl in blockers:
        hit = _line_plane_intersection(p1, p2, pl)
        if hit is not None:
            return False
    return True


def _angles_from_dir(d: NDArray[np.float64]) -> Tuple[float, float]:
    dd = d / np.linalg.norm(d)
    az = float(np.arctan2(dd[1], dd[0]))
    el = float(np.arcsin(dd[2]))
    return az, el


def trace_paths(
    planes: Sequence[Plane],
    tx: AntennaPort,
    rx: AntennaPort,
    freq_hz: NDArray[np.float64],
    max_bounce: int = 2,
    los_blocked: bool = False,
    depol: DepolConfig | None = None,
) -> List[Path]:
    """Enumerate LOS and up-to-2-bounce specular paths."""

    paths: List[Path] = []
    txp = np.asarray(tx.position, dtype=float)
    rxp = np.asarray(rx.position, dtype=float)

    if max_bounce >= 0 and not los_blocked and _segment_clear(txp, rxp, planes):
        d = rxp - txp
        dist = np.linalg.norm(d)
        a = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], len(freq_hz), axis=0)
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
            if any(not _segment_clear(way[i], way[i + 1], []) for i in range(len(way) - 1)):
                continue
            seg_dirs = [way[i + 1] - way[i] for i in range(len(way) - 1)]
            seg_u = [d / np.linalg.norm(d) for d in seg_dirs]
            dist = float(sum(np.linalg.norm(d) for d in seg_dirs))
            a_freq = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], len(freq_hz), axis=0)
            incidence: List[float] = []
            for i, pl in enumerate(pl_seq):
                k_in = seg_u[i]
                k_out = seg_u[i + 1]
                n = pl.unit_normal()
                theta_i = float(np.arccos(np.clip(abs(np.dot(-k_in, n)), 0.0, 1.0)))
                incidence.append(theta_i)
                sp_in = sp_basis(k_in, n)
                sp_out = sp_basis(k_out, n)
                basis_in = tx.transverse_port_basis(seg_u[0]) if i == 0 else sp_basis(seg_u[i], pl_seq[i - 1].unit_normal())
                basis_out = rx.transverse_port_basis(-seg_u[-1]) if i == (b - 1) else sp_basis(seg_u[i + 1], pl_seq[i + 1].unit_normal())
                t_in = projection_matrix(basis_in, sp_in)
                t_out = projection_matrix(sp_out, basis_out)
                gs, gp = fresnel_coefficients(freq_hz, theta_i, pl.material.kind, pl.material.eps_r, pl.material.tan_delta)
                for fi in range(len(freq_hz)):
                    r = np.diag([gs[fi], gp[fi]])
                    a_freq[fi] = t_out.conj().T @ r @ t_in @ a_freq[fi]
                    if depol and depol.enabled and depol.rho > 0:
                        from rt_core.polarization import depol_matrix

                        rng = np.random.default_rng(depol.seed + fi + i if depol.seed is not None else None)
                        dmat = depol_matrix(depol.rho, rng)
                        a_freq[fi] = dmat @ a_freq[fi] if depol.mode == "single" else dmat @ a_freq[fi] @ dmat
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
                        "u_v_basis": {"tx": tx.transverse_port_basis(seg_u[0]), "rx": rx.transverse_port_basis(-seg_u[-1])},
                    },
                )
            )
    return paths
