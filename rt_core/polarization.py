"""Polarization basis transforms and Fresnel/Jones operators.

Example:
    >>> import numpy as np
    >>> from rt_core.polarization import fresnel_coefficients
    >>> f = np.array([1e9])
    >>> gs, gp = fresnel_coefficients(f, np.deg2rad(30.0), kind="PEC")
    >>> np.allclose(gs, -1.0) and np.allclose(gp, -1.0)
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

C0 = 299_792_458.0
EPS0 = 8.854187817e-12

Vector = NDArray[np.float64]
ComplexVec = NDArray[np.complex128]
ComplexMat = NDArray[np.complex128]


@dataclass(frozen=True)
class DepolConfig:
    enabled: bool = False
    mode: str = "single"  # "single" | "inout"
    rho: float = 0.0
    seed: Optional[int] = None


def normalize(v: Vector) -> Vector:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def transverse_basis(k_dir: Vector, reference_up: Optional[Vector] = None) -> Tuple[Vector, Vector]:
    """Build orthonormal transverse basis (u, v) for propagation direction k_dir."""

    k = normalize(np.asarray(k_dir, dtype=float))
    ref = np.array([0.0, 0.0, 1.0]) if reference_up is None else normalize(np.asarray(reference_up, dtype=float))
    if abs(np.dot(ref, k)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(ref, k))
    v = normalize(np.cross(k, u))
    return u, v


def sp_basis(incident_dir: Vector, normal: Vector) -> Tuple[Vector, Vector]:
    """Return local s(TE), p(TM) basis for incident propagation direction."""

    k = normalize(np.asarray(incident_dir, dtype=float))
    n = normalize(np.asarray(normal, dtype=float))
    s = np.cross(n, k)
    if np.linalg.norm(s) < 1e-12:
        alt = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(alt, k)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        s = np.cross(alt, k)
    s = normalize(s)
    p = normalize(np.cross(k, s))
    return s, p


def projection_matrix(from_basis: Tuple[Vector, Vector], to_basis: Tuple[Vector, Vector]) -> ComplexMat:
    """2x2 matrix mapping amplitudes in from_basis to to_basis."""

    f1, f2 = from_basis
    t1, t2 = to_basis
    return np.array(
        [[np.dot(t1, f1), np.dot(t1, f2)], [np.dot(t2, f1), np.dot(t2, f2)]],
        dtype=np.complex128,
    )


def _complex_eps_r(freq_hz: NDArray[np.float64], eps_r: float, tan_delta: float) -> ComplexVec:
    w = 2.0 * np.pi * freq_hz
    sigma = w * EPS0 * eps_r * tan_delta
    return eps_r - 1j * sigma / (w * EPS0)


def fresnel_coefficients(
    freq_hz: NDArray[np.float64],
    incidence_angle_rad: float,
    kind: str,
    eps_r: float = 4.0,
    tan_delta: float = 0.0,
) -> Tuple[ComplexVec, ComplexVec]:
    """Fresnel reflection coefficients (Gamma_s, Gamma_p)."""

    if kind.upper() == "PEC":
        ones = np.ones_like(freq_hz, dtype=np.complex128)
        return -ones, -ones

    sin_t = np.sin(incidence_angle_rad)
    cos_t = np.cos(incidence_angle_rad)
    eps_c = _complex_eps_r(freq_hz.astype(float), eps_r, tan_delta)
    root = np.sqrt(eps_c - sin_t**2)
    gamma_s = (cos_t - root) / (cos_t + root)
    gamma_p = (eps_c * cos_t - root) / (eps_c * cos_t + root)
    return gamma_s.astype(np.complex128), gamma_p.astype(np.complex128)


def depol_matrix(rho: float, rng: np.random.Generator) -> ComplexMat:
    rho = float(np.clip(rho, 0.0, 1.0))
    a = np.sqrt(1.0 - rho)
    b = np.sqrt(rho)
    p1 = rng.uniform(0.0, 2.0 * np.pi)
    p2 = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([[a, b * np.exp(1j * p1)], [b * np.exp(1j * p2), a]], dtype=np.complex128)


def apply_reflection_event(
    a_in: ComplexMat,
    in_basis: Tuple[Vector, Vector],
    out_basis: Tuple[Vector, Vector],
    sp_in: Tuple[Vector, Vector],
    sp_out: Tuple[Vector, Vector],
    gamma_s: ComplexVec,
    gamma_p: ComplexVec,
    depol: Optional[DepolConfig] = None,
) -> NDArray[np.complex128]:
    """Apply one reflection event to path transfer matrix across frequencies.

    Returns shape (Nf, 2, 2).
    """

    n_f = gamma_s.size
    t_in = projection_matrix(in_basis, sp_in)
    t_out = projection_matrix(sp_out, out_basis)
    out = np.zeros((n_f, 2, 2), dtype=np.complex128)
    rng = np.random.default_rng(depol.seed if depol else None)
    for i in range(n_f):
        r = np.diag([gamma_s[i], gamma_p[i]])
        m = t_out.conj().T @ r @ t_in @ a_in
        if depol and depol.enabled and depol.rho > 0:
            d = depol_matrix(depol.rho, rng)
            if depol.mode == "inout":
                m = d @ m @ d
            else:
                m = d @ m
        out[i] = m
    return out
