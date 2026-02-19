"""Step-2 representation/basis utilities and CTF/CIR synthesis.

Example:
    >>> import numpy as np
    >>> from analysis.ctf_cir import synthesize_case
    >>> tau = np.array([0.0])
    >>> a_f = np.ones((1, 8, 2, 2), dtype=np.complex128)
    >>> out = synthesize_case(np.linspace(3e9, 4e9, 8), tau, a_f, nfft=16)
    >>> out["H_f"].shape, out["h_tau"].shape
    ((8, 2, 2), (16, 2, 2))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_C2L = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1j, -1j]], dtype=np.complex128)
_L2C = np.linalg.inv(_C2L)


@dataclass
class SynthesisConfig:
    nfft: Optional[int] = None
    window: str = "hann"  # hann|kaiser|none
    kaiser_beta: float = 8.0
    input_basis: str = "linear"
    output_basis: str = "linear"


def convert_basis_2x2(mats: np.ndarray, in_basis: str, out_basis: str) -> np.ndarray:
    """Convert N×2×2 matrices between linear and circular basis."""

    if in_basis == out_basis:
        return mats
    if in_basis == "linear" and out_basis == "circular":
        u = _L2C
    elif in_basis == "circular" and out_basis == "linear":
        u = _C2L
    else:
        raise ValueError(f"Unsupported basis conversion: {in_basis}->{out_basis}")
    return np.einsum("ab,nbc,cd->nad", u, mats, u.conj().T)


def _window(kind: str, n: int, beta: float = 8.0) -> np.ndarray:
    kind_l = kind.lower()
    if kind_l == "hann":
        return np.hanning(n)
    if kind_l == "kaiser":
        return np.kaiser(n, beta)
    if kind_l in {"none", "rect", "rectangular"}:
        return np.ones(n)
    raise ValueError(f"Unsupported window: {kind}")


def synthesize_ctf(freq_hz: np.ndarray, tau_s: np.ndarray, a_f: np.ndarray) -> np.ndarray:
    """H(f) = Σ_l A_l(f) exp(-j 2π f τ_l)."""

    h = np.zeros((len(freq_hz), 2, 2), dtype=np.complex128)
    for l in range(len(tau_s)):
        h += a_f[l] * np.exp(-1j * 2.0 * np.pi * freq_hz * tau_s[l])[:, None, None]
    return h


def cir_ifft(h_f: np.ndarray, nfft: Optional[int] = None, window: str = "hann", kaiser_beta: float = 8.0) -> np.ndarray:
    n_f = h_f.shape[0]
    nfft = n_f if nfft is None else int(nfft)
    w = _window(window, n_f, beta=kaiser_beta)[:, None, None]
    return np.fft.ifft(h_f * w, n=nfft, axis=0)


def pdp_components(h_tau: np.ndarray) -> Dict[str, np.ndarray]:
    p11 = np.abs(h_tau[:, 0, 0]) ** 2
    p12 = np.abs(h_tau[:, 0, 1]) ** 2
    p21 = np.abs(h_tau[:, 1, 0]) ** 2
    p22 = np.abs(h_tau[:, 1, 1]) ** 2
    return {
        "p11": p11,
        "p12": p12,
        "p21": p21,
        "p22": p22,
        "co_sum": p11 + p22,
        "cross_sum": p12 + p21,
        "total": p11 + p12 + p21 + p22,
    }


def synthesize_case(
    freq_hz: np.ndarray,
    tau_s: np.ndarray,
    a_f: np.ndarray,
    config: Optional[SynthesisConfig] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = config or SynthesisConfig()
    a_use = convert_basis_2x2(a_f, cfg.input_basis, cfg.output_basis)
    h_f = synthesize_ctf(freq_hz, tau_s, a_use)
    h_tau = cir_ifft(h_f, nfft=cfg.nfft, window=cfg.window, kaiser_beta=cfg.kaiser_beta)
    out = {"H_f": h_f, "h_tau": h_tau, "PDP": pdp_components(h_tau)}
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, H_f=h_f, h_tau=h_tau, **out["PDP"])
    return out
