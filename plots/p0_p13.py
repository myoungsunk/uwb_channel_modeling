"""Automated P0~P13 plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, outdir: str, name: str) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    png = Path(outdir) / f"{name}.png"
    pdf = Path(outdir) / f"{name}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    try:
        fig.savefig(pdf, bbox_inches="tight")
    except PermissionError:
        warnings.warn(
            f"Could not write '{pdf}' (permission denied). Saved PNG only.",
            RuntimeWarning,
            stacklevel=2,
        )
    plt.close(fig)
    return str(png)


def p0_geometry_overlay(points: np.ndarray, bounce_count: np.ndarray, outdir: str) -> str:
    fig, ax = plt.subplots()
    sc = ax.scatter(points[:, 0], points[:, 1], c=bounce_count, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="bounce")
    ax.set_title("P0 geometry/ray overlay")
    return _save(fig, outdir, "P0")


def p1_tau_power(tau: np.ndarray, power: np.ndarray, bounce: np.ndarray, outdir: str) -> str:
    fig, ax = plt.subplots()
    sc = ax.scatter(tau * 1e9, 10 * np.log10(power + 1e-15), c=bounce, cmap="plasma")
    fig.colorbar(sc, ax=ax, label="bounce")
    ax.set_xlabel("delay [ns]")
    ax.set_ylabel("power [dB]")
    ax.set_title("P1 path scatter")
    return _save(fig, outdir, "P1")


def p2_h_mag(freq: np.ndarray, h_f: np.ndarray, outdir: str) -> str:
    fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(freq / 1e9, 20 * np.log10(np.abs(h_f[:, i, j]) + 1e-15))
            axs[i, j].set_title(f"|H{i+1}{j+1}|")
    fig.suptitle("P2 |H_ij(f)|")
    return _save(fig, outdir, "P2")


def p3_pdp(tau: np.ndarray, pdp: Dict[str, np.ndarray], outdir: str) -> str:
    fig, ax = plt.subplots()
    ax.plot(tau * 1e9, 10 * np.log10(pdp["co_sum"] + 1e-15), label="co")
    ax.plot(tau * 1e9, 10 * np.log10(pdp["cross_sum"] + 1e-15), label="cross")
    ax.legend()
    ax.set_title("P3 PDP")
    return _save(fig, outdir, "P3")


def p4_zoom_taps(tau: np.ndarray, pdp_total: np.ndarray, outdir: str, n: int = 128) -> str:
    fig, ax = plt.subplots()
    ax.plot(tau[:n] * 1e9, 10 * np.log10(pdp_total[:n] + 1e-15))
    ax.set_title("P4 tap zoom")
    return _save(fig, outdir, "P4")


def p6_parity_box(x_even: np.ndarray, x_odd: np.ndarray, outdir: str) -> str:
    fig, ax = plt.subplots()
    ax.boxplot([x_even, x_odd], labels=["even", "odd"])
    ax.set_title("P6 parity XPD")
    return _save(fig, outdir, "P6")


def p9_subband_mu_sigma(mu: np.ndarray, sigma: np.ndarray, outdir: str) -> str:
    fig, ax = plt.subplots()
    x = np.arange(len(mu))
    ax.errorbar(x, mu, yerr=sigma, fmt="o-")
    ax.set_title("P9 subband mu/sigma")
    return _save(fig, outdir, "P9")


def p12_delay_mu_sigma(mu: np.ndarray, sigma: np.ndarray, outdir: str) -> str:
    fig, ax = plt.subplots()
    x = np.arange(len(mu))
    ax.errorbar(x, mu, yerr=sigma, fmt="o-")
    ax.set_title("P12 delay-conditioned mu/sigma")
    return _save(fig, outdir, "P12")


def p13_kfactor_trend(names: Sequence[str], k_db: Sequence[float], outdir: str) -> str:
    fig, ax = plt.subplots()
    ax.plot(names, k_db, "o-")
    ax.set_title("P13 K-factor trend")
    return _save(fig, outdir, "P13")


def p_generic_hist(idx: int, values: np.ndarray, outdir: str) -> str:
    fig, ax = plt.subplots()
    ax.hist(values, bins=30)
    ax.set_title(f"P{idx}")
    return _save(fig, outdir, f"P{idx}")
