"""Step-3/4 XPD estimators and conditional statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def parity_labels(bounce_count: np.ndarray) -> np.ndarray:
    return (np.asarray(bounce_count, dtype=int) % 2).astype(int)


def pathwise_xpd_db(a_f: np.ndarray, freq_axis: int = 1) -> np.ndarray:
    """Per-path XPD over frequency, returns shape (L,Nf) or reduced by caller."""

    co = np.abs(a_f[..., 0, 0]) ** 2 + np.abs(a_f[..., 1, 1]) ** 2 + 1e-15
    cross = np.abs(a_f[..., 0, 1]) ** 2 + np.abs(a_f[..., 1, 0]) ** 2 + 1e-15
    return 10.0 * np.log10(co / cross)


def pathwise_xpd_summary(a_f: np.ndarray, n_subbands: int = 4) -> Dict[str, np.ndarray]:
    x = pathwise_xpd_db(a_f)
    avg = x.mean(axis=1)
    bands = np.array_split(np.arange(x.shape[1]), n_subbands)
    sub = np.stack([x[:, b].mean(axis=1) for b in bands], axis=1)
    return {"xpd_db_freq": x, "xpd_db_avg": avg, "xpd_db_subband": sub}


def tapwise_xpd_db(
    h_tau: np.ndarray,
    tau_axis_s: np.ndarray,
    win_s: Tuple[float, float] | None = None,
    early_late_split_s: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    p_co = np.abs(h_tau[:, 0, 0]) ** 2 + np.abs(h_tau[:, 1, 1]) ** 2 + 1e-15
    p_cross = np.abs(h_tau[:, 0, 1]) ** 2 + np.abs(h_tau[:, 1, 0]) ** 2 + 1e-15
    xpd = 10.0 * np.log10(p_co / p_cross)
    mask = np.ones_like(xpd, dtype=bool)
    if win_s is not None:
        mask = (tau_axis_s >= win_s[0]) & (tau_axis_s <= win_s[1])
    out = {"xpd_db": xpd, "mask": mask}
    if early_late_split_s is not None:
        out["early_mean_db"] = np.array([np.mean(xpd[(tau_axis_s <= early_late_split_s) & mask])])
        out["late_mean_db"] = np.array([np.mean(xpd[(tau_axis_s > early_late_split_s) & mask])])
    return out


def conditional_normal_fit(samples_db: np.ndarray) -> Dict[str, float]:
    s = np.asarray(samples_db, dtype=float)
    return {"mu": float(np.mean(s)), "sigma": float(np.std(s, ddof=1) if s.size > 1 else 0.0)}


def conditional_stats(
    xpd_db: np.ndarray,
    parity: Optional[np.ndarray] = None,
    delay_bin: Optional[np.ndarray] = None,
    subband_idx: Optional[np.ndarray] = None,
    material_id: Optional[np.ndarray] = None,
    incidence_bin: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    n = len(xpd_db)
    labels: List[Tuple[str, np.ndarray]] = []
    if parity is not None:
        labels.append(("parity", np.asarray(parity)))
    if delay_bin is not None:
        labels.append(("delay", np.asarray(delay_bin)))
    if subband_idx is not None:
        labels.append(("subband", np.asarray(subband_idx)))
    if material_id is not None:
        labels.append(("material", np.asarray(material_id)))
    if incidence_bin is not None:
        labels.append(("incidence", np.asarray(incidence_bin)))

    if not labels:
        return {"all": conditional_normal_fit(xpd_db)}

    out: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        key = "|".join([f"{name}={arr[i]}" for name, arr in labels])
        out.setdefault(key, {"_vals": []})
        out[key]["_vals"].append(float(xpd_db[i]))

    for key in list(out.keys()):
        vals = np.array(out[key].pop("_vals"))
        out[key].update(conditional_normal_fit(vals))
        out[key]["count"] = int(vals.size)
    return out


def save_stats_json(path: str, stats: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
