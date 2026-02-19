"""Path matching/alignment utilities for combining separate TX-port runs.

Use-case: build 2x2 transfer paths from two separate simulations (TX=H and TX=V).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from rt_core.tracer import Path


@dataclass
class MatchConfig:
    tau_tolerance_s: float = 0.25e-9
    allow_nearest: bool = True


def _geometry_key(path: Path) -> Tuple:
    m = path.meta
    bounce = int(m.get("bounce_count", 0))
    surfaces = tuple(int(s) for s in m.get("surface_ids", []))
    incid = tuple(round(float(v), 6) for v in m.get("incidence_angles", []))
    aod = tuple(round(float(v), 5) for v in m.get("AoD", [0.0, 0.0, 0.0]))
    aoa = tuple(round(float(v), 5) for v in m.get("AoA", [0.0, 0.0, 0.0]))
    return bounce, surfaces, incid, aod, aoa


def combine_runs_to_2x2(
    run_h: Sequence[Path],
    run_v: Sequence[Path],
    config: MatchConfig | None = None,
    key_fn: Callable[[Path], Tuple] = _geometry_key,
) -> Tuple[List[Path], List[str]]:
    """Match H/V paths and build combined A with columns [y_H, y_V].

    Assumes each input path has `a_f` shape (Nf, 2) or (Nf,2,1). Output shape is (Nf,2,2).
    Returns (combined_paths, warnings).
    """

    cfg = config or MatchConfig()
    warnings: List[str] = []
    used_v = set()
    v_by_key: Dict[Tuple, List[int]] = {}
    for j, pv in enumerate(run_v):
        v_by_key.setdefault(key_fn(pv), []).append(j)

    combined: List[Path] = []
    for i, ph in enumerate(run_h):
        k = key_fn(ph)
        cand = [j for j in v_by_key.get(k, []) if j not in used_v]
        chosen = None
        if cand:
            chosen = min(cand, key=lambda j: abs(run_v[j].tau_s - ph.tau_s))
            if abs(run_v[chosen].tau_s - ph.tau_s) > cfg.tau_tolerance_s:
                warnings.append(f"key-match tau mismatch too high: H[{i}] V[{chosen}]")
                chosen = None

        if chosen is None and cfg.allow_nearest and run_v:
            free = [j for j in range(len(run_v)) if j not in used_v]
            if free:
                chosen = min(free, key=lambda j: abs(run_v[j].tau_s - ph.tau_s))
                dt = abs(run_v[chosen].tau_s - ph.tau_s)
                if dt > cfg.tau_tolerance_s:
                    warnings.append(f"unmatched H[{i}] nearest V[{chosen}] exceeds tolerance {dt:.3e}s")
                    chosen = None
                else:
                    warnings.append(f"approximate match H[{i}] -> V[{chosen}] with dt={dt:.3e}s")

        if chosen is None:
            warnings.append(f"no match for H[{i}]")
            continue

        used_v.add(chosen)
        pv = run_v[chosen]
        n_f = ph.a_f.shape[0]
        a = np.zeros((n_f, 2, 2), dtype=np.complex128)
        a[:, :, 0] = ph.a_f if ph.a_f.ndim == 2 else ph.a_f[..., :, 0]
        a[:, :, 1] = pv.a_f if pv.a_f.ndim == 2 else pv.a_f[..., :, 0]

        meta = dict(ph.meta)
        meta["matched_with_v_index"] = chosen
        combined.append(Path(tau_s=float((ph.tau_s + pv.tau_s) * 0.5), a_f=a, meta=meta))

    for j in range(len(run_v)):
        if j not in used_v:
            warnings.append(f"unmatched V[{j}]")

    return combined, warnings
