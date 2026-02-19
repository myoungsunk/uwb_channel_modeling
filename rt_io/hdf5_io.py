"""HDF5 schema for polarimetric RT outputs.

The schema stores multiple scenarios and multiple sweep cases per scenario.

Structure:
    /
      meta                       (attrs: created_at, basis, convention)
      frequency                  (Nf,)
      scenarios/{scenario_id}/cases/{case_id}/
          params_json            (scalar utf-8 JSON)
          paths/
              tau_s              (L,)
              A_f                (L,Nf,2,2) complex
              bounce_count       (L,)
              interactions       (L,) variable-length UTF-8 ("|"-joined)
              surface_ids        (L,Smax) int64, -1 padded
              incidence_angles   (L,Smax) float64, nan padded
              AoD                (L,3)
              AoA                (L,3)
              AoD_unit           (L,3)
              AoA_unit           (L,3)
              u_tx               (L,3)
              v_tx               (L,3)
              u_rx               (L,3)
              v_rx               (L,3)

Example:
    >>> import numpy as np
    >>> from rt_core.tracer import Path
    >>> p = Path(1e-8, np.ones((4,2,2), dtype=np.complex128), {
    ...     "bounce_count": 0,
    ...     "interactions": ["LOS"],
    ...     "surface_ids": [],
    ...     "incidence_angles": [],
    ...     "AoD": [1.0,0.0,0.0],
    ...     "AoA": [-1.0,0.0,0.0],
    ... })
    >>> payload = {"C0": {"case0": {"params": {"max_bounce": 0}, "paths": [p]}}}
    >>> save_rt_hdf5("/tmp/rt_example.h5", np.linspace(3e9,4e9,4), payload)
    >>> freq, loaded = load_rt_hdf5("/tmp/rt_example.h5")
    >>> freq.shape, list(loaded.keys())
    ((4,), ['C0'])
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np

from analysis.ctf_cir import cir_ifft, synthesize_ctf
from rt_core.tracer import Path


@dataclass
class CaseData:
    params: Dict[str, Any]
    paths: List[Path]


@dataclass
class Hdf5Meta:
    created_at: str
    basis: str
    convention: str


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unsupported JSON type: {type(obj)}")


def _pad_2d_int(rows: Sequence[Sequence[int]], pad_value: int = -1) -> np.ndarray:
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), pad_value, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = np.asarray(row, dtype=np.int64)
    return out


def _pad_2d_float(rows: Sequence[Sequence[float]], pad_value: float = np.nan) -> np.ndarray:
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), pad_value, dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = np.asarray(row, dtype=np.float64)
    return out


def save_rt_hdf5(
    filepath: str,
    frequency_hz: np.ndarray,
    scenarios: Mapping[str, Mapping[str, CaseData | Mapping[str, Any]]],
    basis: str = "linear",
    convention: str = "IEEE RHCP",
) -> None:
    """Save RT outputs to HDF5 using a fixed schema contract."""

    with h5py.File(filepath, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["basis"] = basis
        meta.attrs["convention"] = convention

        h5.create_dataset("frequency", data=np.asarray(frequency_hz, dtype=np.float64))
        g_scenarios = h5.create_group("scenarios")

        for scenario_id, cases in scenarios.items():
            g_cases = g_scenarios.create_group(str(scenario_id)).create_group("cases")
            for case_id, case in cases.items():
                case_obj = case if isinstance(case, CaseData) else CaseData(params=dict(case["params"]), paths=list(case["paths"]))
                g_case = g_cases.create_group(str(case_id))
                g_case.create_dataset("params_json", data=json.dumps(case_obj.params, default=_json_default))

                paths = case_obj.paths
                g_paths = g_case.create_group("paths")
                g_paths.create_dataset("tau_s", data=np.array([p.tau_s for p in paths], dtype=np.float64))
                g_paths.create_dataset("A_f", data=np.array([p.a_f for p in paths], dtype=np.complex128))

                bounce_count = [int(p.meta.get("bounce_count", 0)) for p in paths]
                interactions = ["|".join(map(str, p.meta.get("interactions", []))) for p in paths]
                surface_rows = [list(map(int, p.meta.get("surface_ids", []))) for p in paths]
                incidence_rows = [list(map(float, p.meta.get("incidence_angles", []))) for p in paths]
                aod = np.array([p.meta.get("AoD", [np.nan, np.nan, np.nan]) for p in paths], dtype=np.float64)
                aoa = np.array([p.meta.get("AoA", [np.nan, np.nan, np.nan]) for p in paths], dtype=np.float64)
                aod_u = np.array([p.meta.get("AoD_unit", [np.nan, np.nan, np.nan]) for p in paths], dtype=np.float64)
                aoa_u = np.array([p.meta.get("AoA_unit", [np.nan, np.nan, np.nan]) for p in paths], dtype=np.float64)

                tx_basis = [p.meta.get("u_v_basis", {}).get("tx", (np.full(3, np.nan), np.full(3, np.nan))) for p in paths]
                rx_basis = [p.meta.get("u_v_basis", {}).get("rx", (np.full(3, np.nan), np.full(3, np.nan))) for p in paths]
                u_tx = np.array([b[0] for b in tx_basis], dtype=np.complex128)
                v_tx = np.array([b[1] for b in tx_basis], dtype=np.complex128)
                u_rx = np.array([b[0] for b in rx_basis], dtype=np.complex128)
                v_rx = np.array([b[1] for b in rx_basis], dtype=np.complex128)

                g_paths.create_dataset("bounce_count", data=np.asarray(bounce_count, dtype=np.int32))
                dt = h5py.string_dtype(encoding="utf-8")
                g_paths.create_dataset("interactions", data=np.asarray(interactions, dtype=dt))
                g_paths.create_dataset("surface_ids", data=_pad_2d_int(surface_rows, -1))
                g_paths.create_dataset("incidence_angles", data=_pad_2d_float(incidence_rows, np.nan))
                g_paths.create_dataset("AoD", data=aod)
                g_paths.create_dataset("AoA", data=aoa)
                g_paths.create_dataset("AoD_unit", data=aod_u)
                g_paths.create_dataset("AoA_unit", data=aoa_u)
                g_paths.create_dataset("u_tx", data=u_tx)
                g_paths.create_dataset("v_tx", data=v_tx)
                g_paths.create_dataset("u_rx", data=u_rx)
                g_paths.create_dataset("v_rx", data=v_rx)


def _unpad_int_row(row: np.ndarray, pad_value: int = -1) -> List[int]:
    return [int(v) for v in row.tolist() if int(v) != pad_value]


def _unpad_float_row(row: np.ndarray) -> List[float]:
    return [float(v) for v in row.tolist() if np.isfinite(v)]


def load_rt_hdf5(filepath: str) -> Tuple[np.ndarray, Dict[str, Dict[str, CaseData]], Hdf5Meta]:
    """Load RT HDF5 and reconstruct cases/paths."""

    scenarios: Dict[str, Dict[str, CaseData]] = {}
    with h5py.File(filepath, "r") as h5:
        meta = Hdf5Meta(
            created_at=str(h5["meta"].attrs.get("created_at", "")),
            basis=str(h5["meta"].attrs.get("basis", "linear")),
            convention=str(h5["meta"].attrs.get("convention", "IEEE RHCP")),
        )
        frequency = np.asarray(h5["frequency"][()], dtype=np.float64)

        for scenario_id, g_scenario in h5["scenarios"].items():
            scenarios[scenario_id] = {}
            for case_id, g_case in g_scenario["cases"].items():
                params = json.loads(g_case["params_json"][()].decode() if isinstance(g_case["params_json"][()], bytes) else g_case["params_json"][()])
                g_paths = g_case["paths"]

                tau_s = np.asarray(g_paths["tau_s"][()], dtype=np.float64)
                a_f = np.asarray(g_paths["A_f"][()], dtype=np.complex128)
                bounce_count = np.asarray(g_paths["bounce_count"][()], dtype=np.int32)
                interactions_raw = g_paths["interactions"][()]
                interactions = [
                    (s.decode() if isinstance(s, bytes) else str(s)).split("|") if (s.decode() if isinstance(s, bytes) else str(s)) else []
                    for s in interactions_raw
                ]
                surface = np.asarray(g_paths["surface_ids"][()], dtype=np.int64)
                incidence = np.asarray(g_paths["incidence_angles"][()], dtype=np.float64)
                aod = np.asarray(g_paths["AoD"][()], dtype=np.float64)
                aoa = np.asarray(g_paths["AoA"][()], dtype=np.float64)
                aod_u = np.asarray(g_paths["AoD_unit"][()], dtype=np.float64) if "AoD_unit" in g_paths else np.full_like(aod, np.nan)
                aoa_u = np.asarray(g_paths["AoA_unit"][()], dtype=np.float64) if "AoA_unit" in g_paths else np.full_like(aoa, np.nan)
                u_tx = np.asarray(g_paths["u_tx"][()], dtype=np.complex128) if "u_tx" in g_paths else np.full((len(tau_s), 3), np.nan, dtype=np.complex128)
                v_tx = np.asarray(g_paths["v_tx"][()], dtype=np.complex128) if "v_tx" in g_paths else np.full((len(tau_s), 3), np.nan, dtype=np.complex128)
                u_rx = np.asarray(g_paths["u_rx"][()], dtype=np.complex128) if "u_rx" in g_paths else np.full((len(tau_s), 3), np.nan, dtype=np.complex128)
                v_rx = np.asarray(g_paths["v_rx"][()], dtype=np.complex128) if "v_rx" in g_paths else np.full((len(tau_s), 3), np.nan, dtype=np.complex128)

                paths: List[Path] = []
                for i in range(len(tau_s)):
                    paths.append(
                        Path(
                            tau_s=float(tau_s[i]),
                            a_f=a_f[i],
                            meta={
                                "bounce_count": int(bounce_count[i]),
                                "interactions": interactions[i],
                                "surface_ids": _unpad_int_row(surface[i]),
                                "incidence_angles": _unpad_float_row(incidence[i]),
                                "AoD": aod[i].tolist(),
                                "AoA": aoa[i].tolist(),
                                "AoD_unit": aod_u[i].tolist(),
                                "AoA_unit": aoa_u[i].tolist(),
                                "u_v_basis": {
                                    "tx": (u_tx[i].tolist(), v_tx[i].tolist()),
                                    "rx": (u_rx[i].tolist(), v_rx[i].tolist()),
                                },
                            },
                        )
                    )

                scenarios[scenario_id][case_id] = CaseData(params=params, paths=paths)

    return frequency, scenarios, meta


def self_test_roundtrip(filepath: str, atol: float = 1e-10) -> bool:
    """Write->read equivalence self-test checking H(f) and CIR reproducibility."""

    f = np.linspace(3e9, 5e9, 64)
    rng = np.random.default_rng(7)
    paths = [
        Path(
            tau_s=12e-9,
            a_f=(rng.standard_normal((64, 2, 2)) + 1j * rng.standard_normal((64, 2, 2))).astype(np.complex128) * 0.1,
            meta={
                "bounce_count": 1,
                "interactions": ["reflection"],
                "surface_ids": [2],
                "incidence_angles": [0.45],
                "AoD": [0.9, 0.1, 0.0],
                "AoA": [-0.9, -0.1, 0.0],
            },
        )
    ]
    payload = {"selftest": {"case0": CaseData(params={"seed": 7}, paths=paths)}}
    save_rt_hdf5(filepath, f, payload, basis="linear", convention="IEEE RHCP")
    f2, scenarios, _ = load_rt_hdf5(filepath)
    p2 = scenarios["selftest"]["case0"].paths

    a1 = np.array([p.a_f for p in paths])
    a2 = np.array([p.a_f for p in p2])
    t1 = np.array([p.tau_s for p in paths])
    t2 = np.array([p.tau_s for p in p2])

    h1 = synthesize_ctf(f, t1, a1)
    h2 = synthesize_ctf(f2, t2, a2)
    c1 = cir_ifft(h1)
    c2 = cir_ifft(h2)
    return bool(np.allclose(h1, h2, atol=atol) and np.allclose(c1, c2, atol=atol))
