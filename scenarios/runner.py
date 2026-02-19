"""Scenario sweep runner + auto plot + validation report."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis.ctf_cir import SynthesisConfig, synthesize_case
from analysis.xpd_stats import conditional_stats, parity_labels, pathwise_xpd_summary
from plots import p0_p13
from rt_io.hdf5_io import CaseData, save_rt_hdf5
from rt_core.tracer import C0

SCENARIO_MODULES = {
    "C0": "scenarios.C0_free_space",
    "A1": "scenarios.A1_los_min_reflect",
    "A2": "scenarios.A2_pec_plane",
    "A3": "scenarios.A3_dihedral",
    "A4": "scenarios.A4_dielectric",
    "A5": "scenarios.A5_depol_stress",
}


def run_all(out_h5: str = "artifacts/rt_sweep.h5", out_plot_dir: str = "artifacts/plots") -> str:
    payload: Dict[str, Dict[str, CaseData]] = {}
    report_lines: List[str] = [
        "# Validation Report",
        "",
        "- power metric: `10*log10(sum(|A_l(f)|^2 over f,2,2) + 1e-15)`",
        "- strongest path range: `r = tau_s * c0`",
        "",
    ]
    all_k = []
    all_names = []
    failures: List[str] = []
    a5_strongest: List[Dict[str, float]] = []

    for sid, mod_name in SCENARIO_MODULES.items():
        mod = import_module(mod_name)
        payload[sid] = {}
        report_lines.append(f"## {sid}")
        for p in mod.build_sweep_params():
            freq, paths = mod.run_case(p)
            case_id = p["case_id"]
            payload[sid][case_id] = CaseData(params=p, paths=paths)

            tau = np.array([x.tau_s for x in paths], dtype=float)
            a_f = np.array([x.a_f for x in paths], dtype=np.complex128)
            syn = synthesize_case(freq, tau, a_f, config=SynthesisConfig(nfft=512, window="hann"))
            pdp = syn["PDP"]
            power = np.sum(np.abs(a_f) ** 2, axis=(1, 2, 3)) if len(paths) else np.array([])
            bounce = np.array([x.meta.get("bounce_count", 0) for x in paths], dtype=int) if len(paths) else np.array([])

            case_dir = str(Path(out_plot_dir) / sid / case_id)
            if len(paths):
                p0_p13.p1_tau_power(tau, power, bounce, case_dir)
                p0_p13.p2_h_mag(freq, syn["H_f"], case_dir)
                tau_axis = np.arange(syn["h_tau"].shape[0]) / (syn["h_tau"].shape[0] * (freq[1] - freq[0]))
                p0_p13.p3_pdp(tau_axis, pdp, case_dir)
                p0_p13.p4_zoom_taps(tau_axis, pdp["total"], case_dir)

                xpds = pathwise_xpd_summary(a_f)["xpd_db_avg"]
                xpd_detail = pathwise_xpd_summary(a_f)
                par = parity_labels(bounce)
                if np.any(par == 0) and np.any(par == 1):
                    p0_p13.p6_parity_box(xpds[par == 0], xpds[par == 1], case_dir)
                stats = conditional_stats(xpds, parity=par)

                strongest = int(np.argmax(power))
                los_exists = bool(np.any(bounce == 0))
                strongest_r_m = float(tau[strongest] * C0)
                strongest_parity = int(bounce[strongest] % 2)
                report_lines.append(f"- case `{case_id}`: paths={len(paths)}, bounce_dist={dict(zip(*np.unique(bounce, return_counts=True)))}")
                report_lines.append(f"  - LOS exists: {los_exists}")
                report_lines.append(
                    f"  - strongest path id={strongest}, tau={tau[strongest]*1e9:.3f} ns, "
                    f"r={strongest_r_m:.3f} m, power={10*np.log10(power[strongest]+1e-15):.2f} dB, "
                    f"bounce_count={int(bounce[strongest])}"
                )
                parity_lines = []
                for k, v in stats.items():
                    parity_lines.append(f"{k}: mu={float(v['mu']):.3f}, sigma={float(v['sigma']):.3f}, n={int(v['count'])}")
                report_lines.append(f"  - parity stats: {parity_lines}")

                xpd_f = np.mean(xpd_detail["xpd_db_freq"], axis=0)
                p0_p13.p10_xpd_freq(freq, xpd_f, case_dir, label=case_id)
                sub = xpd_detail["xpd_db_subband"]
                mu_b = np.mean(sub, axis=0)
                sigma_b = np.std(sub, axis=0)
                p0_p13.p9_subband_mu_sigma(mu_b, sigma_b, case_dir)
                report_lines.append(f"  - subband mu(dB): {[float(x) for x in mu_b]}")
                report_lines.append(f"  - subband sigma(dB): {[float(x) for x in sigma_b]}")
                report_lines.append(f"  - plots: [P1]({case_dir}/P1.png), [P2]({case_dir}/P2.png), [P3]({case_dir}/P3.png), [P4]({case_dir}/P4.png), [P9]({case_dir}/P9.png), [P10]({case_dir}/P10.png)")

                if sid == "A2" and (not np.any(bounce == 1) or np.any(bounce == 0)):
                    report_lines.append("  - WARNING: A2 quality check failed (missing 1-bounce or LOS remained)")

                if sid == "A2" and strongest_parity != 1:
                    failures.append(f"A2:{case_id} strongest parity expected odd(1), got {strongest_parity}")
                if sid == "A3" and p.get("strict", False) and np.any(bounce != 2):
                    failures.append(f"A3 strict:{case_id} contains non-2-bounce paths")
                if sid == "A3" and p.get("strict", False) and strongest_parity != 0:
                    failures.append(f"A3 strict:{case_id} strongest parity expected even(0), got {strongest_parity}")
                if sid == "A5":
                    a5_strongest.append(
                        {
                            "case_id": case_id,
                            "rho": float(p.get("rho", np.nan)),
                            "loss_enabled": bool(p.get("loss_enabled", False)),
                            "strongest_db": float(10 * np.log10(power[strongest] + 1e-15)),
                        }
                    )

                p_los = np.max(power[bounce == 0]) if np.any(bounce == 0) else 1e-15
                p_nlos = np.sum(power[bounce > 0]) + 1e-15
                k_db = 10 * np.log10(p_los / p_nlos)
                all_k.append(k_db)
                all_names.append(f"{sid}:{case_id}")
            else:
                report_lines.append(f"- case `{case_id}`: paths=0")

        report_lines.append("")

    # A5 physics sanity check: with higher rho, strongest power should not increase.
    if a5_strongest:
        groups: Dict[bool, List[Dict[str, float]]] = {}
        for row in a5_strongest:
            groups.setdefault(bool(row["loss_enabled"]), []).append(row)
        for loss_enabled, rows in groups.items():
            rows_sorted = sorted(rows, key=lambda x: x["rho"])
            for i in range(1, len(rows_sorted)):
                if rows_sorted[i]["strongest_db"] > rows_sorted[i - 1]["strongest_db"] + 1e-9:
                    failures.append(
                        f"A5(loss_enabled={loss_enabled}) rho {rows_sorted[i-1]['rho']}->{rows_sorted[i]['rho']} increased strongest power "
                        f"{rows_sorted[i-1]['strongest_db']:.3f}->{rows_sorted[i]['strongest_db']:.3f} dB"
                    )

    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    save_rt_hdf5(out_h5, freq, payload, basis="linear", convention="IEEE RHCP")
    p0_p13.p13_kfactor_trend(all_names, all_k, out_plot_dir)

    report_lines.append("## Failure Checks")
    if failures:
        for msg in failures:
            report_lines.append(f"- FAIL: {msg}")
    else:
        report_lines.append("- PASS: No automatic failure checks triggered.")

    report_path = Path(out_plot_dir).parent / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return str(report_path)


if __name__ == "__main__":
    print(run_all())
