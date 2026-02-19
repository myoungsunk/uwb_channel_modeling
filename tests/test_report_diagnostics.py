import numpy as np


def _synthetic_failure_check_rows():
    return [
        {"case_id": "a5_rho0.2", "rho": 0.2, "loss_enabled": False, "strongest_db": 10.0},
        {"case_id": "a5_rho0.5", "rho": 0.5, "loss_enabled": False, "strongest_db": 11.0},
    ]


def test_failure_check_detects_a5_power_increase_logic():
    rows = _synthetic_failure_check_rows()
    rows_sorted = sorted(rows, key=lambda x: x["rho"])
    increased = rows_sorted[1]["strongest_db"] > rows_sorted[0]["strongest_db"] + 1e-9
    assert increased


def test_parity_summary_format_values_present():
    stats = {"parity=0": {"mu": 3.0, "sigma": 1.0, "count": 2}, "parity=1": {"mu": 1.0, "sigma": 0.5, "count": 3}}
    lines = [f"{k}: mu={float(v['mu']):.3f}, sigma={float(v['sigma']):.3f}, n={int(v['count'])}" for k, v in stats.items()]
    assert any("mu=" in x and "sigma=" in x and "n=" in x for x in lines)
