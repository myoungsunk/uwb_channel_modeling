import numpy as np

from rt_core.antenna import AntennaPort
from rt_core.path_gain import fspl_amplitude
from rt_core.tracer import trace_paths


def _ant_pair(distance_m: float):
    tx = AntennaPort(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "HV")
    rx = AntennaPort(np.array([distance_m, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "HV")
    return tx, rx


def test_fspl_amplitude_ratio_matches_inverse_distance():
    freq = np.array([6.0e9], dtype=float)
    g1 = fspl_amplitude(freq, 2.0)[0]
    g2 = fspl_amplitude(freq, 3.0)[0]
    assert np.isclose(g1 / g2, 3.0 / 2.0, rtol=1e-12, atol=1e-12)


def test_los_power_difference_matches_distance_scaling():
    freq = np.linspace(3e9, 9e9, 16)
    tx1, rx1 = _ant_pair(2.0)
    tx2, rx2 = _ant_pair(3.0)

    p1 = trace_paths([], tx1, rx1, freq, max_bounce=0)[0]
    p2 = trace_paths([], tx2, rx2, freq, max_bounce=0)[0]

    pw1 = float(np.sum(np.abs(p1.a_f) ** 2))
    pw2 = float(np.sum(np.abs(p2.a_f) ** 2))
    observed_db = 10.0 * np.log10(pw1 / pw2)
    expected_db = 20.0 * np.log10(3.0 / 2.0)
    assert np.isclose(observed_db, expected_db, rtol=1e-9, atol=1e-9)


def test_total_power_normalization_is_opt_in():
    freq = np.linspace(3e9, 5e9, 8)
    tx, rx = _ant_pair(5.0)
    paths_no_norm = trace_paths([], tx, rx, freq, max_bounce=0)
    paths_norm = trace_paths([], tx, rx, freq, max_bounce=0, normalize_total_power=True)

    total_no_norm = sum(float(np.sum(np.abs(p.a_f) ** 2)) for p in paths_no_norm)
    total_norm = sum(float(np.sum(np.abs(p.a_f) ** 2)) for p in paths_norm)

    assert total_no_norm != 1.0
    assert np.isclose(total_norm, 1.0, rtol=1e-12, atol=1e-12)
