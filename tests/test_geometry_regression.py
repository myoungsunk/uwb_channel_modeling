import numpy as np

from rt_core.antenna import AntennaPort
from rt_core.geometry import Material, Plane, mirror_point_across_plane
from rt_core.rays import reflect
from rt_core.tracer import C0, _solve_reflection_points, trace_paths


def _antennas(tx_pos: np.ndarray, rx_pos: np.ndarray):
    tx = AntennaPort(tx_pos, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "HV")
    rx = AntennaPort(rx_pos, np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "HV")
    return tx, rx


def _angle_to_normal(direction: np.ndarray, normal: np.ndarray) -> float:
    u = direction / np.linalg.norm(direction)
    n = normal / np.linalg.norm(normal)
    return float(np.arccos(np.clip(abs(np.dot(u, n)), 0.0, 1.0)))


def test_los_tau_matches_distance_over_c():
    tx_pos = np.array([0.0, 0.0, 1.2])
    rx_pos = np.array([3.0, 4.0, 1.2])
    tx, rx = _antennas(tx_pos, rx_pos)
    freq = np.array([4.0e9])

    paths = trace_paths([], tx, rx, freq, max_bounce=0)
    assert len(paths) == 1

    expected_dist = np.linalg.norm(rx_pos - tx_pos)
    expected_tau = expected_dist / C0
    assert np.isclose(paths[0].tau_s, expected_tau, rtol=1e-9, atol=0.0)


def test_one_bounce_plane_reflection_law_and_image_path_length():
    tx_pos = np.array([1.0, 1.5, 1.0])
    rx_pos = np.array([4.0, 1.0, 1.0])
    plane = Plane(point=np.array([0.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), material=Material("PEC"), surface_id="x0")

    pts = _solve_reflection_points(tx_pos, rx_pos, [plane])
    assert pts is not None
    hit = pts[0]

    k_in = (hit - tx_pos) / np.linalg.norm(hit - tx_pos)
    k_out = (rx_pos - hit) / np.linalg.norm(rx_pos - hit)
    expected_k_out = reflect(k_in, plane.unit_normal())

    assert np.allclose(k_out, expected_k_out, rtol=1e-9, atol=1e-12)

    theta_i = _angle_to_normal(-k_in, plane.unit_normal())
    theta_r = _angle_to_normal(k_out, plane.unit_normal())
    assert np.isclose(theta_i, theta_r, rtol=1e-9, atol=1e-12)

    mirrored_rx = mirror_point_across_plane(rx_pos, plane)
    expected_len = np.linalg.norm(mirrored_rx - tx_pos)
    actual_len = np.linalg.norm(hit - tx_pos) + np.linalg.norm(rx_pos - hit)
    assert np.isclose(actual_len, expected_len, rtol=1e-9, atol=1e-12)


def test_two_bounce_corner_reflections_and_image_path_length():
    tx_pos = np.array([2.0, 3.0, 1.0])
    rx_pos = np.array([4.0, 5.0, 1.0])
    p1 = Plane(point=np.array([0.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), material=Material("PEC"), surface_id="x0")
    p2 = Plane(point=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 1.0, 0.0]), material=Material("PEC"), surface_id="y0")

    pts = _solve_reflection_points(tx_pos, rx_pos, [p1, p2])
    assert pts is not None
    h1, h2 = pts

    seg0 = (h1 - tx_pos) / np.linalg.norm(h1 - tx_pos)
    seg1 = (h2 - h1) / np.linalg.norm(h2 - h1)
    seg2 = (rx_pos - h2) / np.linalg.norm(rx_pos - h2)

    assert np.allclose(seg1, reflect(seg0, p1.unit_normal()), rtol=1e-9, atol=1e-12)
    assert np.allclose(seg2, reflect(seg1, p2.unit_normal()), rtol=1e-9, atol=1e-12)

    theta1_i = _angle_to_normal(-seg0, p1.unit_normal())
    theta1_r = _angle_to_normal(seg1, p1.unit_normal())
    theta2_i = _angle_to_normal(-seg1, p2.unit_normal())
    theta2_r = _angle_to_normal(seg2, p2.unit_normal())
    assert np.isclose(theta1_i, theta1_r, rtol=1e-9, atol=1e-12)
    assert np.isclose(theta2_i, theta2_r, rtol=1e-9, atol=1e-12)

    mirror_rx = mirror_point_across_plane(mirror_point_across_plane(rx_pos, p2), p1)
    expected_len = np.linalg.norm(mirror_rx - tx_pos)
    actual_len = np.linalg.norm(h1 - tx_pos) + np.linalg.norm(h2 - h1) + np.linalg.norm(rx_pos - h2)
    assert np.isclose(actual_len, expected_len, rtol=1e-9, atol=1e-12)
