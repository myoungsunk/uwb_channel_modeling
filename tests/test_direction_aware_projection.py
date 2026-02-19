import numpy as np

from rt_core.antenna import AntennaPort
from rt_core.polarization import projection_matrix, transverse_basis


def _fro_energy(m: np.ndarray) -> float:
    return float(np.sum(np.abs(m) ** 2))


def test_direction_aware_projection_preserves_energy_for_basis_change():
    ant = AntennaPort(
        position=np.array([0.0, 0.0, 0.0]),
        boresight=np.array([1.0, 0.0, 0.0]),
        h_axis=np.array([0.0, 1.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        port_basis="HV",
    )
    k1 = np.array([1.0, 0.0, 0.0])
    k2 = np.array([0.7, 0.5, 0.5])
    m = np.array([[1.0 + 0.2j, 0.3 - 0.1j], [0.1 + 0.4j, -0.7 + 0.2j]], dtype=np.complex128)

    b1 = ant.transverse_port_basis(k1)
    b2 = ant.transverse_port_basis(k2)
    t = projection_matrix(b1, b2)
    m2 = t @ m @ t.conj().T

    assert np.isclose(_fro_energy(m2), _fro_energy(m), rtol=1e-10, atol=1e-12)


def test_path_meta_basis_exists_in_trace_paths():
    from rt_core.tracer import trace_paths

    tx = AntennaPort(np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    rx = AntennaPort(np.array([5.0, 0.0, 1.0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), "HV")
    f = np.array([4.0e9])
    paths = trace_paths([], tx, rx, f, max_bounce=0)
    p = paths[0]

    assert "AoD_unit" in p.meta and "AoA_unit" in p.meta
    assert "u_v_basis" in p.meta and "tx" in p.meta["u_v_basis"] and "rx" in p.meta["u_v_basis"]
