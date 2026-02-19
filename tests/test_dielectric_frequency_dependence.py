import numpy as np

from rt_core.polarization import fresnel_coefficients


def test_dielectric_fresnel_is_frequency_dependent():
    f = np.array([3e9, 9e9], dtype=float)
    theta = np.deg2rad(45.0)
    gs, gp = fresnel_coefficients(f, theta, kind="dielectric", eps_r=6.5, tan_delta=0.02)

    assert not np.isclose(gs[0], gs[1])
    assert not np.isclose(gp[0], gp[1])
