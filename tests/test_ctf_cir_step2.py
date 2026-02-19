import numpy as np

from analysis.ctf_cir import SynthesisConfig, convert_basis_2x2, synthesize_case


def test_ctf_synthesis_shapes():
    f = np.linspace(3e9, 4e9, 16)
    tau = np.array([0.0, 5e-9])
    a = np.ones((2, 16, 2, 2), dtype=np.complex128)
    out = synthesize_case(f, tau, a, SynthesisConfig(nfft=32, window="hann"))
    assert out["H_f"].shape == (16, 2, 2)
    assert out["h_tau"].shape == (32, 2, 2)


def test_linear_circular_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((8, 2, 2)) + 1j * rng.standard_normal((8, 2, 2))
    c = convert_basis_2x2(m, "linear", "circular")
    b = convert_basis_2x2(c, "circular", "linear")
    assert np.allclose(m, b)
