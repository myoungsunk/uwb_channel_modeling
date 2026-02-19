import os
import tempfile

import numpy as np

from rt_core.tracer import Path
from rt_io.hdf5_io import CaseData, load_rt_hdf5, save_rt_hdf5, self_test_roundtrip


def _dummy_path(seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    return Path(
        tau_s=10e-9,
        a_f=(rng.standard_normal((16, 2, 2)) + 1j * rng.standard_normal((16, 2, 2))).astype(np.complex128),
        meta={
            "bounce_count": 2,
            "interactions": ["reflection", "reflection"],
            "surface_ids": [1, 3],
            "incidence_angles": [0.3, 0.7],
            "AoD": [1.0, 0.0, 0.0],
            "AoA": [-1.0, 0.0, 0.0],
        },
    )


def test_hdf5_schema_roundtrip():
    f = np.linspace(3e9, 10e9, 16)
    payload = {"A1": {"case_0": CaseData(params={"max_bounce": 1}, paths=[_dummy_path(2)])}}
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "rt.h5")
        save_rt_hdf5(fp, f, payload, basis="linear", convention="IEEE RHCP")
        f2, loaded, meta = load_rt_hdf5(fp)

    assert np.allclose(f, f2)
    assert meta.basis == "linear"
    assert meta.convention == "IEEE RHCP"
    p = loaded["A1"]["case_0"].paths[0]
    assert p.meta["surface_ids"] == [1, 3]
    assert p.a_f.shape == (16, 2, 2)


def test_self_test_roundtrip_function():
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "selftest.h5")
        assert self_test_roundtrip(fp)
