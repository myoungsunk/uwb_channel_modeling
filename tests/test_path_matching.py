import numpy as np

from analysis.path_matching import MatchConfig, combine_runs_to_2x2
from rt_core.tracer import Path


def _p(tau: float, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    vec = (rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))).astype(np.complex128)
    return Path(
        tau_s=tau,
        a_f=vec,
        meta={
            "bounce_count": 1,
            "surface_ids": [4],
            "incidence_angles": [0.5],
            "AoD": [1, 0, 0],
            "AoA": [-1, 0, 0],
        },
    )


def test_match_and_combine():
    run_h = [_p(11e-9, 1)]
    run_v = [_p(11.05e-9, 2)]
    out, warns = combine_runs_to_2x2(run_h, run_v, MatchConfig(tau_tolerance_s=0.1e-9, allow_nearest=True))
    assert len(warns) == 0
    assert len(out) == 1
    assert out[0].a_f.shape == (8, 2, 2)


def test_unmatched_warning():
    run_h = [_p(11e-9, 1)]
    run_v = [_p(15e-9, 2)]
    out, warns = combine_runs_to_2x2(run_h, run_v, MatchConfig(tau_tolerance_s=0.1e-9, allow_nearest=True))
    assert len(out) == 0
    assert any("no match" in w for w in warns)
