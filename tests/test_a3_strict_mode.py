import numpy as np

from scenarios import A3_dihedral
from scenarios.common import default_antennas, make_freq
from rt_core.tracer import trace_paths


def test_a3_strict_only_keeps_two_bounce_paths():
    tx, rx = default_antennas()
    f = make_freq()
    all_paths = trace_paths(A3_dihedral.build_scene(2.0), tx, rx, f, max_bounce=2, los_blocked=True)
    # strict mode should work as a filter over an existing path set
    assert len(all_paths) > 0

    params = {
        "case_id": "a3_even_strict_test",
        "gap": 2.0,
        "max_bounce": 2,
        "los_blocked": True,
        "only_bounce": 2,
        "strict": True,
    }
    _, paths = A3_dihedral.run_case(params)
    bounce = np.array([int(p.meta.get("bounce_count", -1)) for p in paths], dtype=int)
    assert np.all(bounce == 2)
