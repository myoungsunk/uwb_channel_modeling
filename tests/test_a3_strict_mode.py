import numpy as np

from scenarios import A3_dihedral


def test_a3_strict_only_keeps_two_bounce_paths():
    params = {
        "case_id": "a3_even_strict_test",
        "gap": 2.0,
        "max_bounce": 2,
        "los_blocked": True,
        "only_bounce": 2,
        "strict": True,
    }
    _, paths = A3_dihedral.run_case(params)
    assert len(paths) > 0
    bounce = np.array([int(p.meta.get("bounce_count", -1)) for p in paths], dtype=int)
    assert np.all(bounce == 2)
