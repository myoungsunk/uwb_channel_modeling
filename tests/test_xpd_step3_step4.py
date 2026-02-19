import numpy as np

from analysis.xpd_stats import conditional_stats, parity_labels, pathwise_xpd_summary


def test_a1_like_cross_near_zero_high_xpd():
    a = np.zeros((3, 16, 2, 2), dtype=np.complex128)
    a[:, :, 0, 0] = 1.0
    a[:, :, 1, 1] = 1.0
    x = pathwise_xpd_summary(a)["xpd_db_avg"]
    assert np.all(x > 100)


def test_a2_a3_like_parity_separation():
    # even paths high XPD, odd paths lower XPD
    a = np.zeros((6, 8, 2, 2), dtype=np.complex128)
    bounce = np.array([0, 2, 2, 1, 1, 3])
    for i in range(6):
        if bounce[i] % 2 == 0:
            a[i, :, 0, 0] = 1.0
            a[i, :, 1, 1] = 1.0
            a[i, :, 0, 1] = 1e-4
            a[i, :, 1, 0] = 1e-4
        else:
            a[i, :, 0, 0] = 1.0
            a[i, :, 1, 1] = 1.0
            a[i, :, 0, 1] = 0.2
            a[i, :, 1, 0] = 0.2
    x = pathwise_xpd_summary(a)["xpd_db_avg"]
    p = parity_labels(bounce)
    stats = conditional_stats(x, parity=p)
    mu_even = [v["mu"] for k, v in stats.items() if "parity=0" in k][0]
    mu_odd = [v["mu"] for k, v in stats.items() if "parity=1" in k][0]
    assert mu_even > mu_odd
