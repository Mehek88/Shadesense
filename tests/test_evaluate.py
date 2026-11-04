import numpy as np

from shadesense.evaluate import bootstrap_ci, mcnemar, mst_group


def test_mcnemar_and_ci():
    a = np.array([1, 1, 1, 0], bool); b = np.array([1, 0, 0, 0], bool)
    r = mcnemar(a, b)
    assert r["only_a"] == 2 and r["only_b"] == 0 and 0 < r["p_value"] <= 1
    lo, hi = bootstrap_ci(np.array([1.0, 0.0, 1.0, 1.0]), 200)
    assert 0 <= lo <= 0.75 <= hi <= 1


def test_mst_group():
    g = {"light": [1, 3], "medium": [4, 7], "deep": [8, 10]}
    assert [mst_group(m, g) for m in (1, 5, 10)] == ["light", "medium", "deep"]
