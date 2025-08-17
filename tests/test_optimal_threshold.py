import numpy as np
import pytest

from optimal_cutoffs import get_optimal_threshold


y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
y_true = np.array([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("method", ["brute", "minimize", "gradient"])
def test_perfect_separation_threshold(method):
    thr = get_optimal_threshold(y_true=y_true, y_prob=y_prob, method=method)
    assert thr == pytest.approx(0.5, abs=0.2)
