"""Tests for entropy computation in analyze_calibration_result."""

from types import SimpleNamespace

import numpy as np

from examples.data_helpers import analyze_calibration_result


def test_entropy_regression():
    """Verify entropy calculations on a known small matrix."""
    P = np.array([[0.6, 0.4], [0.5, 0.5]])
    Q = np.array([[0.7, 0.3], [0.4, 0.6]])
    M = Q.sum(axis=0)

    result = SimpleNamespace(
        Q=Q, converged=True, iterations=0, final_change=0.0, max_rank_violation=0.0
    )

    analysis = analyze_calibration_result(P, result, M)

    expected_P_entropy = np.mean(-np.sum(P * np.log(P + 1e-10), axis=1))
    expected_Q_entropy = np.mean(-np.sum(Q * np.log(Q + 1e-10), axis=1))

    assert np.isclose(
        analysis["distribution_impact"]["original_entropy"], expected_P_entropy
    )
    assert np.isclose(
        analysis["distribution_impact"]["calibrated_entropy"], expected_Q_entropy
    )

