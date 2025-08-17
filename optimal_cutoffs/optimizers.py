"""Threshold optimization utilities."""
from __future__ import annotations

import numpy as np


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score for binary predictions."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def get_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "brute",
) -> float:
    """Return threshold maximizing F1 score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    method : {"brute", "minimize", "gradient"}, default="brute"
        Optimization strategy. "minimize" and "gradient" rely on
        :func:`scipy.optimize.minimize` if SciPy is available. In its absence,
        they fall back to the "brute" strategy.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if method == "brute":
        thresholds = np.r_[0.0, np.sort(y_prob), 1.0]
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thresholds:
            pred = y_prob >= thr
            f1 = _f1_score(y_true, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        return float(np.clip(best_thr, 0.0, 1.0))

    if method in {"minimize", "gradient"}:
        try:
            from scipy.optimize import minimize
        except Exception:
            # Fall back to brute force search when SciPy is unavailable
            return get_optimal_threshold(y_true, y_prob, method="brute")

        def objective(x: np.ndarray) -> float:
            thr = float(np.clip(x[0], 0.0, 1.0))
            pred = y_prob >= thr
            f1 = _f1_score(y_true, pred)
            return -f1  # negative so that optimizer maximizes F1

        x0 = np.array([0.5])
        bounds = [(0.0, 1.0)]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        thr = float(res.x[0])
        return float(np.clip(thr, 0.0, 1.0))

    raise ValueError(f"Unknown method: {method}")
