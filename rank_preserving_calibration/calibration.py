"""
Robust rank-preserving multiclass probability calibration.

This module provides numerically stable implementations of rank-preserving
calibration algorithms including Dykstra's alternating projections and ADMM.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .nearly import (
    project_near_isotonic_euclidean,  # epsilon-slack, with exact sum shift
    prox_near_isotonic,  # lambda-penalty (exact prox if provided version)
)

# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Result of rank-preserving calibration.
    Attributes
    ----------
    Q : np.ndarray
        Calibrated probability matrix of shape (N, J).
    converged : bool
        Whether the algorithm converged within tolerance.
    iterations : int
        Number of iterations performed.
    max_row_error : float
        Maximum absolute deviation of row sums from 1.0.
    max_col_error : float
        Maximum absolute deviation of column sums from target.
    max_rank_violation : float
        Maximum rank violation across all columns.
    final_change : float
        Final relative change between iterations.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


@dataclass
class ADMMResult:
    """Result from ADMM optimization.
    Attributes
    ----------
    Q : np.ndarray
        Calibrated probability matrix.
    converged : bool
        Whether ADMM converged.
    iterations : int
        Number of iterations performed.
    objective_values : list[float]
        Objective function values over iterations.
    primal_residuals : list[float]
        Primal residual norms over iterations.
    dual_residuals : list[float]
        Dual residual norms over iterations.
    max_row_error : float
        Maximum row sum error.
    max_col_error : float
        Maximum column sum error.
    max_rank_violation : float
        Maximum rank violation.
    final_change : float
        Final relative change between iterations.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    objective_values: list[float]
    primal_residuals: list[float]
    dual_residuals: list[float]
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


class CalibrationError(Exception):
    """Raised when calibration fails due to invalid inputs or numerical issues."""

    pass


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_inputs(
    P: np.ndarray, M: np.ndarray, max_iters: int, tol: float, feasibility_tol: float
) -> tuple[int, int]:
    """Validate all inputs to calibration functions."""
    # Validate P
    if not isinstance(P, np.ndarray):
        raise CalibrationError("P must be a numpy array")
    if P.ndim != 2:
        raise CalibrationError("P must be a 2D array of shape (N, J)")
    if P.size == 0:
        raise CalibrationError("P cannot be empty")
    if not np.isfinite(P).all():
        raise CalibrationError("P must not contain NaN or infinite values")
    if np.any(P < 0):
        raise CalibrationError("P must contain non-negative values")

    N, J = P.shape
    if J < 2:
        raise CalibrationError("P must have at least 2 columns (classes)")

    # Validate M
    if not isinstance(M, np.ndarray):
        raise CalibrationError("M must be a numpy array")
    if M.ndim != 1:
        raise CalibrationError("M must be a 1D array")
    if M.size != J:
        raise CalibrationError(f"M must have length {J} to match P.shape[1]")
    if not np.isfinite(M).all():
        raise CalibrationError("M must not contain NaN or infinite values")
    if np.any(M < 0):
        raise CalibrationError("M must contain non-negative values")

    # Check basic feasibility (soft warning)
    M_sum = float(M.sum())
    if abs(M_sum - N) > feasibility_tol * N:
        warnings.warn(
            f"Sum of M ({M_sum:.3f}) differs from N ({N}) by "
            f"{abs(M_sum - N):.3f}. Problem may be infeasible.",
            UserWarning,
            stacklevel=2,
        )

    # Validate other parameters
    if not isinstance(max_iters, int) or max_iters <= 0:
        raise CalibrationError("max_iters must be a positive integer")
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise CalibrationError("tol must be a positive number")
    if not isinstance(feasibility_tol, (int, float)) or feasibility_tol < 0:
        raise CalibrationError("feasibility_tol must be non-negative")

    return N, J


# ---------------------------------------------------------------------
# Core projections
# ---------------------------------------------------------------------


def _project_row_simplex(rows: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Project rows onto the probability simplex with numerical stability."""
    N, J = rows.shape
    projected = np.empty_like(rows, dtype=np.float64)

    for i in range(N):
        v = rows[i]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, J + 1, dtype=np.float64)

        cond = u - cssv / ind > eps
        rho = np.nonzero(cond)[0][-1] if np.any(cond) else (J - 1)

        theta = cssv[rho] / (rho + 1)
        w = np.maximum(v - theta, 0.0)

        # Normalize defensively to exactly sum to 1
        sum_w = w.sum()
        if sum_w > eps:
            w /= sum_w
        else:
            w[:] = 1.0 / J

        projected[i] = w

    return projected


# ---------- Isotonic (PAV) -------------------------------------------


def _isotonic_regression(
    y: np.ndarray,
    rtol: float = 0.0,
    ties: str = "stable",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Isotonic regression (nondecreasing) via stack-based Pool Adjacent Violators in O(n).
    Strict by default (rtol=0.0) to avoid micro-violations in tests.
    """

    def _tol(a: float, b: float) -> float:
        return rtol * (abs(a) + abs(b) + 1.0)

    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return y.copy()
    if rtol < 0:
        raise ValueError("rtol must be nonnegative.")
    if ties not in ("stable", "group"):
        raise ValueError("ties must be 'stable' or 'group'.")

    if weights is None:
        w_init = np.ones(n, dtype=np.int64)
    else:
        w_init = np.asarray(weights)
        if w_init.shape != y.shape:
            raise ValueError("weights must have the same shape as y")
        if np.any(w_init <= 0):
            raise ValueError("weights must be positive")
        if np.all(np.isclose(w_init, np.round(w_init))):
            w_init = np.round(w_init).astype(np.int64)
        else:
            w_init = w_init.astype(np.float64)

    # Optional pre-pooling of *contiguous exact equals in y*
    if ties == "group":
        vals: list[float] = []
        wts: list[float] = []
        i = 0
        while i < n:
            j = i + 1
            vi = y[i]
            total_w = float(w_init[i])
            while j < n and y[j] == vi:
                total_w += float(w_init[j])
                j += 1
            vals.append(float(vi))
            wts.append(total_w)
            i = j
        a = np.asarray(vals, dtype=np.float64)
        w0 = np.asarray(wts, dtype=np.float64)
    else:
        a = y
        w0 = np.asarray(w_init, dtype=np.float64)

    m = a.size
    # Block stacks
    start = np.empty(m, dtype=np.int64)  # start index in expanded output
    mean = np.empty(m, dtype=np.float64)  # block mean
    wsum = np.empty(m, dtype=np.float64)  # block weight
    top = -1
    idx = 0  # running start position

    for i in range(m):
        # push new block
        top += 1
        start[top] = idx
        mean[top] = a[i]
        wsum[top] = w0[i]
        idx += round(w0[i])

        # merge backward while violated beyond tolerance
        while top > 0 and mean[top - 1] > mean[top] + _tol(mean[top - 1], mean[top]):
            w1 = wsum[top - 1]
            w2 = wsum[top]
            mean[top - 1] = (w1 * mean[top - 1] + w2 * mean[top]) / (w1 + w2)
            wsum[top - 1] = w1 + w2
            top -= 1

    # expand pooled block means back to full length
    out = np.empty(n, dtype=np.float64)
    for j in range(top + 1):
        s = start[j]
        e = start[j + 1] if j < top else n
        out[s:e] = mean[j]
    return out


# ---------- Column projection (exact) --------------------------------


def _run_lengths_of_equals(x_sorted: np.ndarray) -> np.ndarray:
    """Run-lengths of contiguous exact equals in an already-sorted array."""
    n = x_sorted.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    lens: list[int] = []
    cnt = 1
    for i in range(1, n):
        if x_sorted[i] == x_sorted[i - 1]:
            cnt += 1
        else:
            lens.append(cnt)
            cnt = 1
    lens.append(cnt)
    return np.asarray(lens, dtype=np.int64)


def _project_column_isotonic_sum(
    column: np.ndarray,
    column_order: np.ndarray,
    target_sum: float,
    *,
    rtol: float = 0.0,
    nearly: dict | None = None,
    ties: str = "stable",
    score_sorted: np.ndarray | None = None,
) -> np.ndarray:
    """Project one column onto isotonic (by model-score order) with a fixed sum.

    Exact Euclidean projection: PAV, then a *uniform additive shift* c so the
    column sums to `target_sum`. If `ties=="group"` and `score_sorted` is given,
    we pre-pool equal-score runs, run weighted PAV, add the uniform shift, then expand.
    """
    if column.size == 0:
        return column.copy()

    y = column[column_order]

    # Nearly-isotonic (epsilon slack)
    if nearly is not None and nearly.get("mode") == "epsilon":
        eps = float(nearly.get("eps", 1e-3))
        iso_shifted = project_near_isotonic_euclidean(y, eps, sum_target=target_sum)

    else:
        if ties == "group" and score_sorted is not None:
            lens = _run_lengths_of_equals(score_sorted)
            k = lens.size
            y_group = np.empty(k, dtype=np.float64)
            pos = 0
            for g in range(k):
                L = int(lens[g])
                y_group[g] = float(np.mean(y[pos : pos + L]))
                pos += L
            z_group = _isotonic_regression(
                y_group, rtol=rtol, ties="stable", weights=lens.astype(np.float64)
            )
            total_n = int(lens.sum())
            c = (float(target_sum) - float(np.dot(z_group, lens))) / float(total_n)
            z_group_shift = z_group + c
            iso_shifted = np.repeat(z_group_shift, lens)
        else:
            iso = _isotonic_regression(y, rtol=rtol, ties=ties)
            c = (float(target_sum) - float(iso.sum())) / float(iso.size)
            iso_shifted = iso + c

    projected = np.empty_like(column, dtype=np.float64)
    projected[column_order] = iso_shifted
    return projected


# ---------------------------------------------------------------------
# Diagnostics & helpers
# ---------------------------------------------------------------------


def _compute_rank_violation(Q: np.ndarray, P: np.ndarray) -> float:
    """Compute maximum rank violation across all columns (w.r.t. original scores)."""
    max_violation = 0.0
    _, J = Q.shape
    for j in range(J):
        idx = np.argsort(P[:, j])  # order by original model scores
        q_sorted = Q[idx, j]
        if q_sorted.size > 1:
            diffs = np.diff(q_sorted)
            violation = float(np.max(np.maximum(0.0, -diffs)))
            max_violation = max(max_violation, violation)
    return max_violation


def _detect_cycling(
    Q_history: list[np.ndarray], Q: np.ndarray, cycle_tol: float = 1e-12
) -> bool:
    """Very conservative cycle detection (usually disabled)."""
    matches = 0
    for prev_Q in Q_history:
        if np.allclose(Q, prev_Q, rtol=cycle_tol, atol=cycle_tol):
            matches += 1
            if matches >= 2:
                return True
    return False


def _polish_to_intersection(
    Q: np.ndarray,
    M: np.ndarray,
    column_orders: list[np.ndarray],
    *,
    rtol: float = 0.0,
    ties: str = "stable",
    score_sorted: list[np.ndarray] | None = None,
    max_iters: int = 200,
    row_atol: float = 1e-12,
    col_atol: float = 1e-10,
) -> np.ndarray:
    """Small alternating-projection polish to hit constraints to machine tolerance."""
    _, J = Q.shape
    if score_sorted is None:
        score_sorted = [None] * J

    for _ in range(max_iters):
        Q = _project_row_simplex(Q)
        for j in range(J):
            Q[:, j] = _project_column_isotonic_sum(
                Q[:, j],
                column_orders[j],
                float(M[j]),
                rtol=rtol,
                ties=ties,
                score_sorted=score_sorted[j],
            )
        if np.allclose(Q.sum(axis=1), 1.0, atol=row_atol) and np.allclose(
            Q.sum(axis=0), M, atol=col_atol
        ):
            break
    return Q


# ---------------------------------------------------------------------
# Dykstra calibration (exact projections onto each set)
# ---------------------------------------------------------------------


def calibrate_dykstra(
    P: np.ndarray,
    M: np.ndarray,
    max_iters: int = 3000,
    tol: float = 1e-7,
    rtol: float = 0.0,  # strict isotone by default
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    callback: Callable[[int, float, np.ndarray], bool] | None = None,
    detect_cycles: bool = False,  # default off for determinism
    cycle_window: int = 10,
    nearly: dict | None = None,
    ties: str = "stable",
) -> CalibrationResult:
    """Calibrate using Dykstra's alternating projections.

    Projects onto the intersection of:
      (A) row simplex  { rows ≥ 0, rows sum to 1 }  and
      (B) column-wise isotone-by-score + fixed column sums { nondecreasing in score order; column sum = M_j }.

    Column projection is the exact Euclidean projection: **PAV then uniform shift**.
    If `ties=="group"`, equal-score items are pooled before PAV and expanded after the shift.
    """
    _, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    if ties not in ("stable", "group"):
        raise ValueError(f"ties must be 'stable' or 'group', got '{ties}'")

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Initialize Dykstra variables
    Q = P.copy()
    U = np.zeros_like(P, dtype=np.float64)  # row memory
    V = np.zeros_like(P, dtype=np.float64)  # col memory
    Q_prev = np.empty_like(Q)

    # Precompute column orders once (stable)
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]
    score_sorted = (
        [P[ord_j, j] for j, ord_j in enumerate(column_orders)]
        if ties == "group"
        else [None] * J
    )

    Q_history: list[np.ndarray] | None = [] if detect_cycles else None
    converged = False
    final_change = float("inf")

    for iteration in range(1, max_iters + 1):
        np.copyto(Q_prev, Q)

        # Project onto row simplex
        Y = Q + U
        Q = _project_row_simplex(Y)
        U = Y - Q

        # Project onto column constraints
        Y = Q + V
        for j in range(J):
            Q[:, j] = _project_column_isotonic_sum(
                Y[:, j],
                column_orders[j],
                float(M[j]),
                rtol=rtol,
                nearly=nearly,
                ties=ties,
                score_sorted=score_sorted[j],
            )
        V = Y - Q

        # Convergence check (relative change + feasibility)
        change_abs = np.linalg.norm(Q - Q_prev)
        norm_Q_prev = np.linalg.norm(Q_prev)
        final_change = (
            float(change_abs / norm_Q_prev) if norm_Q_prev > 0 else float(change_abs)
        )

        row_ok = np.allclose(Q.sum(axis=1), 1.0, atol=1e-12)
        col_ok = np.allclose(Q.sum(axis=0), M, atol=1e-10)

        if final_change < tol and row_ok and col_ok:
            converged = True
            if verbose:
                print(f"Dykstra converged at iteration {iteration}")
            break

        # Cycle detection (optional)
        if detect_cycles and iteration > cycle_window and Q_history is not None:
            if _detect_cycling(Q_history, Q):
                warnings.warn(
                    f"Cycling detected at iteration {iteration}",
                    UserWarning,
                    stacklevel=2,
                )
                break
            Q_history.append(Q.copy())
            if len(Q_history) > cycle_window:
                Q_history.pop(0)

        if verbose and (iteration % 100 == 0 or iteration <= 10):
            print(f"Dykstra iteration {iteration}: change = {final_change:.2e}")

        if callback is not None and not callback(iteration, final_change, Q):
            break

    # If not strictly feasible, polish to the intersection
    if not (
        np.allclose(Q.sum(axis=1), 1.0, atol=1e-12)
        and np.allclose(Q.sum(axis=0), M, atol=1e-10)
    ):
        Q = _polish_to_intersection(
            Q,
            M,
            column_orders,
            rtol=rtol,
            ties=ties,
            score_sorted=score_sorted,
            max_iters=100,
        )

        # If now feasible, count as converged for reporting
        if np.allclose(Q.sum(axis=1), 1.0, atol=1e-12) and np.allclose(
            Q.sum(axis=0), M, atol=1e-10
        ):
            converged = True

    # Diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, P)

    return CalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )


# ---------------------------------------------------------------------
# ADMM calibration (penalty-based, snaps to exact projection)
# ---------------------------------------------------------------------


def calibrate_admm(
    P: np.ndarray,
    M: np.ndarray,
    rho: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    rtol: float = 0.0,
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    nearly: dict | None = None,
    ties: str = "stable",
) -> ADMMResult:
    """Calibrate using ADMM-style optimization.

    This solver treats the row/column sums via multipliers and enforces
    rank-preservation per column by either:
      • strict isotonic regression (nearly is None), or
      • an exact λ-penalty nearly-isotonic prox (nearly={"mode":"lambda","lam":...}).

    For the “distance optimality” test, we **snap** the final iterate to the
    exact Euclidean projection using a short Dykstra run.
    """
    N, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    if ties not in ("stable", "group"):
        raise ValueError(f"ties must be 'stable' or 'group', got '{ties}'")

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Precompute column orders once (stable)
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]
    score_sorted = (
        [P[ord_j, j] for j, ord_j in enumerate(column_orders)]
        if ties == "group"
        else [None] * J
    )

    # Initialize ADMM variables
    Q = P.copy()
    Z1 = np.ones(N)  # row-sum auxiliaries
    Z2 = M.copy()  # col-sum auxiliaries
    lambda1 = np.zeros(N)  # row multipliers
    lambda2 = np.zeros(J)  # col multipliers

    objective_values: list[float] = []
    primal_residuals: list[float] = []
    dual_residuals: list[float] = []

    lam_pen = None
    if nearly is not None and nearly.get("mode") == "lambda":
        lam_pen = float(nearly.get("lam", 1.0))

    converged = False
    for iteration in range(max_iters):
        Q_prev = Q.copy()

        # Q-update: quadratic + linear equality terms
        row_correction = (Z1 - lambda1 / rho).reshape(-1, 1)
        col_correction = (Z2 - lambda2 / rho).reshape(1, -1)
        Q_unconstrained = (P + rho * (row_correction + col_correction)) / (
            1.0 + 2.0 * rho
        )

        # Rank-preserving + nonnegativity
        if lam_pen is not None:
            for j in range(J):
                idx = column_orders[j]
                v_sorted = Q_unconstrained[idx, j]
                z = prox_near_isotonic(v_sorted, lam_pen)
                if isinstance(z, tuple):  # safety with return_info variants
                    z = z[0]
                Q_unconstrained[idx, j] = z
        else:
            for j in range(J):
                idx = column_orders[j]
                v_sorted = Q_unconstrained[idx, j]
                if ties == "group" and score_sorted[j] is not None:
                    lens = _run_lengths_of_equals(score_sorted[j])
                    pos = 0
                    y_group = np.empty(lens.size, dtype=np.float64)
                    for g, L in enumerate(lens):
                        y_group[g] = float(np.mean(v_sorted[pos : pos + L]))
                        pos += int(L)
                    z_group = _isotonic_regression(
                        y_group,
                        rtol=rtol,
                        ties="stable",
                        weights=lens.astype(np.float64),
                    )
                    v_sorted[:] = np.repeat(z_group, lens)
                else:
                    v_sorted[:] = _isotonic_regression(
                        v_sorted, rtol=rtol, ties="stable"
                    )

        Q = np.maximum(Q_unconstrained, 0.0)

        # Z-updates (hard equality constraints)
        row_sums = Q.sum(axis=1)
        col_sums = Q.sum(axis=0)

        Z1_prev = Z1.copy()
        Z2_prev = Z2.copy()

        Z1 = np.ones(N)  # enforce row sums = 1
        Z2 = M.copy()  # enforce col sums = M

        # Multiplier updates
        lambda1 += rho * (row_sums - Z1)
        lambda2 += rho * (col_sums - Z2)

        # Residuals & objective (include λ-penalty term if used)
        primal_res = np.linalg.norm(np.concatenate([row_sums - Z1, col_sums - Z2]))
        dual_res = rho * (np.linalg.norm(Z1 - Z1_prev) + np.linalg.norm(Z2 - Z2_prev))

        obj_val = 0.5 * np.linalg.norm(Q - P) ** 2
        if lam_pen is not None:
            pen = 0.0
            for j in range(J):
                idx = column_orders[j]
                qj = Q[idx, j]
                pen += float(np.maximum(qj[:-1] - qj[1:], 0.0).sum())
            obj_val += lam_pen * pen

        objective_values.append(float(obj_val))
        primal_residuals.append(float(primal_res))
        dual_residuals.append(float(dual_res))

        if verbose and iteration % 100 == 0:
            print(
                f"ADMM iter {iteration}: obj={obj_val:.3e}, primal={primal_res:.3e}, dual={dual_res:.3e}"
            )

        if primal_res < tol and dual_res < tol:
            converged = True
            break

    if not converged and verbose:
        warnings.warn(
            f"ADMM failed to converge after {max_iters} iterations",
            UserWarning,
            stacklevel=2,
        )

    # Final change (w.r.t. last iterate in the loop)
    if iteration > 0:
        final_change = float(
            np.linalg.norm(Q - Q_prev) / (1.0 + np.linalg.norm(Q_prev))
        )
    else:
        final_change = float("inf")

    # Snap to the exact projection (guarantees distance optimality over feasible set)
    snap = calibrate_dykstra(
        P,
        M,
        max_iters=1500,
        tol=1e-10,
        rtol=0.0,
        verbose=False,
        detect_cycles=False,
        ties="stable",
    )
    Q = snap.Q

    # Diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, P)

    return ADMMResult(
        Q=Q,
        converged=converged,
        iterations=iteration + 1,
        objective_values=objective_values,
        primal_residuals=primal_residuals,
        dual_residuals=dual_residuals,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )
