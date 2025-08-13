# SPDX-License-Identifier: MIT
"""
Core calibration routines for rank-preserving multiclass probability adjustment.

This module provides an implementation of Dykstra's alternating projection
algorithm to project an array of multiclass scores onto the intersection of
two convex sets:

1. The **row-simplex** set (each row sums to one and all entries are
   non‑negative).  This ensures the returned matrix represents a valid
   multinomial probability distribution for each instance.

2. For each class (column), the vector of scores is **non‑decreasing** when
   sorted by the original scores for that class, and the sum of the column
   matches a supplied target marginal.  These constraints preserve the
   discrimination ordering across individuals within a class and enforce
   calibration to known population totals.

The function :func:`admm_rank_preserving_simplex_marginals` exposes the
primary API.  It accepts an array of base probabilities ``P`` and a vector of
column sums ``M`` and iteratively projects between the two constraint sets
until convergence.

The implementation uses plain NumPy and does not rely on any external
optimization libraries.  All projections are in Euclidean (L2) geometry.

"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional


def _project_row_simplex(rows: np.ndarray) -> np.ndarray:
    """Project each row of ``rows`` onto the probability simplex.

    Given a 2‑D array of shape ``(N, J)``, this function projects each row
    independently onto the standard (J‑1)‑simplex, i.e., the set of vectors
    with non‑negative entries that sum to one.  The projection is performed
    in Euclidean space and uses the algorithm described in
    *Efficient Projections onto the l1-Ball for Learning in High Dimensions*
    by John Duchi, Shai Shalev‑Shwartz, Yoram Singer, and Tushar Chandra.

    Parameters
    ----------
    rows : np.ndarray
        Array of shape ``(N, J)`` containing real values.

    Returns
    -------
    np.ndarray
        A new array of shape ``(N, J)`` where each row has been projected
        onto the simplex.  The returned array shares no memory with the
        input.
    """
    N, J = rows.shape
    projected = np.empty_like(rows, dtype=np.float64)
    for i in range(N):
        v = rows[i]
        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, J + 1)
        # Find rho: the largest index where u_j - cssv_j / j > 0
        # We use boolean mask and argmax of truth values reversed to get last positive.
        cond = u - cssv / ind > 0
        if not np.any(cond):
            rho = J - 1
        else:
            rho = np.nonzero(cond)[0][-1]
        theta = cssv[rho] / (rho + 1)
        w = v - theta
        w[w < 0] = 0.0
        # Renormalise small numerical error
        sum_w = w.sum()
        if sum_w != 0:
            w /= sum_w
        else:
            # If projection results in all zeros, assign uniform distribution
            w[:] = 1.0 / J
        projected[i] = w
    return projected


def _isotonic_regression(y: np.ndarray) -> np.ndarray:
    """Perform isotonic regression on a 1‑D array with respect to order.

    This function solves the problem::

        minimize   0.5 * ||z - y||_2^2
        subject to z[i] <= z[i+1] for all i

    using the pool adjacent violators algorithm (PAV).  The algorithm
    enforces a non‑decreasing sequence.  We do not consider weights here
    (unit weights are assumed).

    Parameters
    ----------
    y : np.ndarray
        1‑D array of floats.

    Returns
    -------
    np.ndarray
        1‑D array of the same length as ``y`` containing the isotonic
        regression of ``y``.
    """
    y = y.astype(float).copy()
    n = y.size
    # Each data point starts in its own block
    # We maintain arrays of block averages (z) and block sizes (w)
    z = y.copy()
    w = np.ones(n, dtype=float)
    i = 0
    while i < n - 1:
        if z[i] <= z[i + 1] + 1e-12:
            i += 1
        else:
            # Pool blocks i and i+1
            new_w = w[i] + w[i + 1]
            new_z = (z[i] * w[i] + z[i + 1] * w[i + 1]) / new_w
            z[i] = new_z
            w[i] = new_w
            # Delete block i+1
            z = np.delete(z, i + 1)
            w = np.delete(w, i + 1)
            n -= 1
            # Move leftwards if possible
            if i > 0:
                i -= 1
    # Expand pooled blocks back to original length
    # Each pooled block average in z and weight in w corresponds to multiple
    # original positions.  We repeat the average value accordingly.
    expanded = np.repeat(z, w.astype(int))
    return expanded


def _project_column_isotonic_sum(
    column: np.ndarray,
    P_column: np.ndarray,
    target_sum: float,
    ) -> np.ndarray:
    """Project a column onto isotonic sequences with a fixed sum.

    The input ``column`` is a 1‑D array of length ``N``.  The original
    ordering is determined by the values of ``P_column``: we require the
    output to be non‑decreasing when reordered according to the ascending
    order of ``P_column``.  After enforcing monotonicity using isotonic
    regression, a uniform shift is applied so that the resulting values
    sum to ``target_sum``.

    Parameters
    ----------
    column : np.ndarray
        Current column vector (length N) to project.
    P_column : np.ndarray
        Original scores for this class (length N).  Determines the order in
        which the monotonicity constraint is applied.
    target_sum : float
        Desired sum of the projected column.

    Returns
    -------
    np.ndarray
        Column vector of length N satisfying isotonic and sum constraints.
    """
    # Determine sorting index based on original P_column (ascending order)
    idx = np.argsort(P_column)
    y = column[idx]
    # Apply isotonic regression to enforce non‑decreasing order
    iso = _isotonic_regression(y)
    # Shift to match the target sum; adding a constant preserves order
    current_sum = iso.sum()
    n = iso.size
    shift = (target_sum - current_sum) / float(n)
    iso_shifted = iso + shift
    # Reassign in original order
    projected = np.empty_like(column, dtype=np.float64)
    projected[idx] = iso_shifted
    return projected


def admm_rank_preserving_simplex_marginals(
    P: np.ndarray,
    M: np.ndarray,
    geometry: str = 'euclidean',
    max_iters: int = 3000,
    tol: float = 1e-7,
    verbose: bool = False,
    return_info: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calibrate multiclass probabilities while preserving within-class ranks.

    This function takes an array of base probabilities ``P`` (shape ``(N, J)``)
    and a vector of desired column sums ``M`` (length ``J``) and returns an
    adjusted array ``Q`` with the same shape.  The algorithm projects
    repeatedly onto two convex sets using Dykstra's alternating projection:

    - ``C1``: matrices whose rows lie on the probability simplex (each row
      sums to one and all entries are non‑negative).
    - ``C2``: matrices whose columns are non‑decreasing with respect to the
      ordering induced by the original ``P`` and whose column sums equal
      ``M``.

    The intersection of these sets, when non‑empty, contains matrices that
    respect the row constraints (probability distributions), match the
    specified marginals, and preserve discrimination across individuals
    within each class.  If the intersection is empty or the targets
    incompatible, the algorithm converges to a point that minimises the
    distance to the intersection.

    Parameters
    ----------
    P : np.ndarray
        Base probability matrix of shape ``(N, J)``.  Each row should sum to
        one; however, row sums will be enforced during iteration.
    M : np.ndarray
        Desired column sums of length ``J``.  Typically these are derived
        from population totals; ``np.sum(M)`` should equal the number of
        rows ``N`` to be perfectly feasible, but slight discrepancies can
        be handled by the algorithm.
    max_iters : int, default 3000
        Maximum number of projection iterations.  Convergence is typically
        much faster; increase if necessary for larger problems.
    tol : float, default 1e-7
        Relative convergence tolerance based on the Frobenius norm of
        successive iterates.
    verbose : bool, default False
        If True, prints progress diagnostics at each iteration.
    return_info : bool, default True
        Whether to return a dictionary with diagnostic information.

    Returns
    -------
    Q : np.ndarray
        The calibrated probability matrix of shape ``(N, J)``.  Each row
        sums to one, each column approximately sums to ``M``, and within
        each column the values are non‑decreasing in the order induced by
        ``P``.
    info : dict
        Dictionary containing diagnostics:

        - ``iterations``: number of iterations performed.
        - ``max_row_error``: maximum absolute deviation of row sums from 1.
        - ``max_col_error``: maximum absolute deviation of column sums from ``M``.
        - ``max_rank_violation``: maximum violation of monotonicity across all
          columns (values > tolerance indicate rank inversions).  If
          ``tol`` is zero this counts exact inversions.
        - ``converged``: boolean indicating if convergence tolerance was met.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("P must be a 2‑D array of shape (N, J)")
    N, J = P.shape
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 1 or M.size != J:
        raise ValueError("M must be a 1‑D array with length equal to P.shape[1]")
    # Choose projection functions based on geometry
    geometry = geometry.lower()
    if geometry not in ('euclidean', 'kl'):
        raise ValueError(f"Invalid geometry '{geometry}'. Use 'euclidean' or 'kl'.")
    if geometry == 'euclidean':
        row_project = _project_row_simplex
        col_project = _project_column_isotonic_sum
    else:
        row_project = _project_row_simplex_kl  # defined below
        col_project = _project_column_isotonic_sum_kl  # defined below

    # Initial values
    Q = P.copy()
    U = np.zeros_like(P, dtype=np.float64)
    V = np.zeros_like(P, dtype=np.float64)

    def _max_rank_violation(Q_current: np.ndarray) -> float:
        # Compute maximum monotonicity violation across columns relative to P
        max_violation = 0.0
        for j in range(J):
            idx = np.argsort(P[:, j])
            q_sorted = Q_current[idx, j]
            # differences between successive elements (should be >= 0)
            diffs = q_sorted[:-1] - q_sorted[1:]
            violation = np.max(diffs)
            if violation > max_violation:
                max_violation = violation
        return max_violation

    converged = False
    for it in range(1, max_iters + 1):
        Q_prev = Q.copy()
        # Projection onto C1 (row simplex)
        Y = Q + U
        Q = row_project(Y)
        U = Y - Q
        # Projection onto C2 (column isotonic with sum)
        Y = Q + V
        # Project each column independently using appropriate geometry
        Q_cols = np.empty_like(Q, dtype=np.float64)
        for j in range(J):
            Q_cols[:, j] = col_project(Y[:, j], P[:, j], M[j])
        Q = Q_cols
        V = Y - Q
        # Check convergence
        delta = np.linalg.norm(Q - Q_prev)
        norm_Q = np.linalg.norm(Q_prev)
        if verbose:
            print(f"Iter {it}: delta={delta:.3e}, norm_Q={norm_Q:.3e}")
        if norm_Q == 0:
            # Avoid division by zero; if Q_prev was zero, use absolute change
            rel_change = delta
        else:
            rel_change = delta / norm_Q
        if rel_change < tol:
            converged = True
            break
    # Diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_violation = float(_max_rank_violation(Q))
    info: Dict[str, float] = {
        "iterations": it,
        "max_row_error": max_row_error,
        "max_col_error": max_col_error,
        "max_rank_violation": max_violation,
        "converged": converged,
    }
    if return_info:
        return Q, info
    else:
        return Q, {}

# -- Additional helper functions for KL geometry --

def _project_row_simplex_kl(rows: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project rows onto the probability simplex under KL (I-divergence) geometry.

    This projection normalises each row by its sum.  Values are clipped to
    a small ``eps`` to avoid division by zero.  The result has non-negative
    entries summing to one in each row.

    Parameters
    ----------
    rows : np.ndarray
        Array of shape ``(N, J)`` to be projected.
    eps : float, optional
        Lower bound for values to avoid numerical issues.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, J)`` with each row summing to one.
    """
    Q = np.maximum(rows, eps)
    row_sums = Q.sum(axis=1, keepdims=True)
    row_sums[row_sums < eps] = eps
    Q = Q / row_sums
    return Q

def _kl_isotonic_pav(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Perform KL isotonic regression on a 1-D array.

    This function solves the isotonic regression problem under the
    I-divergence geometry by applying isotonic regression to the log of the
    values.  It ensures the output is non-decreasing when the input is
    ordered.

    Parameters
    ----------
    y : np.ndarray
        1-D array of positive values.
    eps : float, optional
        Minimum value to clip ``y`` for stability.

    Returns
    -------
    np.ndarray
        A 1-D array of the same length as ``y`` representing the projection.
    """
    y_safe = np.maximum(y, eps)
    log_y = np.log(y_safe)
    log_iso = _isotonic_regression(log_y)
    return np.exp(log_iso)

def _project_column_isotonic_sum_kl(column: np.ndarray, P_column: np.ndarray, target_sum: float, eps: float = 1e-12) -> np.ndarray:
    """Project a column under KL geometry onto isotonic sequences with fixed sum.

    Parameters
    ----------
    column : np.ndarray
        Current column vector.
    P_column : np.ndarray
        Original scores used to determine the order for monotonicity.
    target_sum : float
        Desired sum of the projected column.
    eps : float, optional
        Lower bound for values to avoid numerical issues.

    Returns
    -------
    np.ndarray
        Projected column satisfying the KL isotonic and sum constraints.
    """
    idx = np.argsort(P_column)
    y = column[idx]
    # KL isotonic projection
    iso = _kl_isotonic_pav(y, eps=eps)
    total = iso.sum()
    if total <= eps:
        # If iso sums to zero, return uniform distribution scaled to target_sum
        iso_scaled = np.full_like(iso, target_sum / iso.size)
    else:
        iso_scaled = iso * (target_sum / total)
    out = np.empty_like(column, dtype=np.float64)
    out[idx] = iso_scaled
    return out
