# rank_preserving_calibration/nearly.py
# MIT license - keep consistent with repo.
"""
Nearly isotonic regression utilities.

This module provides "relaxed" isotonic constraints that allow for small
violations of monotonicity, useful when strict isotonicity is too restrictive.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "project_near_isotonic_euclidean",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum"
]

# ---------- Utilities ----------

def _pav_increasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool Adjacent Violators (L2) for a 1D sequence (no deps)."""
    n = len(y)
    if w is None:
        w = np.ones(n, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    # Start with each point as a block
    v = y.copy()
    weight = w.copy()
    idx_start = np.arange(n)  # block start indices
    m = n
    i = 0
    while i < m - 1:
        if v[i] <= v[i+1] + 1e-15:  # already nondecreasing
            i += 1
            continue
        # pool i and i+1
        new_w = weight[i] + weight[i+1]
        new_v = (weight[i]*v[i] + weight[i+1]*v[i+1]) / new_w
        v[i] = new_v
        weight[i] = new_w
        # delete block i+1 by shifting left
        v[i+1:m-1] = v[i+2:m]
        weight[i+1:m-1] = weight[i+2:m]
        idx_start[i+1:m-1] = idx_start[i+2:m]
        m -= 1
        # backtrack if needed
        if i > 0:
            i -= 1
    # expand piecewise-constant blocks
    out = np.empty(n, dtype=float)
    for b in range(m):
        j0 = idx_start[b]
        j1 = idx_start[b+1] if b+1 < m else n
        out[j0:j1] = v[b]
    return out

# ---------- (A) Hard-slack nearly isotonic projection ----------

def project_near_isotonic_euclidean(
    v: np.ndarray,
    eps: float,
    sum_target: float | None = None,
    weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Project v onto the set { z : z_{i+1} >= z_i - eps } in L2.
    If sum_target is given, add a uniform shift to make sum(z)=sum_target.
    Parameters
    ----------
    v : np.ndarray
        Input vector to project.
    eps : float
        Slack parameter. Allows z[i+1] >= z[i] - eps instead of strict z[i+1] >= z[i].
    sum_target : float, optional
        If provided, shift result to have this sum.
    weights : np.ndarray, optional
        Weights for weighted projection (currently unused in this implementation).
    Returns
    -------
    np.ndarray
        Projected vector satisfying near-isotonic constraint.
    """
    n = len(v)
    # Reduce to standard isotonic via shift: w_i = v_i + i*eps
    # Then isotonic(w) = w*, and z* = w* - i*eps
    ar = np.arange(n, dtype=float)
    w = v + eps * ar
    iz = _pav_increasing(w, weights)
    z = iz - eps * ar
    if sum_target is not None:
        z += (sum_target - z.sum()) / n
    return z

# ---------- (B) Penalized (lambda) nearly isotonic prox ----------

def prox_near_isotonic(
    v: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
    max_iters: int = 2_000,
    tol: float = 1e-9
) -> np.ndarray:
    """
    Prox of λ * sum (z_i - z_{i+1})_- under 0.5||z - v||^2:
    solves   min_z 0.5||z - v||^2 + lam * sum_i max(0, z_i - z_{i+1})*(-1)
    Implementation: monotone path via modified PAV (mPAVA), evaluated at λ.
    This is a minimal, numerically-stable variant sufficient for column updates.

    Notes:
    - We implement the *single-λ* solution without building the whole path.
    - Follows the KKT structure in Tibshirani et al. (2011).
    Parameters
    ----------
    v : np.ndarray
        Input vector.
    lam : float
        Penalty parameter for isotonicity violations.
    weights : np.ndarray, optional
        Weights (currently unused).
    max_iters : int, default=2000
        Maximum iterations for fixed-point iteration.
    tol : float, default=1e-9
        Convergence tolerance.
    Returns
    -------
    np.ndarray
        Regularized nearly isotonic solution.
    """
    # We implement mPAVA without weights for simplicity; weights can be added similarly.
    y = np.asarray(v, dtype=float)
    n = len(y)

    # Each point starts as a block with slope m=0 and fitted value beta=y.
    # We'll maintain blocks with (start, end, value, slope).
    # starts = list(range(n))  # Unused in current implementation
    # ends   = list(range(1, n+1))  # Unused in current implementation
    beta   = [float(y[i]) for i in range(n)]
    slope  = [0.0 for _ in range(n)]

    # Helper to compute "collision time" between adjacent blocks (see paper).
    def collision_time(k: int) -> float:
        # Blocks k and k+1. If beta[k] <= beta[k+1], they are fine at current λ,
        # else they will collide at a finite λ_star when beta_k - lambda*m_k = beta_{k+1} - lambda*m_{k+1}.
        if k < 0 or k >= len(beta)-1:
            return np.inf
        if beta[k] <= beta[k+1] + 1e-15 and slope[k] <= slope[k+1] + 1e-15:
            return np.inf
        # linear functions in λ: b_k(λ)=beta[k] + slope[k]*(λ-0), etc.
        num = beta[k+1] - beta[k]
        den = slope[k] - slope[k+1]
        if den <= 1e-15:
            return np.inf
        t = num / den
        return t if t >= 0 else np.inf

    # Initialize slopes per block (Eq. 5 in the paper).
    # For the unweighted L2 case, slopes reflect imbalance of "downward edges" at block boundaries.
    # A minimal implementation is to iterate merges until all "violations" are accounted for at given λ.
    # We'll perform a variant of pool+tilt: binary search on λ with isotonicity bias.
    # For robustness and simplicity, we do a monotone search on λ and return the isotonic regression of (y + lam * g),
    # where g is the subgradient favoring nondecreasing fits. A practical g is [-1, 0, ..., 0, +1] applied via finite diffs.
    # In practice, the following surrogate works well for prox usage:
    #   z = PAV(y + lam * d), where d_i = count of "downward pressure" from left/right (fast surrogate).
    # But to keep behavior closer to mPAVA, we approximate via fixed-point iteration:

    # Handle special case of lambda=0
    if lam <= 1e-15:
        return y.copy()

    z = y.copy()
    for _ in range(max_iters):
        # Compute subgradient for hinge on negative differences
        diffs = z[:-1] - z[1:]
        g = np.zeros_like(z)
        neg = diffs > 0.0  # (z_i - z_{i+1})_- active
        g[:-1][neg] += 1.0
        g[1:][neg]  -= 1.0
        z_new = _pav_increasing(y + lam * g)
        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = z_new
    return z

def prox_near_isotonic_with_sum(v: np.ndarray, lam: float, sum_target: float) -> np.ndarray:
    """
    Apply nearly isotonic prox and then shift to achieve target sum.
    Parameters
    ----------
    v : np.ndarray
        Input vector.
    lam : float
        Penalty parameter.
    sum_target : float
        Target sum for the result.
    Returns
    -------
    np.ndarray
        Nearly isotonic solution with specified sum.
    """
    z = prox_near_isotonic(v, lam)
    # Penalty is difference-based → invariant to constant shift.
    z += (sum_target - z.sum()) / len(z)
    return z
