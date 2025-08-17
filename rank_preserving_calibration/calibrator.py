# SPDX-License-Identifier: MIT
"""
Robust rank-preserving multiclass probability calibration using Dykstra's algorithm.

This module provides a numerically stable implementation of rank-preserving
calibration that projects multiclass probability matrices onto the intersection
of two convex sets while maintaining computational efficiency and robustness.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Callable, Union, Dict, Any

import numpy as np


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

    def __post_init__(self):
        """Validate the result after initialization."""
        if not isinstance(self.Q, np.ndarray) or self.Q.ndim != 2:
            raise ValueError("Q must be a 2D numpy array")


class CalibrationError(Exception):
    """Raised when calibration fails due to invalid inputs or numerical issues."""
    pass


def _validate_inputs(P: np.ndarray, M: np.ndarray, max_iters: int, 
                    tol: float, feasibility_tol: float) -> tuple[int, int]:
    """Validate all inputs to the calibration function.
    
    Parameters
    ----------
    P : np.ndarray
        Probability matrix to validate.
    M : np.ndarray
        Target marginals to validate.
    max_iters : int
        Maximum iterations to validate.
    tol : float
        Convergence tolerance to validate.
    feasibility_tol : float
        Feasibility tolerance to validate.
        
    Returns
    -------
    tuple[int, int]
        Shape (N, J) of the validated matrix P.
        
    Raises
    ------
    CalibrationError
        If any input is invalid.
    """
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
    
    # Check basic feasibility
    M_sum = float(M.sum())
    if abs(M_sum - N) > feasibility_tol * N:
        warnings.warn(
            f"Sum of M ({M_sum:.3f}) differs significantly from N ({N}). "
            f"Difference: {abs(M_sum - N):.3f}. Problem may be infeasible.",
            UserWarning
        )
    
    # Validate other parameters
    if not isinstance(max_iters, int) or max_iters <= 0:
        raise CalibrationError("max_iters must be a positive integer")
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise CalibrationError("tol must be a positive number")
    if not isinstance(feasibility_tol, (int, float)) or feasibility_tol < 0:
        raise CalibrationError("feasibility_tol must be non-negative")
        
    return N, J


def _project_row_simplex_vectorized(rows: np.ndarray, 
                                   eps: float = 1e-15) -> np.ndarray:
    """Project rows onto probability simplex with numerical stability.
    
    Uses vectorized operations for better performance and includes
    numerical safeguards against edge cases.
    
    Parameters
    ----------
    rows : np.ndarray
        Array of shape (N, J) to project.
    eps : float
        Minimum threshold for numerical stability.
        
    Returns
    -------
    np.ndarray
        Projected array where each row sums to 1.
    """
    N, J = rows.shape
    projected = np.empty_like(rows, dtype=np.float64)
    
    for i in range(N):
        v = rows[i]
        
        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, J + 1, dtype=np.float64)
        
        # Find rho with numerical tolerance
        cond = u - cssv / ind > eps
        if not np.any(cond):
            rho = J - 1
        else:
            rho = np.nonzero(cond)[0][-1]
            
        theta = cssv[rho] / (rho + 1)
        w = v - theta
        w = np.maximum(w, 0.0)  # Ensure non-negativity
        
        # Robust normalization
        sum_w = w.sum()
        if sum_w > eps:
            w /= sum_w
        else:
            # Uniform distribution if projection fails
            w[:] = 1.0 / J
            
        projected[i] = w
    
    return projected


def _isotonic_regression(y: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Numerically stable isotonic regression using Pool Adjacent Violators.
    
    Parameters
    ----------
    y : np.ndarray
        1D array to make isotonic.
    rtol : float
        Relative tolerance for comparison.
        
    Returns
    -------
    np.ndarray
        Isotonic regression of y.
    """
    if y.size == 0:
        return y.copy()
        
    y = y.astype(np.float64, copy=True)
    n = y.size
    
    if n == 1:
        return y
    
    # PAV algorithm with relative tolerance
    z = y.copy()
    w = np.ones(n, dtype=np.float64)
    i = 0
    
    while i < n - 1:
        # Use relative tolerance for comparison
        abs_tol = rtol * (abs(z[i]) + abs(z[i + 1]) + 1.0)
        if z[i] <= z[i + 1] + abs_tol:
            i += 1
        else:
            # Pool blocks i and i+1
            new_w = w[i] + w[i + 1]
            new_z = (z[i] * w[i] + z[i + 1] * w[i + 1]) / new_w
            z[i] = new_z
            w[i] = new_w
            
            # Remove block i+1
            z = np.delete(z, i + 1)
            w = np.delete(w, i + 1)
            n -= 1
            
            # Move left if possible
            if i > 0:
                i -= 1
    
    # Expand back to original length
    expanded = np.repeat(z, w.astype(int))
    
    # Handle potential floating point errors in weights
    if len(expanded) != len(y):
        # Fallback: simple isotonic averaging
        return _simple_isotonic_fallback(y)
        
    return expanded


def _simple_isotonic_fallback(y: np.ndarray) -> np.ndarray:
    """Simple fallback isotonic regression for edge cases."""
    result = y.copy()
    n = len(result)
    
    # Simple forward pass
    for i in range(1, n):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
            
    return result


def _project_column_isotonic_sum(column: np.ndarray, 
                                P_column: np.ndarray,
                                target_sum: float,
                                rtol: float = 1e-12,
                                eps: float = 1e-15) -> np.ndarray:
    """Project column onto isotonic constraint with fixed sum.
    
    Parameters
    ----------
    column : np.ndarray
        Column to project.
    P_column : np.ndarray
        Original scores determining order.
    target_sum : float
        Target sum for the column.
    rtol : float
        Relative tolerance for isotonic regression.
    eps : float
        Minimum value threshold.
        
    Returns
    -------
    np.ndarray
        Projected column satisfying constraints.
    """
    if column.size == 0:
        return column.copy()
        
    # Get sorting order
    idx = np.argsort(P_column)
    y = column[idx]
    
    # Apply isotonic regression
    iso = _isotonic_regression(y, rtol=rtol)
    
    # Adjust sum while preserving non-negativity
    current_sum = iso.sum()
    n = iso.size
    
    if current_sum > eps:
        # Scale to target sum
        iso_scaled = iso * (target_sum / current_sum)
    else:
        # Uniform distribution if sum is too small
        iso_scaled = np.full_like(iso, target_sum / n)
    
    # Ensure non-negativity and correct sum
    iso_scaled = np.maximum(iso_scaled, 0.0)
    final_sum = iso_scaled.sum()
    
    if final_sum > eps:
        iso_scaled *= (target_sum / final_sum)
    else:
        iso_scaled[:] = target_sum / n
    
    # Return in original order
    projected = np.empty_like(column, dtype=np.float64)
    projected[idx] = iso_scaled
    
    return projected


def _compute_rank_violation(Q: np.ndarray, P: np.ndarray) -> float:
    """Compute maximum rank violation across all columns.
    
    Parameters
    ----------
    Q : np.ndarray
        Current calibrated matrix.
    P : np.ndarray
        Original probability matrix.
        
    Returns
    -------
    float
        Maximum rank violation.
    """
    max_violation = 0.0
    N, J = Q.shape
    
    for j in range(J):
        # Get order based on original P
        idx = np.argsort(P[:, j])
        q_sorted = Q[idx, j]
        
        # Check for violations (negative differences)
        if len(q_sorted) > 1:
            diffs = np.diff(q_sorted)
            violation = float(np.max(-diffs))  # Positive if decreasing
            max_violation = max(max_violation, violation)
    
    return max_violation


def _detect_cycling(Q_history: list, Q: np.ndarray, 
                   cycle_tol: float = 1e-10) -> bool:
    """Detect if algorithm is cycling between solutions.
    
    Parameters
    ----------
    Q_history : list
        Recent history of Q matrices.
    Q : np.ndarray
        Current Q matrix.
    cycle_tol : float
        Tolerance for cycle detection.
        
    Returns
    -------
    bool
        True if cycling is detected.
    """
    for prev_Q in Q_history:
        if np.allclose(Q, prev_Q, rtol=cycle_tol, atol=cycle_tol):
            return True
    return False


def calibrate_rank_preserving(
    P: np.ndarray,
    M: np.ndarray,
    max_iters: int = 3000,
    tol: float = 1e-7,
    rtol: float = 1e-12,
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    callback: Optional[Callable[[int, float, np.ndarray], bool]] = None,
    detect_cycles: bool = True,
    cycle_window: int = 10
) -> CalibrationResult:
    """Calibrate multiclass probabilities preserving within-class ranks.
    
    Uses Dykstra's alternating projection algorithm to find probabilities that:
    1. Have rows summing to 1 (valid probability distributions)
    2. Have columns summing to specified marginals M
    3. Preserve rank ordering within each class
    
    Parameters
    ----------
    P : np.ndarray
        Input probability matrix of shape (N, J). Should have non-negative
        entries, though row normalization is enforced during iteration.
    M : np.ndarray  
        Target column sums of length J. Should sum approximately to N
        for feasible problems.
    max_iters : int, default 3000
        Maximum number of iterations.
    tol : float, default 1e-7
        Convergence tolerance (relative change in Frobenius norm).
    rtol : float, default 1e-12
        Relative tolerance for isotonic regression comparisons.
    feasibility_tol : float, default 0.1
        Tolerance for feasibility warning (fraction of N).
    verbose : bool, default False
        Whether to print iteration progress.
    callback : callable, optional
        Function called each iteration as callback(iter, change, Q).
        Should return True to continue, False to stop early.
    detect_cycles : bool, default True
        Whether to detect and warn about cycling behavior.
    cycle_window : int, default 10
        Number of recent iterations to check for cycles.
        
    Returns
    -------
    CalibrationResult
        Result object containing calibrated matrix and diagnostics.
        
    Raises
    ------
    CalibrationError
        If inputs are invalid or algorithm fails.
        
    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])
    >>> M = np.array([1.5, 1.5])  # Target marginals
    >>> result = calibrate_rank_preserving(P, M)
    >>> print(f"Converged: {result.converged}")
    >>> print(f"Max row error: {result.max_row_error:.2e}")
    """
    # Input validation
    N, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    
    # Convert to working precision
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    
    # Initialize algorithm state
    Q = P.copy()
    U = np.zeros_like(P, dtype=np.float64)
    V = np.zeros_like(P, dtype=np.float64)
    Q_prev = np.empty_like(Q)
    
    # Cycle detection
    Q_history = [] if detect_cycles else None
    cycle_detected = False
    
    # Main iteration loop
    converged = False
    final_change = float('inf')
    
    for iteration in range(1, max_iters + 1):
        # Store previous iterate
        np.copyto(Q_prev, Q)
        
        # Project onto row simplex constraint
        Y = Q + U
        Q = _project_row_simplex_vectorized(Y)
        U = Y - Q
        
        # Project onto column isotonic + sum constraints  
        Y = Q + V
        for j in range(J):
            Q[:, j] = _project_column_isotonic_sum(
                Y[:, j], P[:, j], M[j], rtol=rtol
            )
        V = Y - Q
        
        # Compute convergence metrics
        change_abs = np.linalg.norm(Q - Q_prev)
        norm_Q_prev = np.linalg.norm(Q_prev)
        
        if norm_Q_prev > 0:
            final_change = change_abs / norm_Q_prev
        else:
            final_change = change_abs
            
        # Check convergence
        if final_change < tol:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
            
        # Cycle detection
        if detect_cycles and iteration > cycle_window:
            if _detect_cycling(Q_history, Q):
                cycle_detected = True
                warnings.warn(
                    f"Cycling detected at iteration {iteration}. "
                    "Consider adjusting tolerance or checking problem feasibility.",
                    UserWarning
                )
                break
                
            Q_history.append(Q.copy())
            if len(Q_history) > cycle_window:
                Q_history.pop(0)
        
        # Progress reporting
        if verbose and (iteration % 100 == 0 or iteration <= 10):
            print(f"Iteration {iteration}: change = {final_change:.2e}")
            
        # User callback
        if callback is not None:
            should_continue = callback(iteration, final_change, Q)
            if not should_continue:
                if verbose:
                    print(f"Stopped by callback at iteration {iteration}")
                break
    
    # Final convergence check
    if not converged and not cycle_detected and iteration == max_iters:
        warnings.warn(
            f"Failed to converge after {max_iters} iterations. "
            f"Final change: {final_change:.2e}, tolerance: {tol:.2e}",
            UserWarning
        )
    
    # Compute final diagnostics
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
        final_change=final_change
    )


# Convenience alias for backward compatibility
admm_rank_preserving_simplex_marginals = calibrate_rank_preserving
