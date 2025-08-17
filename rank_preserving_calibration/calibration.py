# SPDX-License-Identifier: MIT
"""
Robust rank-preserving multiclass probability calibration.

This module provides numerically stable implementations of rank-preserving
calibration algorithms including Dykstra's alternating projections and ADMM.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Callable, List

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
    objective_values : List[float]
        Objective function values over iterations.
    primal_residuals : List[float]
        Primal residual norms over iterations.
    dual_residuals : List[float]
        Dual residual norms over iterations.
    max_row_error : float
        Maximum row sum error.
    max_col_error : float
        Maximum column sum error.
    max_rank_violation : float
        Maximum rank violation.
    """
    Q: np.ndarray
    converged: bool
    iterations: int
    objective_values: List[float]
    primal_residuals: List[float]
    dual_residuals: List[float]
    max_row_error: float
    max_col_error: float
    max_rank_violation: float


class CalibrationError(Exception):
    """Raised when calibration fails due to invalid inputs or numerical issues."""
    pass


def _validate_inputs(P: np.ndarray, M: np.ndarray, max_iters: int, 
                    tol: float, feasibility_tol: float) -> tuple[int, int]:
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
    
    # Check basic feasibility
    M_sum = float(M.sum())
    if abs(M_sum - N) > feasibility_tol * N:
        warnings.warn(
            f"Sum of M ({M_sum:.3f}) differs from N ({N}) by "
            f"{abs(M_sum - N):.3f}. Problem may be infeasible.",
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


def _project_row_simplex(rows: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Project rows onto probability simplex with numerical stability."""
    N, J = rows.shape
    projected = np.empty_like(rows, dtype=np.float64)
    
    for i in range(N):
        v = rows[i]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, J + 1, dtype=np.float64)
        
        cond = u - cssv / ind > eps
        if not np.any(cond):
            rho = J - 1
        else:
            rho = np.nonzero(cond)[0][-1]
            
        theta = cssv[rho] / (rho + 1)
        w = np.maximum(v - theta, 0.0)
        
        sum_w = w.sum()
        if sum_w > eps:
            w /= sum_w
        else:
            w[:] = 1.0 / J
            
        projected[i] = w
    
    return projected


def _isotonic_regression(y: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Numerically stable isotonic regression using Pool Adjacent Violators."""
    if y.size <= 1:
        return y.astype(np.float64, copy=True)
        
    y = y.astype(np.float64, copy=True)
    n = y.size
    
    z = y.copy()
    w = np.ones(n, dtype=np.float64)
    i = 0
    
    while i < n - 1:
        abs_tol = rtol * (abs(z[i]) + abs(z[i + 1]) + 1.0)
        if z[i] <= z[i + 1] + abs_tol:
            i += 1
        else:
            # Pool blocks
            new_w = w[i] + w[i + 1]
            new_z = (z[i] * w[i] + z[i + 1] * w[i + 1]) / new_w
            z[i] = new_z
            w[i] = new_w
            
            z = np.delete(z, i + 1)
            w = np.delete(w, i + 1)
            n -= 1
            
            if i > 0:
                i -= 1
    
    # Expand back to original length
    try:
        expanded = np.repeat(z, w.astype(int))
        if len(expanded) != len(y):
            return _simple_isotonic_fallback(y)
        return expanded
    except (ValueError, MemoryError):
        return _simple_isotonic_fallback(y)


def _simple_isotonic_fallback(y: np.ndarray) -> np.ndarray:
    """Simple fallback isotonic regression."""
    result = y.astype(np.float64, copy=True)
    for i in range(1, len(result)):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
    return result


def _project_column_isotonic_sum(column: np.ndarray, 
                                P_column: np.ndarray,
                                target_sum: float,
                                rtol: float = 1e-12,
                                eps: float = 1e-15) -> np.ndarray:
    """Project column onto isotonic constraint with fixed sum."""
    if column.size == 0:
        return column.copy()
        
    idx = np.argsort(P_column)
    y = column[idx]
    
    iso = _isotonic_regression(y, rtol=rtol)
    
    current_sum = iso.sum()
    n = iso.size
    
    if current_sum > eps:
        iso_scaled = iso * (target_sum / current_sum)
    else:
        iso_scaled = np.full_like(iso, target_sum / n)
    
    iso_scaled = np.maximum(iso_scaled, 0.0)
    final_sum = iso_scaled.sum()
    
    if final_sum > eps:
        iso_scaled *= (target_sum / final_sum)
    else:
        iso_scaled[:] = target_sum / n
    
    projected = np.empty_like(column, dtype=np.float64)
    projected[idx] = iso_scaled
    
    return projected


def _compute_rank_violation(Q: np.ndarray, P: np.ndarray) -> float:
    """Compute maximum rank violation across all columns."""
    max_violation = 0.0
    N, J = Q.shape
    
    for j in range(J):
        idx = np.argsort(P[:, j])
        q_sorted = Q[idx, j]
        
        if len(q_sorted) > 1:
            diffs = np.diff(q_sorted)
            violation = float(np.max(-diffs))
            max_violation = max(max_violation, violation)
    
    return max_violation


def _detect_cycling(Q_history: list, Q: np.ndarray, 
                   cycle_tol: float = 1e-10) -> bool:
    """Detect if algorithm is cycling between solutions."""
    for prev_Q in Q_history:
        if np.allclose(Q, prev_Q, rtol=cycle_tol, atol=cycle_tol):
            return True
    return False


def calibrate_dykstra(
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
    """Calibrate using Dykstra's alternating projections.
    
    Projects between row simplex and column isotonic constraints using
    Dykstra's method with memory terms to ensure convergence to intersection.
    
    Parameters
    ----------
    P : np.ndarray
        Input probability matrix of shape (N, J).
    M : np.ndarray  
        Target column sums of length J.
    max_iters : int, default 3000
        Maximum iterations.
    tol : float, default 1e-7
        Convergence tolerance.
    rtol : float, default 1e-12
        Relative tolerance for isotonic regression.
    feasibility_tol : float, default 0.1
        Tolerance for feasibility warning.
    verbose : bool, default False
        Print progress.
    callback : callable, optional
        Progress callback function.
    detect_cycles : bool, default True
        Enable cycle detection.
    cycle_window : int, default 10
        Window for cycle detection.
        
    Returns
    -------
    CalibrationResult
        Result with calibrated matrix and diagnostics.
    """
    N, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    
    # Initialize Dykstra variables
    Q = P.copy()
    U = np.zeros_like(P, dtype=np.float64)  # Row simplex memory
    V = np.zeros_like(P, dtype=np.float64)  # Column constraint memory
    Q_prev = np.empty_like(Q)
    
    Q_history = [] if detect_cycles else None
    converged = False
    final_change = float('inf')
    
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
                Y[:, j], P[:, j], M[j], rtol=rtol
            )
        V = Y - Q
        
        # Check convergence
        change_abs = np.linalg.norm(Q - Q_prev)
        norm_Q_prev = np.linalg.norm(Q_prev)
        
        final_change = change_abs / norm_Q_prev if norm_Q_prev > 0 else change_abs
            
        if final_change < tol:
            converged = True
            if verbose:
                print(f"Dykstra converged at iteration {iteration}")
            break
            
        # Cycle detection
        if detect_cycles and iteration > cycle_window:
            if _detect_cycling(Q_history, Q):
                warnings.warn(f"Cycling detected at iteration {iteration}", UserWarning)
                break
                
            Q_history.append(Q.copy())
            if len(Q_history) > cycle_window:
                Q_history.pop(0)
        
        if verbose and (iteration % 100 == 0 or iteration <= 10):
            print(f"Dykstra iteration {iteration}: change = {final_change:.2e}")
            
        if callback is not None:
            if not callback(iteration, final_change, Q):
                break
    
    if not converged and iteration == max_iters:
        warnings.warn(f"Dykstra failed to converge after {max_iters} iterations", UserWarning)
    
    # Compute diagnostics
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


def calibrate_admm(
    P: np.ndarray,
    M: np.ndarray,
    rho: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    rtol: float = 1e-12,
    feasibility_tol: float = 0.1,
    verbose: bool = False
) -> ADMMResult:
    """Calibrate using ADMM optimization.
    
    Solves the constrained optimization problem using Alternating Direction
    Method of Multipliers with augmented Lagrangian.
    
    Parameters
    ----------
    P : np.ndarray
        Input probability matrix.
    M : np.ndarray
        Target column sums.
    rho : float, default 1.0
        ADMM penalty parameter.
    max_iters : int, default 1000
        Maximum iterations.
    tol : float, default 1e-6
        Convergence tolerance.
    rtol : float, default 1e-12
        Relative tolerance for isotonic regression.
    feasibility_tol : float, default 0.1
        Feasibility tolerance.
    verbose : bool, default False
        Print progress.
        
    Returns
    -------
    ADMMResult
        Result with calibrated matrix and convergence history.
    """
    N, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    
    # Initialize ADMM variables
    Q = P.copy()
    Z1 = np.ones(N)  # Row sum auxiliary variables
    Z2 = M.copy()    # Column sum auxiliary variables
    lambda1 = np.zeros(N)  # Row constraint multipliers
    lambda2 = np.zeros(J)  # Column constraint multipliers
    
    objective_values = []
    primal_residuals = []
    dual_residuals = []
    
    for iteration in range(max_iters):
        Q_prev = Q.copy()
        
        # Q-update: solve quadratic subproblem
        row_correction = (Z1 - lambda1/rho).reshape(-1, 1) @ np.ones((1, J))
        col_correction = np.ones((N, 1)) @ (Z2 - lambda2/rho).reshape(1, -1)
        
        Q_unconstrained = (P + rho * (row_correction + col_correction)) / (1 + 2*rho)
        
        # Apply rank-preserving and non-negativity constraints
        for j in range(J):
            idx = np.argsort(P[:, j])
            iso_vals = _isotonic_regression(Q_unconstrained[idx, j], rtol=rtol)
            Q_unconstrained[idx, j] = iso_vals
            
        Q = np.maximum(Q_unconstrained, 0.0)
        
        # Z-updates (constraint projections)
        row_sums = Q.sum(axis=1)
        col_sums = Q.sum(axis=0)
        
        Z1_prev = Z1.copy()
        Z2_prev = Z2.copy()
        
        Z1 = np.ones(N)  # Row sums constrained to 1
        Z2 = M.copy()    # Column sums constrained to M
        
        # Multiplier updates
        lambda1 += rho * (row_sums - Z1)
        lambda2 += rho * (col_sums - Z2)
        
        # Compute residuals
        primal_res = np.linalg.norm(np.concatenate([row_sums - Z1, col_sums - Z2]))
        dual_res1 = rho * np.linalg.norm(Z1 - Z1_prev)
        dual_res2 = rho * np.linalg.norm(Z2 - Z2_prev)
        dual_res = dual_res1 + dual_res2
        
        obj_val = 0.5 * np.linalg.norm(Q - P)**2
        
        objective_values.append(obj_val)
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)
        
        if verbose and iteration % 100 == 0:
            print(f"ADMM iter {iteration}: obj={obj_val:.3e}, "
                  f"primal={primal_res:.3e}, dual={dual_res:.3e}")
        
        if primal_res < tol and dual_res < tol:
            converged = True
            break
    else:
        converged = False
        
    if not converged and verbose:
        warnings.warn(f"ADMM failed to converge after {max_iters} iterations", UserWarning)
    
    # Final diagnostics
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
        max_rank_violation=max_rank_violation
    )


# Convenience aliases for backward compatibility
calibrate_rank_preserving = calibrate_dykstra
admm_rank_preserving_simplex_marginals = calibrate_dykstra