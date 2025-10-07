"""
Rank-preserving calibration of multiclass probabilities.

This package provides robust implementations of rank-preserving calibration
algorithms including Dykstra's alternating projections and ADMM optimization.

Example
-------
>>> import numpy as np
>>> from rank_preserving_calibration import calibrate_dykstra
>>> from examples.data_helpers import create_test_case
>>>
>>> # Generate test data
>>> P, M = create_test_case("random", N=100, J=4, seed=42)
>>>
>>> # Calibrate probabilities
>>> result = calibrate_dykstra(P, M)
>>> print(f"Converged: {result.converged}")
>>> print(f"Max row error: {result.max_row_error:.2e}")
"""

# Version info
__version__ = "0.5.0"
__author__ = "Gaurav Sood"
__email__ = "gsood07@gmail.com"

# Import main functions and classes
from .calibration import (
    ADMMResult,
    CalibrationError,
    CalibrationResult,
    calibrate_admm,
    calibrate_dykstra,
)

# Import nearly isotonic utilities
from .nearly import (
    project_near_isotonic_euclidean,
    prox_near_isotonic,
    prox_near_isotonic_with_sum,
)

# Legacy aliases for backward compatibility
calibrate_rank_preserving = calibrate_dykstra
admm_rank_preserving_simplex_marginals = calibrate_dykstra

# Define what gets imported with "from rank_preserving_calibration import *"
__all__ = [
    "ADMMResult",
    "CalibrationError",
    "CalibrationResult",
    "admm_rank_preserving_simplex_marginals",
    "calibrate_admm",
    "calibrate_dykstra",
    "calibrate_rank_preserving",
    "project_near_isotonic_euclidean",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum",
]
