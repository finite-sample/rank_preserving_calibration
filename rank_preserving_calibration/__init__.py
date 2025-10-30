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

# Version info - imported dynamically from pyproject.toml
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python < 3.8 compatibility (though project requires 3.11+)
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("rank_preserving_calibration")
except PackageNotFoundError:
    # Fallback for development environments
    __version__ = "unknown"
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

# Define what gets imported with "from rank_preserving_calibration import *"
__all__ = [
    "ADMMResult",
    "CalibrationError",
    "CalibrationResult",
    "calibrate_admm",
    "calibrate_dykstra",
    "project_near_isotonic_euclidean",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum",
]
