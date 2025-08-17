"""
Rank-preserving calibration of multiclass probabilities.

This package provides robust implementations of rank-preserving calibration
algorithms including Dykstra's alternating projections and ADMM optimization.

Example
-------
>>> import numpy as np
>>> from rank_preserving_calibration import calibrate_dykstra, create_test_case
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
__version__ = "0.3.0"
__author__ = "Gaurav Sood"
__email__ = "gsood07@gmail.com"

# Import main functions and classes
from .calibration import (
    calibrate_dykstra,
    calibrate_admm,
    CalibrationResult,
    ADMMResult,
    CalibrationError
)

from .utils import (
    create_test_case,
    create_realistic_classifier_case,
    create_survey_reweighting_case,
    analyze_calibration_result
)

# Legacy aliases for backward compatibility
calibrate_rank_preserving = calibrate_dykstra
admm_rank_preserving_simplex_marginals = calibrate_dykstra

# Define what gets imported with "from rank_preserving_calibration import *"
__all__ = [
    # Main calibration functions
    "calibrate_dykstra",
    "calibrate_admm",
    
    # Result classes
    "CalibrationResult",
    "ADMMResult",
    "CalibrationError",
    
    # Utility functions
    "create_test_case",
    "create_realistic_classifier_case", 
    "create_survey_reweighting_case",
    "analyze_calibration_result",
    
    # Legacy aliases
    "calibrate_rank_preserving",
    "admm_rank_preserving_simplex_marginals",
]