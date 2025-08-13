# SPDX-License-Identifier: MIT
"""
Public API for the rank_preserving_calibration package.

This module exposes the main calibration function and provides package
documentation.  For detailed descriptions of the algorithm and usage
examples, see the docstring on :func:`admm_rank_preserving_simplex_marginals`.
"""
from .calibrator import admm_rank_preserving_simplex_marginals

__all__ = [
    "admm_rank_preserving_simplex_marginals",
]
