Changelog
=========

Version 0.4.1 (2024-08-XX)
---------------------------

**Bug Fixes:**
* Minor PyPI release improvements

Version 0.4.0 (2024-08-XX)
---------------------------

**New Features:**
* Added nearly isotonic calibration with epsilon-slack and lambda-penalty approaches
* New functions: ``project_near_isotonic_euclidean``, ``prox_near_isotonic``, ``prox_near_isotonic_with_sum``
* Enhanced API with ``nearly`` parameter for both Dykstra and ADMM methods

**API Changes:**
* Improved result classes with better diagnostics
* Enhanced error handling and input validation

**Documentation:**
* Added comprehensive Sphinx documentation
* Improved examples and tutorials
* Added mathematical theory section

Version 0.3.x and Earlier
--------------------------

**Core Features:**
* Implementation of Dykstra's alternating projections algorithm
* ADMM-based optimization with convergence tracking
* Row-simplex and isotonic column constraints
* Robust numerical implementation with PAV algorithm
* Comprehensive test suite

