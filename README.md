## Rank Preserving Calibration of Multiclass Probabilities

[![Python application](https://github.com/finite-sample/rank_preserving_calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/rank_preserving_calibration/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rank_preserving_calibration.svg)](https://pypi.org/project/rank_preserving_calibration/)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/rank_preserving_calibration/)
[![PyPI Downloads](https://static.pepy.tech/badge/rank_preserving_calibration)](https://pepy.tech/projects/rank_preserving_calibration)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Survey statisticians and machine learning practitioners often need to adjust the predicted class probabilities from a classifier so that they match known population totals (column marginals). Simple post-hoc methods that apply separate logit shifts or raking to each class can scramble the ranking of individuals within a class when there are three or more classes. This package implements a rank-preserving calibration procedure that projects probabilities onto the intersection of two convex sets:

1. **Row-simplex**: each row sums to one and all entries are non-negative.
2. **Isotonic column marginals**: within each class, values are non-decreasing when instances are sorted by their original scores for that class, and the sum of each column equals a user-supplied target.

The algorithm uses Dykstra's alternating projection method in Euclidean geometry. When the specified column totals are feasible, the procedure returns a matrix that preserves cross-person discrimination within each class, matches the desired totals, and remains a valid probability distribution for each instance. If no such matrix exists, the algorithm converges to the closest point (in L2 sense) satisfying both sets of constraints.

### New: Nearly Isotonic Calibration

This package now supports **nearly isotonic** constraints that allow small violations of strict monotonicity when appropriate:

- **Epsilon-slack constraints**: Allow z[i+1] ≥ z[i] - ε instead of strict z[i+1] ≥ z[i]
- **Lambda-penalty approach**: Penalize isotonicity violations with a tunable parameter

These relaxed constraints can provide better balance between rank preservation and probability calibration when strict isotonic constraints are too restrictive.

An **ADMM optimization** implementation is also provided as an alternative solver that minimizes `||Q - P||²` subject to the same constraints.

## Installation

```bash
pip install rank_preserving_calibration
```

The only runtime dependency is `numpy`. Optional dependencies include `scipy` (for enhanced test case generation) and `matplotlib` (for examples).

## Usage

### Basic Usage

```python
import numpy as np
from rank_preserving_calibration import calibrate_dykstra

P = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7],
])

# Target column sums, e.g. population class frequencies. Must sum to the
# number of rows (3 in this example) for perfect feasibility.
M = np.array([1.0, 1.0, 1.0])

result = calibrate_dykstra(P, M)

print("Adjusted probabilities:\n", result.Q)
print("Converged:", result.converged)
print("Iterations:", result.iterations)
print("Max row error:", result.max_row_error)
print("Max column error:", result.max_col_error)
print("Rank violations:", result.max_rank_violation)
```

### Nearly Isotonic Usage

```python
# Epsilon-slack: Allow small rank violations (recommended)
nearly_params = {"mode": "epsilon", "eps": 0.05}
result = calibrate_dykstra(P, M, nearly=nearly_params)

# Lambda-penalty: Soft isotonic constraint (experimental)
nearly_params = {"mode": "lambda", "lam": 1.0}
result = calibrate_admm(P, M, nearly=nearly_params)
```

The returned `CalibrationResult` contains the calibrated matrix `Q` with the same shape as `P`. Each row of `Q` sums to one, the column sums match `M`, and within each column the entries are sorted in non-decreasing order according to the order implied by the original `P`.

## Functions

### `calibrate_dykstra(P, M, **kwargs)`

Calibrate using Dykstra's alternating projections (recommended). Supports both strict and nearly isotonic constraints.

### `calibrate_admm(P, M, **kwargs)`  

Calibrate using ADMM optimization with penalty parameter `rho`. Supports lambda-penalty nearly isotonic constraints.

### `create_test_case(case_type, N, J, **kwargs)` (in `examples.data_helpers`)

Generate synthetic test data for various scenarios used in examples and tests.

## Arguments

| Parameter | Type | Description |
| --- | --- | --- |
| `P` | `ndarray` of shape `[N, J]` | Base multiclass probabilities or non-negative scores. Rows will be projected to the simplex. |
| `M` | `ndarray` of shape `[J]` | Target column totals (e.g. population class frequencies). The sum of `M` should equal the number of rows `N` for exact feasibility. |
| `max_iters` | `int` | Maximum number of projection iterations (default `3000` for Dykstra, `1000` for ADMM). |
| `tol` | `float` | Relative convergence tolerance (default `1e-7` for Dykstra, `1e-6` for ADMM). |
| `verbose` | `bool` | If `True`, prints convergence diagnostics. |
| `nearly` | `dict` | Nearly isotonic parameters: `{"mode": "epsilon", "eps": 0.05}` or `{"mode": "lambda", "lam": 1.0}`. |
| `rho` | `float` | ADMM penalty parameter (default `1.0`, ADMM only). |

## Returns

### CalibrationResult

Both functions return a `CalibrationResult` object with the following attributes:

* `Q`: NumPy array of shape `[N, J]` containing the calibrated probabilities. Each row sums to one, each column approximately sums to the corresponding entry of `M`, and within each column the values are non-decreasing according to the ordering induced by `P`.
* `converged`: boolean indicating whether the solver met the tolerance criteria.
* `iterations`: number of iterations performed.
* `max_row_error`: maximum absolute deviation of row sums from 1.
* `max_col_error`: maximum absolute deviation of column sums from `M`.
* `max_rank_violation`: maximum violation of monotonicity (should be 0 up to numerical tolerance).
* `final_change`: final relative change between iterations.

### ADMMResult

The ADMM function returns an `ADMMResult` object with additional convergence history:

* All `CalibrationResult` attributes plus:
* `objective_values`: objective function values over iterations.
* `primal_residuals`: primal residual norms over iterations.
* `dual_residuals`: dual residual norms over iterations.

## Algorithm Notes

* **Dykstra's Method**: Uses alternating projections with memory terms to ensure convergence to the intersection of constraint sets. Rows are projected onto the simplex via the algorithm of Duchi et al., and columns are projected via the pool-adjacent-violators algorithm followed by an additive shift to match column totals. This is the recommended method for most applications.

* **Nearly Isotonic Extensions**: 
  - **Epsilon-slack (Dykstra)**: Projects onto the convex set {z : z[i+1] ≥ z[i] - ε} using coordinate transformation. Maintains theoretical convergence guarantees.
  - **Lambda-penalty (ADMM)**: Uses proximal operator to minimize ||Q - P||² + λ∑max(0, z[i] - z[i+1]). More experimental but provides soft constraints.

* **ADMM**: Solves the constrained optimization problem using the Alternating Direction Method of Multipliers. May converge faster for some problems but requires tuning the penalty parameter `rho`. The algorithm minimizes the sum of squared differences `0.5 * ||Q - P||²_F` subject to the calibration constraints.

## Examples

See `examples/` directory for comprehensive examples including:
- `example.ipynb`: Basic usage and visualization
- `focused_nearly_isotonic_example.py`: When to use nearly isotonic calibration
- Real-world classifier calibration scenarios  
- Survey reweighting applications
- Algorithm comparison and performance analysis

### When to Use Nearly Isotonic Calibration

**Use Nearly Isotonic When:**
- Model predictions have good discrimination but need marginal calibration
- Some predictions are already well-calibrated  
- Small rank violations are acceptable in your domain
- You want to preserve model confidence where possible

**Use Strict Isotonic When:**
- Rank order is critical (regulatory, safety applications)
- Model predictions have clear monotonic relationship
- Conservative approach is preferred

## Testing

```bash
python -m pytest tests/ -v
```

## Legacy Compatibility

For backward compatibility, the following aliases are available:
- `calibrate_rank_preserving` → `calibrate_dykstra`
- `admm_rank_preserving_simplex_marginals` → `calibrate_dykstra`

## License

This software is released under the terms of the MIT license.

## Author

Gaurav Sood `<gsood07@gmail.com>`
