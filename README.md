## Rank Preserving Calibration of multiclass probabilities via Dykstra's alternating projections

Survey statisticians and machine learning practitioners often need to adjust
the predicted class probabilities from a classifier so that they match known
population totals (column marginals).  Simple post‑hoc methods that apply
separate logit shifts or raking to each class can scramble the ranking of
individuals within a class when there are three or more classes.  This
package implements a rank‑preserving calibration procedure that projects
probabilities onto the intersection of two convex sets:

1. **Row‑simplex**: each row sums to one and all entries are non‑negative.
2. **Isotonic column marginals**: within each class, values are
   non‑decreasing when instances are sorted by their original scores for
   that class, and the sum of each column equals a user‑supplied target.

The algorithm uses Dykstra's alternating projection method in Euclidean
geometry.  When the specified column totals are feasible, the procedure
returns a matrix that preserves cross‑person discrimination within each
class, matches the desired totals, and remains a valid probability
distribution for each instance.  If no such matrix exists, the algorithm
converges to the closest point (in L2 sense) satisfying both sets of
constraints.

Experimental support for **KL (I‑divergence) geometry** is also provided.
In that mode, row projections normalise each row by its sum and column
projections are based on KL‑isotonic regression (PAV in log space) followed
by multiplicative scaling.  Both geometries enforce the same row and
column constraints and preserve within‑class ranking.

## Installation

To install the package from source, clone the repository and run:

```sh
pip install .
```

The only runtime dependency is `numpy`.

## Usage

```python
import numpy as np
from rank_preserving_calibration import admm_rank_preserving_simplex_marginals

P = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7],
])

# Target column sums, e.g. population class frequencies.  Must sum to the
# number of rows (3 in this example) for perfect feasibility.
M = np.array([1.0, 1.0, 1.0])

Q, info = admm_rank_preserving_simplex_marginals(P, M)

print("Adjusted probabilities:\n", Q)
print("Diagnostics:\n", info)
```

The returned matrix `Q` has the same shape as `P`.  Each row of `Q` sums
to one, the column sums match `M`, and within each column the entries are
sorted in non‑decreasing order according to the order implied by the
original `P`.  The `info` dictionary reports the number of iterations
used, the maximum row and column errors, and any residual rank
violations (at numerical precision).

## Arguments

| Parameter | Type | Description |
| --- | --- | --- |
| `P` | `ndarray` of shape `[N, J]` | Base multiclass probabilities or non‑negative scores.  Rows will be projected to the simplex. |
| `M` | `ndarray` of shape `[J]` | Target column totals (e.g. population class frequencies).  The sum of `M` should equal the number of rows `N` for exact feasibility. |
| `geometry` | `str` | Either `'euclidean'` (default) or `'kl'` (experimental).  Determines the geometry used for projections. |
| `max_iters` | `int` | Maximum number of projection iterations (default `3000`). |
| `tol` | `float` | Relative convergence tolerance (default `1e‑7`). |
| `verbose` | `bool` | If `True`, prints convergence diagnostics. |

## Returns

The function returns a tuple `(Q, info)` where:

* `Q` is a NumPy array of shape `[N, J]` containing the calibrated probabilities.  Each row sums to one, each column approximately sums to the corresponding entry of `M`, and within each column the values are non‑decreasing according to the ordering induced by `P`.
* `info` is a dictionary with diagnostics:
  - `iterations`: number of iterations performed.
  - `max_row_error`: maximum absolute deviation of row sums from 1.
  - `max_col_error`: maximum absolute deviation of column sums from `M`.
  - `max_rank_violation`: maximum violation of monotonicity (should be 0 up to numerical tolerance).
  - `converged`: boolean indicating whether the solver met the tolerance criteria.
  - `geometry`: which geometry was used (`'euclidean'` or `'kl'`).

## Geometry Notes

* **Euclidean (L2)**: The default solver uses Euclidean projections.  Rows are projected onto the simplex via the algorithm of Duchi et al., and columns are projected via the pool‑adjacent‑violators algorithm followed by an additive shift to match the column totals.  This minimises the sum of squared differences `0.5 * ||Q - P||_F^2`.
* **KL (I‑divergence)**: Setting `geometry='kl'` switches to KL‑style projections.  Each row is normalised by its sum (multiplicative projection), and each column is projected by applying isotonic regression to the logarithms of the values (PAV in log space) followed by multiplicative scaling to match the column sum.  This mode is experimental but preserves within‑class ranking and approximately minimises the I‑divergence from `P`.


## License

This software is released under the terms of the MIT license.  See the
`LICENSE` file for details.

## Author

Gaurav Sood `<gsood07@gmail.com>`
