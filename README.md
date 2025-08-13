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
# number of rows for perfect feasibility.
M = np.array([1.5, 1.5, 1.5])

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

## License

This software is released under the terms of the MIT license.  See the
`LICENSE` file for details.

## Author

Gaurav Sood `<gsood07@gmail.com>`
