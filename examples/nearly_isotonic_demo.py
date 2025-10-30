#!/usr/bin/env python3
"""
Demonstration of nearly isotonic calibration functionality.

This script shows how to use the new epsilon-slack and lambda-penalty
nearly isotonic constraints in both Dykstra and ADMM solvers.
"""

import numpy as np
from data_helpers import create_test_case

from rank_preserving_calibration import calibrate_admm, calibrate_dykstra


def compare_isotonic_vs_nearly_isotonic():
    """Compare strict isotonic vs nearly isotonic calibration."""

    # Create test data with some rank violations that strict isotonic would fix
    np.random.seed(42)
    P, M = create_test_case("linear", N=20, J=3, noise_level=0.2)

    print("=== Nearly Isotonic Calibration Demo ===")
    print(f"Input shape: {P.shape}")
    print(f"Target marginals: {M}")
    print(f"Original column sums: {P.sum(axis=0)}")
    print()

    # Standard isotonic calibration with Dykstra
    result_strict = calibrate_dykstra(P, M, verbose=False)

    # Nearly isotonic with epsilon slack
    nearly_epsilon = {"mode": "epsilon", "eps": 0.05}
    result_nearly_eps = calibrate_dykstra(P, M, nearly=nearly_epsilon, verbose=False)

    # Nearly isotonic with ADMM lambda penalty
    nearly_lambda = {"mode": "lambda", "lam": 2.0}
    result_nearly_lam = calibrate_admm(
        P, M, nearly=nearly_lambda, verbose=False, max_iters=500
    )

    print("Results Summary:")
    print("-" * 50)

    methods = [
        ("Strict Isotonic (Dykstra)", result_strict),
        ("Nearly Isotonic ε=0.05 (Dykstra)", result_nearly_eps),
        ("Nearly Isotonic λ=2.0 (ADMM)", result_nearly_lam),
    ]

    for name, result in methods:
        row_error = (
            result.max_row_error
            if hasattr(result, "max_row_error")
            else np.max(np.abs(result.Q.sum(axis=1) - 1.0))
        )
        col_error = (
            result.max_col_error
            if hasattr(result, "max_col_error")
            else np.max(np.abs(result.Q.sum(axis=0) - M))
        )
        rank_viol = (
            result.max_rank_violation if hasattr(result, "max_rank_violation") else 0.0
        )
        converged = result.converged if hasattr(result, "converged") else True
        total_change = np.linalg.norm(result.Q - P)

        print(f"{name}:")
        print(f"  Converged: {converged}")
        print(f"  Max row error: {row_error:.2e}")
        print(f"  Max col error: {col_error:.2e}")
        print(f"  Max rank violation: {rank_viol:.2e}")
        print(f"  Total change from P: {total_change:.3f}")
        print(f"  Column sums: {result.Q.sum(axis=0).round(3)}")
        print()

    # Check rank preservation for each method
    print("Rank Preservation Analysis:")
    print("-" * 30)
    for name, result in methods:
        print(f"\n{name}:")
        for j in range(P.shape[1]):
            original_order = np.argsort(P[:, j])
            calibrated_values = result.Q[original_order, j]

            # Check for violations
            diffs = np.diff(calibrated_values)
            violations = diffs < -1e-10

            if "Nearly" in name and "ε=" in name:
                # For epsilon slack, check against epsilon tolerance
                eps = 0.05
                violations = diffs < -eps - 1e-8

            print(f"  Column {j}: {np.sum(violations)} rank violations")
            if np.sum(violations) > 0:
                print(f"    Worst violation: {np.min(diffs):.6f}")


def visualize_nearly_isotonic_effects():
    """Create visualization showing the effect of nearly isotonic constraints."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
    except ImportError:
        print("Matplotlib/seaborn not available for visualization")
        return

    # Create test case with clear ranking
    np.random.seed(123)
    N = 15
    P = np.zeros((N, 3))

    # Create clear trends with some noise
    for j in range(3):
        trend = np.linspace(0.1 + 0.2 * j, 0.7 + 0.1 * j, N)
        noise = 0.15 * np.random.randn(N)
        P[:, j] = np.clip(trend + noise, 0.05, 0.95)

    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)
    M = np.full(3, N / 3)  # Equal marginals

    # Compare methods
    result_strict = calibrate_dykstra(P, M, verbose=False)

    nearly_eps = {"mode": "epsilon", "eps": 0.1}
    result_nearly = calibrate_dykstra(P, M, nearly=nearly_eps, verbose=False)

    # Create visualization
    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["Original P", "Strict Isotonic", "Nearly Isotonic (ε=0.1)"]
    matrices = [P, result_strict.Q, result_nearly.Q]

    for ax, title, matrix in zip(axes, titles, matrices, strict=False):
        im = ax.imshow(matrix.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Instance")
        ax.set_ylabel("Class")
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"Class {i}" for i in range(3)])
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("nearly_isotonic_comparison.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'nearly_isotonic_comparison.png'")
    plt.show()

    # Print some statistics
    print("\nVisualization Statistics:")
    print(f"Original P - Column sums: {P.sum(axis=0).round(3)}")
    print(f"Strict isotonic - Column sums: {result_strict.Q.sum(axis=0).round(3)}")
    print(f"Nearly isotonic - Column sums: {result_nearly.Q.sum(axis=0).round(3)}")
    print(f"Change from original (strict): {np.linalg.norm(result_strict.Q - P):.3f}")
    print(f"Change from original (nearly): {np.linalg.norm(result_nearly.Q - P):.3f}")


if __name__ == "__main__":
    compare_isotonic_vs_nearly_isotonic()
    print("\n" + "=" * 60 + "\n")
    visualize_nearly_isotonic_effects()
