#!/usr/bin/env python3
"""
Focused example showing when nearly isotonic calibration is beneficial.

This example demonstrates the key trade-offs between strict and nearly isotonic
calibration using a simplified but realistic scenario.
"""

import numpy as np

from rank_preserving_calibration import (
    calibrate_dykstra,
    distance_metrics,
    feasibility_metrics,
    isotonic_metrics,
)


def create_focused_example():
    """
    Create a focused example where nearly isotonic calibration helps.

    Scenario: A classification model with 3 classes where strict isotonic
    constraints force too much change in some well-calibrated predictions.
    """

    np.random.seed(42)

    # Create a realistic case: 20 instances, 3 classes
    N, J = 20, 3

    # Most predictions are well-ordered but need slight marginal adjustment
    P = np.zeros((N, J))

    # Create different types of predictions
    for i in range(N):
        if i < 10:  # Well-behaved predictions
            probs = np.array([0.6 - 0.05 * i, 0.3, 0.1 + 0.05 * i])
        elif i < 15:  # Some uncertainty cases
            probs = np.array([0.4, 0.4, 0.2]) + 0.1 * np.random.randn(3)
        else:  # Clear preferences
            probs = np.array([0.2, 0.2, 0.6])

        probs = np.maximum(probs, 0.05)  # Ensure positive
        P[i] = probs / probs.sum()

    # Target marginals that require adjustment
    M = np.array([8.0, 6.0, 6.0])  # Different from natural distribution

    return P, M


def analyze_rank_preservation():
    """Analyze how rank preservation differs between methods."""

    P, M = create_focused_example()

    # Method 1: Strict isotonic
    result_strict = calibrate_dykstra(P, M, verbose=False, max_iters=1000)

    # Method 2: Nearly isotonic
    result_nearly = calibrate_dykstra(
        P, M, nearly={"mode": "epsilon", "eps": 0.1}, verbose=False, max_iters=1000
    )

    print("=== Rank Preservation Analysis ===")
    print(f"Input shape: {P.shape}, Target marginals: {M}")
    print()

    print("Original column sums:", P.sum(axis=0).round(2))
    print("Target column sums:", M)
    print("Required adjustment:", (M - P.sum(axis=0)).round(2))
    print()

    # Show results for both methods
    methods = [
        ("Strict Isotonic", result_strict),
        ("Nearly Isotonic (ε=0.1)", result_nearly),
    ]

    print("Results Summary:")
    print("-" * 50)

    for name, result in methods:
        print(f"{name}:")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Max row error: {result.max_row_error:.2e}")
        print(f"  Max col error: {result.max_col_error:.2e}")
        print(f"  Max rank violation: {result.max_rank_violation:.4f}")
        print(f"  Final column sums: {result.Q.sum(axis=0).round(2)}")

        # Calculate total change from original
        total_change = np.linalg.norm(result.Q - P)
        print(f"  Total change from P: {total_change:.3f}")
        print()

    # Detailed comparison for a few examples
    print("Detailed Case-by-Case Comparison (first 8 instances):")
    print("-" * 60)
    print(f"{'ID':>3} {'Original':>20} {'Strict':>20} {'Nearly':>20}")
    print("-" * 60)

    for i in range(min(8, len(P))):
        orig_str = f"[{P[i, 0]:.2f}, {P[i, 1]:.2f}, {P[i, 2]:.2f}]"
        strict_str = f"[{result_strict.Q[i, 0]:.2f}, {result_strict.Q[i, 1]:.2f}, {result_strict.Q[i, 2]:.2f}]"
        nearly_str = f"[{result_nearly.Q[i, 0]:.2f}, {result_nearly.Q[i, 1]:.2f}, {result_nearly.Q[i, 2]:.2f}]"

        print(f"{i:>3} {orig_str:>20} {strict_str:>20} {nearly_str:>20}")

        # Check if predictions changed
        orig_pred = np.argmax(P[i])
        strict_pred = np.argmax(result_strict.Q[i])
        nearly_pred = np.argmax(result_nearly.Q[i])

        if orig_pred != strict_pred or orig_pred != nearly_pred:
            print(
                f"    → Predictions: Orig={orig_pred}, Strict={strict_pred}, Nearly={nearly_pred}"
            )

    return P, result_strict, result_nearly


def demonstrate_flexibility():
    """Show how epsilon parameter controls flexibility."""

    print("\n=== Epsilon Parameter Effects ===")

    P, M = create_focused_example()

    epsilon_values = [0.0, 0.05, 0.1, 0.2]
    results = []

    print(
        f"{'Epsilon':>8} {'Converged':>10} {'Iterations':>11} {'Total Change':>13} {'Max Rank Viol':>14}"
    )
    print("-" * 60)

    for eps in epsilon_values:
        if eps == 0.0:
            result = calibrate_dykstra(P, M, verbose=False, max_iters=500)
            method_name = "Strict"
        else:
            result = calibrate_dykstra(
                P,
                M,
                nearly={"mode": "epsilon", "eps": eps},
                verbose=False,
                max_iters=500,
            )
            method_name = f"ε={eps}"

        total_change = np.linalg.norm(result.Q - P)

        print(
            f"{method_name:>8} {'Yes' if result.converged else 'No':>10} {result.iterations:>11} {total_change:>13.3f} {result.max_rank_violation:>14.4f}"
        )
        results.append((eps, result))

    print()
    print("Key Observations:")
    print("• Larger ε allows more flexibility (smaller total change)")
    print("• ε=0 enforces strict isotonic constraints")
    print("• Small ε (0.05-0.1) often provides good balance")
    print("• Convergence may be faster with appropriate ε")


def show_practical_recommendations():
    """Provide practical guidance on when to use nearly isotonic calibration."""

    print("\n=== Practical Recommendations ===")
    print()
    print("✓ Use Nearly Isotonic Calibration When:")
    print("  • Your model has good discrimination but needs marginal calibration")
    print("  • Some predictions are already well-calibrated")
    print("  • Small rank violations are acceptable in your domain")
    print("  • You want to preserve model confidence where possible")
    print("  • Strict constraints cause over-adjustment")
    print()

    print("✓ Use Strict Isotonic Calibration When:")
    print("  • Rank order is critical (regulatory, safety applications)")
    print("  • Model predictions have clear monotonic relationship")
    print("  • Conservative approach is preferred")
    print("  • Original predictions have large rank violations")
    print()

    print("⚙️  Parameter Guidelines:")
    print("  • Start with ε = 0.05 to 0.1 for most applications")
    print("  • Use ε = 0.01 to 0.03 for conservative relaxation")
    print("  • Use ε = 0.1 to 0.2 for more flexibility")
    print("  • Monitor max_rank_violation in results")
    print()

    print("📊 Evaluation Metrics:")
    print("  • Compare total_change: lower usually better")
    print("  • Check prediction_changes: fewer changes often better")
    print("  • Verify max_rank_violation stays within tolerance")
    print("  • Ensure convergence (increase max_iters if needed)")


def demonstrate_comprehensive_metrics():
    """Show comprehensive evaluation using the metrics module."""

    print("\n=== Comprehensive Metrics Evaluation ===")
    print()

    # Get test data and calibration results
    P, M = create_focused_example()

    # Compare strict vs nearly isotonic
    result_strict = calibrate_dykstra(P, M, max_iters=1000, verbose=False)
    result_nearly = calibrate_dykstra(
        P, M, nearly={"mode": "epsilon", "eps": 0.05}, max_iters=1000, verbose=False
    )

    print("Detailed Metrics Comparison:")
    print("-" * 70)

    for label, result in [
        ("Strict Isotonic", result_strict),
        ("Nearly Isotonic", result_nearly),
    ]:
        print(f"\n{label} Results:")
        print("-" * 30)

        # 1. Feasibility metrics
        feasibility = feasibility_metrics(result.Q, M)
        print("  Constraint Satisfaction:")
        print(f"    Max row error: {feasibility['row']['max_abs_error']:.2e}")
        print(f"    Max column error: {feasibility['col']['max_abs_error']:.2e}")
        print(f"    Min probability: {feasibility['row']['min_value']:.3f}")

        # 2. Isotonic metrics
        isotonic = isotonic_metrics(result.Q, P)
        print("  Rank Preservation:")
        print(f"    Max rank violation: {isotonic['max_rank_violation']:.4f}")
        print(f"    Total violation mass: {isotonic['total_violation_mass']:.4f}")
        print(f"    Mean flat fraction: {isotonic['mean_flat_fraction']:.3f}")

        # 3. Distance metrics
        distances = distance_metrics(result.Q, P)
        print("  Calibration Changes:")
        print(f"    Frobenius distance: {distances['frobenius']:.3f}")
        print(f"    Mean absolute change: {distances['mean_abs']:.4f}")
        print(f"    Max absolute change: {distances['max_abs']:.4f}")

        # 4. Algorithm performance
        print("  Algorithm Performance:")
        print(f"    Converged: {result.converged}")
        print(f"    Iterations: {result.iterations}")
        print(f"    Final change: {result.final_change:.2e}")

    print("\n" + "=" * 70)
    print("Metrics Interpretation Guide:")
    print("• Row/Column errors should be < 1e-6 for well-converged solutions")
    print("• Rank violations measure departure from monotonicity (0 = perfect)")
    print("• Smaller distance metrics = less change from original predictions")
    print("• Nearly isotonic often achieves similar constraint satisfaction")
    print("  with less change from original predictions")


if __name__ == "__main__":
    P, result_strict, result_nearly = analyze_rank_preservation()
    demonstrate_flexibility()
    show_practical_recommendations()
    demonstrate_comprehensive_metrics()
