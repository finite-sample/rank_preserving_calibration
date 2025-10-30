#!/usr/bin/env python3
"""
Realistic example demonstrating when nearly isotonic calibration is beneficial.

This example shows a credit scoring scenario where strict isotonic constraints
may be too restrictive, and nearly isotonic calibration provides a better balance
between rank preservation and probability calibration.
"""

import numpy as np
from data_helpers import analyze_calibration_result

from rank_preserving_calibration import calibrate_admm, calibrate_dykstra


def create_credit_scoring_scenario():
    """
    Create a realistic credit scoring scenario where nearly isotonic helps.

    Scenario: A bank has a machine learning model that predicts credit risk across
    different risk categories (Low, Medium, High). The model shows good discrimination
    but needs calibration to match observed default rates in their portfolio.

    The challenge: Due to economic conditions and regulatory requirements, the bank
    needs to adjust their risk predictions, but strict isotonic constraints might
    be too rigid for certain edge cases where model confidence varies.
    """

    np.random.seed(12345)
    N = 500  # 500 loan applications

    # True risk categories with realistic proportions
    # In practice, these might come from historical data
    true_categories = np.random.choice(
        [0, 1, 2], size=N, p=[0.6, 0.3, 0.1]
    )  # Most are low-risk

    # Simulate ML model predictions that are generally good but not perfectly calibrated
    P = np.zeros((N, 3))

    for i in range(N):
        true_cat = true_categories[i]

        if true_cat == 0:  # Low risk - model is fairly confident
            base_probs = [0.7, 0.25, 0.05]
            noise_scale = 0.15
        elif true_cat == 1:  # Medium risk - model less certain
            base_probs = [0.3, 0.5, 0.2]
            noise_scale = 0.25
        else:  # High risk - model confident but some uncertainty
            base_probs = [0.1, 0.3, 0.6]
            noise_scale = 0.2

        # Add noise to simulate model uncertainty and create violations
        noisy_probs = np.array(base_probs) + noise_scale * np.random.randn(3)
        noisy_probs = np.maximum(noisy_probs, 0.01)  # Ensure positive
        P[i] = noisy_probs / noisy_probs.sum()  # Normalize

    # Target marginals based on regulatory requirements or portfolio targets
    # Bank wants to match observed default rates in their data
    M = np.array([280, 150, 70])  # Slightly different from natural model distribution

    return P, M, true_categories


def demonstrate_nearly_isotonic_benefits():
    """Show how nearly isotonic calibration helps in realistic scenarios."""

    print("=== Realistic Credit Scoring Example ===")
    print("Scenario: ML model predictions need calibration to match portfolio targets")
    print("Challenge: Strict isotonic constraints may be too rigid\n")

    P, M, true_categories = create_credit_scoring_scenario()

    print(f"Dataset: {P.shape[0]} loan applications, {P.shape[1]} risk categories")
    print(f"Original model predictions sum by category: {P.sum(axis=0).round(1)}")
    print(f"Target marginals (regulatory/portfolio): {M}")
    print(f"Adjustment needed: {(M - P.sum(axis=0)).round(1)}")
    print()

    # Method 1: Strict isotonic calibration
    print("Method 1: Strict Isotonic Calibration")
    print("-" * 40)
    result_strict = calibrate_dykstra(P, M, verbose=False)
    analysis_strict = analyze_calibration_result(P, result_strict, M)

    print(f"Converged: {result_strict.converged}")
    print(
        f"Prediction changes: {analysis_strict['prediction_impact']['prediction_changes']:.1%}"
    )
    print(
        f"Confidence change: {analysis_strict['prediction_impact']['confidence_change']:.3f}"
    )
    print(
        f"Total probability change: {analysis_strict['distribution_impact']['total_change']:.3f}"
    )
    print(f"Rank violations: {result_strict.max_rank_violation:.2e}")
    print()

    # Method 2: Nearly isotonic with small slack
    print("Method 2: Nearly Isotonic (ε = 0.02)")
    print("-" * 40)
    nearly_params = {"mode": "epsilon", "eps": 0.02}
    result_nearly = calibrate_dykstra(P, M, nearly=nearly_params, verbose=False)
    analysis_nearly = analyze_calibration_result(P, result_nearly, M)

    print(f"Converged: {result_nearly.converged}")
    print(
        f"Prediction changes: {analysis_nearly['prediction_impact']['prediction_changes']:.1%}"
    )
    print(
        f"Confidence change: {analysis_nearly['prediction_impact']['confidence_change']:.3f}"
    )
    print(
        f"Total probability change: {analysis_nearly['distribution_impact']['total_change']:.3f}"
    )
    print(
        f"Max rank violation: {result_nearly.max_rank_violation:.3f} (within ε tolerance)"
    )
    print()

    # Method 3: Nearly isotonic with larger slack for comparison
    print("Method 3: Nearly Isotonic (ε = 0.05)")
    print("-" * 40)
    nearly_params_loose = {"mode": "epsilon", "eps": 0.05}
    result_loose = calibrate_dykstra(P, M, nearly=nearly_params_loose, verbose=False)
    analysis_loose = analyze_calibration_result(P, result_loose, M)

    print(f"Converged: {result_loose.converged}")
    print(
        f"Prediction changes: {analysis_loose['prediction_impact']['prediction_changes']:.1%}"
    )
    print(
        f"Confidence change: {analysis_loose['prediction_impact']['confidence_change']:.3f}"
    )
    print(
        f"Total probability change: {analysis_loose['distribution_impact']['total_change']:.3f}"
    )
    print(
        f"Max rank violation: {result_loose.max_rank_violation:.3f} (within ε tolerance)"
    )
    print()

    # Analysis: When is nearly isotonic better?
    print("=== Analysis: When Nearly Isotonic Helps ===")
    print()

    print("1. Preservation of Model Confidence:")
    print(f"   Original avg confidence: {np.mean(np.max(P, axis=1)):.3f}")
    print(
        f"   Strict isotonic:        {analysis_strict['prediction_impact']['calibrated_confidence']:.3f}"
    )
    print(
        f"   Nearly isotonic (ε=0.02): {analysis_nearly['prediction_impact']['calibrated_confidence']:.3f}"
    )
    print(
        f"   Nearly isotonic (ε=0.05): {analysis_loose['prediction_impact']['calibrated_confidence']:.3f}"
    )
    print()

    print("2. Flexibility vs. Constraints:")
    strict_change = analysis_strict["distribution_impact"]["total_change"]
    nearly_change = analysis_nearly["distribution_impact"]["total_change"]
    loose_change = analysis_loose["distribution_impact"]["total_change"]

    print("   Total change from original (lower = better):")
    print(f"   Strict isotonic:        {strict_change:.3f}")
    print(
        f"   Nearly isotonic (ε=0.02): {nearly_change:.3f} ({((nearly_change - strict_change) / strict_change * 100):+.1f}%)"
    )
    print(
        f"   Nearly isotonic (ε=0.05): {loose_change:.3f} ({((loose_change - strict_change) / strict_change * 100):+.1f}%)"
    )
    print()

    print("3. Individual Case Analysis:")
    # Find cases where nearly isotonic makes different decisions
    diff_decisions = np.argmax(result_strict.Q, axis=1) != np.argmax(
        result_nearly.Q, axis=1
    )
    n_diff = np.sum(diff_decisions)

    print(
        f"   Cases with different predictions: {n_diff} ({n_diff / len(P) * 100:.1f}%)"
    )

    if n_diff > 0:
        # Look at a few examples
        diff_indices = np.where(diff_decisions)[0][:3]
        print("   Example cases where methods differ:")

        for idx in diff_indices:
            orig_pred = np.argmax(P[idx])
            strict_pred = np.argmax(result_strict.Q[idx])
            nearly_pred = np.argmax(result_nearly.Q[idx])
            true_cat = true_categories[idx]

            print(
                f"     Case {idx}: True={true_cat}, Original={orig_pred}, Strict={strict_pred}, Nearly={nearly_pred}"
            )
            print(f"       Original probs: {P[idx].round(3)}")
            print(f"       Strict result:  {result_strict.Q[idx].round(3)}")
            print(f"       Nearly result:  {result_nearly.Q[idx].round(3)}")

    print()
    print("=== Recommendations ===")
    print()
    print("Use Nearly Isotonic Calibration When:")
    print("• Model predictions have good discrimination but need marginal adjustment")
    print(
        "• Strict rank preservation is less important than preserving model confidence"
    )
    print("• You have domain knowledge that small rank violations are acceptable")
    print(
        "• The model has inherent uncertainty that strict isotonic constraints ignore"
    )
    print()
    print("Use Strict Isotonic Calibration When:")
    print("• Rank preservation is critical (e.g., regulatory requirements)")
    print("• Model has clear monotonic relationship with outcomes")
    print("• Uncertainty in original predictions is low")
    print("• Conservative approach is preferred")


def compare_computational_aspects():
    """Compare computational performance of different methods."""

    print("\n=== Computational Performance Comparison ===")

    P, M, _ = create_credit_scoring_scenario()

    import time

    methods = [
        ("Strict Isotonic", lambda: calibrate_dykstra(P, M, verbose=False)),
        (
            "Nearly Isotonic (ε=0.02)",
            lambda: calibrate_dykstra(
                P, M, nearly={"mode": "epsilon", "eps": 0.02}, verbose=False
            ),
        ),
        (
            "Nearly Isotonic (ADMM λ=1.0)",
            lambda: calibrate_admm(
                P,
                M,
                nearly={"mode": "lambda", "lam": 1.0},
                verbose=False,
                max_iters=200,
            ),
        ),
    ]

    for name, method in methods:
        times = []
        for _ in range(5):  # Run multiple times for timing
            start = time.time()
            result = method()
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(
            f"{name:30s}: {avg_time:.4f} ± {std_time:.4f} seconds, {result.iterations} iterations"
        )


if __name__ == "__main__":
    demonstrate_nearly_isotonic_benefits()
    compare_computational_aspects()
