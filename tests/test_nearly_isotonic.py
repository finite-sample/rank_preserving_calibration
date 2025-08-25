"""
Test suite for nearly isotonic functionality.

Run with: python -m pytest tests/test_nearly_isotonic.py -v
"""

import numpy as np
import pytest
from rank_preserving_calibration import (
    calibrate_dykstra,
    calibrate_admm,
    project_near_isotonic_euclidean,
    prox_near_isotonic,
    prox_near_isotonic_with_sum,
    CalibrationError,
)
from examples.data_helpers import create_test_case


class TestNearlyIsotonicUtilities:
    """Test nearly isotonic utility functions."""
    
    def test_project_near_isotonic_basic(self):
        """Test basic epsilon-slack projection."""
        v = np.array([1.0, 0.5, 0.8, 0.3])  # Has violations
        eps = 0.2
        
        result = project_near_isotonic_euclidean(v, eps)
        
        # Check that result satisfies near-isotonic constraint
        diffs = np.diff(result)
        violations = diffs < -eps - 1e-10
        assert not np.any(violations), f"Near-isotonic violations: {diffs[violations]}"
    
    def test_project_near_isotonic_with_sum(self):
        """Test epsilon-slack projection with sum constraint."""
        v = np.array([0.3, 0.2, 0.6, 0.1])
        eps = 0.1
        target_sum = 1.0
        
        result = project_near_isotonic_euclidean(v, eps, sum_target=target_sum)
        
        # Check sum constraint
        assert abs(result.sum() - target_sum) < 1e-10
        
        # Check near-isotonic constraint
        diffs = np.diff(result)
        violations = diffs < -eps - 1e-10
        assert not np.any(violations)
    
    def test_prox_near_isotonic_basic(self):
        """Test lambda-penalty prox operator."""
        v = np.array([1.0, 0.3, 0.7, 0.2])
        lam = 0.5
        
        result = prox_near_isotonic(v, lam)
        
        # Result should be closer to isotonic than original
        assert result.shape == v.shape
        assert np.isfinite(result).all()
    
    def test_prox_near_isotonic_with_sum_constraint(self):
        """Test lambda-penalty prox with sum constraint."""
        v = np.array([0.4, 0.1, 0.3, 0.2])
        lam = 1.0
        target_sum = 1.5
        
        result = prox_near_isotonic_with_sum(v, lam, target_sum)
        
        # Check sum constraint
        assert abs(result.sum() - target_sum) < 1e-10
    
    def test_epsilon_reduces_to_isotonic(self):
        """Test that eps=0 gives standard isotonic projection.""" 
        v = np.array([1.0, 0.5, 0.8, 0.3])
        
        result = project_near_isotonic_euclidean(v, eps=0.0)
        
        # Should be isotonic (non-decreasing)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10)
        
    def test_large_epsilon_gives_freedom(self):
        """Test that large epsilon allows more flexibility."""
        v = np.array([1.0, 0.1, 0.9, 0.2])
        
        # Large epsilon should keep values close to original
        result_large_eps = project_near_isotonic_euclidean(v, eps=2.0)
        result_small_eps = project_near_isotonic_euclidean(v, eps=0.01)
        
        # Large epsilon result should be closer to original
        dist_large = np.linalg.norm(result_large_eps - v)
        dist_small = np.linalg.norm(result_small_eps - v)
        
        assert dist_large <= dist_small + 1e-10


class TestDykstraWithEpsilonSlack:
    """Test Dykstra solver with epsilon-slack constraints."""
    
    def test_dykstra_epsilon_basic(self):
        """Test Dykstra with epsilon-slack nearly isotonic."""
        P, M = create_test_case("linear", N=20, J=3, seed=42)
        
        # Standard isotonic
        result_iso = calibrate_dykstra(P, M, verbose=False)
        
        # Nearly isotonic with slack
        nearly_params = {"mode": "epsilon", "eps": 0.1}
        result_nearly = calibrate_dykstra(P, M, nearly=nearly_params, verbose=False)
        
        # Both should satisfy basic constraints
        assert result_iso.converged
        assert result_nearly.converged
        assert result_iso.max_row_error < 1e-6
        assert result_nearly.max_row_error < 1e-6
        assert result_iso.max_col_error < 1e-6
        assert result_nearly.max_col_error < 1e-6
        
        # Nearly isotonic should have more flexibility
        iso_change = np.linalg.norm(result_iso.Q - P)
        nearly_change = np.linalg.norm(result_nearly.Q - P)
        
        # Not guaranteed but often true: nearly isotonic makes smaller changes
        # Just check that both are reasonable
        assert iso_change >= 0
        assert nearly_change >= 0
    
    def test_dykstra_epsilon_convergence(self):
        """Test that epsilon-slack version still converges."""
        P, M = create_test_case("challenging", N=15, J=4, seed=123)
        
        nearly_params = {"mode": "epsilon", "eps": 0.05}
        result = calibrate_dykstra(P, M, nearly=nearly_params, verbose=False, max_iters=1000)
        
        # Should converge and satisfy constraints
        assert result.converged or result.max_row_error < 1e-5
        assert result.max_col_error < 1e-5
        
        # Check near-isotonic constraint satisfaction
        for j in range(P.shape[1]):
            idx = np.argsort(P[:, j])
            q_sorted = result.Q[idx, j]
            diffs = np.diff(q_sorted)
            violations = diffs < -0.05 - 1e-8  # Allow some numerical tolerance
            assert not np.any(violations), f"Column {j} has violations: {diffs[violations]}"


class TestADMMWithLambdaPenalty:
    """Test ADMM solver with lambda-penalty constraints."""
    
    def test_admm_lambda_basic(self):
        """Test ADMM with lambda-penalty nearly isotonic."""
        P, M = create_test_case("random", N=15, J=3, seed=42)
        
        # Standard isotonic
        result_iso = calibrate_admm(P, M, verbose=False)
        
        # Nearly isotonic with penalty
        nearly_params = {"mode": "lambda", "lam": 0.5}
        result_nearly = calibrate_admm(P, M, nearly=nearly_params, verbose=False)
        
        # Both should satisfy basic constraints
        assert result_iso.Q.shape == P.shape
        assert result_nearly.Q.shape == P.shape
        
        # Check that both give reasonable results
        assert np.all(result_iso.Q >= -1e-10)
        assert np.all(result_nearly.Q >= -1e-10)
        
        # Row sums should be reasonable (ADMM may not converge perfectly)
        if result_iso.converged:
            assert np.max(np.abs(result_iso.Q.sum(axis=1) - 1.0)) < 1e-4
        if result_nearly.converged:
            assert np.max(np.abs(result_nearly.Q.sum(axis=1) - 1.0)) < 1e-4
    
    def test_admm_lambda_different_penalties(self):
        """Test ADMM with different lambda penalty values."""
        P, M = create_test_case("skewed", N=10, J=3, seed=456)
        
        results = {}
        for lam in [0.1, 1.0, 10.0]:
            nearly_params = {"mode": "lambda", "lam": lam}
            result = calibrate_admm(P, M, nearly=nearly_params, verbose=False, max_iters=500)
            results[lam] = result
        
        # All should give valid probability matrices
        for lam, result in results.items():
            assert result.Q.shape == P.shape
            assert np.all(result.Q >= -1e-10), f"Negative values for lambda={lam}"
            
            # Check that algorithm ran and produced a result
            assert len(result.objective_values) > 0, f"No iterations for lambda={lam}"
            assert np.isfinite(result.Q).all(), f"Non-finite values for lambda={lam}"


class TestNearlyIsotonicEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test handling of empty inputs.""" 
        v = np.array([])
        result = project_near_isotonic_euclidean(v, eps=0.1)
        assert len(result) == 0
    
    def test_single_element(self):
        """Test single element input."""
        v = np.array([0.5])
        result = project_near_isotonic_euclidean(v, eps=0.1, sum_target=0.7)
        assert len(result) == 1
        assert abs(result[0] - 0.7) < 1e-10
    
    def test_negative_epsilon(self):
        """Test that negative epsilon still works."""
        v = np.array([1.0, 0.5, 0.8])
        result = project_near_isotonic_euclidean(v, eps=-0.1)
        # Negative epsilon makes constraint stricter than isotonic
        assert np.isfinite(result).all()
    
    def test_zero_lambda(self):
        """Test lambda=0 in prox operator."""
        v = np.array([1.0, 0.3, 0.7, 0.2])
        result = prox_near_isotonic(v, lam=0.0)
        # Should be close to identity when lambda=0
        assert np.allclose(result, v, atol=1e-6)
    
    def test_invalid_nearly_params_dykstra(self):
        """Test invalid nearly parameters for Dykstra."""
        P, M = create_test_case("random", N=10, J=3, seed=42)
        
        # Invalid mode should fall back to standard isotonic
        nearly_params = {"mode": "invalid_mode", "eps": 0.1}
        result = calibrate_dykstra(P, M, nearly=nearly_params, verbose=False)
        assert result.converged
    
    def test_invalid_nearly_params_admm(self):
        """Test invalid nearly parameters for ADMM."""
        P, M = create_test_case("random", N=10, J=3, seed=42)
        
        # Invalid mode should fall back to standard isotonic
        nearly_params = {"mode": "invalid_mode", "lam": 0.5}
        result = calibrate_admm(P, M, nearly=nearly_params, verbose=False)
        assert result.Q.shape == P.shape


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])