Examples and Use Cases
=====================

This section provides practical examples of using rank-preserving calibration in different scenarios.

Basic Example: Overconfident Classifier
----------------------------------------

A common use case is calibrating an overconfident multiclass classifier:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from rank_preserving_calibration import calibrate_dykstra

   # Create overconfident predictions
   np.random.seed(42)
   n_samples, n_classes = 500, 3
   
   # Generate predictions that are too concentrated
   true_labels = np.random.choice(n_classes, n_samples, p=[0.4, 0.35, 0.25])
   
   # Overconfident probabilities
   P = np.random.dirichlet([0.1, 0.1, 0.1], size=n_samples)
   for i, label in enumerate(true_labels):
       P[i, label] = 0.7 + 0.2 * np.random.random()  # Make correct class very confident
       remaining = 1 - P[i, label]
       other_classes = [j for j in range(n_classes) if j != label]
       P[i, other_classes] = np.random.dirichlet([1, 1], size=len(other_classes)) * remaining

   # Target marginals from validation set
   M = np.array([200.0, 175.0, 125.0])  # True class distribution
   
   # Calibrate
   result = calibrate_dykstra(P, M, verbose=True)
   
   print(f"Original column sums: {P.sum(axis=0)}")
   print(f"Calibrated column sums: {result.Q.sum(axis=0)}")
   print(f"Target column sums: {M}")

Comparing Algorithms
--------------------

Compare Dykstra's method with ADMM:

.. code-block:: python

   from rank_preserving_calibration import calibrate_dykstra, calibrate_admm
   import time

   # Use same data as above
   
   # Dykstra's method
   start_time = time.time()
   result_dykstra = calibrate_dykstra(P, M, max_iters=3000, tol=1e-7)
   dykstra_time = time.time() - start_time
   
   # ADMM method  
   start_time = time.time()
   result_admm = calibrate_admm(P, M, max_iters=1000, tol=1e-6)
   admm_time = time.time() - start_time
   
   print(f"Dykstra: {result_dykstra.n_iter} iterations, {dykstra_time:.3f}s")
   print(f"ADMM: {result_admm.n_iter} iterations, {admm_time:.3f}s")
   print(f"Final errors - Dykstra: {result_dykstra.final_error:.2e}, ADMM: {result_admm.final_error:.2e}")

Nearly Isotonic Calibration
----------------------------

When strict isotonic constraints are too restrictive:

.. code-block:: python

   # Create a challenging case where strict isotonicity might be problematic
   np.random.seed(123)
   P_challenging = np.random.dirichlet([1, 1, 1], size=100)
   
   # Add some noise that violates isotonic ordering
   noise = 0.1 * np.random.normal(0, 1, P_challenging.shape)
   P_challenging = np.maximum(0, P_challenging + noise)
   P_challenging = P_challenging / P_challenging.sum(axis=1, keepdims=True)
   
   M_challenging = np.array([35.0, 35.0, 30.0])
   
   # Strict isotonic calibration
   result_strict = calibrate_dykstra(P_challenging, M_challenging)
   
   # Nearly isotonic with epsilon-slack
   nearly_params = {"mode": "epsilon", "eps": 0.05}
   result_nearly = calibrate_dykstra(P_challenging, M_challenging, nearly=nearly_params)
   
   print(f"Strict isotonic error: {result_strict.final_error:.2e}")
   print(f"Nearly isotonic error: {result_nearly.final_error:.2e}")
   
   # Compare distances from original
   strict_distance = np.linalg.norm(result_strict.Q - P_challenging, 'fro')
   nearly_distance = np.linalg.norm(result_nearly.Q - P_challenging, 'fro')
   
   print(f"Distance from original - Strict: {strict_distance:.3f}, Nearly: {nearly_distance:.3f}")

Realistic Classification Scenario
----------------------------------

Working with actual classifier outputs:

.. code-block:: python

   from scipy.special import softmax

   # Simulate neural network logits
   np.random.seed(456)
   n_samples, n_classes = 1000, 5
   
   # Create features and true labels
   X = np.random.normal(0, 1, (n_samples, 10))
   true_labels = np.random.choice(n_classes, n_samples)
   
   # Simulate biased classifier (overconfident, biased toward certain classes)
   W = np.random.normal(0, 0.5, (10, n_classes))
   W[:, 0] += 0.5  # Bias toward class 0
   W[:, 1] -= 0.3  # Bias against class 1
   
   logits = X @ W + np.random.normal(0, 0.1, (n_samples, n_classes))
   P_realistic = softmax(logits, axis=1)
   
   # True class proportions (from held-out validation set)
   true_proportions = np.array([0.18, 0.22, 0.20, 0.20, 0.20])
   M_realistic = true_proportions * n_samples
   
   print(f"Original class proportions: {P_realistic.sum(axis=0) / n_samples}")
   print(f"Target class proportions: {true_proportions}")
   
   # Calibrate
   result = calibrate_dykstra(P_realistic, M_realistic, max_iters=5000)
   
   print(f"Calibrated class proportions: {result.Q.sum(axis=0) / n_samples}")
   
   # Check rank preservation
   original_rankings = P_realistic.argsort(axis=1)
   calibrated_rankings = result.Q.argsort(axis=1) 
   rank_agreement = np.mean(original_rankings == calibrated_rankings)
   
   print(f"Rank agreement: {rank_agreement:.3f}")

Handling Edge Cases
-------------------

Dealing with infeasible or near-infeasible problems:

.. code-block:: python

   # Case 1: Slightly infeasible target sums
   P_edge = np.random.dirichlet([2, 1, 1], size=100)
   M_infeasible = np.array([30.0, 35.0, 40.0])  # Sums to 105, not 100
   
   print(f"Target sum: {M_infeasible.sum()}, Number of samples: {P_edge.shape[0]}")
   
   # This will issue a feasibility warning but still attempt calibration
   result_infeasible = calibrate_dykstra(P_edge, M_infeasible)
   print(f"Final error: {result_infeasible.final_error:.2e}")
   
   # Case 2: Using nearly isotonic to handle difficult constraints
   nearly_params = {"mode": "epsilon", "eps": 0.1}
   result_nearly_infeasible = calibrate_dykstra(P_edge, M_infeasible, nearly=nearly_params)
   print(f"Nearly isotonic final error: {result_nearly_infeasible.final_error:.2e}")

Performance Monitoring
----------------------

Tracking convergence and performance:

.. code-block:: python

   # Use ADMM to get detailed convergence history
   result_admm = calibrate_admm(P, M, verbose=True, max_iters=1000)
   
   # Plot convergence
   import matplotlib.pyplot as plt
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
   
   # Primal and dual residuals
   ax1.semilogy(result_admm.primal_residuals, label='Primal residual')
   ax1.semilogy(result_admm.dual_residuals, label='Dual residual')
   ax1.set_xlabel('Iteration')
   ax1.set_ylabel('Residual')
   ax1.legend()
   ax1.set_title('ADMM Convergence')
   
   # Objective function values
   ax2.plot(result_admm.objective_values)
   ax2.set_xlabel('Iteration')
   ax2.set_ylabel('Objective Value')
   ax2.set_title('Objective Function')
   
   plt.tight_layout()
   plt.show()

Batch Processing
----------------

Processing multiple calibration problems efficiently:

.. code-block:: python

   # Multiple datasets with different target marginals
   datasets = []
   targets = []
   
   for i in range(5):
       # Generate different problems
       P_i = np.random.dirichlet([2, 1, 1], size=100)
       M_i = np.random.dirichlet([1, 1, 1]) * 100  # Random but valid targets
       
       datasets.append(P_i)
       targets.append(M_i)
   
   # Process each dataset
   results = []
   for i, (P_i, M_i) in enumerate(zip(datasets, targets)):
       print(f"Processing dataset {i+1}/5...")
       result = calibrate_dykstra(P_i, M_i, verbose=False)
       results.append(result)
       
       if result.converged:
           print(f"  Converged in {result.n_iter} iterations")
       else:
           print(f"  Did not converge (error: {result.final_error:.2e})")

Integration with scikit-learn
-----------------------------

Using rank-preserving calibration in a machine learning pipeline:

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_classification

   # Generate synthetic dataset
   X, y = make_classification(n_samples=1000, n_features=10, n_classes=4, 
                             n_informative=8, random_state=42)
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
   # Train classifier
   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)
   
   # Get uncalibrated probabilities
   P_uncalibrated = clf.predict_proba(X_test)
   
   # Estimate true class proportions from validation set (in practice, use separate data)
   true_proportions = np.bincount(y_test) / len(y_test)
   M_estimated = true_proportions * len(X_test)
   
   # Apply rank-preserving calibration
   result = calibrate_dykstra(P_uncalibrated, M_estimated)
   P_calibrated = result.Q
   
   # Compare predictions
   pred_uncalibrated = P_uncalibrated.argmax(axis=1)
   pred_calibrated = P_calibrated.argmax(axis=1)
   
   from sklearn.metrics import accuracy_score, classification_report
   
   print("Uncalibrated accuracy:", accuracy_score(y_test, pred_uncalibrated))
   print("Calibrated accuracy:", accuracy_score(y_test, pred_calibrated))
   print("\nClass proportions:")
   print("True:", true_proportions)
   print("Uncalibrated:", P_uncalibrated.sum(axis=0) / len(X_test))
   print("Calibrated:", P_calibrated.sum(axis=0) / len(X_test))