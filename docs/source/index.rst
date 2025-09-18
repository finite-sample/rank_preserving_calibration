Rank-Preserving Calibration Documentation
==========================================

.. image:: https://github.com/finite-sample/rank_preserving_calibration/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/finite-sample/rank_preserving_calibration/actions/workflows/ci.yml
   :alt: Python application

.. image:: https://img.shields.io/pypi/v/rank_preserving_calibration.svg
   :target: https://pypi.org/project/rank_preserving_calibration/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/rank_preserving_calibration.svg
   :target: https://pypi.org/project/rank_preserving_calibration/
   :alt: Python versions

Welcome to the documentation for **rank_preserving_calibration**, a Python package for rank-preserving calibration of multiclass probabilities. This package implements algorithms to project probability matrices onto the intersection of row-simplex constraints and isotonic column marginals.

Key Features
------------

* **Two main algorithms**: Dykstra's alternating projections (recommended) and ADMM optimization
* **Rank preservation**: Maintains the original ordering of predictions while satisfying calibration constraints
* **Nearly isotonic calibration**: Support for relaxed isotonic constraints with epsilon-slack or lambda-penalty approaches
* **Robust implementation**: Numerically stable algorithms with comprehensive error handling
* **Comprehensive testing**: Extensive test suite ensuring mathematical correctness

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install rank_preserving_calibration

Basic usage:

.. code-block:: python

   import numpy as np
   from rank_preserving_calibration import calibrate_dykstra

   # Your probability matrix (N samples × J classes)
   P = np.random.dirichlet([1, 1, 1], size=100)
   
   # Target column sums
   M = np.array([30.0, 40.0, 30.0])
   
   # Calibrate probabilities
   result = calibrate_dykstra(P, M)
   calibrated_probs = result.Q

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   theory
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`