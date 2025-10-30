Installation
============

Requirements
------------

* Python 3.11 or higher
* NumPy 1.18 or higher

Basic Installation
------------------

Install from PyPI using pip:

.. code-block:: bash

   pip install rank_preserving_calibration

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/finite_sample/rank_preserving_calibration.git
   cd rank_preserving_calibration
   pip install -e .

Optional Dependencies
---------------------

For running examples and notebooks:

.. code-block:: bash

   pip install rank_preserving_calibration[examples]

For development and testing:

.. code-block:: bash

   pip install rank_preserving_calibration[testing]

For all optional dependencies:

.. code-block:: bash

   pip install rank_preserving_calibration[all]

Verification
------------

To verify your installation, run:

.. code-block:: python

   import rank_preserving_calibration
   print(rank_preserving_calibration.__version__)

You can also run the test suite:

.. code-block:: bash

   python -m pytest tests/ -v