Time Domain Modal Estimation Documentation
============================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

A Python package for extracting modal parameters from time-domain response data using the Complex Exponential Algorithm (CEA) and Eigensystem Realization Algorithm (ERA).

Overview
--------

This package implements time-domain modal estimation techniques for identifying modal parameters (natural frequencies, damping ratios, and mode shapes) from measured response data. Two primary methods are implemented:

1. **Complex Exponential Algorithm (CEA)** - for free decay response data
2. **Eigensystem Realization Algorithm (ERA)** - for impulse response data

Both methods are derived from:

    Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques. 
    *Experimental Techniques*, 22(6), 45-49.

Features
--------

* **Complex Exponential Algorithm (CEA)**: Extract modal parameters from free decay response
* **Eigensystem Realization Algorithm (ERA)**: Extract modal parameters from impulse response via Hankel matrices and SVD
* **Automatic pole identification**: Find system poles from characteristic polynomial or eigenvalues
* **Modal parameter extraction**: Calculate natural frequencies, damping ratios, and mode shapes
* **Response reconstruction**: Validate identified models by reconstructing the original response
* **Stabilization diagrams**: Assist with model order selection

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Generate synthetic response data
   dt = 0.01  # Time step (seconds)
   t = np.arange(0, 5, dt)
   f = 5.0  # Frequency (Hz)
   zeta = 0.03  # Damping ratio
   omega_n = 2 * np.pi * f
   y = np.exp(-zeta * omega_n * t) * np.cos(omega_n * np.sqrt(1 - zeta**2) * t)

   # Apply CEA algorithm
   results = complex_exponential_algorithm(y=y, dt=dt, n_modes=1)

   print(f"Estimated frequency: {results['frequencies'][0]:.4f} Hz")
   print(f"Estimated damping: {results['damping_ratios'][0]:.4f}")

Installation
------------

From source:

.. code-block:: bash

   git clone https://github.com/daleas0120/time_domain_modal_estimation.git
   cd time_domain_modal_estimation
   pip install -e .

With documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

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

   api/modules
   api/complex_exp
   api/era

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   references
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
