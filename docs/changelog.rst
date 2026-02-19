Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Version 0.1.0 (2026-02-18)
--------------------------

Initial release.

Added
~~~~~

* **Complex Exponential Algorithm (CEA)** implementation

  * ``complex_exponential_algorithm()`` - Main function for modal parameter estimation
  * ``build_toeplitz_matrix()`` - Construct Toeplitz matrix from time series
  * ``solve_polynomial_coefficients()`` - Least squares solution for characteristic polynomial
  * ``find_system_poles()`` - Extract poles from characteristic polynomial
  * ``poles_to_modal_frequencies()`` - Convert poles to modal frequencies
  * ``build_vandermonde_matrix()`` - Construct modal matrix
  * ``solve_mode_shapes()`` - Extract modal participation factors
  * ``reconstruct_response()`` - Reconstruct time series from modal parameters

* **Documentation**

  * Complete API documentation with NumPy-style docstrings
  * Theoretical background and algorithm description
  * Quick start guide and examples
  * Sphinx documentation with Read the Docs theme

* **Examples**

  * ``basic_usage.py`` - Simple single-mode example
  * ``demo_cea.py`` - Comprehensive demonstration with visualization
  * Test scripts for validation

* **Testing**

  * Test suite with multiple validation cases
  * Single-mode tests
  * Multi-mode tests
  * Phase handling tests

* **Package Infrastructure**

  * Modern ``pyproject.toml`` configuration
  * Proper package structure with ``src/`` layout
  * Development and documentation dependencies
  * Build system configuration
  * README with installation and usage instructions
  * MIT License

Known Limitations
~~~~~~~~~~~~~~~~~

* Algorithm is sensitive to measurement noise
* Requires manual selection of number of modes
* May struggle with closely spaced modes
* Assumes proportional damping for real mode shapes

Future Plans
------------

Version 0.2.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

* Automatic model order selection
* Stabilization diagrams
* Multi-reference implementation
* Additional time-domain methods (ERA, ITD)

Version 0.3.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

* Frequency-domain methods
* Uncertainty quantification
* Signal pre-processing utilities
* Extended visualization tools

Contributing
~~~~~~~~~~~~

Contributions are welcome! Please see the repository for guidelines.
