Complex Exponential Algorithm Module
=====================================

.. automodule:: time_domain_modal_estimation.complex_exp
   :members:
   :undoc-members:
   :show-inheritance:

Main Function
-------------

.. autofunction:: time_domain_modal_estimation.complex_exp.complex_exponential_algorithm

Helper Functions
----------------

Building Matrices
~~~~~~~~~~~~~~~~~

.. autofunction:: time_domain_modal_estimation.complex_exp.build_toeplitz_matrix

.. autofunction:: time_domain_modal_estimation.complex_exp.build_vandermonde_matrix

Solving for Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: time_domain_modal_estimation.complex_exp.solve_polynomial_coefficients

.. autofunction:: time_domain_modal_estimation.complex_exp.find_system_poles

.. autofunction:: time_domain_modal_estimation.complex_exp.poles_to_modal_frequencies

.. autofunction:: time_domain_modal_estimation.complex_exp.solve_mode_shapes

Reconstruction
~~~~~~~~~~~~~~

.. autofunction:: time_domain_modal_estimation.complex_exp.reconstruct_response
