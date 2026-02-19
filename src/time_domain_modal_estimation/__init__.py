"""
Time Domain Modal Estimation

A Python package for extracting modal parameters from time-domain response data.
Implements techniques such as the Complex Exponential Algorithm (CEA) and
Eigensystem Realization Algorithm (ERA).
"""

from .complex_exp import (
    complex_exponential_algorithm,
    build_toeplitz_matrix,
    solve_polynomial_coefficients,
    find_system_poles,
    poles_to_modal_frequencies,
    build_vandermonde_matrix,
    solve_mode_shapes,
    reconstruct_response,
)

from .era import (
    eigensystem_realization_algorithm,
    build_hankel_matrix,
    generate_impulse_response,
    stabilization_diagram,
)

__version__ = "0.1.0"
__author__ = "Ashley"

__all__ = [
    # CEA functions
    "complex_exponential_algorithm",
    "build_toeplitz_matrix",
    "solve_polynomial_coefficients",
    "find_system_poles",
    "poles_to_modal_frequencies",
    "build_vandermonde_matrix",
    "solve_mode_shapes",
    "reconstruct_response",
    # ERA functions
    "eigensystem_realization_algorithm",
    "build_hankel_matrix",
    "generate_impulse_response",
    "stabilization_diagram",
]
