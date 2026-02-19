"""
Test CEA with phase offset
"""

import numpy as np
import pytest
from time_domain_modal_estimation import complex_exponential_algorithm


def test_single_mode_with_phase():
    """Test CEA with a single mode having phase offset."""
    # Generate test case with phase
    dt = 0.01
    t = np.arange(0, 3, dt)
    
    # True parameters
    f_true = 5.0  # Hz
    zeta_true = 0.03
    omega_n_true = 2 * np.pi * f_true
    omega_d_true = omega_n_true * np.sqrt(1 - zeta_true**2)
    A_true = 1.0
    phi_true = np.pi / 4
    
    # Generate response with cosine (like in demo)
    y = A_true * np.exp(-zeta_true * omega_n_true * t) * np.cos(omega_d_true * t + phi_true)
    
    # Apply CEA
    results = complex_exponential_algorithm(y, dt, n_modes=1)
    
    # Assert frequency estimation
    freq_error = abs(results['frequencies'][0] - f_true) / f_true
    assert freq_error < 0.01, f"Frequency error {freq_error*100:.2f}% exceeds 1%"
    
    # Assert damping estimation
    damp_error = abs(results['damping_ratios'][0] - zeta_true) / zeta_true
    assert damp_error < 0.05, f"Damping error {damp_error*100:.2f}% exceeds 5%"
    
    # Check reconstruction
    recon_error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
    assert recon_error < 1e-6, f"Reconstruction error {recon_error:.2e} exceeds tolerance"


@pytest.mark.parametrize("phase", [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
def test_various_phase_offsets(phase):
    """Test CEA with various phase offsets."""
    dt = 0.01
    t = np.arange(0, 2, dt)
    
    f_true = 4.0
    zeta_true = 0.04
    omega_n = 2 * np.pi * f_true
    omega_d = omega_n * np.sqrt(1 - zeta_true**2)
    
    y = np.exp(-zeta_true * omega_n * t) * np.cos(omega_d * t + phase)
    
    results = complex_exponential_algorithm(y, dt, n_modes=1)
    
    # Frequency should be accurate regardless of phase
    freq_error = abs(results['frequencies'][0] - f_true) / f_true
    assert freq_error < 0.01, f"Phase {phase:.2f} rad: frequency error too large"
