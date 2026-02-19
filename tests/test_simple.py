"""
Test CEA with clean exponential data (single mode)
"""

import numpy as np
import pytest
from time_domain_modal_estimation import complex_exponential_algorithm


def test_single_mode_clean_data():
    """Test CEA with a single clean exponential mode."""
    # Generate simple test case with known parameters
    dt = 0.01
    t = np.arange(0, 2, dt)  # 2 seconds
    
    # True parameters for ONE mode
    f_true = 5.0  # Hz
    zeta_true = 0.05
    omega_n_true = 2 * np.pi * f_true
    lambda_true = -zeta_true * omega_n_true + 1j * omega_n_true * np.sqrt(1 - zeta_true**2)
    A_true = 1.0
    
    # Generate response with BOTH complex conjugate modes
    y = np.real(A_true * np.exp(lambda_true * t) + np.conj(A_true) * np.exp(np.conj(lambda_true) * t))
    
    # Apply CEA
    results = complex_exponential_algorithm(y, dt, n_modes=1)
    
    # Assert frequency estimation within 1%
    freq_error = abs(results['frequencies'][0] - f_true) / f_true
    assert freq_error < 0.01, f"Frequency error {freq_error*100:.2f}% exceeds 1%"
    
    # Assert damping estimation within 5%
    damp_error = abs(results['damping_ratios'][0] - zeta_true) / zeta_true
    assert damp_error < 0.05, f"Damping error {damp_error*100:.2f}% exceeds 5%"
    
    # Assert reconstruction quality
    recon_error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
    assert recon_error < 1e-10, f"Reconstruction error {recon_error:.2e} exceeds tolerance"
    
    # Assert output shapes
    assert len(results['frequencies']) == 1
    assert len(results['damping_ratios']) == 1
    assert len(results['mode_shapes']) == 1
    assert results['lambda'].shape[0] == 1


def test_single_mode_reconstruction():
    """Test that single mode reconstruction is accurate."""
    dt = 0.01
    t = np.arange(0, 1, dt)
    
    f_true = 3.0
    zeta_true = 0.03
    omega_n = 2 * np.pi * f_true
    lambda_k = -zeta_true * omega_n + 1j * omega_n * np.sqrt(1 - zeta_true**2)
    
    y = np.real(np.exp(lambda_k * t) + np.exp(np.conj(lambda_k) * t))
    
    results = complex_exponential_algorithm(y, dt, n_modes=1)
    
    # Reconstruction should be nearly perfect for clean data
    recon_error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
    assert recon_error < 1e-6
