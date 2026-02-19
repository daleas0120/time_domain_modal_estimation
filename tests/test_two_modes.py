"""
Test CEA with two modes
"""

import numpy as np
import pytest
from time_domain_modal_estimation import complex_exponential_algorithm


def test_two_modes_clean_data():
    """Test CEA with two well-separated modes."""
    # Generate test case with TWO modes
    dt = 0.01
    t = np.arange(0, 3, dt)
    
    # True parameters for TWO modes
    f_true = np.array([2.5, 5.0])  # Hz
    zeta_true = np.array([0.02, 0.03])
    A_true = [1.0, 0.6]
    
    # Generate response
    y = np.zeros(len(t))
    for f, zeta, A in zip(f_true, zeta_true, A_true):
        omega_n = 2 * np.pi * f
        lambda_k = -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta**2)
        # Add both conjugate modes
        y_mode = np.real(A * np.exp(lambda_k * t) + np.conj(A) * np.exp(np.conj(lambda_k) * t))
        y += y_mode
    
    # Apply CEA
    results = complex_exponential_algorithm(y, dt, n_modes=2)
    
    # Sort by frequency for comparison
    sort_idx = np.argsort(results['frequencies'])
    freqs_est = results['frequencies'][sort_idx]
    damp_est = results['damping_ratios'][sort_idx]
    
    # Assert frequency estimation within 1%
    for i in range(2):
        freq_error = abs(freqs_est[i] - f_true[i]) / f_true[i]
        assert freq_error < 0.01, f"Mode {i+1} frequency error {freq_error*100:.2f}% exceeds 1%"
    
    # Assert damping estimation within 10%
    for i in range(2):
        damp_error = abs(damp_est[i] - zeta_true[i]) / zeta_true[i]
        assert damp_error < 0.10, f"Mode {i+1} damping error {damp_error*100:.2f}% exceeds 10%"
    
    # Assert reconstruction quality
    recon_error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
    assert recon_error < 1e-6, f"Reconstruction error {recon_error:.2e} exceeds tolerance"
    
    # Assert output shapes
    assert len(results['frequencies']) == 2
    assert len(results['damping_ratios']) == 2
    assert len(results['mode_shapes']) == 2


def test_two_modes_different_amplitudes():
    """Test CEA with two modes having different amplitudes."""
    dt = 0.01
    t = np.arange(0, 2, dt)
    
    # Second mode much smaller
    f_true = np.array([3.0, 7.0])
    zeta_true = np.array([0.025, 0.035])
    A_true = [1.0, 0.2]
    
    y = np.zeros(len(t))
    for f, zeta, A in zip(f_true, zeta_true, A_true):
        omega_n = 2 * np.pi * f
        lambda_k = -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta**2)
        y += np.real(A * np.exp(lambda_k * t) + np.conj(A) * np.exp(np.conj(lambda_k) * t))
    
    results = complex_exponential_algorithm(y, dt, n_modes=2)
    
    sort_idx = np.argsort(results['frequencies'])
    freqs_est = results['frequencies'][sort_idx]
    
    # Both modes should be identified
    freq_errors = np.abs(freqs_est - f_true) / f_true
    assert np.all(freq_errors < 0.02), "One or more modes not accurately identified"
