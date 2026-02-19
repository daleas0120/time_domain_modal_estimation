"""
Test CEA with demo parameters (three modes)
"""

import numpy as np
import pytest
from time_domain_modal_estimation import complex_exponential_algorithm


@pytest.fixture
def demo_parameters():
    """Fixture providing demo parameters."""
    return {
        'dt': 0.01,
        'duration': 10.0,
        'frequencies': np.array([2.5, 5.0, 8.3]),
        'damping': np.array([0.02, 0.03, 0.05]),
        'amplitudes': [1.0, 0.6, 0.3],
        'phases': [0.0, np.pi/4, -np.pi/6]
    }


def test_three_modes_demo_params(demo_parameters):
    """Test CEA with three modes using demo parameters."""
    np.random.seed(42)
    
    params = demo_parameters
    t = np.arange(0, params['duration'], params['dt'])
    
    # Generate response (same as demo)
    y = np.zeros_like(t)
    for f, zeta, A, phi in zip(params['frequencies'], params['damping'], 
                                params['amplitudes'], params['phases']):
        omega_n = 2 * np.pi * f
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        y += A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    
    # Apply CEA
    results = complex_exponential_algorithm(y, params['dt'], n_modes=3)
    
    # Sort by frequency
    sort_idx = np.argsort(results['frequencies'])
    freqs_est = results['frequencies'][sort_idx]
    damp_est = results['damping_ratios'][sort_idx]
    
    # Assert frequency estimation within 2%
    for i in range(3):
        freq_error = abs(freqs_est[i] - params['frequencies'][i]) / params['frequencies'][i]
        assert freq_error < 0.02, f"Mode {i+1} frequency error {freq_error*100:.2f}% exceeds 2%"
    
    # Assert damping estimation within 20%
    for i in range(3):
        damp_error = abs(damp_est[i] - params['damping'][i]) / params['damping'][i]
        assert damp_error < 0.20, f"Mode {i+1} damping error {damp_error*100:.2f}% exceeds 20%"
    
    # Assert reconstruction quality
    recon_error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
    assert recon_error < 1e-4, f"Reconstruction error {recon_error:.2e} exceeds tolerance"
    
    # Assert good R² score
    r_squared = 1 - recon_error**2
    assert r_squared > 0.99, f"R² score {r_squared:.4f} below threshold"


def test_three_modes_output_structure(demo_parameters):
    """Test that output structure is correct for three modes."""
    params = demo_parameters
    t = np.arange(0, params['duration'], params['dt'])
    
    y = np.zeros_like(t)
    for f, zeta, A, phi in zip(params['frequencies'], params['damping'],
                                params['amplitudes'], params['phases']):
        omega_n = 2 * np.pi * f
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        y += A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    
    results = complex_exponential_algorithm(y, params['dt'], n_modes=3)
    
    # Check output structure
    assert len(results['frequencies']) == 3
    assert len(results['damping_ratios']) == 3
    assert len(results['mode_shapes']) == 3
    assert results['lambda'].shape[0] == 3
    assert results['reconstruction'].shape == y.shape
    assert 'poles' in results
