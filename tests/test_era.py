"""
Test Eigensystem Realization Algorithm (ERA)
"""

import numpy as np
import pytest
from time_domain_modal_estimation import (
    eigensystem_realization_algorithm,
    generate_impulse_response,
    build_hankel_matrix
)


@pytest.fixture
def single_mode_params():
    """Fixture for single mode parameters."""
    return {
        'frequencies': [5.0],
        'damping': [0.03],
        'mode_shapes': [[1.0]],
        'dt': 0.01
    }


@pytest.fixture
def multi_mode_params():
    """Fixture for multiple mode parameters."""
    return {
        'frequencies': [2.5, 5.0, 8.0],
        'damping': [0.02, 0.03, 0.05],
        'mode_shapes': [[1.0], [0.8], [0.5]],
        'dt': 0.01
    }


def test_era_single_mode(single_mode_params):
    """Test ERA with a single mode."""
    params = single_mode_params
    t = np.arange(0, 5, params['dt'])
    
    # Generate impulse response
    Y = generate_impulse_response(
        params['frequencies'],
        params['damping'],
        params['mode_shapes'],
        t,
        n_outputs=1
    )
    
    # Apply ERA
    results = eigensystem_realization_algorithm(
        Y=Y,
        dt=params['dt'],
        n_modes=1,
        r=100,
        s=100
    )
    
    # Assert frequency estimation within 1%
    freq_error = abs(results['frequencies'][0] - params['frequencies'][0]) / params['frequencies'][0]
    assert freq_error < 0.01, f"Frequency error {freq_error*100:.2f}% exceeds 1%"
    
    # Assert damping estimation within 10%
    damp_error = abs(results['damping_ratios'][0] - params['damping'][0]) / params['damping'][0]
    assert damp_error < 0.10, f"Damping error {damp_error*100:.2f}% exceeds 10%"
    
    # Check output structure
    assert len(results['frequencies']) == 1
    assert len(results['damping_ratios']) == 1
    assert results['mode_shapes'].shape[1] == 1
    assert results['A'].shape[0] > 0
    assert results['B'].shape[0] > 0
    assert results['C'].shape[0] > 0


def test_era_multi_mode(multi_mode_params):
    """Test ERA with multiple modes."""
    params = multi_mode_params
    t = np.arange(0, 5, params['dt'])
    
    # Generate impulse response
    Y = generate_impulse_response(
        params['frequencies'],
        params['damping'],
        params['mode_shapes'],
        t,
        n_outputs=1
    )
    
    # Apply ERA
    results = eigensystem_realization_algorithm(
        Y=Y,
        dt=params['dt'],
        n_modes=3,
        r=min(len(t)//3, 150),
        s=min(len(t)//3, 150)
    )
    
    # Sort by frequency
    sort_idx = np.argsort(np.abs(results['frequencies']))
    freqs_est = np.abs(results['frequencies'][sort_idx])
    damp_est = np.abs(results['damping_ratios'][sort_idx])
    
    # Assert frequency estimation
    for i in range(3):
        freq_error = abs(freqs_est[i] - params['frequencies'][i]) / params['frequencies'][i]
        assert freq_error < 0.02, f"Mode {i+1} frequency error {freq_error*100:.2f}% exceeds 2%"
    
    # Assert damping estimation (more tolerance for multiple modes)
    for i in range(3):
        damp_error = abs(damp_est[i] - params['damping'][i]) / params['damping'][i]
        assert damp_error < 0.20, f"Mode {i+1} damping error {damp_error*100:.2f}% exceeds 20%"
    
    # Check output structure
    assert len(results['frequencies']) == 3
    assert len(results['damping_ratios']) == 3
    assert results['mode_shapes'].shape[1] == 3


def test_era_with_noise(single_mode_params):
    """Test ERA robustness with noisy data."""
    params = single_mode_params
    t = np.arange(0, 5, params['dt'])
    
    # Generate impulse response
    Y = generate_impulse_response(
        params['frequencies'],
        params['damping'],
        params['mode_shapes'],
        t,
        n_outputs=1
    )
    
    # Add noise
    noise_level = 0.01
    Y += np.random.normal(0, noise_level, Y.shape)
    
    # Apply ERA
    results = eigensystem_realization_algorithm(
        Y=Y,
        dt=params['dt'],
        n_modes=1,
        r=100,
        s=100
    )
    
    # With noise, allow more tolerance
    freq_error = abs(results['frequencies'][0] - params['frequencies'][0]) / params['frequencies'][0]
    assert freq_error < 0.05, f"Frequency error {freq_error*100:.2f}% exceeds 5%"


def test_build_hankel_matrix():
    """Test Hankel matrix construction."""
    # Simple impulse response
    Y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    H = build_hankel_matrix(Y, r=3, s=4)
    
    # Check shape
    assert H.shape == (3, 4)
    
    # Check structure (Hankel property: constant along anti-diagonals)
    expected = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ])
    np.testing.assert_array_equal(H, expected)


def test_generate_impulse_response():
    """Test impulse response generation."""
    frequencies = [5.0]
    damping = [0.03]
    mode_shapes = [[1.0]]
    t = np.arange(0, 1, 0.01)
    
    Y = generate_impulse_response(frequencies, damping, mode_shapes, t, n_outputs=1)
    
    # Check shape
    assert Y.shape == (1, len(t))
    
    # Check it's decaying
    assert Y[0, 0] > abs(Y[0, -1])
    
    # Check initial amplitude is reasonable
    assert Y[0, 0] > 0


def test_era_state_space_matrices(single_mode_params):
    """Test that ERA produces valid state-space matrices."""
    params = single_mode_params
    t = np.arange(0, 3, params['dt'])
    
    Y = generate_impulse_response(
        params['frequencies'],
        params['damping'],
        params['mode_shapes'],
        t,
        n_outputs=1
    )
    
    results = eigensystem_realization_algorithm(
        Y=Y,
        dt=params['dt'],
        n_modes=1,
        r=50,
        s=50
    )
    
    # Check matrix dimensions are consistent
    n_states = results['A'].shape[0]
    n_outputs = results['C'].shape[0]
    
    assert results['A'].shape == (n_states, n_states)
    # B can be either (n_states,) or (n_outputs, n_states) depending on formulation
    assert results['B'].size >= n_states
    assert results['C'].shape == (n_outputs, n_states)
    
    # Check eigenvalues are within unit circle (stable system)
    eigenvalues = np.linalg.eigvals(results['A'])
    assert np.all(np.abs(eigenvalues) <= 1.0), "System eigenvalues outside unit circle"


@pytest.mark.parametrize("n_modes", [1, 2, 3])
def test_era_different_model_orders(n_modes):
    """Test ERA with different model orders."""
    frequencies = [3.0, 6.0, 9.0][:n_modes]
    damping = [0.02, 0.03, 0.04][:n_modes]
    mode_shapes = [[1.0], [0.8], [0.6]][:n_modes]
    
    t = np.arange(0, 5, 0.01)
    
    Y = generate_impulse_response(frequencies, damping, mode_shapes, t, n_outputs=1)
    
    results = eigensystem_realization_algorithm(
        Y=Y,
        dt=0.01,
        n_modes=n_modes,
        r=100,
        s=100
    )
    
    # Check correct number of modes identified
    assert len(results['frequencies']) == n_modes
    assert len(results['damping_ratios']) == n_modes
