"""
Demonstration of Eigensystem Realization Algorithm (ERA)
Using synthetic impulse response data with known modal parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from time_domain_modal_estimation import (
    eigensystem_realization_algorithm,
    generate_impulse_response
)


def print_comparison(true_params: dict, estimated_params: dict):
    """
    Print comparison between true and estimated parameters.
    """
    print("\n" + "="*70)
    print("MODAL PARAMETER COMPARISON")
    print("="*70)
    
    n_modes = len(true_params['frequencies'])
    
    print(f"\n{'Mode':<10} {'True Freq':<15} {'Est. Freq':<15} {'Error %':<12}")
    print("-" * 70)
    
    for i in range(n_modes):
        true_f = true_params['frequencies'][i]
        est_f = np.abs(estimated_params['frequencies'][i])
        error = 100 * abs(est_f - true_f) / true_f
        print(f"{i+1:<10} {true_f:<15.4f} {est_f:<15.4f} {error:<12.2f}")
    
    print(f"\n{'Mode':<10} {'True Damp':<15} {'Est. Damp':<15} {'Error %':<12}")
    print("-" * 70)
    
    for i in range(n_modes):
        true_d = true_params['damping_ratios'][i]
        est_d = np.abs(estimated_params['damping_ratios'][i])
        error = 100 * abs(est_d - true_d) / true_d if true_d > 0 else 0
        print(f"{i+1:<10} {true_d:<15.4f} {est_d:<15.4f} {error:<12.2f}")
    
    print("="*70)


def reconstruct_impulse_response(A: np.ndarray, B: np.ndarray, C: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Reconstruct impulse response from state-space model.
    
    Parameters
    ----------
    A : np.ndarray
        State matrix
    B : np.ndarray
        Input matrix (reshaped to column vector)
    C : np.ndarray
        Output matrix
    n_steps : int
        Number of time steps
        
    Returns
    -------
    y_recon : np.ndarray
        Reconstructed impulse response (n_outputs x n_steps)
    """
    n_states = A.shape[0]
    n_outputs = C.shape[0]
    
    # Reshape B to be a column vector
    B_vec = B.flatten()[:n_states].reshape(-1, 1)
    
    # Initialize state with impulse input
    x = B_vec.copy()
    
    # Initialize output
    y_recon = np.zeros((n_outputs, n_steps))
    
    # Simulate system
    for k in range(n_steps):
        # Output: y[k] = C*x[k]
        y_recon[:, k] = (C @ x).flatten()
        
        # State update: x[k+1] = A*x[k]
        x = A @ x
    
    return y_recon


def plot_results(Y: np.ndarray, t: np.ndarray, results: dict, true_params: dict):
    """
    Plot the results of ERA analysis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Reconstruct impulse response from state-space model
    n_steps = Y.shape[1]
    Y_recon = reconstruct_impulse_response(results['A'], results['B'], results['C'], n_steps)
    
    # Extract parameters for title
    freq = true_params['frequencies'][0]
    damp = true_params['damping_ratios'][0]
    period = 1.0 / freq
    
    # Plot 1: Time domain response comparison
    ax = axes[0]
    ax.plot(t, Y[0, :], 'b-', label='True Response', linewidth=1.5)
    ax.plot(t, Y_recon[0, :], 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response')
    ax.set_title(f'Impulse Response Comparison (Period = {period:.1f} s, ζ = {damp:.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pole locations in complex plane
    ax = axes[1]
    eigenvalues = results['eigenvalues']
    ax.plot(np.real(eigenvalues), np.imag(eigenvalues), 'ro', markersize=10, label='Estimated Poles')
    
    # Unit circle (for discrete-time system)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('System Poles in Complex Plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Reconstruction error
    ax = axes[2]
    error = Y[0, :] - Y_recon[0, :]
    ax.plot(t, error, 'g-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error')
    ax.set_title('Reconstruction Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('era_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'era_results.png'")
    plt.show()


def main():
    """
    Main demonstration script.
    """
    print("\n" + "="*70)
    print("EIGENSYSTEM REALIZATION ALGORITHM (ERA) DEMONSTRATION")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Use similar function to CEA: f(x) = exp{-1E-3*x/2}*cos(2*pi/50*x)
    # This represents a single mode impulse response with:
    # - Exponential decay: exp(-0.0005*t)
    # - Frequency: 1/50 Hz = 0.02 Hz
    # - Period: 50 seconds
    
    # Extract modal parameters from the function
    frequency_hz = 1.0 / 50.0  # 0.02 Hz
    omega_n = 2 * np.pi * frequency_hz
    damping_coeff = 0.0005  # ζ*ω_n
    damping_ratio = damping_coeff / omega_n
    
    true_frequencies = [frequency_hz]  # Hz
    true_damping = [damping_ratio]   # Fraction of critical damping
    true_mode_shapes = [[1.0]]  # Single output, single mode
    
    print("\nUSING FUNCTION: f(x) = exp{-1E-3*x/2}*cos(2*pi/50*x)")
    print("="*70)
    print("\nTRUE MODAL PARAMETERS:")
    print("-" * 70)
    for i, (f, d) in enumerate(zip(true_frequencies, true_damping)):
        print(f"Mode {i+1}: f = {f:.6f} Hz, ζ = {d:.6f}")
    print(f"Period: {1/frequency_hz:.2f} seconds")
    
    # Time parameters
    dt = 0.5  # Time step (s)
    duration = 5000.0  # Duration (s)
    t = np.arange(0, duration, dt)
    n_outputs = 1
    
    print(f"\nSAMPLING PARAMETERS:")
    print("-" * 70)
    print(f"Time step: {dt} s")
    print(f"Duration: {duration} s")
    print(f"Number of samples: {len(t)}")
    print(f"Sampling frequency: {1/dt} Hz")
    print(f"Number of outputs: {n_outputs}")
    
    # Generate synthetic impulse response
    Y = generate_impulse_response(
        true_frequencies,
        true_damping,
        true_mode_shapes,
        t,
        n_outputs=n_outputs
    )
    
    # Add small amount of noise
    noise_level = 0.0  # No noise for very low frequency demonstration
    if noise_level > 0:
        Y += np.random.normal(0, noise_level, Y.shape)
    print(f"Noise level: {noise_level} (standard deviation)")
    print(f"Signal amplitude range: [{np.min(Y):.4f}, {np.max(Y):.4f}]")
    
    # For very long signals, use a representative subset for analysis
    max_samples = 2000  # Use up to 2000 samples
    if len(t) > max_samples:
        print(f"\nNote: Using {max_samples} samples out of {len(t)} for ERA analysis")
        print(f"      (covers {max_samples*dt/50:.1f} periods)")
        Y_analysis = Y[:, :max_samples]
        t_analysis = t[:max_samples]
    else:
        Y_analysis = Y
        t_analysis = t
    
    # Apply ERA algorithm
    n_modes = len(true_frequencies)
    print(f"\nAPPLYING ERA ALGORITHM...")
    print(f"Estimating {n_modes} mode(s)...")
    
    # Use reasonable Hankel matrix dimensions
    r = min(len(t_analysis) // 3, 100)
    s = min(len(t_analysis) // 3, 100)
    print(f"Hankel matrix dimensions: {n_outputs * r} x {s}")
    
    results = eigensystem_realization_algorithm(
        Y=Y_analysis,
        dt=dt,
        n_modes=n_modes,
        r=r,
        s=s
    )
    
    # Sort results by frequency
    sort_idx = np.argsort(np.abs(results['frequencies']))
    results['frequencies'] = results['frequencies'][sort_idx]
    results['damping_ratios'] = results['damping_ratios'][sort_idx]
    results['mode_shapes'] = results['mode_shapes'][:, sort_idx]
    
    # Print comparison
    true_params = {
        'frequencies': true_frequencies,
        'damping_ratios': true_damping
    }
    estimated_params = results
    
    print_comparison(true_params, estimated_params)
    
    # Additional information
    print("\nESTIMATED MODE SHAPES:")
    print("-" * 70)
    for i in range(n_modes):
        mode_shape = results['mode_shapes'][:, i]
        print(f"Mode {i+1}: φ = {np.abs(mode_shape[0]):.4f}")
    
    print("\nSYSTEM MATRICES:")
    print("-" * 70)
    print(f"State matrix A: {results['A'].shape}")
    print(f"Input matrix B: {results['B'].shape}")
    print(f"Output matrix C: {results['C'].shape}")
    print(f"Model order (realized): {results['model_order']}")
    
    # Calculate reconstruction quality
    Y_recon = reconstruct_impulse_response(results['A'], results['B'], results['C'], Y_analysis.shape[1])
    reconstruction_error = np.linalg.norm(Y_analysis - Y_recon) / np.linalg.norm(Y_analysis)
    print(f"\nReconstruction Error (normalized): {reconstruction_error:.6f}")
    print(f"R² score: {1 - reconstruction_error**2:.6f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(Y_analysis, t_analysis, results, true_params)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
