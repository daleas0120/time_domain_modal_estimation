"""
Demonstration of Complex Exponential Algorithm (CEA)
Using a toy dataset with known modal parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from time_domain_modal_estimation import complex_exponential_algorithm, reconstruct_response


def generate_synthetic_response(
    frequencies: list,
    damping_ratios: list,
    amplitudes: list,
    phases: list,
    t: np.ndarray,
    noise_level: float = 0.0
) -> np.ndarray:
    """
    Generate synthetic multi-mode response data.
    
    Parameters
    ----------
    frequencies : list
        Natural frequencies in Hz
    damping_ratios : list
        Damping ratios (fraction of critical damping)
    amplitudes : list
        Mode amplitudes
    phases : list
        Phase angles in radians
    t : np.ndarray
        Time vector
    noise_level : float
        Standard deviation of measurement noise
        
    Returns
    -------
    y : np.ndarray
        Synthetic response time series
    """
    y = np.zeros_like(t)
    
    for f, zeta, A, phi in zip(frequencies, damping_ratios, amplitudes, phases):
        omega_n = 2 * np.pi * f  # Natural frequency in rad/s
        omega_d = omega_n * np.sqrt(1 - zeta**2)  # Damped frequency
        
        # Free decay response: y(t) = A * exp(-ζω_n*t) * cos(ω_d*t + φ)
        y += A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)
    
    # Add measurement noise
    if noise_level > 0:
        y += np.random.normal(0, noise_level, len(t))
    
    return y


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


def plot_results(t: np.ndarray, y_true: np.ndarray, results: dict):
    """
    Plot the results of CEA analysis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Time domain response
    ax = axes[0]
    ax.plot(t, y_true, 'b-', label='True Response', linewidth=1.5)
    ax.plot(t, results['reconstruction'], 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response')
    ax.set_title('Time Domain Response Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pole locations in complex plane
    ax = axes[1]
    poles = results['poles']
    ax.plot(np.real(poles), np.imag(poles), 'ro', markersize=10, label='Estimated Poles')
    
    # Draw unit circle
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
    error = y_true - results['reconstruction']
    ax.plot(t, error, 'g-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error')
    ax.set_title('Reconstruction Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cea_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'cea_results.png'")
    plt.show()


def main():
    """
    Main demonstration script.
    """
    print("\n" + "="*70)
    print("COMPLEX EXPONENTIAL ALGORITHM (CEA) DEMONSTRATION")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Use the specified function: f(x) = exp{-1E-3*x/2}*cos(2*pi/50*x)
    # This represents a single mode with:
    # - Exponential decay: exp(-0.0005*t)
    # - Frequency: 1/50 Hz = 0.02 Hz
    # - Period: 50 seconds
    
    # Extract modal parameters from the function
    # f(t) = exp(-0.0005*t) * cos(2*pi*t/50)
    # Compare to: A * exp(-ζ*ω_n*t) * cos(ω_d*t + φ)
    
    frequency_hz = 1.0 / 50.0  # 0.02 Hz
    omega_n = 2 * np.pi * frequency_hz
    damping_coeff = 0.0005  # ζ*ω_n
    damping_ratio = damping_coeff / omega_n
    
    true_frequencies = [frequency_hz]  # Hz
    true_damping = [damping_ratio]   # Fraction of critical damping
    true_amplitudes = [1.0]
    true_phases = [0.0]  # radians
    
    print("\nUSING FUNCTION: f(x) = exp{-1E-3*x/2}*cos(2*pi/50*x)")
    print("="*70)
    print("\nTRUE MODAL PARAMETERS:")
    print("-" * 70)
    for i, (f, d, a, p) in enumerate(zip(true_frequencies, true_damping, 
                                          true_amplitudes, true_phases)):
        print(f"Mode {i+1}: f = {f:.6f} Hz, ζ = {d:.6f}, A = {a:.2f}, φ = {p:.2f} rad")
    print(f"Period: {1/frequency_hz:.2f} seconds")
    
    # Time parameters
    # Need sufficient duration to capture several periods (at least 3-5 periods recommended)
    dt = 0.5  # Time step (s)
    duration = 5000.0  # Duration (s) - 100 periods for 50s period signal
    t = np.arange(0, duration, dt)
    
    print(f"\nSAMPLING PARAMETERS:")
    print("-" * 70)
    print(f"Time step: {dt} s")
    print(f"Duration: {duration} s")
    print(f"Number of samples: {len(t)}")
    print(f"Sampling frequency: {1/dt} Hz")
    
    # Generate synthetic response
    # Note: CEA is sensitive to noise, especially for multiple modes
    noise_level = 0.0  # No noise for very low frequency demonstration
    y = generate_synthetic_response(
        true_frequencies,
        true_damping,
        true_amplitudes,
        true_phases,
        t,
        noise_level=noise_level
    )
    
    print(f"Noise level: {noise_level} (standard deviation)")
    print(f"Signal amplitude range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    
    # For very long signals, CEA can have numerical issues
    # Use a representative subset for analysis (still covering multiple periods)
    max_samples = 2000  # Use up to 2000 samples
    if len(y) > max_samples:
        print(f"\nNote: Using {max_samples} samples out of {len(y)} for CEA analysis")
        print(f"      (covers {max_samples*dt/50:.1f} periods)")
        y_analysis = y[:max_samples]
        t_analysis = t[:max_samples]
    else:
        y_analysis = y
        t_analysis = t
    
    # Apply CEA algorithm
    n_modes = len(true_frequencies)
    print(f"\nAPPLYING CEA ALGORITHM...")
    print(f"Estimating {n_modes} mode(s)...")
    
    results = complex_exponential_algorithm(
        y=y_analysis,
        dt=dt,
        n_modes=n_modes,
        t_start=0.0
    )
    
    # Sort results by frequency
    sort_idx = np.argsort(np.abs(results['frequencies']))
    results['frequencies'] = results['frequencies'][sort_idx]
    results['damping_ratios'] = results['damping_ratios'][sort_idx]
    results['mode_shapes'] = results['mode_shapes'][sort_idx]
    
    # Print comparison
    true_params = {
        'frequencies': true_frequencies,
        'damping_ratios': true_damping
    }
    estimated_params = results
    
    print_comparison(true_params, estimated_params)
    
    # Additional information
    print("\nESTIMATED MODE SHAPES (Participation Factors):")
    print("-" * 70)
    for i, A in enumerate(results['mode_shapes']):
        print(f"Mode {i+1}: A = {A:.4f}")
    
    # Calculate fit quality
    reconstruction_error = np.linalg.norm(y_analysis - results['reconstruction']) / np.linalg.norm(y_analysis)
    print(f"\nReconstruction Error (normalized): {reconstruction_error:.6f}")
    print(f"R² score: {1 - reconstruction_error**2:.6f}")
    
    # Generate full reconstruction for plotting
    # Get all lambda values for reconstruction
    lambda_all = results['lambda']
    A_all = results['mode_shapes']
    # For complex conjugate completion
    lambda_full = np.concatenate([lambda_all, np.conj(lambda_all)])
    A_full = np.concatenate([A_all, np.conj(A_all)])
    y_full_recon = reconstruct_response(lambda_full, A_full, t)
    
    # Plot results
    print("\nGenerating plots...")
    # Update results with full reconstruction for plotting
    results_plot = results.copy()
    results_plot['time'] = t
    results_plot['reconstruction'] = y_full_recon
    plot_results(t, y, results_plot)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
