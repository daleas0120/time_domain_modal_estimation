"""
Basic usage example of time_domain_modal_estimation package
"""

import numpy as np
from time_domain_modal_estimation import complex_exponential_algorithm

# Generate synthetic response data for a single mode
dt = 0.01  # Time step (seconds)
t = np.arange(0, 5, dt)

# Define true parameters
f_true = 5.0  # Frequency (Hz)
zeta_true = 0.03  # Damping ratio
omega_n = 2 * np.pi * f_true
omega_d = omega_n * np.sqrt(1 - zeta_true**2)

# Generate response: A * exp(-ζ*ω_n*t) * cos(ω_d*t)
y = 1.0 * np.exp(-zeta_true * omega_n * t) * np.cos(omega_d * t)

# Apply CEA algorithm
print("Applying Complex Exponential Algorithm...")
results = complex_exponential_algorithm(
    y=y,
    dt=dt,
    n_modes=1
)

# Display results
print(f"\nTrue Parameters:")
print(f"  Frequency: {f_true:.4f} Hz")
print(f"  Damping ratio: {zeta_true:.4f}")

print(f"\nEstimated Parameters:")
print(f"  Frequency: {results['frequencies'][0]:.4f} Hz")
print(f"  Damping ratio: {results['damping_ratios'][0]:.4f}")
print(f"  Mode shape: {results['mode_shapes'][0]:.4f}")

# Calculate errors
freq_error = abs(results['frequencies'][0] - f_true) / f_true * 100
damp_error = abs(results['damping_ratios'][0] - zeta_true) / zeta_true * 100

print(f"\nErrors:")
print(f"  Frequency: {freq_error:.2f}%")
print(f"  Damping: {damp_error:.2f}%")

# Check reconstruction quality
recon_error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
print(f"\nReconstruction error: {recon_error:.6f}")
print(f"R² score: {1 - recon_error**2:.6f}")
