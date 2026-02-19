"""
Complex Exponential Algorithm (CEA) for Time Domain Modal Estimation

Implements the algorithm from equations 7-13 for extracting modal parameters
from single reference-response pairs.

References
----------
Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques.
Experimental Techniques, 22(6), 45-49.

@article{fahey1998time,
  title={Time domain modal estimation techniques},
  author={Fahey, S O'F and Pratt, J},
  journal={Experimental techniques},
  volume={22},
  number={6},
  pages={45--49},
  year={1998},
  publisher={Springer}
}
"""

import numpy as np
from typing import Tuple, Optional


def build_toeplitz_matrix(y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Toeplitz matrix from response data (Equation 7).
    
    Parameters
    ----------
    y : np.ndarray
        Response time series data of length N
    n : int
        Number of modes to estimate (order of the system)
        
    Returns
    -------
    Y : np.ndarray
        Toeplitz matrix of shape (N-2n, 2n)
    y_target : np.ndarray
        Target vector of length (N-2n)
        
    Notes
    -----
    From eq (7), the Toeplitz matrix structure is:
    Y = [y1    y2    ... y2n+1  ]
        [y2    y3    ... y2n+2  ]
        [...   ...   ... ...    ]
        [yN-2n yN-2n+1 ... yN-1 ]
    
    And we solve: Y * α = -yN
    """
    N = len(y)
    
    if N < 2*n + 1:
        raise ValueError(f"Need at least {2*n + 1} data points for {n} modes")
    
    # Number of rows in Toeplitz matrix
    num_rows = N - 2*n
    
    # Build Toeplitz matrix (eq 7, left side)
    Y = np.zeros((num_rows, 2*n))
    for i in range(num_rows):
        Y[i, :] = y[i:i+2*n]
    
    # Target vector (eq 7, right side) - note the negative sign in eq (11)
    y_target = -y[2*n:N]
    
    return Y, y_target


def solve_polynomial_coefficients(Y: np.ndarray, y_target: np.ndarray) -> np.ndarray:
    """
    Solve for polynomial coefficients using least squares (Equation 11).
    
    Parameters
    ----------
    Y : np.ndarray
        Toeplitz matrix
    y_target : np.ndarray
        Target vector
        
    Returns
    -------
    alpha : np.ndarray
        Polynomial coefficients [α0, α1, ..., α_{2n-1}]
    """
    # Least squares solution: α = (Y^T Y)^-1 Y^T y_target
    alpha, residuals, rank, s = np.linalg.lstsq(Y, y_target, rcond=None)
    return alpha


def find_system_poles(alpha: np.ndarray) -> np.ndarray:
    """
    Find system poles from characteristic polynomial (Equation 8).
    
    Parameters
    ----------
    alpha : np.ndarray
        Polynomial coefficients [α0, α1, ..., α_{2n-1}]
        
    Returns
    -------
    z_k : np.ndarray
        System poles (roots of characteristic polynomial)
        
    Notes
    -----
    From eq (8): Π(z - z_k) = 0
    We solve: z^{2n} + α_{2n-1}*z^{2n-1} + ... + α_1*z + α_0 = 0
    """
    # Build polynomial coefficients [1, α_{2n-1}, ..., α_1, α_0]
    poly_coeffs = np.concatenate([[1], alpha[::-1]])
    
    # Find roots (poles)
    z_k = np.roots(poly_coeffs)
    
    return z_k


def poles_to_modal_frequencies(z_k: np.ndarray, dt: float) -> np.ndarray:
    """
    Convert poles to modal frequencies (Equations 9 and 10).
    
    Parameters
    ----------
    z_k : np.ndarray
        System poles
    dt : float
        Time step (sampling interval)
        
    Returns
    -------
    lambda_k : np.ndarray
        Modal frequencies (complex, with real = damping, imag = frequency)
        
    Notes
    -----
    From eq (9): λ_k = (1/(2π*Δt)) * ln(z_k)  [in Hz]
    From eq (10): λ_k = (1/Δt) * ln(z_k)      [in rad/s]
    
    We use eq (10) for rad/s convention.
    """
    lambda_k = np.log(z_k) / dt
    return lambda_k


def build_vandermonde_matrix(lambda_k: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build Vandermonde-like matrix Λ (Equation 12).
    
    Parameters
    ----------
    lambda_k : np.ndarray
        Modal frequencies [λ_1, λ_2, ..., λ_n]
    t : np.ndarray
        Time vector
        
    Returns
    -------
    Lambda : np.ndarray
        Matrix with exp[λ_k * t_j] entries
        
    Notes
    -----
    From eq (12):
    Λ_{1,N} = [exp[λ_1*t_1]  exp[λ_1*t_2]  ... exp[λ_1*t_N]]
              [exp[λ_2*t_1]  exp[λ_2*t_2]  ... exp[λ_2*t_N]]
              [     ...           ...       ...      ...     ]
              [exp[λ_n*t_1]  exp[λ_n*t_2]  ... exp[λ_n*t_N]]
    """
    n = len(lambda_k)
    N = len(t)
    
    Lambda = np.zeros((n, N), dtype=complex)
    
    for i, lam in enumerate(lambda_k):
        Lambda[i, :] = np.exp(lam * t)
    
    return Lambda


def solve_mode_shapes(Lambda: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve for mode shapes/participation factors (Equation 13).
    
    Parameters
    ----------
    Lambda : np.ndarray
        Vandermonde matrix of shape (n, N)
    y : np.ndarray
        Response data of length N
        
    Returns
    -------
    A : np.ndarray
        Mode shape coefficients/participation factors
        
    Notes
    -----
    From eq (13): [A_1, A_2, ..., A_n]^T = Λ_{1,N}^{-1} * y
    
    The response is modeled as: y = Λ^T * A
    Where Lambda is (n x N), A is (n,), and y is (N,)
    
    Solving: A = (Λ * Λ^T)^{-1} * Λ * y
    Or using pseudo-inverse: A = pinv(Λ^T) * y
    """
    # Pseudo-inverse solution: A = pinv(Λ.T) @ y
    A = np.linalg.pinv(Lambda.T) @ y
    
    return A


def complex_exponential_algorithm(
    y: np.ndarray,
    dt: float,
    n_modes: int,
    t_start: Optional[float] = None
) -> dict:
    """
    Complete Complex Exponential Algorithm (CEA) implementation.
    
    Parameters
    ----------
    y : np.ndarray
        Response time series data
    dt : float
        Time step (sampling interval)
    n_modes : int
        Number of modes to estimate
    t_start : float, optional
        Starting time (default: 0)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'frequencies': Natural frequencies (Hz)
        - 'damping_ratios': Damping ratios (fraction of critical)
        - 'mode_shapes': Mode shape coefficients
        - 'poles': System poles z_k
        - 'lambda': Modal frequencies λ_k (complex)
        
    Notes
    -----
    Implements the CEA algorithm from equations 7-13:
    1. Build Toeplitz matrix (eq 7)
    2. Solve for polynomial coefficients (eq 11)
    3. Find poles (eq 8)
    4. Convert to modal frequencies (eq 9-10)
    5. Build Λ matrix (eq 12)
    6. Solve for mode shapes (eq 13)
    
    References
    ----------
    Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques.
    Experimental Techniques, 22(6), 45-49.
    """
    # Time vector
    N = len(y)
    if t_start is None:
        t_start = 0.0
    t = t_start + np.arange(N) * dt
    
    # Step 1: Build Toeplitz matrix (eq 7)
    Y, y_target = build_toeplitz_matrix(y, n_modes)
    
    # Step 2: Solve for polynomial coefficients (eq 11)
    alpha = solve_polynomial_coefficients(Y, y_target)
    
    # Step 3: Find system poles (eq 8)
    z_k = find_system_poles(alpha)
    
    # Step 4: Convert poles to modal frequencies (eq 9-10)
    lambda_k_all = poles_to_modal_frequencies(z_k, dt)
    
    # Step 5: Build Λ matrix using ALL poles (eq 12)
    # For real-valued signals, we need complex conjugate pairs
    Lambda = build_vandermonde_matrix(lambda_k_all, t)
    
    # Step 6: Solve for mode shapes (eq 13)
    A_all = solve_mode_shapes(Lambda, y)
    
    # Select unique modes (complex conjugate pairs) for reporting
    # Keep poles with positive imaginary parts
    positive_imag_idx = np.imag(lambda_k_all) > 0
    
    if np.sum(positive_imag_idx) >= n_modes:
        lambda_k_unique = lambda_k_all[positive_imag_idx]
        A_unique = A_all[positive_imag_idx]
        # Sort by magnitude of imaginary part (oscillation frequency)
        sorted_indices = np.argsort(np.abs(np.imag(lambda_k_unique)))
        lambda_k_unique = lambda_k_unique[sorted_indices[:n_modes]]
        A_unique = A_unique[sorted_indices[:n_modes]]
    else:
        # Otherwise, just take the first n_modes poles sorted by imaginary part
        sorted_indices = np.argsort(np.imag(lambda_k_all))[::-1]
        lambda_k_unique = lambda_k_all[sorted_indices[:n_modes]]
        A_unique = A_all[sorted_indices[:n_modes]]
    
    
    # Extract physical parameters (from unique modes only)
    # For complex conjugate pairs: λ = -ζω_n ± i*ω_d
    # where ω_n is natural frequency, ζ is damping ratio, ω_d is damped frequency
    damping_ratios = -np.real(lambda_k_unique) / np.abs(lambda_k_unique)
    omega_n = np.abs(lambda_k_unique)
    frequencies_hz = omega_n / (2 * np.pi)
    
    results = {
        'frequencies': frequencies_hz,
        'damping_ratios': damping_ratios,
        'mode_shapes': A_unique,
        'poles': z_k,
        'lambda': lambda_k_unique,
        'alpha': alpha,
        'time': t,
        'reconstruction': reconstruct_response(lambda_k_all, A_all, t)  # Use all modes for reconstruction
    }
    
    return results


def reconstruct_response(lambda_k: np.ndarray, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Reconstruct the response from modal parameters.
    
    Parameters
    ----------
    lambda_k : np.ndarray
        Modal frequencies
    A : np.ndarray
        Mode shape coefficients
    t : np.ndarray
        Time vector
        
    Returns
    -------
    y_reconstructed : np.ndarray
        Reconstructed response
        
    Notes
    -----
    Response is: y = Λ^T @ A = Σ A_k * exp(λ_k * t)
    """
    y_recon = np.zeros(len(t), dtype=complex)
    
    for lam, a in zip(lambda_k, A):
        y_recon += a * np.exp(lam * t)
    
    # Return real part (for real-valued responses)
    return np.real(y_recon)
