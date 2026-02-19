"""
Eigensystem Realization Algorithm (ERA) for Time Domain Modal Estimation

Implements the ERA from equations 27-35 for extracting modal parameters
from impulse response data with multiple references and responses.

References
----------
Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques.
Experimental Techniques, 22(6), 45-49.

Juang, J. N., & Pappa, R. S. (1985). An eigensystem realization algorithm for
modal parameter identification and model reduction. Journal of guidance,
control, and dynamics, 8(5), 620-627.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def build_hankel_matrix(
    Y: np.ndarray,
    r: int,
    s: int,
    block_rows: Optional[int] = None
) -> np.ndarray:
    """
    Build generalized Hankel matrix from impulse response data (Equation 29).
    
    Parameters
    ----------
    Y : np.ndarray
        Impulse response matrix of shape (p, N) where p is number of outputs
        and N is number of time steps
    r : int
        Number of block rows
    s : int
        Number of block columns
    block_rows : int, optional
        Explicit number of block rows (if different from r)
        
    Returns
    -------
    H : np.ndarray
        Hankel matrix of shape (p*r, s)
        
    Notes
    -----
    From eq (29), the Hankel matrix H_r(k-1) is:
    H(k-1) = [Y(k)      Y(k+1)    ... Y(k+s-1)    ]
             [Y(k+1)    Y(k+2)    ... Y(k+s)      ]
             [  ...       ...     ...    ...      ]
             [Y(k+r-1)  Y(k+r)    ... Y(k+s+r-2)  ]
    
    Each block Y(i) can be a matrix itself for multiple outputs.
    """
    if block_rows is None:
        block_rows = r
        
    p = Y.shape[0]  # Number of outputs
    N = Y.shape[1]  # Number of time steps
    
    # Check if we have enough data
    if N < block_rows + s - 1:
        raise ValueError(f"Need at least {block_rows + s - 1} time steps, got {N}")
    
    # Build Hankel matrix
    H = np.zeros((p * block_rows, s))
    
    for i in range(block_rows):
        for j in range(s):
            H[i*p:(i+1)*p, j] = Y[:, i + j]
    
    return H


def eigensystem_realization_algorithm(
    Y: np.ndarray,
    dt: float,
    n_modes: int,
    r: Optional[int] = None,
    s: Optional[int] = None
) -> Dict:
    """
    Complete Eigensystem Realization Algorithm (ERA) implementation.
    
    Parameters
    ----------
    Y : np.ndarray
        Impulse response data of shape (p, N) where p is number of outputs
        and N is number of time steps
    dt : float
        Time step (sampling interval)
    n_modes : int
        Number of modes to extract (model order)
    r : int, optional
        Number of block rows in Hankel matrix (default: N//2)
    s : int, optional
        Number of block columns in Hankel matrix (default: N//2)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'frequencies': Natural frequencies (Hz)
        - 'damping_ratios': Damping ratios (fraction of critical)
        - 'mode_shapes': Mode shapes (complex)
        - 'A': State matrix
        - 'B': Input matrix
        - 'C': Output matrix
        - 'eigenvalues': System eigenvalues
        - 'eigenvectors': System eigenvectors
        - 'singular_values': Singular values from SVD
        
    Notes
    -----
    Implements the ERA algorithm from equations 27-35:
    1. Build Hankel matrices H(0) and H(1) (eq 29-30)
    2. SVD of H(0) (eq 31)
    3. Construct reduced-order system matrices (eq 32-33)
    4. Extract eigenvalues and eigenvectors (eq 34)
    5. Compute mode shapes, poles, and amplitudes (eq 35)
    
    References
    ----------
    Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques.
    Experimental Techniques, 22(6), 45-49.
    
    Juang, J. N., & Pappa, R. S. (1985). An eigensystem realization algorithm.
    Journal of guidance, control, and dynamics, 8(5), 620-627.
    """
    p, N = Y.shape
    
    # Set default Hankel matrix dimensions
    if r is None:
        r = min(N // 2, 100)  # Limit for computational efficiency
    if s is None:
        s = min(N - r, 100)
    
    # Ensure we have enough data
    if N < r + s:
        raise ValueError(f"Need at least {r + s} time steps for r={r}, s={s}")
    
    # Step 1: Build Hankel matrices H(0) and H(1) (equations 29-30)
    H0 = build_hankel_matrix(Y, r, s, block_rows=r)
    H1 = build_hankel_matrix(Y[:, 1:], r, s, block_rows=r)  # Shifted by one time step
    
    # Step 2: SVD of H(0) (equation 31)
    # H(0) ≈ P * D * Q^T where D is diagonal with singular values
    U, Sigma, VT = np.linalg.svd(H0, full_matrices=False)
    
    # Keep only the first n_modes singular values for model reduction
    # This determines the dimension of the realized system
    n = min(n_modes * 2, len(Sigma))  # Use 2*n_modes for complex conjugate pairs
    
    P = U[:, :n]
    D = np.diag(Sigma[:n])
    Q = VT[:n, :].T
    
    # D^(-1/2) for scaling
    D_inv_sqrt = np.diag(1.0 / np.sqrt(Sigma[:n]))
    
    # Step 3: Construct state matrix A (equation 32-33)
    # A = D^(-1/2) * P^T * H(1) * Q * D^(-1/2)
    A = D_inv_sqrt @ P.T @ H1 @ Q @ D_inv_sqrt
    
    # Construct output matrix C
    # C is the first p rows of P * D^(1/2)
    D_sqrt = np.diag(np.sqrt(Sigma[:n]))
    C = (P @ D_sqrt)[:p, :]
    
    # Construct input matrix B
    # B is the first column(s) of D^(1/2) * Q^T
    B = (D_sqrt @ Q.T)[:, :p].T
    
    # Step 4: Extract eigenvalues and eigenvectors (equation 34)
    # Ψ^(-1) * A * Ψ = Λ (diagonal eigenvalue matrix)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Step 5: Extract modal parameters
    # Convert discrete eigenvalues to continuous modal frequencies
    lambda_c = np.log(eigenvalues) / dt
    
    # Compute mode shapes from eigenvectors (equation 35)
    # Mode shapes = C * eigenvectors
    mode_shapes_full = C @ eigenvectors
    
    # Select modes with positive imaginary parts (for complex conjugate pairs)
    positive_imag = np.imag(lambda_c) > 0
    
    if np.sum(positive_imag) >= n_modes:
        # Take modes with positive imaginary parts
        indices = np.where(positive_imag)[0]
        # Sort by imaginary part (frequency)
        sorted_indices = indices[np.argsort(np.abs(np.imag(lambda_c[indices])))]
        selected_indices = sorted_indices[:n_modes]
    else:
        # If not enough positive modes, take the largest by absolute imaginary part
        sorted_indices = np.argsort(np.abs(np.imag(lambda_c)))[::-1]
        selected_indices = sorted_indices[:n_modes]
    
    lambda_selected = lambda_c[selected_indices]
    mode_shapes_selected = mode_shapes_full[:, selected_indices]
    
    # Extract physical parameters
    omega_n = np.abs(lambda_selected)
    frequencies_hz = omega_n / (2 * np.pi)
    damping_ratios = -np.real(lambda_selected) / omega_n
    
    results = {
        'frequencies': frequencies_hz,
        'damping_ratios': damping_ratios,
        'mode_shapes': mode_shapes_selected,
        'A': A,
        'B': B,
        'C': C,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'lambda': lambda_selected,
        'singular_values': Sigma,
        'model_order': n,
    }
    
    return results


def generate_impulse_response(
    frequencies: list,
    damping_ratios: list,
    mode_shapes: list,
    t: np.ndarray,
    n_outputs: int = 1
) -> np.ndarray:
    """
    Generate synthetic impulse response data for testing ERA.
    
    Parameters
    ----------
    frequencies : list
        Natural frequencies in Hz
    damping_ratios : list
        Damping ratios (fraction of critical)
    mode_shapes : list
        Mode shape amplitudes for each output
    t : np.ndarray
        Time vector
    n_outputs : int
        Number of output channels
        
    Returns
    -------
    Y : np.ndarray
        Impulse response matrix of shape (n_outputs, len(t))
    """
    n_modes = len(frequencies)
    Y = np.zeros((n_outputs, len(t)), dtype=complex)
    
    for i, (f, zeta, shape) in enumerate(zip(frequencies, damping_ratios, mode_shapes)):
        omega_n = 2 * np.pi * f
        lambda_k = -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta**2)
        
        # Each mode contributes to each output channel
        if np.isscalar(shape):
            shape = [shape] * n_outputs
        
        for output_idx in range(n_outputs):
            # Add both the mode and its complex conjugate
            Y[output_idx, :] += shape[output_idx] * np.exp(lambda_k * t)
            Y[output_idx, :] += np.conj(shape[output_idx]) * np.exp(np.conj(lambda_k) * t)
    
    return np.real(Y)


def stabilization_diagram(
    Y: np.ndarray,
    dt: float,
    max_order: int,
    r: Optional[int] = None,
    s: Optional[int] = None,
    freq_tol: float = 0.01,
    damp_tol: float = 0.05
) -> Dict:
    """
    Generate stabilization diagram by running ERA at multiple model orders.
    
    Parameters
    ----------
    Y : np.ndarray
        Impulse response data
    dt : float
        Time step
    max_order : int
        Maximum model order to test
    r : int, optional
        Hankel matrix block rows
    s : int, optional
        Hankel matrix block columns
    freq_tol : float
        Frequency tolerance for stability (default: 1%)
    damp_tol : float
        Damping tolerance for stability (default: 5%)
        
    Returns
    -------
    diagram : dict
        Contains 'orders', 'frequencies', 'damping_ratios', 'stability'
    """
    orders = range(1, max_order + 1)
    all_freqs = []
    all_damps = []
    all_orders = []
    stability = []
    
    prev_freqs = None
    prev_damps = None
    
    for order in orders:
        try:
            results = eigensystem_realization_algorithm(Y, dt, order, r, s)
            
            for f, d in zip(results['frequencies'], results['damping_ratios']):
                all_freqs.append(f)
                all_damps.append(d)
                all_orders.append(order)
                
                # Check stability: compare with previous order
                is_stable = False
                if prev_freqs is not None:
                    for pf, pd in zip(prev_freqs, prev_damps):
                        if (abs(f - pf) / pf < freq_tol and 
                            abs(d - pd) / (pd + 1e-10) < damp_tol):
                            is_stable = True
                            break
                
                stability.append(is_stable)
            
            prev_freqs = results['frequencies']
            prev_damps = results['damping_ratios']
            
        except Exception as e:
            print(f"Warning: Order {order} failed: {e}")
            continue
    
    return {
        'orders': np.array(all_orders),
        'frequencies': np.array(all_freqs),
        'damping_ratios': np.array(all_damps),
        'stability': np.array(stability)
    }
