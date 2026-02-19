Examples
========

This section provides detailed examples demonstrating the use of the time-domain modal estimation package.

Example 1: Single Mode Estimation
----------------------------------

This example demonstrates basic usage for a single-mode system.

.. code-block:: python

   """
   Single mode estimation example
   """
   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Define parameters
   dt = 0.01  # Time step (seconds)
   duration = 5.0  # Duration (seconds)
   t = np.arange(0, duration, dt)

   # True modal parameters
   f_true = 5.0  # Natural frequency (Hz)
   zeta_true = 0.03  # Damping ratio
   A_true = 1.0  # Amplitude

   # Generate response
   omega_n = 2 * np.pi * f_true
   omega_d = omega_n * np.sqrt(1 - zeta_true**2)
   y = A_true * np.exp(-zeta_true * omega_n * t) * np.cos(omega_d * t)

   # Apply CEA
   results = complex_exponential_algorithm(y=y, dt=dt, n_modes=1)

   # Display results
   print("True vs Estimated Parameters:")
   print(f"Frequency: {f_true:.4f} Hz -> {results['frequencies'][0]:.4f} Hz")
   print(f"Damping:   {zeta_true:.4f} -> {results['damping_ratios'][0]:.4f}")

   # Check reconstruction quality
   error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
   print(f"Reconstruction error: {error:.6f}")

Example 2: Multi-Mode System
-----------------------------

Estimating parameters for a system with multiple modes.

.. code-block:: python

   """
   Multi-mode estimation example
   """
   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Time parameters
   dt = 0.01
   t = np.arange(0, 10, dt)

   # Define three modes
   modes = [
       {'f': 2.5, 'zeta': 0.02, 'A': 1.0},
       {'f': 5.0, 'zeta': 0.03, 'A': 0.6},
       {'f': 8.3, 'zeta': 0.05, 'A': 0.3},
   ]

   # Generate combined response
   y = np.zeros_like(t)
   for mode in modes:
       omega_n = 2 * np.pi * mode['f']
       omega_d = omega_n * np.sqrt(1 - mode['zeta']**2)
       y += mode['A'] * np.exp(-mode['zeta'] * omega_n * t) * np.cos(omega_d * t)

   # Estimate parameters
   results = complex_exponential_algorithm(y=y, dt=dt, n_modes=3)

   # Sort by frequency
   sort_idx = np.argsort(results['frequencies'])
   
   print("\\nMode Identification Results:")
   print("-" * 60)
   for i, idx in enumerate(sort_idx):
       true_f = modes[i]['f']
       true_zeta = modes[i]['zeta']
       est_f = results['frequencies'][idx]
       est_zeta = results['damping_ratios'][idx]
       
       print(f"Mode {i+1}:")
       print(f"  Frequency: {true_f:.2f} Hz -> {est_f:.2f} Hz")
       print(f"  Damping:   {true_zeta:.4f} -> {est_zeta:.4f}")

Example 3: Noisy Data
---------------------

Handling measurement noise in modal estimation.

.. code-block:: python

   """
   Modal estimation with noisy data
   """
   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   np.random.seed(42)  # For reproducibility

   # Generate clean signal
   dt = 0.01
   t = np.arange(0, 8, dt)
   f_true = 5.0
   zeta_true = 0.03
   omega_n = 2 * np.pi * f_true

   y_clean = np.exp(-zeta_true * omega_n * t) * np.cos(omega_n * np.sqrt(1 - zeta_true**2) * t)

   # Add noise (5% of signal amplitude)
   noise_level = 0.05
   y_noisy = y_clean + np.random.normal(0, noise_level, len(t))

   # Estimate from noisy data
   results = complex_exponential_algorithm(y=y_noisy, dt=dt, n_modes=1)

   print("Results with 5% noise:")
   print(f"Frequency error: {abs(results['frequencies'][0] - f_true)/f_true * 100:.2f}%")
   print(f"Damping error: {abs(results['damping_ratios'][0] - zeta_true)/zeta_true * 100:.2f}%")

Example 4: Visualization
-------------------------

Complete example with visualization.

.. code-block:: python

   """
   Complete example with plotting
   """
   import numpy as np
   import matplotlib.pyplot as plt
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Generate data
   dt = 0.01
   t = np.arange(0, 5, dt)
   f = 5.0
   zeta = 0.03
   omega_n = 2 * np.pi * f
   y = np.exp(-zeta * omega_n * t) * np.cos(omega_n * np.sqrt(1 - zeta**2) * t)

   # Estimate parameters
   results = complex_exponential_algorithm(y=y, dt=dt, n_modes=1)

   # Create visualization
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))

   # Time series
   ax = axes[0, 0]
   ax.plot(t, y, 'b-', label='Measured', linewidth=1.5)
   ax.plot(t, results['reconstruction'], 'r--', label='Reconstructed', linewidth=1.5)
   ax.set_xlabel('Time (s)')
   ax.set_ylabel('Response')
   ax.set_title('Time Domain Response')
   ax.legend()
   ax.grid(True, alpha=0.3)

   # Error
   ax = axes[0, 1]
   error = y - results['reconstruction']
   ax.plot(t, error, 'g-', linewidth=1)
   ax.set_xlabel('Time (s)')
   ax.set_ylabel('Error')
   ax.set_title('Reconstruction Error')
   ax.grid(True, alpha=0.3)

   # Poles
   ax = axes[1, 0]
   poles = results['poles']
   ax.plot(np.real(poles), np.imag(poles), 'ro', markersize=10, label='Poles')
   theta = np.linspace(0, 2*np.pi, 100)
   ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
   ax.set_xlabel('Real Part')
   ax.set_ylabel('Imaginary Part')
   ax.set_title('System Poles')
   ax.legend()
   ax.grid(True, alpha=0.3)
   ax.axis('equal')

   # Info text
   ax = axes[1, 1]
   ax.axis('off')
   info_text = f"""
   Estimated Parameters:
   
   Frequency: {results['frequencies'][0]:.4f} Hz
   Damping:   {results['damping_ratios'][0]:.4f}
   
   True Parameters:
   
   Frequency: {f:.4f} Hz
   Damping:   {zeta:.4f}
   
   Errors:
   
   Freq: {abs(results['frequencies'][0] - f):.6f} Hz
   Damp: {abs(results['damping_ratios'][0] - zeta):.6f}
   """
   ax.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
           verticalalignment='center')

   plt.tight_layout()
   plt.show()

Example 5: Low Frequency Mode
------------------------------

Working with very low frequency oscillations.

.. code-block:: python

   """
   Low frequency mode example
   """
   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Very low frequency: 0.02 Hz (50 second period)
   dt = 0.5  # Coarser sampling for long duration
   t = np.arange(0, 1000, dt)  # 1000 seconds (20 periods)

   # Modal parameters
   f_true = 0.02  # 0.02 Hz
   damping_coeff = 0.0005  # exp(-0.0005*t)
   omega_n = 2 * np.pi * f_true
   zeta_true = damping_coeff / omega_n

   # Generate response: f(x) = exp(-0.0005*t) * cos(2*pi*t/50)
   y = np.exp(-damping_coeff * t) * np.cos(2 * np.pi * t / 50)

   # Use subset for analysis (to avoid numerical issues with huge matrices)
   n_samples = min(2000, len(y))
   y_analysis = y[:n_samples]
   
   # Estimate
   results = complex_exponential_algorithm(y=y_analysis, dt=dt, n_modes=1)

   print(f"Low frequency mode:")
   print(f"True frequency:      {f_true:.6f} Hz")
   print(f"Estimated frequency: {results['frequencies'][0]:.6f} Hz")
   print(f"True damping:        {zeta_true:.6f}")
   print(f"Estimated damping:   {results['damping_ratios'][0]:.6f}")

Eigensystem Realization Algorithm Examples
===========================================

Example 6: Basic ERA Usage
---------------------------

Estimating modal parameters from impulse response data.

.. code-block:: python

   """
   Basic ERA example
   """
   import numpy as np
   from time_domain_modal_estimation import (
       eigensystem_realization_algorithm,
       generate_impulse_response
   )

   # Time parameters
   dt = 0.01  # Time step (seconds)
   t = np.arange(0, 5, dt)
   
   # True modal parameters
   frequencies = [5.0]  # Hz
   damping = [0.03]     # Fraction of critical
   mode_shapes = [[1.0]]  # Single output

   # Generate impulse response
   Y = generate_impulse_response(
       frequencies,
       damping,
       mode_shapes,
       t,
       n_outputs=1
   )

   # Apply ERA
   results = eigensystem_realization_algorithm(
       Y=Y,
       dt=dt,
       n_modes=1,
       r=100,  # Hankel matrix rows
       s=100   # Hankel matrix columns
   )

   print(f"Estimated frequency: {results['frequencies'][0]:.4f} Hz")
   print(f"Estimated damping: {results['damping_ratios'][0]:.4f}")

Example 7: Multi-Mode ERA
--------------------------

Identifying multiple modes from impulse response.

.. code-block:: python

   """
   Multi-mode ERA example
   """
   import numpy as np
   from time_domain_modal_estimation import (
       eigensystem_realization_algorithm,
       generate_impulse_response
   )

   # Time parameters
   dt = 0.01
   t = np.arange(0, 10, dt)
   
   # Multiple modes
   frequencies = [2.5, 5.0, 8.0]
   damping = [0.02, 0.03, 0.05]
   mode_shapes = [[1.0], [0.8], [0.5]]

   # Generate impulse response
   Y = generate_impulse_response(
       frequencies,
       damping,
       mode_shapes,
       t,
       n_outputs=1
   )
   
   # Add noise
   Y += np.random.normal(0, 0.01, Y.shape)

   # Apply ERA
   results = eigensystem_realization_algorithm(
       Y=Y,
       dt=dt,
       n_modes=3,
       r=min(len(t)//3, 150),
       s=min(len(t)//3, 150)
   )

   # Sort by frequency
   sort_idx = np.argsort(results['frequencies'])
   
   print("\\nMulti-mode ERA Results:")
   for i, idx in enumerate(sort_idx):
       print(f"Mode {i+1}: f = {results['frequencies'][idx]:.2f} Hz, "
             f"ζ = {results['damping_ratios'][idx]:.4f}")

Example 8: ERA with Stabilization Diagram
------------------------------------------

Using stabilization diagrams for model order selection.

.. code-block:: python

   """
   ERA with stabilization diagram
   """
   import numpy as np
   import matplotlib.pyplot as plt
   from time_domain_modal_estimation import (
       generate_impulse_response,
       stabilization_diagram
   )

   # Parameters
   dt = 0.01
   t = np.arange(0, 5, dt)
   
   frequencies = [3.0, 7.0]
   damping = [0.025, 0.035]
   mode_shapes = [[1.0], [0.7]]

   # Generate data
   Y = generate_impulse_response(frequencies, damping, mode_shapes, t, n_outputs=1)
   Y += np.random.normal(0, 0.005, Y.shape)

   # Create stabilization diagram
   diagram = stabilization_diagram(
       Y=Y,
       dt=dt,
       max_order=20,
       r=100,
       s=100,
       freq_tol=0.01,
       damp_tol=0.05
   )

   # Plot
   fig, ax = plt.subplots(figsize=(10, 6))
   
   # Unstable poles
   unstable = ~diagram['stability']
   ax.plot(diagram['frequencies'][unstable], 
           diagram['orders'][unstable],
           'o', color='lightgray', markersize=4, label='Unstable')
   
   # Stable poles
   stable = diagram['stability']
   ax.plot(diagram['frequencies'][stable], 
           diagram['orders'][stable],
           'ro', markersize=6, label='Stable')
   
   ax.set_xlabel('Frequency (Hz)')
   ax.set_ylabel('Model Order')
   ax.set_title('Stabilization Diagram')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.show()

   print("Stable poles typically align vertically at true frequencies")

Running the Examples
--------------------

The package includes ready-to-run example scripts in the ``examples/`` directory:

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   python examples/basic_usage.py

CEA Demonstration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/demo_cea.py

This generates a comprehensive demonstration with visualization saved as ``cea_results.png``.

ERA Demonstration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/demo_era.py

This generates a comprehensive ERA demonstration with visualization saved as ``era_results.png``.
