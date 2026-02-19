Quick Start Guide
=================

This guide will help you get started with the Complex Exponential Algorithm (CEA) for modal parameter estimation.

Basic Usage
-----------

Single Mode Example
~~~~~~~~~~~~~~~~~~~

Here's a simple example estimating parameters for a single mode:

.. code-block:: python

   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Time parameters
   dt = 0.01  # Time step (seconds)
   t = np.arange(0, 5, dt)

   # True modal parameters
   f_true = 5.0  # Frequency (Hz)
   zeta_true = 0.03  # Damping ratio
   omega_n = 2 * np.pi * f_true
   omega_d = omega_n * np.sqrt(1 - zeta_true**2)

   # Generate synthetic response
   y = 1.0 * np.exp(-zeta_true * omega_n * t) * np.cos(omega_d * t)

   # Apply CEA algorithm
   results = complex_exponential_algorithm(y=y, dt=dt, n_modes=1)

   # Display results
   print(f"Estimated frequency: {results['frequencies'][0]:.4f} Hz")
   print(f"Estimated damping: {results['damping_ratios'][0]:.4f}")

Multi-Mode Example
~~~~~~~~~~~~~~~~~~

For systems with multiple modes:

.. code-block:: python

   import numpy as np
   from time_domain_modal_estimation import complex_exponential_algorithm

   # Generate multi-mode response
   dt = 0.01
   t = np.arange(0, 10, dt)

   # Mode 1: 2.5 Hz, 2% damping
   f1, zeta1 = 2.5, 0.02
   omega_n1 = 2 * np.pi * f1
   lambda1 = -zeta1 * omega_n1 + 1j * omega_n1 * np.sqrt(1 - zeta1**2)
   y1 = np.real(1.0 * np.exp(lambda1 * t) + 1.0 * np.exp(np.conj(lambda1) * t))

   # Mode 2: 5.0 Hz, 3% damping
   f2, zeta2 = 5.0, 0.03
   omega_n2 = 2 * np.pi * f2
   lambda2 = -zeta2 * omega_n2 + 1j * omega_n2 * np.sqrt(1 - zeta2**2)
   y2 = np.real(0.6 * np.exp(lambda2 * t) + 0.6 * np.exp(np.conj(lambda2) * t))

   # Combined response
   y = y1 + y2

   # Estimate parameters
   results = complex_exponential_algorithm(y=y, dt=dt, n_modes=2)

   # Sort by frequency
   sort_idx = np.argsort(results['frequencies'])
   for i in sort_idx:
       print(f"Mode {i+1}:")
       print(f"  Frequency: {results['frequencies'][i]:.4f} Hz")
       print(f"  Damping: {results['damping_ratios'][i]:.4f}")

Understanding Results
---------------------

The ``complex_exponential_algorithm`` function returns a dictionary with:

* ``frequencies``: Natural frequencies in Hz
* ``damping_ratios``: Damping ratios (fraction of critical damping)
* ``mode_shapes``: Modal participation factors (complex)
* ``poles``: System poles in the z-domain
* ``lambda``: Modal frequencies in the continuous domain (complex)
* ``reconstruction``: Reconstructed time series

Reconstruction Quality
~~~~~~~~~~~~~~~~~~~~~~

Check how well the model fits the data:

.. code-block:: python

   import numpy as np

   # Compute normalized error
   error = np.linalg.norm(y - results['reconstruction']) / np.linalg.norm(y)
   r_squared = 1 - error**2

   print(f"Reconstruction error: {error:.6f}")
   print(f"R² score: {r_squared:.6f}")

Visualization
-------------

Basic Time Series Plot
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(12, 4))
   plt.plot(t, y, 'b-', label='Measured', linewidth=1.5)
   plt.plot(t, results['reconstruction'], 'r--', label='Reconstructed', linewidth=1.5)
   plt.xlabel('Time (s)')
   plt.ylabel('Response')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Pole Plot
~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   poles = results['poles']
   
   plt.figure(figsize=(6, 6))
   plt.plot(np.real(poles), np.imag(poles), 'ro', markersize=10, label='Poles')
   
   # Unit circle
   theta = np.linspace(0, 2*np.pi, 100)
   plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
   
   plt.xlabel('Real Part')
   plt.ylabel('Imaginary Part')
   plt.title('System Poles')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.axis('equal')
   plt.show()

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

* Ensure adequate sampling rate (at least 10x highest mode frequency)
* Record enough cycles (minimum 3-5, preferably more)
* Minimize measurement noise when possible

Model Order Selection
~~~~~~~~~~~~~~~~~~~~~

* Start with expected number of modes
* Check reconstruction quality
* Be aware that noise can lead to spurious modes

Noise Considerations
~~~~~~~~~~~~~~~~~~~~

The CEA algorithm is sensitive to noise. For noisy data:

* Use longer time series
* Consider pre-filtering
* Validate results against known physics

Complete Examples
-----------------

See the ``examples/`` directory for complete working examples:

* ``basic_usage.py``: Simple single-mode example
* ``demo_cea.py``: Comprehensive demonstration with visualization

Run them with:

.. code-block:: bash

   python examples/basic_usage.py
   python examples/demo_cea.py
