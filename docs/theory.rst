Theory
======

This section describes the theoretical foundation of the Complex Exponential Algorithm (CEA) for time-domain modal estimation.

Mathematical Background
-----------------------

Free Vibration Response
~~~~~~~~~~~~~~~~~~~~~~~

For a multi-degree-of-freedom linear system with viscous damping, the free vibration response can be expressed as a sum of decaying exponentials:

.. math::

   y(t) = \sum_{k=1}^{n} A_k e^{\lambda_k t}

where:

* :math:`A_k` are the modal participation factors (complex)
* :math:`\lambda_k` are the modal frequencies (complex)
* :math:`n` is the number of modes

Modal Frequencies
~~~~~~~~~~~~~~~~~

The modal frequencies are complex:

.. math::

   \lambda_k = -\zeta_k \omega_{n,k} \pm i\omega_{d,k}

where:

* :math:`\omega_{n,k}` is the natural frequency of mode :math:`k`
* :math:`\zeta_k` is the damping ratio (fraction of critical damping)
* :math:`\omega_{d,k} = \omega_{n,k}\sqrt{1-\zeta_k^2}` is the damped frequency

For real-valued responses, the modal frequencies occur in complex conjugate pairs.

Complex Exponential Algorithm
------------------------------

The CEA extracts modal parameters through the following steps:

Step 1: Toeplitz Matrix (Equation 7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a Toeplitz matrix from the response time series:

.. math::

   Y = \begin{bmatrix}
   y_1 & y_2 & \cdots & y_{2n} \\
   y_2 & y_3 & \cdots & y_{2n+1} \\
   \vdots & \vdots & \ddots & \vdots \\
   y_{N-2n} & y_{N-2n+1} & \cdots & y_{N-1}
   \end{bmatrix}

where :math:`N` is the total number of samples and :math:`n` is the number of modes.

Step 2: Polynomial Coefficients (Equation 11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve the least squares problem:

.. math::

   Y \boldsymbol{\alpha} = -\mathbf{y}_{target}

where :math:`\mathbf{y}_{target} = [y_{2n+1}, y_{2n+2}, \ldots, y_N]^T` and :math:`\boldsymbol{\alpha} = [\alpha_0, \alpha_1, \ldots, \alpha_{2n-1}]^T` are the characteristic polynomial coefficients.

Step 3: System Poles (Equation 8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find the roots of the characteristic polynomial:

.. math::

   \prod_{k=1}^{2n} (z - z_k) = z^{2n} + \alpha_{2n-1}z^{2n-1} + \cdots + \alpha_1 z + \alpha_0 = 0

The roots :math:`z_k` are the system poles in the discrete-time domain.

Step 4: Modal Frequencies (Equations 9-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert discrete-time poles to continuous-time modal frequencies:

.. math::

   \lambda_k = \frac{1}{\Delta t} \ln(z_k)

where :math:`\Delta t` is the sampling interval.

In Hz:

.. math::

   f_k = \frac{1}{2\pi\Delta t} \ln(z_k)

Step 5: Vandermonde Matrix (Equation 12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build the modal matrix:

.. math::

   \Lambda_{1,N} = \begin{bmatrix}
   e^{\lambda_1 t_1} & e^{\lambda_1 t_2} & \cdots & e^{\lambda_1 t_N} \\
   e^{\lambda_2 t_1} & e^{\lambda_2 t_2} & \cdots & e^{\lambda_2 t_N} \\
   \vdots & \vdots & \ddots & \vdots \\
   e^{\lambda_n t_1} & e^{\lambda_n t_2} & \cdots & e^{\lambda_n t_N}
   \end{bmatrix}

Step 6: Mode Shapes (Equation 13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve for the modal participation factors:

.. math::

   \mathbf{A} = \Lambda_{1,N}^{-1} \mathbf{y}

where :math:`\mathbf{A} = [A_1, A_2, \ldots, A_n]^T`.

Since :math:`\Lambda` is typically not square, the pseudo-inverse is used:

.. math::

   \mathbf{A} = (\Lambda^T \Lambda)^{-1} \Lambda^T \mathbf{y}

Physical Interpretation
-----------------------

Natural Frequency
~~~~~~~~~~~~~~~~~

The natural frequency :math:`f_k` is:

.. math::

   f_k = \frac{|\lambda_k|}{2\pi}

in Hz, or:

.. math::

   \omega_{n,k} = |\lambda_k|

in rad/s.

Damping Ratio
~~~~~~~~~~~~~

The damping ratio is:

.. math::

   \zeta_k = -\frac{\text{Re}(\lambda_k)}{|\lambda_k|}

A positive damping ratio indicates stable (decaying) modes.

Damped Frequency
~~~~~~~~~~~~~~~~

The damped frequency is:

.. math::

   \omega_{d,k} = \text{Im}(\lambda_k)

or in Hz:

.. math::

   f_{d,k} = \frac{\omega_{d,k}}{2\pi}

Assumptions and Limitations
----------------------------

Assumptions
~~~~~~~~~~~

1. **Linear system**: The structure behaves linearly
2. **Viscous damping**: Damping is proportional to velocity
3. **Free vibration**: No external forcing during measurement
4. **Observable modes**: All modes of interest are excited
5. **Stationary**: System properties don't change during measurement

Limitations
~~~~~~~~~~~

1. **Noise sensitivity**: CEA is sensitive to measurement noise
2. **Model order**: Requires knowing or estimating the number of modes
3. **Closely spaced modes**: May have difficulty distinguishing modes with similar frequencies
4. **Non-proportional damping**: Assumes classical (proportional) damping for real mode shapes

Practical Considerations
------------------------

Sampling Requirements
~~~~~~~~~~~~~~~~~~~~~

* **Nyquist criterion**: Sample at least 2x the highest frequency of interest
* **Recommended**: 10x the highest frequency for good accuracy
* **Duration**: Record at least 3-5 complete cycles of the lowest frequency

Model Order Selection
~~~~~~~~~~~~~~~~~~~~~

The model order :math:`n` must be chosen appropriately:

* Too low: Misses important modes
* Too high: Introduces spurious (noise-driven) poles

Consider using stabilization diagrams or information criteria for order selection.

Noise Mitigation
~~~~~~~~~~~~~~~~

For noisy data:

* Use longer time series
* Apply appropriate filters before analysis
* Average multiple measurements if possible
* Validate results against physical expectations

Eigensystem Realization Algorithm
----------------------------------

The ERA extracts modal parameters from impulse response data by constructing a minimal state-space realization.

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~

For a linear time-invariant system, the impulse response can be expressed as:

.. math::

   y_k = C A^{k-1} B

where:

* :math:`A` is the state matrix
* :math:`B` is the input matrix
* :math:`C` is the output matrix
* :math:`k` is the discrete time index

The modal parameters are extracted from the eigenvalues and eigenvectors of the state matrix :math:`A`.

ERA Algorithm Steps
~~~~~~~~~~~~~~~~~~~

Step 1: Hankel Matrix (Equation 29)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build a Hankel matrix from the impulse response:

.. math::

   H(k) = \begin{bmatrix}
   y_k & y_{k+1} & \cdots & y_{k+s-1} \\
   y_{k+1} & y_{k+2} & \cdots & y_{k+s} \\
   \vdots & \vdots & \ddots & \vdots \\
   y_{k+r-1} & y_{k+r} & \cdots & y_{k+r+s-2}
   \end{bmatrix}

where :math:`r` and :math:`s` determine the size of the Hankel matrix.

Step 2: Singular Value Decomposition (Equation 30)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform SVD on the Hankel matrix:

.. math::

   H(0) = U \Sigma V^T

where :math:`U` and :math:`V` are orthogonal matrices and :math:`\Sigma` contains the singular values.

Step 3: Reduced-Order Projection (Equation 31-32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Project onto the dominant subspace:

.. math::

   U_n = U[:, :n], \quad \Sigma_n = \Sigma[:n, :n], \quad V_n = V[:, :n]

where :math:`n` is the model order (typically :math:`2 \times` number of modes).

Step 4: State-Space Realization (Equation 33-35)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recover the system matrices:

.. math::

   A = \Sigma_n^{-1/2} U_n^T H(1) V_n \Sigma_n^{-1/2}

.. math::

   B = \Sigma_n^{1/2} V_n^T e_1

.. math::

   C = e_1^T U_n \Sigma_n^{1/2}

where :math:`e_1` is the first unit vector.

Step 5: Modal Parameter Extraction (Equations 27-28)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract modal parameters from eigenvalues of :math:`A`:

.. math::

   \mu_k = \frac{\ln(\lambda_k)}{\Delta t}

where :math:`\lambda_k` are the eigenvalues of :math:`A` and :math:`\Delta t` is the sampling interval.

Natural frequencies (Hz):

.. math::

   f_k = \frac{|\mu_k|}{2\pi}

Damping ratios:

.. math::

   \zeta_k = -\frac{\text{Re}(\mu_k)}{|\mu_k|}

ERA vs CEA Comparison
~~~~~~~~~~~~~~~~~~~~~

+----------------------+-------------------------------+--------------------------------+
| Feature              | CEA                           | ERA                            |
+======================+===============================+================================+
| Input data           | Free decay response           | Impulse response               |
+----------------------+-------------------------------+--------------------------------+
| Matrix formulation   | Toeplitz + polynomial roots   | Hankel + SVD                   |
+----------------------+-------------------------------+--------------------------------+
| Noise robustness     | Moderate                      | Good (SVD filtering)           |
+----------------------+-------------------------------+--------------------------------+
| Model order          | Must specify exactly          | Flexible via SVD truncation    |
+----------------------+-------------------------------+--------------------------------+
| Computational cost   | Lower                         | Higher (SVD)                   |
+----------------------+-------------------------------+--------------------------------+
| Multiple outputs     | Single output                 | Multiple outputs native        |
+----------------------+-------------------------------+--------------------------------+

References
----------

The implementation is based on:

    Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques. 
    *Experimental Techniques*, 22(6), 45-49.

    Juang, J.-N., & Pappa, R. S. (1985). An eigensystem realization algorithm for modal parameter identification and model reduction. 
    *Journal of Guidance, Control, and Dynamics*, 8(5), 620-627.

See also classical modal analysis literature for additional context and extensions.
