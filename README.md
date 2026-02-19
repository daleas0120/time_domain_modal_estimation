# Time Domain Modal Estimation

[![Tests](https://github.com/daleas0120/time_domain_modal_estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/daleas0120/time_domain_modal_estimation/actions/workflows/tests.yml)
[![Documentation](https://github.com/daleas0120/time_domain_modal_estimation/actions/workflows/docs.yml/badge.svg)](https://daleas0120.github.io/time_domain_modal_estimation/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Is Vibe Coded](https://img.shields.io/badge/is-vibe--coded-purple.svg)](https://github.com/danielrosehill/Is-Vibe-Coded)

A Python package for extracting modal parameters from time-domain response data using the Complex Exponential Algorithm (CEA) and Eigensystem Realization Algorithm (ERA).

## Overview

This package implements time-domain modal estimation techniques for identifying modal parameters (natural frequencies, damping ratios, and mode shapes) from measured response data. Two primary methods are implemented:

1. **Complex Exponential Algorithm (CEA)** - for free decay response data
2. **Eigensystem Realization Algorithm (ERA)** - for impulse response data

Both methods are derived from equations in classical modal analysis literature.

## Installation

### From source

```bash
git clone https://github.com/daleas0120/time_domain_modal_estimation.git
cd time_domain_modal_estimation
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from time_domain_modal_estimation import complex_exponential_algorithm

# Generate synthetic response data
dt = 0.01  # Time step (seconds)
t = np.arange(0, 5, dt)
f = 5.0  # Frequency (Hz)
zeta = 0.03  # Damping ratio
omega_n = 2 * np.pi * f
y = np.exp(-zeta * omega_n * t) * np.cos(omega_n * np.sqrt(1 - zeta**2) * t)

# Apply CEA algorithm
results = complex_exponential_algorithm(
    y=y,
    dt=dt,
    n_modes=1
)

print(f"Estimated frequency: {results['frequencies'][0]:.4f} Hz")
print(f"Estimated damping: {results['damping_ratios'][0]:.4f}")
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0 (for visualization)

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{dale2026time_domain_modal_estimation,
  author = {Dale, Ashley S.},
  title = {Time Domain Modal Estimation},
  year = {2026},
  url = {https://github.com/daleas0120/time_domain_modal_estimation},
  version = {0.1.0}
}
```

A `CITATION.cff` file is also provided for automatic citation generation on GitHub.

## References

Fahey, S. O'F., & Pratt, J. (1998). Time domain modal estimation techniques. *Experimental Techniques*, 22(6), 45-49.

Juang, J.-N., & Pappa, R. S. (1985). An eigensystem realization algorithm for modal parameter identification and model reduction. *Journal of Guidance, Control, and Dynamics*, 8(5), 620-627.

```bibtex
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

@article{juang1985eigensystem,
  title={An eigensystem realization algorithm for modal parameter identification and model reduction},
  author={Juang, Jer-Nan and Pappa, Richard S},
  journal={Journal of guidance, control, and dynamics},
  volume={8},
  number={5},
  pages={620--627},
  year={1985}
}
```
