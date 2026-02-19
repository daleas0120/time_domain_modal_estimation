# Contributing to Time Domain Modal Estimation

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/time_domain_modal_estimation.git
cd time_domain_modal_estimation
```

1. Create a virtual environment and install development dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,docs]"
```

1. Create a new branch for your feature:

```bash
git checkout -b feature/your-feature-name
```

## Running Tests

Run the test suite locally before submitting:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ -v --cov=src/time_domain_modal_estimation --cov-report=html
```

## Code Style

This project uses [Black](https://github.com/psf/black) for code formatting:

```bash
black src/ tests/ examples/
```

Check for linting issues:

```bash
flake8 src/ --max-line-length=127
```

## Documentation

Build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

## Continuous Integration

All pull requests are automatically tested via GitHub Actions:

- **Tests**: Run on Python 3.8-3.12 across Ubuntu, macOS, and Windows
- **Linting**: Code is checked with flake8 for syntax errors
- **Documentation**: Documentation build is verified

Make sure your changes pass all CI checks before requesting review.

## Pull Request Process

1. Update tests to cover your changes
2. Update documentation if you're adding/changing functionality
3. Ensure all tests pass locally
4. Update the CHANGELOG.md if applicable
5. Submit a pull request with a clear description of changes

## Commit Messages

Use clear, descriptive commit messages:

- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 50 characters
- Add detailed description after a blank line if needed

Examples:

```
Add ERA stabilization diagram feature

Implement stabilization diagram generation for model order
selection in the ERA algorithm. Includes tolerance parameters
for frequency and damping stability checks.
```

## Adding New Algorithms

If you're adding a new modal estimation algorithm:

1. Create a new module in `src/time_domain_modal_estimation/`
2. Follow the existing code structure and documentation style
3. Add comprehensive tests in `tests/`
4. Create a demonstration script in `examples/`
5. Add documentation in `docs/`
6. Update the main `__init__.py` to export key functions
7. Add references to the algorithm in `docs/references.rst`

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue for:

- Bug reports
- Feature requests
- Documentation improvements
- Questions about usage or contributing

Thank you for contributing!
