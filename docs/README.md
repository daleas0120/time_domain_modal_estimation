# Documentation

This directory contains the Sphinx documentation for the time-domain-modal-estimation package.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install individually:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Build HTML Documentation

On Linux/macOS:

```bash
cd docs
make html
```

On Windows:

```bash
cd docs
make.bat html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Other Formats

Build PDF documentation (requires LaTeX):

```bash
make latexpdf
```

Build other formats:

```bash
make epub      # EPUB format
make text      # Plain text
make man       # Manual pages
```

### Clean Build

Remove all built files:

```bash
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation page
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start guide
- `theory.rst` - Theoretical background
- `examples.rst` - Usage examples
- `api/` - API reference documentation
- `references.rst` - References and citations
- `changelog.rst` - Version history
- `license.rst` - License information

## Viewing Locally

After building, you can start a simple HTTP server to view the docs:

```bash
cd _build/html
python -m http.server 8000
```

Then open <http://localhost:8000> in your browser.

## Read the Docs

This documentation is configured for automatic building on Read the Docs.
