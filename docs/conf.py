# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Time Domain Modal Estimation'
copyright = '2026, Ashley S. Dale'
author = 'Ashley S. Dale'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autosummary_generate = True

# MyST settings
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/daleas0120/time_domain_modal_estimation/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = f"{project} v{release}"

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
}
