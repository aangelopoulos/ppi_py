# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ppi_py'
copyright = '2023, Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, Tijana Zrnic, and others'
author = 'Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, Tijana Zrnic, and others'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode']

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'logo': 'ppi.svg',
    'github_user': 'aangelopoulos',
    'github_repo': 'ppi_py',
    'github_button': True,
    'github_type': 'star',
    'analytics_id': 'G-S4WVQMJCM9',
    'sidebar_collapse': True,
    'page_width': '75%',
    'sidebar_width': '25%',
    'description': 'A python package for powerful statistical inference using machine learning',
}
