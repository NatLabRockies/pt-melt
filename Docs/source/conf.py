# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pt-melt"
copyright = "2023, Alliance for Sustainable Energy"
author = "Nicholas T. Wimer"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_rtd_theme",
]

autosummary_generate = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,  # Do not collapse the navigation bar
    "sticky_navigation": True,  # Make the navigation bar sticky
    "navigation_depth": 5,  # Maximum depth of the navigation tree
    "titles_only": False,  # Show all titles in the navigation bar
    "includehidden": False,  # Do not include hidden documents
    "style_external_links": True,  # Add icons to external links
    "style_nav_header_background": "#2980B9",  # Set the navigation bar color
}
