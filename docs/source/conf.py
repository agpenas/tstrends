# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TStrends"
copyright = "2025, Abel G Penas"
author = "Abel G Penas"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx extension for API documentation
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "nbsphinx",  # Jupyter notebook integration
    "sphinx_sitemap",  # Generate sitemap.xml
]

# Set up IPython lexer for code highlighting
from sphinx.highlighting import lexers
from ipython_pygments_lexers import IPythonLexer

lexers["ipython3"] = IPythonLexer()

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
    "bayes_opt": (
        "https://bayesian-optimization.github.io/BayesianOptimization/master/",
        None,
    ),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type hints configuration
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = {
    "NDArray": "numpy.ndarray",
    "Labels": "tstrends.trend_labelling.Labels",
}
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Add notebooks to the path
import os

notebooks_path = os.path.abspath("../../notebooks")
sys.path.insert(0, notebooks_path)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# Disable the Python module index
html_domain_indices = (
    []
)  # Empty list disables all domain-specific indices, including the Python Module Index
html_use_modindex = False  # Explicitly disable the module index
html_use_index = False  # Explicitly disable the index

# Copy robots.txt to the output directory
html_extra_path = ["_static/robots.txt"]

# Sitemap configuration
html_baseurl = "https://tstrends.xyz/en/latest/"
sitemap_filename = "sitemap.xml"
sitemap_url_scheme = "{link}"
sitemap_locales = [None]
# Exclude pages that shouldn't be indexed
sitemap_excludes = [
    "search.html",
    "genindex.html",
    "py-modindex.html",
    "_modules/index.html",
    "_modules/*",  # Exclude all module source pages
    "_sources/*",  # Exclude source files
]

# Theme options for controlling the sidebar and navigation
html_theme_options = {
    # Navigation depth - how many levels to display in the left sidebar
    "navigation_depth": 3,  # Set to -1 for unlimited depth or a positive integer to limit depth
    # Other useful options for navigation control
    "collapse_navigation": False,  # Don't collapse navigation sections
    "sticky_navigation": True,  # Make navigation bar sticky (always visible)
    "titles_only": False,  # Show full section titles in navigation, not just page titles
}

# -- Options for nitpicky mode ----------------------------------------------
nitpicky = True  # This will warn about all broken references
nitpick_ignore = [
    ("py:class", "None"),
    ("py:class", "optional"),
    ("py:class", "typing.Union"),
    ("py:class", "BayesianOptimization"),
    ("py:class", "acquisition.AcquisitionFunction"),
    ("py:class", "tstrends.trend_labelling.base_labeller.BaseLabeller"),
    ("py:class", "tstrends.trend_labelling.oracle_labeller.BaseOracleTrendLabeller"),
    ("py:class", "BaseLabeller"),
    ("py:class", "BaseOracleTrendLabeller"),
    ("py:class", "BaseReturnEstimator"),
    ("py:class", "tstrends.returns_estimation.returns_estimation.BaseReturnEstimator"),
    ("py:class", "TrendState"),
    ("py:class", "binary_CTL.TrendState"),
    # Notebook section references that can be ignored
    ("ref", "/notebooks/labellers_catalogue.ipynb#1-using-the-binary-ctl-labeller"),
    ("ref", "/notebooks/labellers_catalogue.ipynb#optimize-different-labellers"),
    ("ref", "/notebooks/labellers_catalogue.ipynb#3-using-the-binary-oracle-labeller"),
    ("ref", "/notebooks/labellers_catalogue.ipynb#4-using-the-ternary-oracle-labeller"),
]

# nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during the build - include as-is
nbsphinx_allow_errors = True  # Continue build even if notebooks have errors
nbsphinx_timeout = 300  # Reduced timeout to improve build speed
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png'}",  # Use PNG only for smaller file sizes
    "--InlineBackend.rc={'figure.dpi': 72}",  # Lower DPI for smaller images
]
nbsphinx_kernel_name = "python3"  # Use the Python 3 kernel

# Skip using Pandoc for conversion
nbsphinx_pandoc_args = ["--no-highlight"]  # Use this minimal option
nbsphinx_use_xhtml = True

# Additional nbsphinx settings for better integration and performance
nbsphinx_prompt_width = "0"  # Remove the prompt column
nbsphinx_thumbnails = {}  # Enable thumbnails for notebooks

# Disable the download link to reduce unnecessary processing
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::
   
   This page was generated from a Jupyter notebook.
"""

# -- Optimize HTML output for speed ------------------------------------------
html_scaled_image_link = False  # Don't create scaled image links
html_copy_source = False  # Don't copy source files
html_show_sourcelink = False  # Don't show source links

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Custom function to handle internal links better
from docutils import nodes
from docutils.parsers.rst import roles


def notebook_section_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Custom role to handle notebook section links.
    This resolves links like `/notebooks/labellers_catalogue.ipynb#section` properly.
    """
    url = text
    if url.startswith("/notebooks/"):
        url = url[1:]  # Remove leading slash
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []


# Register the custom role
roles.register_local_role("notebook_section", notebook_section_role)

# -- Options for autodoc extension ------------------------------------------
# Add an autodoc_overrides dictionary to add :no-index: to the duplicate entries
autodoc_overrides = {
    "tstrends.optimization.OptimizationBounds.implemented_labellers": {
        "no-index": True
    },
    "tstrends.returns_estimation.FeesConfig.lp_transaction_fees": {"no-index": True},
    "tstrends.returns_estimation.FeesConfig.sp_transaction_fees": {"no-index": True},
    "tstrends.returns_estimation.FeesConfig.lp_holding_fees": {"no-index": True},
    "tstrends.returns_estimation.FeesConfig.sp_holding_fees": {"no-index": True},
}
