# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from datetime import datetime

from kikuchipy import release

sys.path.append('../')

# -- Project information -----------------------------------------------------

project = 'KikuchiPy'
copyright = str(datetime.now().year) + ', ' + release.author
author = release.author

# The full version, including alpha/beta/rc tags
release = release.version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Create links to references within KikuchiPy's documentation to these
# packages.
intersphinx_mapping = {
    'dask': ('https://docs.dask.org/en/latest', None),
    'hyperspy': ('http://hyperspy.org/hyperspy-doc/current', None),
    'matplotlib': ('https://matplotlib.org', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'python': ('https://docs.python.org/3', None),
    'pyxem': ('https://pyxem.github.io/pyxem-website/docstring', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', ]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', ]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', ]

master_doc = 'index'
pygments_style = 'solarized-light'

# Logo
#html_logo = '_static/'
