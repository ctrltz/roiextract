# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ROIextract"
copyright = "2024, Nikolai Kapralov"
author = "Nikolai Kapralov"
release = "0.0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]


# -- Copybutton settings -----------------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#strip-and-configure-input-prompts-for-code-cells

copybutton_exclude = ".linenos, .gp"


# -- Autodoc settings --------------------------------------------------------

autodoc_typing_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
}


# -- Intersphinx settings ----------------------------------------------------

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


# -- Napoleon settings -------------------------------------------------------

napoleon_numpy_docstring = True
