# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import roiextract

from intersphinx_registry import get_intersphinx_mapping


project = "ROIextract"
copyright = "2024-2025, ROIextract contributors"
author = "ROIextract contributors"
release = roiextract.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # contrib
    "numpydoc",
    "sphinx_copybutton",
    # "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False


# -- Copybutton settings -----------------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#strip-and-configure-input-prompts-for-code-cells

copybutton_exclude = ".linenos, .gp"


# -- Autodoc settings --------------------------------------------------------

autodoc_typing_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
}


# -- Intersphinx settings ----------------------------------------------------

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
intersphinx_mapping.update(
    get_intersphinx_mapping(packages={"matplotlib", "mne", "numpy", "python"})
)

# Numpydoc
numpydoc_attributes_as_param_list = True
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "bool": ":ref:`bool <python:typebool>`",
    # MNE-Python
    "Forward": "mne.Forward",
    "InverseOperator": "mne.minimum_norm.InverseOperator",
    "Label": "mne.Label",
    "Raw": "mne.io.Raw",
    "SourceSpaces": "mne.SourceSpaces",
    # ROIextract
    "SpatialFilter": "roiextract.filter.SpatialFilter",
}
numpydoc_xref_ignore = {
    "type",
    "optional",
    "default",
    "or",
    "shape",
    "n_series",
    "n_times",
}
