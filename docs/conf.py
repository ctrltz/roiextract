# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import roiextract

from intersphinx_registry import get_intersphinx_mapping


year = datetime.datetime.now().year
project = "ROIextract"
copyright = f"2024-{year}, ROIextract contributors"
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
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/ctrltz/roiextract",
            icon="fa-brands fa-square-github fa-fw",
        ),
    ],
    # include class methods in the per-page TOC
    "show_toc_level": 2,
}

# -- Copybutton settings -----------------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#strip-and-configure-input-prompts-for-code-cells

copybutton_exclude = ".linenos, .gp"


# -- Autodoc settings --------------------------------------------------------

autodoc_typehints = "none"
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
    "Info": "mne.Info",
    "Covariance": "mne.Covariance",
    "SourceSpaces": "mne.SourceSpaces",
    "SourceEstimate": "mne.SourceEstimate",
    # ROIextract
    "SpatialFilter": "roiextract.filter.SpatialFilter",
}
numpydoc_xref_ignore = {
    "type",
    "optional",
    "default",
    "or",
    "of",
    "shape",
    "n_times",
    "n_sources",
    "n_sensors",
    "n_labels",
}
