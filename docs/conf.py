import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "VAE mixin PyTorch"
copyright = "2023, Xiao Yuan"
author = "Xiao Yuan"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

toc_object_entries_show_parents = "hide"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
}

autodoc_member_order = "bysource"
autodoc_mock_imports = ["torch"]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True