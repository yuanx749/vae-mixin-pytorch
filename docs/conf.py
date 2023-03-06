import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "VAE mixin PyTorch"
copyright = "2023, yuanx749"
author = "yuanx749"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
toc_object_entries_show_parents = "hide"

html_theme = "furo"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_mock_imports = ["torch"]
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
