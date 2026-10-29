# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Sphinx configuration for the DynkinLab documentation site."""

import importlib.metadata

release = importlib.metadata.version("dynkinlab")
version = release
project = "DynkinLab"
author = "Mario Abarca"
copyright = "2026-present Mario Abarca"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

html_theme = "furo"
html_title = f"{project} {release}"
html_theme_options = {
    "source_repository": "https://github.com/knkillname/dynkinlab/",
    "source_branch": "main",
    "source_directory": "docs/",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
