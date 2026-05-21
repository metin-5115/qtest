# ============================================================================
#  qtest — Sphinx configuration
#  https://www.sphinx-doc.org/en/master/usage/configuration.html
# ============================================================================
from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

# ----------------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------------
# Make the package importable so autodoc can introspect it without an install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ----------------------------------------------------------------------------
# Project information
# ----------------------------------------------------------------------------
project = "qtest"
author = "Metin Tuncbilek"
copyright = f"{datetime.now():%Y}, {author}"

try:
    release = metadata.version("qtest")
except metadata.PackageNotFoundError:
    # Editable / un-installed checkout — fall back to the source __version__.
    from qtest import __version__ as release  # type: ignore[no-redef]

version = ".".join(release.split(".")[:2])

# ----------------------------------------------------------------------------
# General configuration
# ----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source files — accept both reStructuredText and Markdown.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The default role for `text` in rst — render as code rather than emphasis.
default_role = "code"

# Allow `todo::` directives to render.
todo_include_todos = True

# ----------------------------------------------------------------------------
# Autodoc / autosummary
# ----------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True

# Mock heavy / optional imports so RTD builds do not require Qiskit, Aer, etc.
autodoc_mock_imports = [
    "qiskit",
    "qiskit_aer",
    "cirq",
    "pennylane",
    "matplotlib",
]

# ----------------------------------------------------------------------------
# Napoleon (NumPy / Google docstring support)
# ----------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# ----------------------------------------------------------------------------
# Intersphinx — cross-link to upstream docs
# ----------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "hypothesis": ("https://hypothesis.readthedocs.io/en/latest/", None),
}
intersphinx_timeout = 10

# ----------------------------------------------------------------------------
# MyST parser
# ----------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# ----------------------------------------------------------------------------
# HTML output — Furo theme
# ----------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"qtest {version}"
html_short_title = "qtest"
html_show_sourcelink = True
html_copy_source = False

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/metin-5115/qtest/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/metin-5115/qtest",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
    "light_css_variables": {
        "color-brand-primary": "#5b3a86",
        "color-brand-content": "#5b3a86",
        "font-stack": (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
            "Helvetica, Arial, sans-serif"
        ),
        "font-stack--monospace": (
            "'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, "
            "'Liberation Mono', monospace"
        ),
    },
    "dark_css_variables": {
        "color-brand-primary": "#b08fdc",
        "color-brand-content": "#b08fdc",
    },
}

# ----------------------------------------------------------------------------
# Copy-button
# ----------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------
nitpicky = False
suppress_warnings = ["myst.header"]

# If we are building on Read the Docs, expose that so templates can react.
on_rtd = os.environ.get("READTHEDOCS", "False") == "True"
