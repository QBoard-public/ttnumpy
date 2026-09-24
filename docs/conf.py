# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "TTNumPy"
copyright = "2026, Cloud Quantum Technologies Limited Liability Company"
author = "Oleg Tatarkin, Sergey Usmanov"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

_INTERSPHINX_CANDIDATES = {
    "python": "https://docs.python.org/3",
    "numpy": "https://numpy.org/doc/stable",
    "scipy": "https://docs.scipy.org/doc/scipy",
}


def _reachable_inventories() -> dict[str, tuple[str, None]]:
    import requests

    reachable = {}
    for name, base in _INTERSPHINX_CANDIDATES.items():
        try:
            response = requests.get(f"{base}/objects.inv", timeout=3, stream=True)
            response.close()
            if response.ok:
                reachable[name] = (base, None)
        except Exception:  # offline, proxied, DNS-blocked -- all equivalent here
            pass
    return reachable


intersphinx_mapping = _reachable_inventories()
intersphinx_timeout = 15

# Fail the build on any cross-reference that cannot be resolved, so the API
# reference cannot silently rot. Everything genuinely unresolvable is listed
# below rather than tolerated globally.
nitpicky = True

nitpick_ignore = [
    # CPython's own inventory registers typing.Union as py:obj, not py:data,
    # so the annotation sphinx-autodoc-typehints emits can never resolve.
    ("py:data", "typing.Union"),
]

nitpick_ignore_regex = [
    # Third-party and stdlib names that autodoc emits into type annotations.
    # Intersphinx resolves these into real links whenever it can reach the
    # inventories; ignoring them here keeps a build without network access
    # from failing, so nitpicky enforces *our* references rather than the
    # runner's connectivity.
    (r"py:.*", r"(numpy|scipy|typing|enum|logging|pathlib|warnings)(\..*)?$"),
]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../branding/logo/ttnumpy-mark.svg"
html_favicon = "../branding/logo/ttnumpy-mark.svg"
html_css_files = ["custom.css"]
latex_logo = "../branding/logo/ttnumpy-logo-print.png"

latex_elements = {
    "extraclassoptions": "openany",
    # With no \chapter in the document the report class leaves its chapter
    # counter at zero, which numbers sections 0.1, 0.2, ... and would number
    # figures and tables the same way. Drop the chapter prefix from each.
    "preamble": "\n".join(
        [
            # pdflatex aborts on any Unicode character it has no mapping for.
            # Sphinx ships mappings for punctuation (dashes, times, umlauts)
            # but not for Greek, which is easy to type into a docstring in a
            # numerics library. Map the likely ones to their math equivalents
            # so a stray character renders instead of breaking the build.
            *[
                rf"\DeclareUnicodeCharacter{{{ord(ch):04X}}}{{\ensuremath{{\{name}}}}}"
                for ch, name in [
                    ("\u0394", "Delta"),
                    ("\u03a9", "Omega"),
                    ("\u03a6", "Phi"),
                    ("\u03a8", "Psi"),
                    ("\u03a3", "Sigma"),
                    ("\u03a0", "Pi"),
                    ("\u03b1", "alpha"),
                    ("\u03b2", "beta"),
                    ("\u03b3", "gamma"),
                    ("\u03b4", "delta"),
                    ("\u03b5", "varepsilon"),
                    ("\u03b8", "theta"),
                    ("\u03ba", "kappa"),
                    ("\u03bb", "lambda"),
                    ("\u03bc", "mu"),
                    ("\u03bd", "nu"),
                    ("\u03c0", "pi"),
                    ("\u03c1", "rho"),
                    ("\u03c3", "sigma"),
                    ("\u03c4", "tau"),
                    ("\u03c6", "varphi"),
                    ("\u03c8", "psi"),
                    ("\u03c9", "omega"),
                ]
            ],
            r"\renewcommand{\thesection}{\arabic{section}}",
            r"\renewcommand{\thefigure}{\arabic{figure}}",
            r"\renewcommand{\thetable}{\arabic{table}}",
            r"\renewcommand{\theequation}{\arabic{equation}}",
        ]
    ),
}

# The PDF is a continuous document rather than a chaptered book: at this length
# a forced page break per top-level document costs more than it helps. Set this
# back to "chapter" for conventional chapter-per-page manual layout (and drop
# the \thesection override above, which exists only because of this choice).
latex_toplevel_sectioning = "section"
