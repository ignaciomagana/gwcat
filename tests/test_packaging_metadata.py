"""Packaging metadata: ONE version authority, and a license that exists.

Not cosmetics.  Every export stamps ``writer_version`` (see
``gwcat/export/writers_gwcat2.py``), and that value comes from the INSTALLED
distribution metadata, not from the imported module.  While the version was
written twice -- a literal in ``pyproject.toml`` and another in
``gwcat/__init__.py`` -- bumping one and not the other made a file's provenance
disagree with the code that wrote it, silently and forever after.  So the
version is declared dynamically from ``gwcat.__version__``, and these tests pin
that there is nowhere else to bump it.

The build also required ``setuptools-scm`` and never configured it (the version
was a hardcoded string), so it was pure build-time cost with no effect on the
metadata -- and it made "where does the version come from?" ambiguous, which is
the same defect one layer up.
"""
import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT_PY = REPO_ROOT / "gwcat" / "__init__.py"

needs_source_tree = pytest.mark.skipif(
    not PYPROJECT.exists(),
    reason="packaging metadata lives in the source tree, not the installed pkg")


def _pyproject():
    try:
        import tomllib
    except ImportError:                       # py3.9/3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(PYPROJECT.read_text())


def _version_literal_from_init():
    """``__version__`` as a STATIC literal -- the way setuptools reads it."""
    tree = ast.parse(INIT_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__version__":
                    return ast.literal_eval(node.value)
    raise AssertionError("gwcat/__init__.py declares no __version__ literal")


# ==========================================================================
# 1. One version authority
# ==========================================================================
@needs_source_tree
def test_the_version_is_declared_once_and_read_from_the_package():
    """``pyproject.toml`` must carry no version literal of its own."""
    data = _pyproject()
    project = data["project"]
    assert "version" not in project, (
        "pyproject.toml declares a static version; it must stay dynamic so "
        "gwcat.__version__ is the only place to bump.")
    assert "version" in project.get("dynamic", []), \
        "the version must be declared dynamic"
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "gwcat.__version__"


@needs_source_tree
def test_the_declared_attr_is_statically_readable_and_is_the_running_version():
    """setuptools reads the attr WITHOUT importing gwcat, so it must be a plain
    literal -- and it must be the value the imported package reports."""
    import gwcat

    assert _version_literal_from_init() == gwcat.__version__


@needs_source_tree
def test_setuptools_can_resolve_the_dynamic_version():
    """The real resolution path, not a re-implementation of it."""
    expand = pytest.importorskip("setuptools.config.expand")
    import gwcat

    assert expand.read_attr("gwcat.__version__",
                            root_dir=str(REPO_ROOT)) == gwcat.__version__


@needs_source_tree
def test_no_other_file_declares_the_package_version():
    """A second literal is a second authority, whatever it is called."""
    literal = _version_literal_from_init()
    pattern = re.compile(r'^\s*(version|__version__)\s*=\s*["\']'
                         + re.escape(literal) + r'["\']', re.M)
    offenders = []
    for path in REPO_ROOT.glob("*.toml"):
        if pattern.search(path.read_text()):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    for path in (REPO_ROOT / "gwcat").rglob("*.py"):
        if path == INIT_PY:
            continue
        if pattern.search(path.read_text()):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"version also declared in {offenders}"


@needs_source_tree
def test_the_build_does_not_require_setuptools_scm():
    """It was required and never configured: build cost, no effect."""
    requires = _pyproject()["build-system"]["requires"]
    assert not any("setuptools" in r and "scm" in r.replace("_", "-")
                   for r in requires), \
        f"unconfigured setuptools-scm is still a build requirement: {requires}"


# ==========================================================================
# 2. The declared license has a file
# ==========================================================================
@needs_source_tree
def test_the_declared_license_ships_with_its_text():
    """The package has declared MIT since the first commit, with no LICENSE
    file in the tree -- so the one thing that grants anyone the right to use
    these exports was a string in a build file."""
    project = _pyproject()["project"]
    assert project["license"] == "MIT"
    assert "LICENSE" in project.get("license-files", []), \
        "the declared license must name the file that carries its text"

    text = (REPO_ROOT / "LICENSE").read_text()
    assert text.lstrip().startswith("MIT License")
    assert "Ignacio Magana Hernandez" in text
    # The grant and the disclaimer, i.e. an actual MIT license rather than a
    # header claiming to be one.
    assert "Permission is hereby granted, free of charge" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text


# ==========================================================================
# Review 2026-10-20: every module-scope import of `import gwcat` is declared
# ==========================================================================
#: Import name -> distribution name, where they differ.
_DIST_OF = {"yaml": "pyyaml"}


@needs_source_tree
def test_import_gwcat_needs_only_declared_dependencies():
    """scipy was a hard import of ``import gwcat`` (gwcat/chi_eff_exact.py)
    but not a declared dependency, so a plain ``pip install .`` could not
    import gwcat.  Every third-party package imported at MODULE scope by a
    module that ``import gwcat`` loads must be in [project].dependencies."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, gwcat; print('\\n'.join(sorted("
         "m.__file__ for n, m in sys.modules.items() if n == 'gwcat' or "
         "n.startswith('gwcat.'))))"],
        capture_output=True, text=True, check=True, cwd=str(REPO_ROOT))
    files = [Path(x) for x in out.stdout.split() if x.endswith(".py")]
    assert files, "import gwcat loaded no gwcat modules?"
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    third = set()
    for f in files:
        for node in ast.parse(f.read_text()).body:     # module scope only
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            else:
                continue
            third |= {n.split(".")[0] for n in names} - stdlib - {"gwcat"}
    if not getattr(sys, "stdlib_module_names", None):  # py < 3.10
        third -= {"typing", "dataclasses", "contextlib", "contextvars"}
    declared = {re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower()
                for d in _pyproject()["project"]["dependencies"]}
    missing = sorted(n for n in third
                     if _DIST_OF.get(n, n).lower() not in declared)
    assert not missing, (
        f"`import gwcat` imports {missing} at module scope, but "
        f"pyproject.toml does not declare them: a plain `pip install .` "
        f"would fail to import gwcat.")
