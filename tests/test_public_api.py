"""Guards on the public API surface.

These tests are deliberately about *contracts*, not behaviour: if a symbol is
removed from ``__all__`` or stops being importable from the top level, that is
a breaking change and should fail loudly here.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import radtrax


SUBMODULES = [
    "exceptions",
]

# The layout of radtrax · 3. Each subpackage owns one stage of the forward
# model; adding or removing one is a design change, so it fails here first.
SUBPACKAGES = [
    "atmosphere",
    "geo",
    "instrument",
    "optics",
    "retrieval",
    "retrieval.matched_filter",
    "retrieval.var3d",
    "slab",
    "solvers",
    "spectroscopy",
    "spectroscopy.hitran",
    "surface",
]


def test_version_is_a_dotted_string() -> None:
    assert isinstance(radtrax.__version__, str)
    major, minor, patch = radtrax.__version__.split(".")[:3]
    assert all(part.isdigit() for part in (major, minor, patch))


def _isort_key(name: str) -> tuple[int, str]:
    """Ruff's RUF022 order: SCREAMING_CASE, then CamelCase, then the rest."""
    if name.isupper():
        return (0, name)
    return (1 if name[:1].isupper() else 2, name)


def test_all_is_sorted_and_unique() -> None:
    assert radtrax.__all__ == sorted(radtrax.__all__, key=_isort_key)
    assert len(radtrax.__all__) == len(set(radtrax.__all__))


@pytest.mark.parametrize("name", radtrax.__all__)
def test_every_exported_name_is_importable(name: str) -> None:
    assert hasattr(radtrax, name), f"{name} is in __all__ but not defined"


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports_cleanly(module: str) -> None:
    assert importlib.import_module(f"radtrax.{module}") is not None


def test_no_unexpected_submodules() -> None:
    """A new module must be added to SUBMODULES (and given an API doc page)."""
    found = {
        info.name for info in pkgutil.iter_modules(radtrax.__path__) if not info.ispkg
    }
    assert found == set(SUBMODULES)


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodules_declare_all(module: str) -> None:
    imported = importlib.import_module(f"radtrax.{module}")
    assert hasattr(imported, "__all__"), f"radtrax.{module} is missing __all__"
    assert imported.__all__ == sorted(imported.__all__, key=_isort_key)


def test_package_is_typed() -> None:
    """PEP 561 marker must ship so downstream type checkers see annotations."""
    marker = Path(radtrax.__path__[0]) / "py.typed"
    assert marker.is_file(), f"missing PEP 561 marker at {marker}"


def test_subpackages_match_the_layout() -> None:
    found = {
        info.name.removeprefix("radtrax.")
        for info in pkgutil.walk_packages(radtrax.__path__, prefix="radtrax.")
        if info.ispkg
    }
    assert found == set(SUBPACKAGES)


@pytest.mark.parametrize("package", SUBPACKAGES)
def test_subpackages_import_and_declare_all(package: str) -> None:
    imported = importlib.import_module(f"radtrax.{package}")
    assert imported.__doc__, f"radtrax.{package} needs a docstring saying what it owns"
    assert imported.__all__ == sorted(imported.__all__, key=_isort_key)
