"""The exception hierarchy, and a scan that keeps every raise inside it."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

import radtrax
from radtrax import MissingTableError, RadtraxError, exceptions as exc


SRC = Path(radtrax.__file__).parent

BUILTIN_EXCEPTIONS = {
    name
    for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, BaseException)
}

# Raises of a built-in exception that stay built-in: (path under src/radtrax,
# enclosing function, exception) -> reason.
ALLOWED_BUILTIN_RAISES = {
    ("geo/__init__.py", "__getattr__", "AttributeError"): "module __getattr__ protocol",
    (
        "retrieval/matched_filter/__init__.py",
        "__getattr__",
        "AttributeError",
    ): "module __getattr__ protocol",
    (
        "retrieval/var3d/__init__.py",
        "__getattr__",
        "AttributeError",
    ): "module __getattr__ protocol",
    (
        "instrument/srf.py",
        "_build_parametric_matrix",
        "AssertionError",
    ): "unreachable branch, an internal invariant rather than a user error",
}


def test_radtrax_error_is_an_exception() -> None:
    with pytest.raises(RadtraxError, match="boom"):
        raise RadtraxError("boom")


def test_float64_is_enabled_in_tests() -> None:
    import jax.numpy as jnp

    assert jnp.zeros(1).dtype == jnp.float64


def test_missing_table_error_is_a_file_not_found_error() -> None:
    with pytest.raises(FileNotFoundError, match="build"):
        raise MissingTableError("build it")
    assert issubclass(MissingTableError, exc.MissingFileError)


@pytest.mark.parametrize("name", [n for n in exc.__all__ if n != "RadtraxError"])
def test_each_error_is_a_radtrax_error_and_a_builtin(name: str) -> None:
    """Contract 3: ``except RadtraxError`` and ``except <builtin>`` both catch it."""
    cls = getattr(exc, name)
    assert issubclass(cls, RadtraxError)
    builtin_bases = [b for b in cls.__mro__ if b.__name__ in BUILTIN_EXCEPTIONS]
    assert builtin_bases[0] not in (Exception, BaseException)
    assert getattr(radtrax, name) is cls


def _builtin_raises() -> set[tuple[str, str, str]]:
    """Each ``raise <builtin>``: (path under src/radtrax, enclosing function, name)."""
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()

        def visit(node: ast.AST, scope: str, rel: str = rel) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                scope = node.name
            if isinstance(node, ast.Raise) and node.exc is not None:
                target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(target, ast.Name) and target.id in BUILTIN_EXCEPTIONS:
                    found.add((rel, scope, target.id))
            for child in ast.iter_child_nodes(node):
                visit(child, scope)

        visit(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    return found


def test_no_builtin_exception_is_raised_outside_the_allow_list() -> None:
    """Every eager error radtrax raises is a ``RadtraxError``.

    Raise a class from ``radtrax.exceptions`` instead of a built-in; an entry in
    ``ALLOWED_BUILTIN_RAISES`` needs a reason. ``eqx.error_if`` checks under
    ``jit`` raise a ``RuntimeError`` at run time and are out of scope.
    """
    found = _builtin_raises()
    assert found - ALLOWED_BUILTIN_RAISES.keys() == set()


def test_the_allow_list_has_no_stale_entries() -> None:
    assert ALLOWED_BUILTIN_RAISES.keys() - _builtin_raises() == set()
