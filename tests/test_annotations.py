"""Every public array argument and return carries a jaxtyping shape.

An AST scan of ``src/radtrax``: a public function or method (``__init__``
included) fails when an argument or return annotation names an array type
(``np.ndarray``, ``jax.Array``, ``Array``, ``ArrayLike``, ``NDArray``) outside
a jaxtyping dtype (``Float[...]``, ``Int[...]``, …), or when a dimension string
is not quoted (``Float[Array, B]``). Module aliases such as
``Spectrum = Float[np.ndarray, " B"]`` pass. The dimension names are listed in
the "Shapes" section of AGENTS.md.

``ALLOWED`` is empty since #81: annotate a new signature rather than list it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import radtrax


SRC = Path(radtrax.__file__).parent

ARRAY_NAMES = {"Array", "ArrayLike", "NDArray", "ndarray"}
JAXTYPING_DTYPES = {
    "Bool",
    "Complex",
    "Float",
    "Inexact",
    "Int",
    "Integer",
    "Num",
    "Real",
    "Shaped",
    "UInt",
}

# Public signatures without shapes, as "path::qualname".
ALLOWED: frozenset[str] = frozenset()


def _is_public(path: Path) -> bool:
    parts = path.relative_to(SRC).with_suffix("").parts
    return not any(p.startswith("_") and p != "__init__" for p in parts)


def _problems(node: ast.expr | None) -> Iterator[str]:
    """Bare array types and unquoted dimension names in one annotation."""
    if node is None:
        return
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in JAXTYPING_DTYPES
    ):
        dims = node.slice.elts[1] if isinstance(node.slice, ast.Tuple) else None
        if not (isinstance(dims, ast.Constant) and isinstance(dims.value, str)):
            yield "unquoted dimensions"
        return
    if isinstance(node, ast.Name) and node.id in ARRAY_NAMES:
        yield f"bare {node.id}"
        return
    if isinstance(node, ast.Attribute) and node.attr in ARRAY_NAMES:
        yield f"bare {node.attr}"
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            yield from _problems(child)


def _signatures(
    body: list[ast.stmt], prefix: str = ""
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_") or node.name == "__init__":
                yield prefix + node.name, node
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield from _signatures(node.body, f"{prefix}{node.name}.")


def _unshaped() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        if not _is_public(path):
            continue
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for qualname, func in _signatures(tree.body):
            args = func.args
            every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            every += [a for a in (args.vararg, args.kwarg) if a is not None]
            problems = [f"{a.arg}: {p}" for a in every for p in _problems(a.annotation)]
            problems += [f"return: {p}" for p in _problems(func.returns)]
            if problems:
                found[f"{rel}::{qualname}"] = problems
    return found


def test_public_array_annotations_carry_shapes() -> None:
    new = {k: v for k, v in _unshaped().items() if k not in ALLOWED}
    assert new == {}, "annotate with a jaxtyping shape (AGENTS.md, Shapes)"


def test_the_allow_list_has_no_stale_entries() -> None:
    assert ALLOWED - _unshaped().keys() == set()
