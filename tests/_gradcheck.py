"""Gradient check: reverse-mode autodiff against central finite differences.

radtrax · 3 makes this the one test every release runs: every public
differentiable function is checked on a reduced state. Import it in a test
module as ``from _gradcheck import assert_gradients_match``.

For a function F: ℝⁿ → ℝᵐ (any output shape, flattened) and state x, column j
of the finite-difference Jacobian is

    ∂F/∂xⱼ ≈ (F(x + hⱼ eⱼ) − F(x − hⱼ eⱼ)) / (2 hⱼ),   hⱼ = ε^{1/3} max(1, |xⱼ|),

with ε the float64 machine epsilon, the step that balances the O(h²)
truncation error against the O(ε / h) rounding error. The step never falls
below ε^{1/3} ≈ 6e-6, so check a physically small variable (a VMR of 1e-7, a
column in kg/m²) in units that make it O(1), e.g. ppm. Each entry is compared
as |J^ad − J^fd| / max(|J^fd|, floor), where floor = ``floor_ratio`` ·
max|J^fd|. The central difference carries an absolute error near
ε^{2/3} |F| ≈ 4e-11 |F|, so an entry far below the Jacobian's scale cannot be
checked relatively; with the default floor of 1e-3, an entry is judged
against 1e-3 of the scale, and an rtol of 1e-6 still catches any error above
1e-9 of the scale.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ["GradCheck", "assert_gradients_match", "check_gradients"]


@dataclass(frozen=True, slots=True)
class GradCheck:
    """The outcome of a gradient check.

    Attributes:
        max_rel_error: The largest entrywise relative error.
        worst: The ``(output, input)`` flat indices where it occurs.
        worst_values: The autodiff and finite-difference entries there.
        jac_autodiff: The reverse-mode Jacobian, shape ``(m, k)`` for the
            ``k`` checked inputs.
        jac_fd: The central-difference Jacobian, same shape.
    """

    max_rel_error: float
    worst: tuple[int, int]
    worst_values: tuple[float, float]
    jac_autodiff: np.ndarray
    jac_fd: np.ndarray


def check_gradients(
    f: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    indices: Sequence[int] | None = None,
    floor_ratio: float = 1e-3,
) -> GradCheck:
    """Compare ``jax.jacrev(f)`` with central differences at ``x``.

    Args:
        f: A function of a one-dimensional float64 state; its output may have
            any shape and is flattened.
        x: The state, shape ``(n,)``, float64.
        indices: The inputs to check (a reduced state); all of them when
            ``None``.
        floor_ratio: The relative-error floor, as a fraction of the largest
            finite-difference entry.

    Returns:
        The largest relative error, where it occurs, and both Jacobians.

    Raises:
        ValueError: If ``x`` is not a one-dimensional float64 array, since
            central differences in float32 cannot resolve a gradient to the
            tolerances radtrax tests at.
    """
    x = jnp.asarray(x)
    if x.ndim != 1 or x.dtype != jnp.float64:
        msg = f"'x' must be a one-dimensional float64 array, got {x.dtype}{x.shape}"
        raise ValueError(msg)
    cols = list(range(x.shape[0])) if indices is None else list(indices)

    def flat(z: jax.Array) -> jax.Array:
        return jnp.ravel(f(z))

    jac_ad = np.asarray(jax.jacrev(flat)(x))[:, cols]

    eps = np.finfo(np.float64).eps
    columns = []
    for j in cols:
        h = eps ** (1 / 3) * max(1.0, abs(float(x[j])))
        e = jnp.zeros_like(x).at[j].set(h)
        columns.append((np.asarray(flat(x + e)) - np.asarray(flat(x - e))) / (2 * h))
    jac_fd = np.stack(columns, axis=1)

    floor = floor_ratio * max(float(np.max(np.abs(jac_fd))), np.finfo(np.float64).tiny)
    rel = np.abs(jac_ad - jac_fd) / np.maximum(np.abs(jac_fd), floor)
    i, k = np.unravel_index(int(np.argmax(rel)), rel.shape)
    return GradCheck(
        max_rel_error=float(rel[i, k]),
        worst=(int(i), cols[k]),
        worst_values=(float(jac_ad[i, k]), float(jac_fd[i, k])),
        jac_autodiff=jac_ad,
        jac_fd=jac_fd,
    )


def assert_gradients_match(
    f: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    rtol: float = 1e-6,
    indices: Sequence[int] | None = None,
    floor_ratio: float = 1e-3,
) -> GradCheck:
    """Fail unless autodiff and central differences agree to ``rtol``.

    The default 1e-6 leaves room above the O(ε^{2/3}) ≈ 4e-11 error of the
    central difference itself; radtrax · 3's v0 slice reached 2.3e-8.

    Returns:
        The ``GradCheck``, for tests that also assert on the Jacobians.
    """
    result = check_gradients(f, x, indices=indices, floor_ratio=floor_ratio)
    (i, j), (ad, fd) = result.worst, result.worst_values
    assert result.max_rel_error <= rtol, (
        f"autodiff and finite differences disagree: relative error "
        f"{result.max_rel_error:.3e} > {rtol:.0e} at output {i}, input {j} "
        f"(autodiff {ad:.9e}, finite differences {fd:.9e})"
    )
    return result
