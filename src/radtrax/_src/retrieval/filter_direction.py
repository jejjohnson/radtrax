"""The filter direction g = Σ_y⁻¹ t and its target-energy check.

Shared by every matched-filter kernel (the score, its noise and its
sensitivity), so they share one solve and one fail-fast check. Moved from
``radtrax.retrieval.matched_filter.core``, which re-imports both names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from radtrax.exceptions import DegenerateInputError


__all__: list[str] = []


if TYPE_CHECKING:
    LinearOperator = lx.AbstractLinearOperator
else:
    LinearOperator = object


def _precompute_filter(
    cov_op: LinearOperator, target: Float[ArrayLike, " B"]
) -> tuple[Float[Array, " B"], Float[Array, ""]]:
    """Return ``(w, tᵀw)`` with ``w = Σ⁻¹ t``.

    Caller precomputes these once; every pixel then costs a single dot
    product against ``w``. Validates the target energy ``tᵀ Σ⁻¹ t`` so
    callers get a clear :class:`ValueError` instead of silent NaNs/infs
    when the target is zero or the covariance is singular/indefinite
    along the target direction. The check is a no-op under
    :func:`jax.jit` tracing; users who want the same fail-fast behaviour
    inside a jit-compiled pipeline should call
    :func:`validate_mf_inputs` explicitly before jitting.
    """
    w = gx.solve(cov_op, target)
    target_norm_sq = jnp.dot(target, w)
    _check_target_energy(target, target_norm_sq)
    return w, target_norm_sq


def _check_target_energy(
    target: Float[ArrayLike, " B"], target_norm_sq: Float[Array, ""]
) -> None:
    """Fail-fast validation — eager only; silently returns under jit tracing."""
    try:
        t_max = float(jnp.max(jnp.abs(target)))
        tnorm_sq_val = float(target_norm_sq)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return  # traced call: no concrete values to inspect.
    if t_max == 0.0:
        raise DegenerateInputError(
            "matched_filter: target signature is all zero — the score "
            "(x−μ)ᵀΣ⁻¹t / (tᵀΣ⁻¹t) is undefined."
        )
    if not tnorm_sq_val > 0.0:
        raise DegenerateInputError(
            f"matched_filter: target energy tᵀΣ⁻¹t = {tnorm_sq_val!r} is not "
            "strictly positive. The covariance operator is singular or "
            "indefinite along the target direction. Add a Tikhonov ridge or "
            "use a PD estimator such as estimate_cov_shrunk (Ledoit–Wolf)."
        )
