"""Layers from levels by hydrostatic balance.

A profile is given on ``L + 1`` levels (pressure, temperature and mixing
ratios at layer boundaries, surface first); radiative transfer wants ``L``
layers. Between levels ``k`` (bottom) and ``k + 1`` (top), with ``T̄`` the
mean of the two level temperatures and ``m_air`` the mass of an air
molecule, hydrostatic balance ``dp/dz = −p m_air g / (k_B T)`` gives the
thickness

.. math::
    Δz = \\frac{k_B \\bar T}{m_{air} g_0} \\ln\\frac{p_k}{p_{k+1}} ,

and the layer pressure is the log-mean

.. math::
    \\bar p = \\frac{p_k - p_{k+1}}{\\ln(p_k / p_{k+1})} ,

the choice that makes the ideal-gas column ``n̄ Δz = p̄ Δz / (k_B T̄)``
equal the hydrostatic one, ``(p_k − p_{k+1}) / (m_air g_0)``, exactly and
whatever the temperature. Mixing ratios are the mean of the two levels.

``g_0`` is constant (standard gravity), so the altitudes are geopotential
(short of geometric altitude by ~0.3 % per 20 km) and the column is
``Δp / (m_air g_0)``: about 0.2 % below an integration with ``g(z)`` over the
US Standard profile, and surface gravity varies with latitude by ±0.27 %.

Inputs are cast to floating point (integer pressures would truncate the
means). Pressures must be positive and strictly decreasing: checked eagerly
with a [`GridError`][radtrax.GridError], and under ``jit`` / ``vmap`` with
``eqx.error_if``, which raises a ``RuntimeError`` at run time.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from radtrax.atmosphere.gas import (
    AVOGADRO_PER_MOL,
    BOLTZMANN_J_PER_K,
    DRY_AIR_MOLAR_MASS_KG_PER_MOL,
    STANDARD_GRAVITY_M_PER_S2,
)
from radtrax.atmosphere.state import AtmosphericState
from radtrax.exceptions import GridError, ShapeError


__all__ = ["layers_from_levels"]

#: Below this relative gap ``(p_k − p_{k+1}) / p_{k+1}`` the log-mean pressure
#: is taken from its series, ``p_{k+1} (1 + u/2 − u²/12)``: the closed form
#: is 0/0 at ``u = 0`` and loses its gradient to cancellation near it (2 % at
#: ``u = 1e-7``). The series' next term is ``u³/24 ≈ 4e-20`` here.
_SERIES_GAP = 1e-6


def layers_from_levels(
    p_levels_pa: Real[ArrayLike, " L1"],
    t_levels_k: Real[ArrayLike, " L1"],
    vmr_levels: dict[str, Float[Array, " L1"]],
    *,
    z_surface_m: float | Float[Array, ""] = 0.0,
) -> AtmosphericState:
    """An :class:`AtmosphericState` of ``L`` layers from ``L + 1`` levels.

    Args:
        p_levels_pa: Level pressures [Pa], surface first, strictly decreasing,
            ``(L + 1,)``.
        t_levels_k: Level temperatures [K], ``(L + 1,)``.
        vmr_levels: Level mixing ratios [mol/mol] per gas, each ``(L + 1,)``.
        z_surface_m: Altitude of the lowest level [m].

    Returns:
        The layers: log-mean pressure, mean temperature and mixing ratios,
        hydrostatic thickness, and the altitude of each layer's middle.

    Raises:
        ValueError: If the shapes disagree, there are fewer than two levels,
            or (for concrete inputs) the pressures are not strictly
            decreasing and positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> atm = layers_from_levels(
        ...     jnp.array([101325.0, 50662.5]), jnp.array([288.15, 255.0]), {}
        ... )
        >>> f"{float(atm.dz_m[0]):.1f}"
        '5510.2'
    """
    p = jnp.asarray(p_levels_pa)
    p = p.astype(jnp.result_type(p, float))
    t = jnp.asarray(t_levels_k, dtype=p.dtype)
    if p.ndim != 1 or p.shape[0] < 2:
        raise ShapeError(
            f"layers_from_levels: need (L + 1,) levels with L ≥ 1, got {p.shape}."
        )
    for name, value in {"t_levels_k": t, **vmr_levels}.items():
        if jnp.shape(value) != p.shape:
            raise ShapeError(
                f"layers_from_levels: {name} has shape {jnp.shape(value)}, "
                f"p_levels_pa has {p.shape}."
            )
    _check_decreasing(p)
    p = eqx.error_if(
        p,
        jnp.any(p[1:] >= p[:-1]) | jnp.any(p <= 0),
        "layers_from_levels: level pressures must be positive and strictly "
        "decreasing upwards (surface first).",
    )
    p_bot, p_top = p[:-1], p[1:]
    gap = (p_bot - p_top) / p_top  # u, relative gap
    log_ratio = jnp.log1p(gap)
    # Log-mean pressure, double-where around the 0/0 at u = 0.
    small = gap < _SERIES_GAP
    safe_log = jnp.where(small, 1.0, log_ratio)
    p_mean = jnp.where(
        small,
        p_top * (1.0 + gap / 2 - gap**2 / 12),
        (p_bot - p_top) / safe_log,
    )
    t_mean = 0.5 * (t[:-1] + t[1:])
    m_air = DRY_AIR_MOLAR_MASS_KG_PER_MOL / AVOGADRO_PER_MOL
    dz = BOLTZMANN_J_PER_K * t_mean / (m_air * STANDARD_GRAVITY_M_PER_S2) * log_ratio
    z_bottom = z_surface_m + jnp.concatenate(
        [jnp.zeros(1, dz.dtype), jnp.cumsum(dz)[:-1]]
    )
    return AtmosphericState(
        p_pa=p_mean,
        t_k=t_mean,
        z_m=z_bottom + 0.5 * dz,
        dz_m=dz,
        vmr={
            gas: 0.5
            * (jnp.asarray(x, dtype=p.dtype)[:-1] + jnp.asarray(x, dtype=p.dtype)[1:])
            for gas, x in vmr_levels.items()
        },
    )


def _check_decreasing(p: Float[Array, " L1"]) -> None:
    """Fail fast on concrete pressures; a traced call has nothing to inspect."""
    try:
        ok = bool(jnp.all(p[1:] < p[:-1])) and bool(jnp.all(p > 0))
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        return
    if not ok:
        raise GridError(
            "layers_from_levels: level pressures must be positive and strictly "
            "decreasing upwards (surface first)."
        )
