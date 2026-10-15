"""Per-layer optical depth: the ``OpacityProvider`` protocol and table opacity.

The optical depth of each layer at each wavenumber, ``τ (L, G)``, is
radtrax's canonical interface between the atmosphere and a solver
(*radtrax · 3*). v0 computes it from cross-section tables σ_g(ν; p, T),
interpolated bilinearly in ``(ln p, T)`` at each layer's pressure and
temperature:

.. math::
    σ_{g,l}(ν) = \\operatorname{bilinear}(σ_g;\\ \\ln p_l, T_l), \\qquad
    τ_l(ν) = \\sum_g σ_{g,l}(ν)\\, n_l\\, x_{g,l}\\, Δz_l ,

with ``n_l`` the air number density, ``x_{g,l}`` the gas's mixing ratio and
``Δz_l`` the thickness (:class:`~radtrax.atmosphere.AtmosphericState`).

Ported from the radtrax · 3 design notebook's ``TableOpacity`` and
generalised to a ``dict`` of per-gas tables mapped with ``jax.tree.map``, so
the gas list lives in the data. The interpolation is
:class:`jax.scipy.interpolate.RegularGridInterpolator`. Shapes: ``L``
layers, ``G`` wavenumbers, ``P`` and ``T`` table pressures and
temperatures.
"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Array, Float

from radtrax.atmosphere.gas import ATM_TO_PA
from radtrax.atmosphere.state import AtmosphericState
from radtrax.exceptions import GridError, MissingKeyError, ParameterError, ShapeError


if TYPE_CHECKING:
    import xarray as xr


__all__ = ["CrossSectionTable", "OpacityProvider", "TableOpacity"]


class OpacityProvider(eqx.Module):
    """Protocol: an atmospheric state in, per-layer optical depth out.

    An abstract Equinox module (an ``eqx.AbstractVar`` field and an
    ``abc``-abstract ``__call__``), so ``filter_jit`` compiles each concrete
    provider and a forward model can hold any of them.

    Attributes:
        nu_cm: The wavenumber grid [cm⁻¹], ``(G,)``.
    """

    nu_cm: eqx.AbstractVar[Float[Array, " G"]]

    @abc.abstractmethod
    def __call__(self, state: AtmosphericState) -> Float[Array, "L G"]:
        """Optical depth of each layer at each wavenumber (dimensionless)."""


class CrossSectionTable(eqx.Module):
    """One gas's cross sections σ(ν; p, T) on a ``(ln p, T)`` grid.

    Attributes:
        log_p_pa: Natural log of the table pressures [ln Pa], strictly
            increasing, ``(P,)``.
        t_k: Table temperatures [K], strictly increasing, ``(T,)``.
        sigma_cm2: Cross sections [cm²/molecule], ``(P, T, G)``.
    """

    log_p_pa: Float[Array, " P"]
    t_k: Float[Array, " T"]
    sigma_cm2: Float[Array, "P T G"]

    def __check_init__(self) -> None:
        n_p, n_t = jnp.shape(self.log_p_pa)[0], jnp.shape(self.t_k)[0]
        shape = jnp.shape(self.sigma_cm2)
        if len(shape) != 3 or shape[:2] != (n_p, n_t):
            raise ShapeError(
                f"CrossSectionTable: sigma_cm2 must be (P, T, G) = ({n_p}, {n_t}, G), "
                f"got {shape}."
            )
        if n_p < 2 or n_t < 2:
            raise GridError(
                "CrossSectionTable: need at least two pressures and temperatures."
            )
        _check_increasing("log_p_pa", self.log_p_pa)
        _check_increasing("t_k", self.t_k)

    def __call__(
        self, p_pa: Float[Array, " L"], t_k: Float[Array, " L"]
    ) -> Float[Array, "L G"]:
        """σ at each layer's ``(p, T)``, bilinear in ``(ln p, T)``.

        Exact at the grid nodes, where the derivative is the full one-sided
        slope. Outside the grid the nearest edge is used (constant
        extrapolation, zero gradient); keep layers inside the table's range.
        """
        points = jnp.stack(
            [_clamp(jnp.log(p_pa), self.log_p_pa), _clamp(t_k, self.t_k)], axis=-1
        )
        interpolate = RegularGridInterpolator(
            (self.log_p_pa, self.t_k), self.sigma_cm2, method="linear", fill_value=None
        )
        return interpolate(points)


class TableOpacity(OpacityProvider):
    """Optical depth from per-gas cross-section tables.

    ``τ_l(ν) = Σ_g σ_g(ν; p_l, T_l) n_l x_{g,l} Δz_l``. The tables are a
    ``dict`` keyed by gas and mapped with ``jax.tree.map``; every table
    gas must be in the state's ``vmr`` (gases only in the state are
    transparent here). Tables are leaves: to keep them out of a gradient,
    split them off with ``eqx.partition``. Table opacity needs float64
    (``jax_enable_x64``); a float32 grid warns.

    Attributes:
        nu_cm: Wavenumber grid [cm⁻¹], ``(G,)``, shared by every table.
        tables: One :class:`CrossSectionTable` per gas.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax.atmosphere import layers_from_levels
        >>> table = CrossSectionTable(
        ...     log_p_pa=jnp.log(jnp.array([1e4, 1e5])),
        ...     t_k=jnp.array([200.0, 300.0]),
        ...     sigma_cm2=jnp.full((2, 2, 3), 1e-21),
        ... )
        >>> opacity = TableOpacity(
        ...     jnp.array([4300.0, 4300.1, 4300.2]), {"CH4": table}
        ... )
        >>> atm = layers_from_levels(
        ...     jnp.array([1e5, 5e4]),
        ...     jnp.array([288.0, 255.0]),
        ...     {"CH4": jnp.full(2, 1.9e-6)},
        ... )
        >>> opacity(atm).shape
        (1, 3)
    """

    nu_cm: Float[Array, " G"]
    tables: dict[str, CrossSectionTable]

    def __init__(
        self, nu_cm: Float[Array, " G"], tables: dict[str, CrossSectionTable]
    ) -> None:
        self.nu_cm = jnp.asarray(nu_cm)
        if self.nu_cm.ndim != 1:
            raise ShapeError(
                f"TableOpacity: nu_cm must be (G,), got {self.nu_cm.shape}."
            )
        if self.nu_cm.dtype != jnp.float64:
            warnings.warn(
                f"TableOpacity: nu_cm is {self.nu_cm.dtype}; table opacity needs "
                "float64 (float32 spaces 4300 cm⁻¹ by 4.9e-4 cm⁻¹). Enable "
                "jax_enable_x64 before building the tables.",
                stacklevel=2,
            )
        if not tables:
            raise ParameterError("TableOpacity: need at least one gas table.")
        for gas, table in tables.items():
            n_nu = jnp.shape(table.sigma_cm2)[-1]
            if n_nu != self.nu_cm.shape[0]:
                raise GridError(
                    f"TableOpacity: the {gas!r} table has {n_nu} wavenumbers, "
                    f"nu_cm has {self.nu_cm.shape[0]}."
                )
        self.tables = dict(tables)

    def __call__(self, state: AtmosphericState) -> Float[Array, "L G"]:
        missing = sorted(set(self.tables) - set(state.vmr))
        if missing:
            raise MissingKeyError(
                f"TableOpacity: the state has no mixing ratio for {missing}."
            )
        air = state.air_column()  # n Δz, molecules/cm²                      (L,)
        sigma = jax.tree.map(
            lambda table: table(state.p_pa, state.t_k),
            self.tables,
            is_leaf=lambda node: isinstance(node, CrossSectionTable),
        )  # gas → σ (L, G)
        per_gas = [
            sigma[gas] * (air * state.vmr[gas])[:, None] for gas in sorted(sigma)
        ]
        return sum(per_gas[1:], start=per_gas[0])

    @classmethod
    def from_datasets(
        cls,
        datasets: dict[str, xr.Dataset],
        *,
        var: str | dict[str, str] = "absorption_cross_section",
    ) -> TableOpacity:
        """Build from per-gas look-up tables in radtrax's HITRAN format.

        Each dataset is what :func:`radtrax.spectroscopy.hitran.build_lut_dataset`
        writes: ``var`` on ``(wavenumber, temperature, pressure)`` with the
        pressure in atm, in any order (pressure and temperature are sorted).
        ``var`` may be a ``dict`` per gas, e.g. ``{"CH4": "CH4_cross_section"}``
        for the table of :func:`radtrax.spectroscopy.hitran.create_combined_lut`.
        Every gas must share one wavenumber grid.

        Raises:
            KeyError: If a dataset has no such variable.
            ValueError: If the wavenumber grids differ.
        """
        nu: np.ndarray | None = None
        tables: dict[str, CrossSectionTable] = {}
        for gas, ds in datasets.items():
            name = var[gas] if isinstance(var, dict) else var
            if name not in ds:
                raise MissingKeyError(
                    f"TableOpacity: {gas!r} has no {name!r} (have {list(ds.data_vars)})."
                )
            da = (
                ds[name]
                .sortby(["pressure", "temperature"])
                .transpose("pressure", "temperature", "wavenumber")
            )
            grid = np.asarray(da["wavenumber"].values, dtype=float)
            if nu is None:
                nu = grid
            elif grid.shape != nu.shape or not np.array_equal(grid, nu):
                raise GridError(
                    f"TableOpacity: {gas!r} is on a different wavenumber grid."
                )
            tables[gas] = CrossSectionTable(
                log_p_pa=jnp.log(
                    jnp.asarray(da["pressure"].values, dtype=float) * ATM_TO_PA
                ),
                t_k=jnp.asarray(da["temperature"].values, dtype=float),
                sigma_cm2=jnp.asarray(da.values, dtype=float),
            )
        if nu is None:
            raise ParameterError("TableOpacity: need at least one gas table.")
        return cls(jnp.asarray(nu), tables)


def _clamp(x: Float[Array, " L"], grid: Float[Array, " N"]) -> Float[Array, " L"]:
    """``x`` held inside ``[grid[0], grid[-1]]``.

    ``jnp.where``, not ``jnp.clip``: on a bound ``clip`` passes half the
    gradient, which would halve ``∂σ`` at the first and last nodes.
    """
    return jnp.where(x < grid[0], grid[0], jnp.where(x > grid[-1], grid[-1], x))


def _check_increasing(name: str, grid: Float[Array, " N"]) -> None:
    """Fail fast on a concrete grid that is not strictly increasing."""
    try:
        ok = bool(jnp.all(jnp.diff(grid) > 0))
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        return
    if not ok:
        raise GridError(f"CrossSectionTable: {name} must be strictly increasing.")
