"""The layered atmospheric state: the input of every forward model.

``AtmosphericState`` holds, per layer, the pressure, temperature, altitude
and thickness, and a volume mixing ratio per gas. Everything is a JAX array
leaf, so a forward model differentiates with respect to any of it, and the
gases are a ``dict`` keyed by name, so the gas list lives in the data rather
than in the code (*radtrax · 3*). Build one from levels with
:func:`radtrax.atmosphere.layers_from_levels`, or take the AFGL US Standard
profile, :func:`radtrax.atmosphere.afgl_us_standard`.

Shapes: ``L`` layers.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from radtrax.atmosphere.gas import (
    ATM_TO_PA,
    AVOGADRO_PER_MOL,
    CM2_PER_M2,
    number_density,
)
from radtrax.exceptions import ShapeError


__all__ = ["AtmosphericState"]

_CM_PER_M = 100.0


class AtmosphericState(eqx.Module):
    """Pressure, temperature, altitude, thickness and gases, per layer.

    Layers are ordered from the surface up. Every field is a leaf: choose
    what is not differentiated with ``eqx.partition`` or
    ``jax.lax.stop_gradient``.

    Attributes:
        p_pa: Layer pressure [Pa], ``(L,)``.
        t_k: Layer temperature [K], ``(L,)``.
        z_m: Altitude of the layer's middle [m], ``(L,)``.
        dz_m: Layer thickness [m], ``(L,)``.
        vmr: Volume mixing ratio [mol/mol] per gas, each ``(L,)``, keyed by
            gas name (``"CH4"``, ``"H2O"``, …).

    Examples:
        >>> import jax.numpy as jnp
        >>> slab = AtmosphericState(
        ...     p_pa=jnp.array([101325.0]),
        ...     t_k=jnp.array([273.15]),
        ...     z_m=jnp.array([500.0]),
        ...     dz_m=jnp.array([1000.0]),
        ...     vmr={"CH4": jnp.array([1.9e-6])},
        ... )
        >>> f"{float(slab.column('CH4')):.4g}"
        '5.105e+18'
    """

    p_pa: Real[Array, " L"]
    t_k: Real[Array, " L"]
    z_m: Real[Array, " L"]
    dz_m: Real[Array, " L"]
    vmr: dict[str, Float[Array, " L"]]

    def __check_init__(self) -> None:
        shape = jnp.shape(self.p_pa)
        if len(shape) != 1:
            raise ShapeError(f"AtmosphericState: p_pa must be (L,), got {shape}.")
        fields = {"t_k": self.t_k, "z_m": self.z_m, "dz_m": self.dz_m}
        fields |= {f"vmr[{gas!r}]": x for gas, x in self.vmr.items()}
        for name, value in fields.items():
            if jnp.shape(value) != shape:
                raise ShapeError(
                    f"AtmosphericState: {name} has shape {jnp.shape(value)}, "
                    f"p_pa has {shape}."
                )

    @property
    def n_layers(self) -> int:
        """Number of layers ``L``."""
        return int(jnp.shape(self.p_pa)[0])

    def number_density(self) -> Float[Array, " L"]:
        """Air number density per layer [molecules/cm³], ``n = p / (k_B T)``."""
        return jnp.asarray(number_density(self.p_pa / ATM_TO_PA, self.t_k))

    def air_column(self) -> Float[Array, " L"]:
        """Air column per layer [molecules/cm²], ``n Δz``."""
        return self.number_density() * self.dz_m * _CM_PER_M

    def column(self, gas: str) -> Float[Array, ""]:
        """Total column of ``gas`` [molecules/cm²], ``Σ_l n_l x_l Δz_l``.

        Raises:
            KeyError: If the state carries no ``gas``.
        """
        return jnp.sum(self.air_column() * self.vmr[gas])

    def add_column(
        self,
        gas: str,
        delta_column_mol_m2: Float[ArrayLike, ""],
        profile: Float[ArrayLike, " L"] | None = None,
    ) -> AtmosphericState:
        """A copy with ``ΔX`` [mol/m²] more ``gas``, spread over the layers.

        Layer ``l`` gains ``w_l ΔX`` with ``w = profile / Σ profile``, so the
        column grows by exactly ``ΔX``:
        ``Δx_l = w_l ΔX N_A / (n_l Δz_l)`` with the air column ``n_l Δz_l``
        in molecules/m². Differentiable in ``ΔX`` and ``profile``; a plume's
        unit-absorption target is ``∂ ln y / ∂ΔX`` through this.

        Args:
            gas: The gas, which the state must carry.
            delta_column_mol_m2: Column enhancement ``ΔX`` [mol/m²].
            profile: Relative weight per layer, ``(L,)``; the default puts
                all of ``ΔX`` in the bottom layer.

        Raises:
            KeyError: If the state carries no ``gas``.
            ValueError: If ``profile`` is not ``(L,)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> from radtrax.atmosphere import (
            ...     AVOGADRO_PER_MOL,
            ...     CM2_PER_M2,
            ...     afgl_us_standard,
            ... )
            >>> atm = afgl_us_standard(z_top_km=3.0, gases=["CH4"])
            >>> plume = atm.add_column("CH4", 0.1)
            >>> added = (plume.column("CH4") - atm.column("CH4")) * CM2_PER_M2
            >>> round(float(added / AVOGADRO_PER_MOL), 4)  # mol/m²
            0.1
        """
        x = self.vmr[gas]
        if profile is None:
            weights = jnp.zeros_like(x).at[0].set(1.0)
        else:
            weights = jnp.asarray(profile, dtype=x.dtype)
            if weights.shape != x.shape:
                raise ShapeError(
                    f"AtmosphericState.add_column: profile must be (L,) = "
                    f"{x.shape}, got {weights.shape}."
                )
            weights = weights / jnp.sum(weights)
        # Air column per layer [mol/m²]: n Δz [molecules/cm²] × 1e4 / N_A.
        air_mol_m2 = self.air_column() * CM2_PER_M2 / AVOGADRO_PER_MOL
        return self.with_vmr(gas, x + weights * delta_column_mol_m2 / air_mol_m2)

    def with_vmr(self, gas: str, vmr: float | Float[Array, " L"]) -> AtmosphericState:
        """A copy with ``gas`` set to ``vmr`` [mol/mol] (a scalar fills every layer)."""
        x = jnp.asarray(vmr)
        value = jnp.broadcast_to(
            x.astype(jnp.result_type(x, self.p_pa, float)), self.p_pa.shape
        )
        return AtmosphericState(
            p_pa=self.p_pa,
            t_k=self.t_k,
            z_m=self.z_m,
            dz_m=self.dz_m,
            vmr={**self.vmr, gas: value},
        )
