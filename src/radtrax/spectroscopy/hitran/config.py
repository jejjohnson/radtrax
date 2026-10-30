"""Gas + grid configuration for HAPI absorption-cross-section look-up tables.

A LUT is an N-dimensional array of absorption cross-sections

    σ(ν, T, P) = Σ_l S_l(T) f_V(ν − ν_l; Γ_D, Γ_L) · 1[|ν − ν_l| ≤ w]    [cm² / molecule]

computed line-by-line via HAPI's Voigt routine, over the lines l of every
isotopologue of the gas, each cut at w = ±25 cm⁻¹ from its centre, the
convention continua are defined against (Korkin et al. 2025, eqs. 16–18).
HITRAN's intensities S_l are already weighted by each isotopologue's natural
abundance, so the sum needs no rescaling. ``GasConfig`` identifies the
HITRAN line set (molecule + isotopologues + spectral range) and
``LUTGridConfig`` describes the common (T, P, ν-step, cut-off) grid used
across gases.

The ``ATMOSPHERIC_GASES`` registry covers the six species most commonly needed
for SWIR/NIR remote sensing (CH₄, CO₂, H₂O, O₂, N₂O, CO). Add entries for
other HITRAN species by constructing a ``GasConfig`` directly.

Ported from plumax 0.1.2 at 28f257e (``src/plumax/hapi_lut/config.py``); every
isotopologue and the ±25 cm⁻¹ cut-off since radtrax#172.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from radtrax.exceptions import MissingExtraError, MissingKeyError, ParameterError


# Typical dry-air volume mixing ratios, used as the "self" fraction when
# building HAPI's Diluent dict. Small errors here only affect self-vs-air
# collisional broadening and are second-order for most retrievals.
DEFAULT_VMR_NOMINAL: dict[str, float] = {
    "CH4": 2e-6,
    "CO2": 400e-6,
    "H2O": 1e-2,
    "O2": 0.21,
    "N2O": 0.3e-6,
    "CO": 0.1e-6,
}


@dataclass(frozen=True)
class GasConfig:
    """Configuration for a single HITRAN gas species.

    Attributes:
        name:             Short name used as HAPI SourceTable id (e.g. 'CH4').
        molecule_id:      HITRAN molecule id (see `hitran.org`).
        nu_min:           Lower wavenumber bound [cm⁻¹].
        nu_max:           Upper wavenumber bound [cm⁻¹].
        description:      Free-text description for dataset metadata.
        isotopologue_ids: HITRAN *global* isotopologue ids (the ``id`` of
                          ``hapi.ISO``, e.g. 32–35 for CH₄); ``None`` (the
                          default) takes every isotopologue of the molecule.
                          Restrict it only on purpose, e.g. for an
                          isotopologue retrieval.
        isotopologue_id:  Deprecated: one *local* isotopologue number
                          (1 = most abundant); use ``isotopologue_ids``.

    Raises:
        ParameterError: If both ``isotopologue_ids`` and ``isotopologue_id``
            are given, or ``description`` is not a string (before radtrax#172
            the positional order was name, molecule_id, isotopologue_id,
            nu_min, nu_max).
    """

    name: str
    molecule_id: int
    nu_min: float
    nu_max: float
    description: str = ""
    isotopologue_ids: tuple[int, ...] | None = field(default=None, kw_only=True)
    isotopologue_id: int | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.description, str):
            raise ParameterError(
                f"GasConfig({self.name!r}): description must be a string, got "
                f"{self.description!r}; since radtrax#172 the positional order "
                "is (name, molecule_id, nu_min, nu_max, description)."
            )
        if self.isotopologue_id is not None:
            if self.isotopologue_ids is not None:
                raise ParameterError(
                    f"GasConfig({self.name!r}): give isotopologue_ids, not both "
                    "it and the deprecated isotopologue_id."
                )
            warnings.warn(
                "GasConfig(isotopologue_id=...) is deprecated: pass "
                "isotopologue_ids=(global ids, ...), or leave it None for every "
                "isotopologue.",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.isotopologue_ids is not None:
            ids = tuple(int(i) for i in self.isotopologue_ids)
            object.__setattr__(self, "isotopologue_ids", ids)

    def resolve_isotopologue_ids(self) -> tuple[int, ...]:
        """The HITRAN global isotopologue ids to fetch (needs ``[hitran]``).

        ``isotopologue_ids`` as given; else the deprecated local
        ``isotopologue_id`` mapped through ``hapi.ISO``; else every global id
        whose ``hapi.ISO`` entry has this ``molecule_id``, in HITRAN's
        isotopologue order.

        Raises:
            MissingExtraError: If the ids must come from ``hapi.ISO`` and the
                ``[hitran]`` extra is not installed.
            KeyError: If ``hapi.ISO`` has no such molecule or isotopologue.
        """
        if self.isotopologue_ids is not None:
            return self.isotopologue_ids
        try:
            from hapi import ISO
        except ModuleNotFoundError as err:
            if err.name != "hapi":
                raise
            raise MissingExtraError(
                f"GasConfig({self.name!r}).resolve_isotopologue_ids() needs the "
                "[hitran] extra: pip install 'radtrax[hitran]'"
            ) from err

        global_ids = {
            local: int(entry[0])
            for (molecule, local), entry in ISO.items()
            if molecule == self.molecule_id and self.isotopologue_id in (None, local)
        }
        if not global_ids:
            number = "" if self.isotopologue_id is None else f" {self.isotopologue_id}"
            raise MissingKeyError(
                f"GasConfig({self.name!r}): hapi.ISO has no isotopologue{number} "
                f"of molecule {self.molecule_id}."
            )
        return tuple(global_ids[local] for local in sorted(global_ids))

    def __repr__(self) -> str:
        if self.isotopologue_ids is not None:
            isotopologues = "ids " + ",".join(map(str, self.isotopologue_ids))
        elif self.isotopologue_id is not None:
            isotopologues = f"I{self.isotopologue_id}"
        else:
            isotopologues = "all isotopologues"
        return f"GasConfig({self.name}, M{self.molecule_id}, {isotopologues})"


@dataclass
class LUTGridConfig:
    """Grid configuration shared across gases for a single LUT build.

    Attributes:
        T_grid:  Temperature knots [K]. Should bracket the atmospheric range
                 of interest (upper-troposphere ~200 K to surface ~320 K).
        P_grid:  Pressure knots [atm]. Should bracket the altitude range of
                 interest (upper-tropo ~0.1 atm to surface ~1.0 atm).
        nu_step: Spectral step [cm⁻¹] on the output wavenumber grid. HITRAN
                 lines have HWHMs down to ~0.005 cm⁻¹ at low pressure — step
                 ≤ 0.01 cm⁻¹ for hyperspectral work, 0.05 cm⁻¹ for tutorials.
        diluent_composition: Optional override for the air diluent fractions.
                 If None, each gas gets {'air': 1-VMR, 'self': VMR}.
        wing_cm: Absolute line cut-off w [cm⁻¹]: each line is evaluated
                 within ±w of its centre (HAPI's ``WavenumberWing``). The
                 default 25 cm⁻¹ is the convention HITRAN line parameters are
                 used with and continua (MT_CKD) are defined against.
        wing_halfwidths: Relative cut-off in half-widths max(Γ_L, Γ_D)
                 (HAPI's ``WavenumberWingHW``); the larger of the two applies.
                 0 by default, so the absolute cut-off alone applies.
    """

    T_grid: np.ndarray = field(
        default_factory=lambda: np.array(
            [200, 220, 240, 260, 280, 300, 320], dtype=float
        )
    )
    P_grid: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    )
    nu_step: float = 0.01
    diluent_composition: dict[str, float] | None = None
    wing_cm: float = 25.0
    wing_halfwidths: float = 0.0

    def get_diluent_for_gas(
        self,
        gas_name: str,
        vmr_nominal: float | None = None,
    ) -> dict[str, float]:
        """Return the HAPI ``Diluent`` dict for a given gas.

        If an explicit ``diluent_composition`` override was supplied at
        construction time it is returned verbatim; otherwise a standard
        ``{'air': 1 - VMR, 'self': VMR}`` split is built from
        ``DEFAULT_VMR_NOMINAL[gas_name]`` (falling back to 1e-6).
        """
        if self.diluent_composition is not None:
            return dict(self.diluent_composition)
        if vmr_nominal is None:
            vmr_nominal = DEFAULT_VMR_NOMINAL.get(gas_name, 1e-6)
        return {"air": 1.0 - vmr_nominal, "self": vmr_nominal}


# Canonical SWIR/NIR gases for methane/CO₂ remote sensing.
ATMOSPHERIC_GASES: dict[str, GasConfig] = {
    "CH4": GasConfig(
        name="CH4",
        molecule_id=6,
        nu_min=4000.0,  # SWIR (2500 nm)
        nu_max=8000.0,  # NIR (1250 nm)
        description="Methane — primary greenhouse-gas retrieval target.",
    ),
    "CO2": GasConfig(
        name="CO2",
        molecule_id=2,
        nu_min=4000.0,
        nu_max=8000.0,
        description="Carbon dioxide — reference gas for column retrievals.",
    ),
    "H2O": GasConfig(
        name="H2O",
        molecule_id=1,
        nu_min=4000.0,
        nu_max=25000.0,
        description="Water vapour — dominant atmospheric absorber.",
    ),
    "O2": GasConfig(
        name="O2",
        molecule_id=7,
        nu_min=12500.0,  # O2 A-band (~760 nm)
        nu_max=14000.0,
        description="Oxygen — surface-pressure retrievals via A-band.",
    ),
    "N2O": GasConfig(
        name="N2O",
        molecule_id=4,
        nu_min=4000.0,
        nu_max=5000.0,
        description="Nitrous oxide — greenhouse-gas side channel.",
    ),
    "CO": GasConfig(
        name="CO",
        molecule_id=5,
        nu_min=4000.0,
        nu_max=5000.0,
        description="Carbon monoxide — pollution / combustion tracer.",
    ),
}
