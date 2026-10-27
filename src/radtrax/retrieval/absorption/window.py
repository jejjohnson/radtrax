"""A gas's absorption window: its fitting bands, its pairing bands and its background cut.

A window is configuration, not state, so it is a frozen dataclass, as
`radtrax.spectroscopy.hitran`'s configs are. Its masks are NumPy booleans:
they select bands before anything is traced, and nothing differentiates
through them.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from jaxtyping import ArrayLike, Bool, Float

from radtrax.atmosphere.gas import AVOGADRO_PER_MOL, BOLTZMANN_J_PER_K
from radtrax.exceptions import ParameterError


__all__ = ["AbsorptionWindow"]

# The bottom layer of `afgl_us_standard(z_top_km=20.0)`, ~500 m: its pressure
# [Pa] and temperature [K] as that profile gives them.
_BOTTOM_LAYER_P_PA = 95476.19736570618
_BOTTOM_LAYER_T_K = 284.95
# 1 ppm·m in mol/m²: 10⁻⁶ p / (k_B T N_A) × 1 m in that layer, ≈ 4.030e-5.
# Other conventions differ: 4.23e-5 at the surface (1013.25 hPa, 288.15 K),
# 4.46e-5 at STP (273.15 K).
_MOL_M2_PER_PPM_M = (
    1e-6 * _BOTTOM_LAYER_P_PA / (BOLTZMANN_J_PER_K * _BOTTOM_LAYER_T_K)
) / AVOGADRO_PER_MOL

Ranges = tuple[tuple[float, float], ...]


def _check_ranges(name: str, ranges: Ranges) -> None:
    """Raise unless every ``(lo, hi)`` in ``ranges`` is finite with ``lo < hi``."""
    for lo, hi in ranges:
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ParameterError(f"{name}: range ({lo}, {hi}) nm is not finite")
        if lo >= hi:
            raise ParameterError(
                f"{name}: range ({lo}, {hi}) nm is empty or reversed; need lo < hi"
            )


def _inside(
    wavelength_nm: Float[ArrayLike, " B"], ranges: Ranges
) -> Bool[np.ndarray, " B"]:
    """Bands inside any of ``ranges``, inclusive at both ends."""
    wl = np.asarray(wavelength_nm)
    inside = np.zeros(wl.shape, dtype=bool)
    for lo, hi in ranges:
        inside |= (wl >= lo) & (wl <= hi)
    return inside


@dataclasses.dataclass(frozen=True)
class AbsorptionWindow:
    """A gas's fitting bands, the bands kept out of pairing, and the background cut.

    Ranges are in nm and inclusive at both ends, as in the reference code's
    configuration (``pv/config/config.yaml``). Pairing uses every band outside
    ``absorbing_nm``, so a window whose fitting ranges lie inside its
    absorbing ranges has disjoint fitting and pairing masks.

    Attributes:
        gas: The gas whose enhancement is fitted, as named in the forward
            model's state (``"CH4"``).
        fit_nm: The ``(lo, hi)`` ranges, in nm, of the bands the
            transmittance is fitted on; at least one.
        absorbing_nm: The ``(lo, hi)`` ranges, in nm, where the gas absorbs;
            pairing uses only bands outside all of them.
        mf_background_mol_m2: The matched-filter score, in mol/m², below
            which a pixel counts as background; the absorption-shape test
            does not use it, the code that chooses the out-pixels does
            (``plumax.vetting``'s background pool). The presets convert
            EMIT's ppm·m with 1 ppm·m ≈ 4.03 × 10⁻⁵ mol/m², the molar
            density of air in the bottom layer of the US standard
            atmosphere, 954.8 hPa and 285.0 K, times 1 m.

    Raises:
        ParameterError: If ``fit_nm`` is empty, a range is not finite or
            has ``lo >= hi``, or the threshold is negative or not finite.

    Examples:
        >>> import numpy as np
        >>> from radtrax.retrieval.absorption import AbsorptionWindow
        >>> window = AbsorptionWindow.ch4_emit()
        >>> window.fit_nm, window.absorbing_nm
        (((2100.0, 2440.0),), ((1640.0, 1690.0), (2100.0, 2440.0)))
        >>> print(f"{window.mf_background_mol_m2:.3e} mol/m²")
        1.209e-03 mol/m²
        >>> wl = np.array([1500.0, 1650.0, 2100.0, 2300.0, 2440.0, 2450.0])
        >>> window.fit_bands(wl)
        array([False, False,  True,  True,  True, False])
        >>> window.pairing_bands(wl)
        array([ True, False, False, False, False,  True])
    """

    gas: str
    fit_nm: Ranges
    absorbing_nm: Ranges
    mf_background_mol_m2: float

    def __post_init__(self) -> None:
        if not self.fit_nm:
            raise ParameterError("fit_nm needs at least one (lo, hi) range")
        _check_ranges("fit_nm", self.fit_nm)
        _check_ranges("absorbing_nm", self.absorbing_nm)
        threshold = self.mf_background_mol_m2
        if not (math.isfinite(threshold) and threshold >= 0):
            raise ParameterError(
                f"mf_background_mol_m2 must be finite and >= 0, got {threshold}"
            )

    @classmethod
    def ch4_emit(cls) -> AbsorptionWindow:
        """EMIT's CH₄ window, as the reference code's configuration sets it.

        Fits 2100–2440 nm; keeps CH₄'s 1640–1690 and 2100–2440 nm bands out
        of pairing; counts pixels below 30 ppm·m as background.
        """
        return cls(
            gas="CH4",
            fit_nm=((2100.0, 2440.0),),
            absorbing_nm=((1640.0, 1690.0), (2100.0, 2440.0)),
            mf_background_mol_m2=30.0 * _MOL_M2_PER_PPM_M,
        )

    @classmethod
    def co_emit(cls) -> AbsorptionWindow:
        """EMIT's CO window, from the source report's Table S1 (Růžička et al.).

        Fits 2278–2441 nm and keeps the same range out of pairing; counts
        pixels below 300 ppm·m as background.
        """
        return cls(
            gas="CO",
            fit_nm=((2278.0, 2441.0),),
            absorbing_nm=((2278.0, 2441.0),),
            mf_background_mol_m2=300.0 * _MOL_M2_PER_PPM_M,
        )

    def fit_bands(
        self, wavelength_nm: Float[ArrayLike, " B"]
    ) -> Bool[np.ndarray, " B"]:
        """The bands inside any fitting range.

        Args:
            wavelength_nm: Band centres, in nm, ``(B,)``.

        Returns:
            A boolean mask, ``(B,)``.
        """
        return _inside(wavelength_nm, self.fit_nm)

    def pairing_bands(
        self, wavelength_nm: Float[ArrayLike, " B"]
    ) -> Bool[np.ndarray, " B"]:
        """The bands outside every absorbing range, which pairing compares.

        Args:
            wavelength_nm: Band centres, in nm, ``(B,)``.

        Returns:
            A boolean mask, ``(B,)``.
        """
        return ~_inside(wavelength_nm, self.absorbing_nm)
