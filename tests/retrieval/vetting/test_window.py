"""VettingWindow and the CH₄ and CO presets (radtrax#155)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from radtrax import ParameterError
from radtrax.atmosphere import afgl_us_standard
from radtrax.atmosphere.gas import AVOGADRO_PER_MOL, BOLTZMANN_J_PER_K
from radtrax.retrieval.vetting import VettingWindow
from radtrax.retrieval.vetting.window import _MOL_M2_PER_PPM_M


# EMIT's 285 band centres, uniformly spaced between its first and last
# (381.0 and 2493.3 nm in the L1B sensor-band parameters).
EMIT_NM = np.linspace(381.00558, 2493.2576, 285)
PRESETS = [VettingWindow.ch4_emit(), VettingWindow.co_emit()]


def test_ch4_fit_bands_are_exactly_those_inside_2100_2440() -> None:
    fit = VettingWindow.ch4_emit().fit_bands(EMIT_NM)
    assert fit.shape == (285,) and fit.dtype == bool
    np.testing.assert_array_equal(fit, (EMIT_NM >= 2100.0) & (EMIT_NM <= 2440.0))


def test_ch4_pairing_excludes_both_absorbing_ranges() -> None:
    pairing = VettingWindow.ch4_emit().pairing_bands(EMIT_NM)
    in_1650 = (EMIT_NM >= 1640.0) & (EMIT_NM <= 1690.0)
    in_2300 = (EMIT_NM >= 2100.0) & (EMIT_NM <= 2440.0)
    assert in_1650.any() and in_2300.any()
    np.testing.assert_array_equal(pairing, ~(in_1650 | in_2300))


@pytest.mark.parametrize("window", PRESETS, ids=["ch4", "co"])
def test_preset_masks_are_disjoint_and_nonempty(window: VettingWindow) -> None:
    fit, pairing = window.fit_bands(EMIT_NM), window.pairing_bands(EMIT_NM)
    assert fit.any() and pairing.any()
    assert not (fit & pairing).any()


def test_co_preset() -> None:
    window = VettingWindow.co_emit()
    assert window.gas == "CO"
    np.testing.assert_array_equal(
        window.fit_bands(EMIT_NM), (EMIT_NM >= 2278.0) & (EMIT_NM <= 2441.0)
    )
    assert window.mf_background_mol_m2 == pytest.approx(300.0 * _MOL_M2_PER_PPM_M)


def test_ranges_are_inclusive_at_both_ends() -> None:
    window = VettingWindow("CH4", ((2100.0, 2440.0),), ((2100.0, 2440.0),), 0.0)
    wl = np.array([2099.999, 2100.0, 2440.0, 2440.001])
    np.testing.assert_array_equal(window.fit_bands(wl), [False, True, True, False])
    np.testing.assert_array_equal(window.pairing_bands(wl), [True, False, False, True])


def test_no_absorbing_range_pairs_on_every_band() -> None:
    window = VettingWindow("CH4", ((2100.0, 2440.0),), (), 0.0)
    assert window.pairing_bands(EMIT_NM).all()


def test_ppm_m_is_the_afgl_bottom_layer() -> None:
    """1 ppm·m = 10⁻⁶ p / (k_B T N_A) × 1 m in the notebook's surface layer."""
    state = afgl_us_standard(z_top_km=20.0, gases=["CH4"])
    p, t = float(state.p_pa[0]), float(state.t_k[0])
    expected = 1e-6 * p / (BOLTZMANN_J_PER_K * t) / AVOGADRO_PER_MOL
    # rel=1e-3: the constant is the computed 4.0299e-5 rounded to three figures
    assert pytest.approx(expected, rel=1e-3) == _MOL_M2_PER_PPM_M
    assert VettingWindow.ch4_emit().mf_background_mol_m2 == pytest.approx(
        30.0 * expected, rel=1e-3
    )


def test_windows_with_equal_fields_are_equal_and_hashable() -> None:
    a = VettingWindow("CH4", ((2100.0, 2440.0),), ((1640.0, 1690.0),), 1e-3)
    b = VettingWindow("CH4", ((2100.0, 2440.0),), ((1640.0, 1690.0),), 1e-3)
    assert a == b and hash(a) == hash(b)


def test_window_is_frozen() -> None:
    window = VettingWindow.ch4_emit()
    with pytest.raises(dataclasses.FrozenInstanceError):
        window.gas = "CO"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("fit", "absorbing", "threshold", "match"),
    [
        ((), (), 0.0, "at least one"),
        (((2440.0, 2100.0),), (), 0.0, "empty or reversed"),
        (((2100.0, 2100.0),), (), 0.0, "empty or reversed"),
        (((2100.0, 2440.0),), ((1690.0, 1640.0),), 0.0, "empty or reversed"),
        (((2100.0, float("nan")),), (), 0.0, "not finite"),
        (((2100.0, 2440.0),), (), -1e-3, ">= 0"),
        (((2100.0, 2440.0),), (), float("nan"), ">= 0"),
    ],
)
def test_invalid_windows_raise(
    fit: tuple, absorbing: tuple, threshold: float, match: str
) -> None:
    with pytest.raises(ParameterError, match=match):
        VettingWindow("CH4", fit, absorbing, threshold)
