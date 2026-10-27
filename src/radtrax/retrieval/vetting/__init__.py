"""Plume vetting: a spectral test for matched-filter candidates.

Xiang et al.'s method on arrays: seeds and a background pool from the
matched filter, one-to-one pairs on bands the gas does not absorb, the ratio
spectrum, its fit to the forward model's transmittance, ``D_norm`` and a
p-value against the same scene's clutter. Theory: the plume-vetting page.

- `window`: `VettingWindow`, a gas's fitting and pairing bands and its
  background-pool cut, with EMIT's CH₄ and CO presets (#155).
- `transmittance`: `transmittance_table`, the band transmittance of a
  column enhancement from `ForwardModel`, exact or constant-coefficient
  (#154).
- `seeds`: `select_seeds` and `background_pool`, a candidate's in-plume
  seeds and the background pool they are paired from (#148).
- `pairing`: `pair_pixels` and `ratio_spectrum`, one-to-one pairs on the
  pairing bands and the ratio of their mean radiances (#149).
- `fit`: `TransmittanceFit`, `fit_transmittance` and `d_norm`, the profile
  fit of the ratio to a Legendre continuum times the transmittance, and its
  score (#150).
- `null`: `null_masks` and `null_pvalue`, rotated and translated copies of
  a candidate's mask off every candidate, and the candidate's p-value
  among their scores (#152).
"""

from __future__ import annotations

from radtrax.retrieval.vetting.fit import TransmittanceFit, d_norm, fit_transmittance
from radtrax.retrieval.vetting.null import null_masks, null_pvalue
from radtrax.retrieval.vetting.pairing import pair_pixels, ratio_spectrum
from radtrax.retrieval.vetting.seeds import background_pool, select_seeds
from radtrax.retrieval.vetting.transmittance import transmittance_table
from radtrax.retrieval.vetting.window import VettingWindow


__all__ = [
    "TransmittanceFit",
    "VettingWindow",
    "background_pool",
    "d_norm",
    "fit_transmittance",
    "null_masks",
    "null_pvalue",
    "pair_pixels",
    "ratio_spectrum",
    "select_seeds",
    "transmittance_table",
]
