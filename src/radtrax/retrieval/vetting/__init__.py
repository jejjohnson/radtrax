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
"""

from __future__ import annotations

from radtrax.retrieval.vetting.pairing import pair_pixels, ratio_spectrum
from radtrax.retrieval.vetting.seeds import background_pool, select_seeds
from radtrax.retrieval.vetting.transmittance import transmittance_table
from radtrax.retrieval.vetting.window import VettingWindow


__all__ = [
    "VettingWindow",
    "background_pool",
    "pair_pixels",
    "ratio_spectrum",
    "select_seeds",
    "transmittance_table",
]
