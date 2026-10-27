"""Plume vetting: a spectral test for matched-filter candidates.

Xiang et al.'s method on arrays: seeds and a background pool from the
matched filter, one-to-one pairs on bands the gas does not absorb, the ratio
spectrum, its fit to the forward model's transmittance, ``D_norm`` and a
p-value against the same scene's clutter. Theory: the plume-vetting page.

- `window`: `VettingWindow`, a gas's fitting and pairing bands and its
  background-pool cut, with EMIT's CH₄ and CO presets (#155).
"""

from __future__ import annotations

from radtrax.retrieval.vetting.window import VettingWindow


__all__ = ["VettingWindow"]
