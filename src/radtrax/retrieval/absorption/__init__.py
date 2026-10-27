"""The absorption-shape test: does a ratio spectrum have a gas's shape?

A plume-agnostic retrieval on arrays. Given two sets of pixels of one
radiance cube (in-pixels and out-pixels, as flat indices), it pairs them one
to one on bands the gas does not absorb, divides the pairs' mean radiances
into a ratio spectrum, fits that ratio with a smooth continuum times the
gas's transmittance from `radtrax.ForwardModel`, and returns the fitted
column ``ΔX̂`` and the misfit ``D_norm`` (Xiang et al.'s eq. 11): how far the
ratio's shape is from the transmittance's. It knows nothing of plumes or
candidates; ``plumax.vetting`` chooses the pixel sets from plume candidates
and decides on the scores. Theory: the absorption-shape test page.

- `window`: `AbsorptionWindow`, a gas's fitting and pairing bands and its
  matched-filter background cut, with EMIT's CH₄ and CO presets (#155).
"""

from __future__ import annotations

from radtrax.retrieval.absorption.window import AbsorptionWindow


__all__ = ["AbsorptionWindow"]
