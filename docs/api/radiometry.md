# Radiometry

Radiometric source functions, the lowest layer every emitting or measuring
stage reaches (#246): the Planck radiance `B(ν̃, T)` per unit wavenumber in
W m⁻² sr⁻¹ (cm⁻¹)⁻¹, its log, their closed-form derivatives, the radiation
constants, and the per-µm and per-GHz views. Wavenumber is the one internal
axis; wavelength and frequency are views. Brightness temperature inverts Planck: in
closed form per wavenumber, and per band of an `Instrument` by Newton with
implicit differentiation.

The Planck function is checked against independent oracles (gate TH12): the
constants against the SI 2019 exact `h`, `c` and `k`, the Stefan–Boltzmann law
from the spectral integral, the Wien peaks, and pyradi's `ryplanck` (radiance,
derivatives and Widger–Woodall band integrals, rtol 1e-12). pyradi is never
imported: `scripts/make_planck_oracle.py` evaluates `ryplanck.py` at commit
`c7cfe9df96fdf0e6c258ee6fd624f62f19ebab06` (MIT licence at the root, BSD-3
per file) in an isolated interpreter and stores the values, with the commit,
the NumPy and SciPy versions and SciPy's constants, in
`tests/data/planck_pyradi_c7cfe9d.npz`.

::: radtrax.radiometry
    options:
      members: false

## `planck`

::: radtrax.radiometry.planck
    options:
      show_root_heading: false
      heading_level: 3

## `brightness`

::: radtrax.radiometry.brightness
    options:
      show_root_heading: false
      heading_level: 3
