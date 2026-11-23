# Radiometry

What an instrument adds to Planck's law: the band-integrated Planck radiance
and the band brightness temperature that inverts it, by Newton with implicit
differentiation. Planck's law in wavenumber, its logarithm, its per-µm and
per-GHz views and the monochromatic brightness temperature are
[radiomax](https://github.com/jejjohnson/radiomax)'s (`radiomax.thermal`).

radiomax's Planck function is held here to the checks that radtrax's own was
(gate TH12): the constants against the SI 2019 exact `h`, `c` and `k`, the
Stefan–Boltzmann law from the spectral integral, the Wien peaks, and pyradi's
`ryplanck` (radiance, derivatives and Widger–Woodall band integrals, rtol
1e-12). pyradi is never imported: `scripts/make_planck_oracle.py` evaluates
`ryplanck.py` at commit `c7cfe9df96fdf0e6c258ee6fd624f62f19ebab06` (MIT licence
at the root, BSD-3 per file) in an isolated interpreter and stores the values,
with the commit, the NumPy and SciPy versions and SciPy's constants, in
`tests/data/planck_pyradi_c7cfe9d.npz`.

::: radtrax.radiometry
    options:
      members: false

## `brightness`

::: radtrax.radiometry.brightness
    options:
      show_root_heading: false
      heading_level: 3
