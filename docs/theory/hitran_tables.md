---
title: "Beer–Lambert radiative transfer and the HAPI absorption LUT"
---

# Beer–Lambert radiative transfer and the HAPI absorption LUT

*Ported from plumax 0.1.2 (`28f257e`); module and link paths updated for radtrax.*

This note derives the forward model behind `radtrax.spectroscopy.hitran` and `radtrax.spectroscopy.tables`. We start from a line-by-line sum of Voigt profiles, which gives the absorption cross-section $\sigma(\nu; T, p)$ of a gas, store it in a look-up table (LUT) over temperature and pressure, and then use it in a single-layer Beer–Lambert calculation of the transmittance. So the LUT is the expensive object we precompute once, and the forward model is an interpolation in $(T, p)$ followed by one exponential.

The theory follows Gordon et al. [^gordon2022] for the line-by-line physics (intensity, shape and cross-section) and Rothman et al. [^rothman2013] for the HITRAN database conventions. The LUT design follows the decoupled-VOD recipe used in operational SWIR retrievals (e.g. Frankenberg et al. [^frankenberg2005]).

[^gordon2022]: Gordon, I. E. et al. (2022). *The HITRAN2020 molecular spectroscopic database*. JQSRT **277**, 107949.
[^rothman2013]: Rothman, L. S. et al. (2013). *HITRAN: 40 years and counting*. JQSRT **130**, 4–50.
[^frankenberg2005]: Frankenberg, C. et al. (2005). *Iterative maximum a posteriori (IMAP)-DOAS for retrieval of strongly absorbing trace gases*. Atmos. Chem. Phys. **5**, 9–22.

**Symbols**

| Symbol | Meaning | Shape | Unit |
|---|---|---|---|
| $\nu$ | wavenumber; $G$ is the number of points on a wavenumber grid | scalar | cm⁻¹ |
| $T$, $p$ | temperature and pressure of the layer | scalar | K, atm |
| $\mathrm{VMR}$ | volume mixing ratio of the absorbing gas | scalar | mol/mol |
| $i$ | index of a line (one rovibrational transition) | - | - |
| $\nu_{0,i}$ | line centre | scalar | cm⁻¹ |
| $S_i(T)$ | line intensity | scalar | cm⁻¹/(molecule cm⁻²) |
| $E_i''$ | lower-state energy | scalar | cm⁻¹ |
| $Q(T)$ | total internal partition sum | scalar | dimensionless |
| $c_2$ | second radiation constant, $hc/k_B$ | scalar | cm K |
| $m$ | molecular mass of the gas | scalar | kg |
| $\gamma_{D,i}$, $\gamma_{L,i}$ | Doppler and Lorentz half-widths at half maximum | scalar | cm⁻¹ |
| $f_{V,i}(\nu)$ | Voigt profile of line $i$, normalised to unit area | scalar | cm |
| $\sigma(\nu; T, p)$ | absorption cross-section | $\mathbb{R}^{G}$ on a grid | cm²/molecule |
| $n_\text{total}$, $n_\text{gas}$ | number density of air and of the gas | scalar | molecules/cm³ |
| $k_\text{abs}(\nu)$ | absorption coefficient | $\mathbb{R}^{G}$ on a grid | cm⁻¹ |
| $\Delta z$ | vertical thickness of the layer | scalar | cm |
| $\mathrm{SZA}$, $\mathrm{VZA}$ | solar and viewing zenith angles | scalar | degrees |
| $\mathrm{AMF}$ | two-way air-mass factor | scalar | dimensionless |
| $\tau(\nu)$ | slant optical depth of the layer | $\mathbb{R}^{G}$ on a grid | dimensionless |
| $e^{-\tau(\nu)}$ | transmittance of the layer | $\mathbb{R}^{G}$ on a grid | dimensionless |
| $L(\nu)$ | radiance at the sensor | scalar | W m⁻² sr⁻¹ (cm⁻¹)⁻¹ |
| $F_0(\nu)$ | solar irradiance at the top of the atmosphere | scalar | W m⁻² (cm⁻¹)⁻¹ |
| $\rho(\nu)$ | surface reflectance | scalar | dimensionless |
| $\tau_\text{atm}(\nu)$ | optical depth of everything but the target gas (other gases, aerosol) | scalar | dimensionless |
| $\mathrm{VMR}_\text{bg}$, $\Delta \mathrm{VMR}$ | background mixing ratio and the plume's enhancement over it | scalar | mol/mol |

Pressures are in atm here because the tables are built in atm (HITRAN's broadening coefficients are per atm); elsewhere radtrax uses Pa unless a name says otherwise. On this page $\rho$ is a reflectance, not an air density, and $\tau$ is always an optical depth, never a transmittance.

## 1. Physical setup

Consider a pencil of sunlight crossing a plane-parallel atmospheric layer of vertical thickness $\Delta z$, which holds a single absorbing gas at temperature $T$, pressure $p$ and volume mixing ratio $\mathrm{VMR}$. We want the transmittance $e^{-\tau(\nu)} \in [0, 1]$ (the fraction of photons at wavenumber $\nu$ that get through untouched) as a function of the atmospheric state.

Two viewing angles enter: the solar zenith angle $\mathrm{SZA}$ (light coming in) and the viewing zenith angle $\mathrm{VZA}$ (light going out to the sensor after reflection at the ground). The plane-parallel two-way air-mass factor

$$
\mathrm{AMF}(\mathrm{SZA}, \mathrm{VZA}) = \frac{1}{\cos \mathrm{SZA}} + \frac{1}{\cos \mathrm{VZA}}
$$

converts the vertical path through the layer into the slant path the photons actually travel. It is valid for zenith angles below about 75° and gives $\mathrm{AMF} = 2$ at nadir ($\mathrm{SZA} = \mathrm{VZA} = 0$). In radtrax it is [`air_mass_factor`](xref:api#radtrax.atmosphere.air_mass_factor).

## 2. Line-by-line absorption cross-section

The HITRAN database lists every rovibrational transition $i$ of every atmospherically relevant isotopologue by its line centre $\nu_{0,i}$, a reference line intensity $S_i(T_\text{ref})$ at $T_\text{ref} = 296$ K and a lower-state energy $E_i''$. For a given $T$, each line's intensity is the reference value scaled by the Boltzmann and stimulated-emission factors, so we will have:

$$
S_i(T) = S_i(T_\text{ref}) \frac{Q(T_\text{ref})}{Q(T)} \frac{\exp(-c_2 E_i'' / T)}{\exp(-c_2 E_i'' / T_\text{ref})} \frac{1 - \exp(-c_2 \nu_{0,i} / T)}{1 - \exp(-c_2 \nu_{0,i} / T_\text{ref})}
$$

where $Q(T)$ is the total internal partition sum (HAPI ships the TIPS-2021 tables) and $c_2 = hc/k_B \approx 1.4388$ cm K is the second radiation constant.

Each line also has a shape $f_{V,i}(\nu; T, p)$, set by two broadening mechanisms. Thermal motion of the molecules gives a Gaussian (Doppler) shape whose half-width grows with the line centre and with $\sqrt{T/m}$:

$$
\gamma_{D,i}(T) = \frac{\nu_{0,i}}{c} \sqrt{\frac{2 \ln 2 \ k_B T}{m}}
$$

Collisions give a Lorentzian shape whose half-width is a sum over the broadening gases $d \in \{\text{air}, \text{self}\}$, each with its HITRAN coefficient $\gamma_{d,i}$ [cm⁻¹/atm], temperature exponent $n_{d,i}$ and fraction $x_d$ of the gas mixture:

$$
\gamma_{L,i}(T, p) = \frac{p}{p_\text{ref}} \sum_{d} x_d \gamma_{d,i} \left(\frac{T_\text{ref}}{T}\right)^{n_{d,i}}
$$

with $p_\text{ref} = 1$ atm. For CH₄ at 4300 cm⁻¹ and 296 K the Doppler half-width is about $6.6 \times 10^{-3}$ cm⁻¹, whereas at 1 atm the Lorentz half-width (several hundredths of a cm⁻¹) dominates; high in the atmosphere the two become comparable. HITRAN also gives a pressure shift $\delta_{\text{air},i}$, which moves each centre to $\nu_{0,i} + \delta_{\text{air},i} p$.

The real line shape is the convolution of the two, the Voigt profile:

$$
f_{V,i}(\nu; T, p) = \int_{-\infty}^{\infty} f_{G,i}(\nu'; T) f_{L,i}(\nu - \nu'; T, p) \mathrm{d}\nu'
$$

where $f_{G,i}$ and $f_{L,i}$ are the normalised Gaussian and Lorentzian profiles with half-widths $\gamma_{D,i}$ and $\gamma_{L,i}$. HAPI evaluates it with the Humlíček algorithm. Summing intensity times shape over all lines within the requested wavenumber window gives the absorption cross-section:

$$
\sigma(\nu; T, p) = \sum_{i} S_i(T) f_{V,i}(\nu; T, p)
$$

:::{important} Result · the absorption cross-section
$\sigma(\nu; T, p)$, in cm²/molecule, is what HAPI's `absorptionCoefficient_Voigt` returns with `HITRAN_units=True` (despite the name, that option gives $\sigma$, not the absorption coefficient). It is the one quantity the LUT stores.
:::

Practically speaking, the builder cuts each line at $\pm 25$ cm⁻¹ from its centre and sums every isotopologue of the gas; §8 explains why those two conventions matter.

## 3. Absorption coefficient and Beer–Lambert

The absorption coefficient follows from $\sigma$ and the number density of the absorbing gas. From the ideal-gas law ([`number_density`](xref:api#radtrax.atmosphere.number_density)), we will have:

$$
n_\text{total}(T, p) = \frac{p}{k_B T}
$$

$$
n_\text{gas} = \mathrm{VMR} \ n_\text{total}
$$

and the absorption coefficient is their product:

$$
k_\text{abs}(\nu; T, p, \mathrm{VMR}) = \sigma(\nu; T, p) n_\text{gas}
$$

The Beer–Lambert law says that each small path element $\mathrm{d}l$ removes a fraction $k_\text{abs} \mathrm{d}l$ of the light. So integrating over a homogeneous layer along the slant path, of length $\Delta z \ \mathrm{AMF}$, gives the optical depth and the transmittance:

$$
\tau(\nu) = k_\text{abs}(\nu; T, p, \mathrm{VMR}) \Delta z \ \mathrm{AMF}
$$

$$
e^{-\tau(\nu)} = \exp\left(-k_\text{abs}(\nu; T, p, \mathrm{VMR}) \Delta z \ \mathrm{AMF}\right)
$$

:::{important} Result · single-layer Beer–Lambert
$e^{-\tau(\nu)}$ is the single-layer forward model of [`beers_law_from_lut`](xref:api#radtrax.spectroscopy.beers_law_from_lut). Optical depths add along the path and transmittances multiply, which is what the plume ratio in §5 uses.
:::

The code names the transmittance `tau` in places (the docstring of [`transmittance`](xref:api#radtrax.spectroscopy.transmittance) writes $\tau(\nu) = \exp(-\alpha L_\text{vert} \mathrm{AMF})$, with its own $\alpha$ for our $k_\text{abs}$ and $L_\text{vert}$ for our $\Delta z$). What it returns is the transmittance $e^{-\tau}$ in the notation of this page.

**Units**

| Quantity | Units | |
|---|---|---|
| $S_i f_{V,i}$ | cm⁻¹/(molecule cm⁻²) × cm | cm²/molecule ✓ |
| $n_\text{gas} = \mathrm{VMR} \ p / (k_B T)$ | mol/mol × molecules/cm³ | molecules/cm³ ✓ |
| $k_\text{abs} = \sigma n_\text{gas}$ | cm²/molecule × molecules/cm³ | cm⁻¹ ✓ |
| $\tau = k_\text{abs} \Delta z \ \mathrm{AMF}$ | cm⁻¹ × cm × 1 | dimensionless ✓ |

## 4. Why a look-up table

Evaluating $\sigma(\nu; T, p)$ at run time for every pixel is very expensive, since each call sums thousands of Voigt profiles over a dense $\nu$ grid. However, at fixed $\nu$, $\sigma$ is a smooth function of $(T, p)$: the Doppler width scales as $\sqrt{T}$, the Lorentz width as $p \ T^{-n}$, and the line intensity $S_i(T)$ is entire in $T$. So we precompute $\sigma$ once on a coarse $(T, p)$ grid and interpolate at run time.

The natural axes are:

- **Wavenumber $\nu$**, set by the instrument's native resolution. Hyperspectral work (TROPOMI, EMIT) needs a step of 0.01 cm⁻¹ or less, since HITRAN half-widths go down to about 0.005 cm⁻¹ at low pressure; coarser is fine for tutorials.
- **Temperature $T$**, from the upper troposphere (about 200 K) to the surface (about 320 K). Steps of about 20 K keep the bilinear interpolation error below 1 %.
- **Pressure $p$**, from the upper troposphere (about 0.1 atm) to the surface (about 1.0 atm), logarithmic or uniform depending on where accuracy matters most.

The builder's defaults ([`LUTGridConfig`](xref:api#radtrax.spectroscopy.hitran.LUTGridConfig)) follow these ranges: seven temperatures from 200 to 320 K every 20 K, six pressures between 0.1 and 1.0 atm, and a 0.01 cm⁻¹ step. The tables of the layered forward model (v0's grid) reach higher in the atmosphere: 4000–6500 cm⁻¹ at 0.01 cm⁻¹, sixteen pressures log-spaced from 0.01 to 1.1 atm and eight temperatures from 180 to 320 K, interpolated bilinearly in $(\ln p, T)$ by `TableOpacity`.

Viewing geometry ($\mathrm{SZA}$, $\mathrm{VZA}$) is not a LUT axis: the $\mathrm{AMF}$ multiplies $k_\text{abs} \Delta z$ after the lookup, so one cross-section table serves every satellite geometry.

## 5. Differential Beer–Lambert: the plume enhancement

For plume retrievals from point sources (landfills, oil and gas, industrial stacks) we care about the enhancement above the regional background, not the absolute column. Let $\mathrm{VMR}_\text{bg}$ be the background mixing ratio and $\mathrm{VMR}_\text{bg} + \Delta \mathrm{VMR}$ the mixing ratio in the plume pixel. Since $k_\text{abs}$ is linear in the mixing ratio, the optical depth splits additively into a background part and an enhancement part:

$$
\tau_\text{tot}(\nu) = \tau_\text{bg}(\nu) + \Delta \tau(\nu)
$$

$$
\Delta \tau(\nu) = \sigma(\nu; T, p) \Delta \mathrm{VMR} \ n_\text{total} \Delta z \ \mathrm{AMF}
$$

and so the transmittances multiply:

$$
e^{-\tau_\text{tot}(\nu)} = e^{-\tau_\text{bg}(\nu)} e^{-\Delta \tau(\nu)}
$$

On the measurement side, dividing the radiance of the plume pixel by that of a background pixel cancels every factor the two pixels share:

$$
\frac{L_\text{plume}(\nu)}{L_\text{bg}(\nu)} = \frac{F_0(\nu) \rho(\nu) e^{-\tau_\text{atm}(\nu)} e^{-\tau_\text{bg}(\nu)} e^{-\Delta \tau(\nu)}}{F_0(\nu) \rho(\nu) e^{-\tau_\text{atm}(\nu)} e^{-\tau_\text{bg}(\nu)}} = e^{-\Delta \tau(\nu)}
$$

The solar irradiance $F_0$, the broadband surface reflectance $\rho$, the aerosol and other-gas optical depth $\tau_\text{atm}$ and the target gas's own background all drop out of the ratio, which leaves only the narrow-band absorption features of the enhancement. This cancellation is what makes plume retrievals from space possible, even when an absolute column retrieval would be dominated by nuisance parameters. It assumes the two pixels share their reflectance, which is the assumption the [absorption-shape test](absorption_test.md) makes explicit and pairs pixels to satisfy.

The retrieval then fits the simulated ratio to the measured one:

$$
\widehat{\Delta \mathrm{VMR}} = \arg\min_{\Delta \mathrm{VMR}} \Vert L_\text{plume} / L_\text{bg} - e^{-\Delta \tau(\Delta \mathrm{VMR})} \Vert_2^2
$$

where the norm runs over the wavenumber grid. [`plume_ratio_spectrum`](xref:api#radtrax.spectroscopy.plume_ratio_spectrum) computes the model side, $e^{-\tau_\text{tot}} / e^{-\tau_\text{bg}}$, with `vmr_total` $= \mathrm{VMR}_\text{bg} + \Delta \mathrm{VMR}$ (its docstring writes `vmr_total = vmr_background * (1 + enhancement)`, a fractional enhancement). The same Beer–Lambert kernel is the forward model of the [matched filter](matched_filter.md) and of the [3D-Var retrieval](var3d.md) downstream.

## 6. Assumptions and caveats

The forward model above inherits a stack of approximations, roughly in decreasing order of severity:

1. **Absorption-only radiative transfer.** Scattering by molecules (Rayleigh), aerosols and clouds is ignored. For hazy or cloudy pixels a full scalar or vector RTM (e.g. VLIDORT, SASKTRAN) is needed.
2. **A single homogeneous layer.** Real atmospheres have $T(z)$, $p(z)$ and $\mathrm{VMR}(z)$ profiles. Operational retrievals discretise into 20–40 layers and sum the layers' vertical optical depths (VODs); the same LUT machinery applies per layer, which is what radtrax's layered [`ForwardModel`](xref:api#radtrax.ForwardModel) does with `TableOpacity`.
3. **Plane-parallel geometry.** It breaks down past about 75° zenith, where the $\sec$ factors in the $\mathrm{AMF}$ are replaced by a spherical-shell air mass (radtrax's `ViewingGeometry.layer_air_mass`, radtrax#104).
4. **Local thermodynamic equilibrium (LTE).** It holds in the troposphere and lower stratosphere, and fails at very high altitudes, where radiative excitation competes with collisional excitation.
5. **No line mixing.** HAPI's `_Voigt` profile does not account for line mixing, which matters at very high pressure for some species (the CO₂ 4.3 µm band, for instance). The `_HT` (Hartmann–Tran) profile is available but costs more.
6. **Diluent composition.** We pin the diluent to `{'air': 1-VMR, 'self': VMR}`, with dry-air broadening coefficients. Humid conditions change $\gamma_L$ slightly through water-vapour broadening, which is second order for most retrievals.

## 7. Software surface

Three functions carry the physics:

- [`compute_absorption_lut(gas_config, grid_config, *, cache_dir, progress)`](xref:api#radtrax.spectroscopy.hitran.compute_absorption_lut) evaluates $\sigma(\nu; T, p)$ on the full LUT grid by calling `hapi.absorptionCoefficient_Voigt` at each $(T, p)$ knot, for a [`GasConfig`](xref:api#radtrax.spectroscopy.hitran.GasConfig) and a [`LUTGridConfig`](xref:api#radtrax.spectroscopy.hitran.LUTGridConfig). [`build_table`](xref:api#radtrax.spectroscopy.hitran.build_table) wraps it for v0's tables, with the command line `python -m radtrax.spectroscopy.hitran GAS [GAS ...]`.
- [`beers_law_from_lut(ds, *, vmr, T_K, p_atm, l_vert_cm, sza_deg, vza_deg)`](xref:api#radtrax.spectroscopy.beers_law_from_lut) interpolates the LUT at $(T, p)$, multiplies by $n_\text{gas}$ to get $k_\text{abs}$, applies the $\mathrm{AMF}$ and the exponential, and returns the transmittance $e^{-\tau(\nu)}$.
- [`plume_ratio_spectrum(ds, *, vmr_background, vmr_total, T_K, p_atm, l_vert_cm, sza_deg, vza_deg)`](xref:api#radtrax.spectroscopy.plume_ratio_spectrum) returns the differential form $e^{-\tau_\text{tot}(\nu)} / e^{-\tau_\text{bg}(\nu)} = e^{-\Delta \tau(\nu)}$, used by the enhancement retrieval in [03_beers_law_with_lut.ipynb](https://github.com/jejjohnson/plumax/blob/main/docs/notebooks/hapi_lut/03_beers_law_with_lut.ipynb).

The sketch below runs the chain of §3 and §5 on radtrax's synthetic CH₄ table, which has the same layout as a HITRAN-built LUT, so it needs no HAPI and no download. The layer is 1 km at 1 atm and 288 K with 1.9 ppm of CH₄, the sun at 35° and the sensor at 10°.

```python
from radtrax.spectroscopy import (
    absorption_coefficient,
    beers_law_from_lut,
    interpolate_cross_section,
    plume_ratio_spectrum,
    transmittance,
)
from radtrax.spectroscopy.synthetic import synthetic_table

# Shapes: G = 14,000 wavenumbers (4230–4370 cm⁻¹ at 0.01 cm⁻¹); T = 6, P = 10 knots

# today (radtrax)
# σ(ν; T, p) on the table's knots [cm²/molecule]
table = synthetic_table()                                    # → (G, T, P)

# today (radtrax)
# σ(ν) at (T, p), bilinear between the knots
sigma = interpolate_cross_section(table, T_K=288.0, p_atm=1.0)    # (G, T, P) → (G,)

# today (radtrax)
# k_abs(ν) = σ(ν) VMR p / (k_B T)   [cm⁻¹]
k_abs = absorption_coefficient(sigma, vmr=1.9e-6, p_atm=1.0, T_K=288.0)  # (G,) → (G,)

# today (radtrax)
# e^{-τ(ν)},  τ(ν) = k_abs(ν) Δz AMF
trans = transmittance(k_abs, l_vert_cm=1e5, sza_deg=35.0, vza_deg=10.0)  # (G,) → (G,)

# today (radtrax)
# the same three steps in one call
trans_lut = beers_law_from_lut(                              # → (G,)
    table, vmr=1.9e-6, T_K=288.0, p_atm=1.0,
    l_vert_cm=1e5, sza_deg=35.0, vza_deg=10.0,
)

# today (radtrax)
# e^{-τ_tot(ν)} / e^{-τ_bg(ν)} = e^{-Δτ(ν)}
ratio = plume_ratio_spectrum(                                # → (G,)
    table, vmr_background=1.9e-6, vmr_total=2.9e-6, T_K=288.0, p_atm=1.0,
    l_vert_cm=1e5, sza_deg=35.0, vza_deg=10.0,
)
```

`interpolate_cross_section`, `absorption_coefficient` and `transmittance` are the three steps of §3, and `beers_law_from_lut` is the same chain in one call (the two agree exactly). On this layer the $\mathrm{AMF}$ is 2.24, the strongest line leaves a transmittance of 0.64 ($\tau \approx 0.45$), and an enhancement of 1 ppm over the 1 km layer (1000 ppm·m) brings the plume ratio down to 0.79 at its deepest. These numbers come from the synthetic table, so they illustrate the chain rather than real CH₄.

The notebooks walk through building the LUT ([01](../notebooks/hitran_ch4_table.ipynb)), extending it to several gases ([02](../notebooks/hitran_multigas_tables.ipynb)) and, in plumax because it simulates the plume, using it as the forward model of a plume-enhancement retrieval ([03](https://github.com/jejjohnson/plumax/blob/main/docs/notebooks/hapi_lut/03_beers_law_with_lut.ipynb)).

## 8. Benchmarks against published numbers

The builder is tested against numbers it did not produce, the benchmarks of Korkin et al. [^korkin2025] (`tests/spectroscopy/test_benchmarks_korkin2025.py`, radtrax#171).

- **One line** (their §4.6, Table 4): a single HITRAN2020 O₂ record, whose peak cross-section on a 0.01 cm⁻¹ grid at 270, 300 and 330 K and 0.9, 1.0 and 1.1 atm must match the published seven digits to $10^{-4}$ relative, the accuracy of HAPI's Voigt algorithm. It catches unit, partition-sum and line-shift errors.
- **Gas cells** (their §5.4.1 and §5.4.2): pure O₂ in the A band and pure CH₄ at 2.3 µm, whose optical thickness $\tau = n_\text{col} \sigma$ (with $n_\text{col}$ the cell's column density, in molecules/cm²) must be within 1 % of `gcell`'s wherever $\tau > 10^{-3}$, and, for CH₄, whose filter-weighted transmittance $e^{-\tau} f$ must be within $10^{-3}$ of SpectralCalc's at every wavenumber.

Three conventions decide whether the cells can pass. The cells are pure gas, so only self-broadening applies, and `gcell` shifts each line by $\delta_\text{air} p$ whatever the broadener (HITRAN has no self shift, so HAPI would shift a pure gas by nothing). `gcell` sums every isotopologue and cuts each line at $\pm 25$ cm⁻¹. And the cells were computed from HITRAN2020, while HITRANonline now serves a later edition whose CH₄ lines differ by several per cent, so the tests read the authors' own line lists rather than fetching lines.

Since radtrax#172 the builder follows the same two line conventions by default: it fetches every isotopologue of a gas (`GasConfig.isotopologue_ids = None`; HITRAN's intensities already carry each isotopologue's natural abundance) and cuts each line at $\pm 25$ cm⁻¹ (`LUTGridConfig.wing_cm`). Built the old way, with the main isotopologue alone and HAPI's default cut-off of 50 half-widths, the O₂ cell's $\tau$ was up to 100 % off `gcell`'s (median 8.5 %) and the CH₄ cell's up to 8.2 % (median 2.9 %, on 4300–4305 cm⁻¹). Built the new way, they agree to $7 \times 10^{-5}$ and $7 \times 10^{-7}$, and the CH₄ filter-weighted transmittance is within $7 \times 10^{-4}$ of SpectralCalc's.

[^korkin2025]: Korkin, S., Sayer, A. M., Ibrahim, A. & Lyapustin, A. (2025). *A practical guide to coding line-by-line trace gas absorption in Earth's atmosphere*. JQSRT **337**, 109345.
