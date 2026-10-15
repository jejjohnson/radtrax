"""The synthetic Sentinel-2 scenes of the GeoStack notes *radtrax · 1* and *· 2*.

Rebuilt draw for draw from ``notebooks/geomodels/radtrax_01_column_retrieval.py``
(stages 1–3) and ``radtrax_02_mbmp.py`` (stages 2–3) in jejjohnson/geostack,
with the spectrum and band responses from radtrax, so tests can check that
radtrax's retrievals reproduce the notes' numbers. Both need georeader's
Sentinel-2 response table (the ``[geo]`` extra), and build in a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from radtrax.geo.srf import sentinel2_band_responses
from radtrax.slab.nb_lut import NBLookup, inject_plume
from radtrax.spectroscopy.synthetic import band_mean, methane_cross_section


SEED = 0
SIG12, RATIO11 = 0.40, 0.10
OMEGA_BG = 1.9e-6 * (16.04 / 28.96) * 101325.0 / 9.80665  # kg/m²
SZA, VZA = 35.0, 5.0
M_AMF = 1 / np.cos(np.radians(SZA)) + 1 / np.cos(np.radians(VZA))
M_CH4 = 0.01604  # kg/mol, radtrax · 2's


def _transmittance(sigma: np.ndarray, m, omega, f: np.ndarray) -> np.ndarray:
    """T_b(m, Ω) = ⟨exp(−m σ Ω)⟩_b, element-wise over broadcast m and Ω."""
    m, omega = np.broadcast_arrays(np.asarray(m, float), np.asarray(omega, float))
    out = np.empty(m.shape)
    sel = f > 0
    sig_b, f_b = sigma[sel], f[sel]
    for i in np.ndindex(m.shape):
        out[i] = (np.exp(-m[i] * sig_b * omega[i]) * f_b).sum() / f_b.sum()
    return out


@dataclass
class Note1:
    """radtrax · 1, stages 1–3: the single-pass scene and its MBSP observable."""

    sigma: np.ndarray
    f11: np.ndarray
    f12: np.ndarray
    s11: float
    s12: float
    d_omega: np.ndarray
    plume: np.ndarray
    suspect: np.ndarray
    ring: np.ndarray
    tau_b: np.ndarray  # MBSP observable −ln(c R₁₂ / R₁₁), whole-scene c
    pix: float = 20.0

    def transmittance(self, m, omega, f: np.ndarray) -> np.ndarray:
        return _transmittance(self.sigma, m, omega, f)

    def score(self, omega_hat: np.ndarray) -> dict[str, float]:
        """The note's score: plume mass above the ring's median, against the truth."""
        off = np.median(omega_hat[self.ring])
        mass = float((omega_hat[self.plume] - off).sum() * self.pix**2)
        true = float(self.d_omega[self.plume].sum() * self.pix**2)
        rmse = np.sqrt(
            np.mean((omega_hat[self.plume] - off - self.d_omega[self.plume]) ** 2)
        )
        return {"ratio": mass / true, "offset": float(off), "rmse": float(rmse)}


def note1() -> Note1:
    rng = np.random.default_rng(SEED)
    f11, f12 = sentinel2_band_responses("S2A")
    sigma = methane_cross_section(rng, f11, f12, SIG12, RATIO11)
    s11, s12 = float(band_mean(sigma, f11)), float(band_mean(sigma, f12))

    n, pix = 200, 20.0
    yy, xx = np.mgrid[0:n, 0:n] * pix
    smooth = np.cumsum(np.cumsum(rng.normal(size=(n, n)), 0), 1)
    smooth = (smooth - smooth.min()) / np.ptp(smooth)
    rho11 = 0.20 + 0.20 * smooth
    veg = (np.sin(xx / 900.0) + np.cos(yy / 700.0)) > 0.9
    ratio = np.where(veg, 0.80, 0.84) * (1 + 0.01 * rng.normal(size=(n, n)))
    rho12 = rho11 * ratio
    x0, y0 = 1400.0, 2000.0
    u = (xx - x0) * np.cos(np.radians(30)) + (yy - y0) * np.sin(np.radians(30))
    v = -(xx - x0) * np.sin(np.radians(30)) + (yy - y0) * np.cos(np.radians(30))
    d_omega = 0.3 * np.exp(-0.5 * ((u / 420.0) ** 2 + (v / 140.0) ** 2))
    noise = 0.0015
    grid = np.linspace(0.0, 0.4, 401)
    t11 = _transmittance(sigma, M_AMF, OMEGA_BG + grid, f11)
    t12 = _transmittance(sigma, M_AMF, OMEGA_BG + grid, f12)
    r11 = rho11 * np.interp(d_omega, grid, t11) + noise * rng.normal(size=(n, n))
    r12 = rho12 * np.interp(d_omega, grid, t12) + noise * rng.normal(size=(n, n))
    suspect = d_omega > 0.001
    c = float((r11 * r12).sum() / (r12 * r12).sum())
    return Note1(
        sigma=sigma,
        f11=f11,
        f12=f12,
        s11=s11,
        s12=s12,
        d_omega=d_omega,
        plume=d_omega > 0.005,
        suspect=suspect,
        ring=~suspect & (np.hypot(u / 420.0, v / 140.0) < 6.0),
        tau_b=-np.log(c * r12 / r11),
    )


@dataclass
class Note2:
    """radtrax · 2, stages 2–3: the two-pass scene and its exact band tables."""

    grid: np.ndarray  # ΔΩ [kg/m²], radtrax · 2's (quadratic in index)
    tau11: np.ndarray  # τ₁₁(ΔΩ) on the grid
    tau12: np.ndarray  # τ₁₂(ΔΩ) on the grid
    a11: float
    a12: float
    plume: np.ndarray
    regions: dict[str, np.ndarray]
    obs: np.ndarray  # (2, NY, NX): B11, B12 on the plume pass
    ref: np.ndarray  # (2, NY, NX) on the reference pass
    scene: object  # (a11, r, d_omega, h2o12) → (2, NY, NX), for noise studies
    a11_map: np.ndarray
    ratio: np.ndarray


def note2() -> Note2:
    rng = np.random.default_rng(SEED)
    f11, f12 = sentinel2_band_responses("S2A")
    sigma = methane_cross_section(rng, f11, f12, SIG12, RATIO11)

    def t_band(omega, f):
        return np.atleast_1d(_transmittance(sigma, M_AMF, np.atleast_1d(omega), f))

    def alpha(f, h=1e-6):
        t = t_band([OMEGA_BG + h, OMEGA_BG - h], f)
        return float(-(np.log(t[0]) - np.log(t[1])) / (2 * h) / M_AMF)

    d_max = 0.45
    grid = (d_max / M_CH4) * np.linspace(0.0, 1.0, 181) ** 2 * M_CH4
    nb_bg = NBLookup(
        delta_column=grid / M_CH4,
        nB=np.stack(
            [t_band(OMEGA_BG + grid, f) / t_band(OMEGA_BG, f) for f in (f11, f12)]
        ),
        band_names=("B11", "B12"),
    )
    t_bg = np.array([t_band(OMEGA_BG, f11)[0], t_band(OMEGA_BG, f12)[0]])

    ny = nx = 200
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    plume = 0.3 * np.exp(-0.5 * (((xx - 70) / 25) ** 2 + ((yy - 100) / 8) ** 2))
    plume[xx < 20] *= np.exp(-(((20 - xx[xx < 20]) / 6) ** 2))
    srng = np.random.default_rng(SEED + 1)

    def texture(scale: float) -> np.ndarray:
        g = gaussian_filter(srng.normal(size=(ny, nx)), scale)
        return (g - g.mean()) / g.std()

    a11_map = 0.25 * (1 + 0.15 * texture(6.0))
    ratio = 0.80 * (1 + 0.04 * texture(3.0))
    regions = {
        "mineral": (yy >= 140) & (yy < 185) & (xx >= 10) & (xx < 90),
        "wetted": (yy >= 20) & (yy < 60) & (xx >= 130) & (xx < 180),
        "ratio_change": (yy >= 140) & (yy < 180) & (xx >= 130) & (xx < 190),
    }
    ratio = np.where(regions["mineral"], 0.85 * ratio, ratio)
    regions["plume"] = plume > 0.01
    regions["clean"] = (
        ~(gaussian_filter(regions["plume"].astype(float), 4) > 0.01)
        & ~regions["mineral"]
        & ~regions["wetted"]
        & ~regions["ratio_change"]
    )

    def scene(a11, r, d_omega, h2o12=1.0):
        clean = np.stack([a11 * t_bg[0], a11 * r * t_bg[1] * h2o12])
        return inject_plume(clean, d_omega / M_CH4, nb_bg)

    ref = scene(a11_map, ratio, 0.0 * plume)
    wet = np.where(regions["wetted"], 0.8, 1.0)
    r_p = np.where(regions["ratio_change"], 0.97 * ratio, ratio)
    obs = scene(1.05 * wet * a11_map, r_p, plume, h2o12=0.99)
    return Note2(
        grid=grid,
        tau11=-np.log(nb_bg.nB[0]),
        tau12=-np.log(nb_bg.nB[1]),
        a11=alpha(f11),
        a12=alpha(f12),
        plume=plume,
        regions=regions,
        obs=obs,
        ref=ref,
        scene=scene,
        a11_map=a11_map,
        ratio=ratio,
    )
