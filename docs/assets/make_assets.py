"""Draw radtrax's README and docs assets as light and dark SVGs.

The SVGs next to this file are generated; edit this script and re-run it::

    uv run python docs/assets/make_assets.py

It writes the icon (``icon.svg``), the logo (``logo-{light,dark}.svg``),
the forward-model figure (``forward-model-{light,dark}.svg``), the layer
stack (``architecture-{light,dark}.svg``) and the place in the stack
(``ecosystem-{light,dark}.svg``). Standard library only; the style follows gaussx's
assets (a rounded square with a gradient and a white glyph, chips in a
monospace face).
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SERIF = "'Times New Roman', Georgia, serif"

# The brand: sunlight on a slant path, amber to deep orange.
GRAD = ("#F59E0B", "#EA580C")

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "ink": "#1F2937",
        "muted": "#6B7280",
        "panel": "#F8FAFC",
        "panel_stroke": "#E5E7EB",
        "chip": "#FFFFFF",
        "chip_stroke": "#D1D5DB",
        "accent": "#EA580C",
        "accent_ink": "#C2410C",
        "edge": "#9CA3AF",
        "sky": "#EFF6FF",
        "ground": "#A8A29E",
        "faint": "#CBD5E1",
    },
    "dark": {
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "panel": "#111827",
        "panel_stroke": "#374151",
        "chip": "#1F2937",
        "chip_stroke": "#4B5563",
        "accent": "#FB923C",
        "accent_ink": "#FDBA74",
        "edge": "#6B7280",
        "sky": "#0F172A",
        "ground": "#78716C",
        "faint": "#475569",
    },
}

# Layer colours of the architecture stack, light then dark.
LAYERS = {
    "L3": ("#E11D48", "#FB7185"),
    "L2": ("#EA580C", "#FB923C"),
    "L1": ("#D97706", "#FBBF24"),
    "L0": ("#0D9488", "#2DD4BF"),
}


# ── primitives ───────────────────────────────────────────────────────────────


def svg(width: float, height: float, body: str, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" '
        f'height="{height:g}" viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-label="{escape(label)}">\n<title>{escape(label)}</title>\n'
        f"{body}\n</svg>\n"
    )


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
    family: str = SANS,
    style: str = "",
) -> str:
    st = f' font-style="{style}"' if style else ""
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="{family}" font-size="{size:g}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>'
        f"{s}</text>"
    )


def rect(x, y, w, h, *, rx=8, fill="none", stroke="none", sw=1.5, dash="", op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{op:g}"' if op != 1.0 else ""
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
        f'fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw:g}"{d}/>'
    )


def path(
    d: str,
    *,
    stroke: str,
    sw: float = 2,
    dash: str = "",
    marker: str = "",
    fill: str = "none",
    op: float = 1.0,
) -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    o = f' stroke-opacity="{op:g}"' if op != 1.0 else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
        f'stroke-linecap="round" stroke-linejoin="round"{da}{mk}{o}/>'
    )


def arrowhead(mid: str, color: str) -> str:
    return (
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{color}"/></marker>'
    )


def gradient(gid: str) -> str:
    return (
        f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{GRAD[0]}"/>'
        f'<stop offset="1" stop-color="{GRAD[1]}"/></linearGradient>'
    )


def mono_width(s: str, size: float) -> float:
    return 0.61 * size * len(s)


def chip(
    x: float,
    y: float,
    s: str,
    t: dict[str, str],
    *,
    size: float = 13,
    dashed: bool = False,
    h: float = 26,
) -> tuple[str, float]:
    w = mono_width(s, size) + 22
    body = rect(
        x,
        y,
        w,
        h,
        rx=6,
        fill=t["chip"],
        stroke=t["chip_stroke"],
        sw=1.2,
        dash="5 4" if dashed else "",
    )
    body += text(
        x + 11,
        y + h / 2 + size * 0.36,
        escape(s),
        size=size,
        fill=t["ink"],
        family=MONO,
    )
    return body, w


def spectrum_points(x0: float, x1: float, base: float, lines, n: int = 120):
    """Polyline of a continuum at ``base`` with Lorentzian dips ``(c, depth, w)``."""
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = base + sum(d * w * w / ((x - c) ** 2 + w * w) for c, d, w in lines)
        pts.append((x, y))
    return "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)


# ── the icon (128 × 128, the glyph in white on the gradient) ─────────────────


def glyph_spectrum(ox: float, oy: float) -> str:
    """An absorption spectrum over its bands, with ∂: spectroscopy you can
    differentiate (gaussx's curve-over-grid, in radtrax's terms)."""
    out = []
    heights = [30, 30, 12, 26, 8, 22, 30]  # band values: deep where the lines are
    for i, h in enumerate(heights):
        x = ox + 22 + i * 12.5
        op = "0.92" if h < 20 else "0.22"
        out.append(
            f'<rect x="{x:.1f}" y="{oy + 106 - h:.1f}" width="10" '
            f'height="{h}" rx="2" fill="#FFFFFF" fill-opacity="{op}"/>'
        )
    lines = [
        (ox + 52, 24, 2.6),
        (ox + 59, 14, 2.2),
        (ox + 78, 27, 2.8),
        (ox + 86, 16, 2.2),
        (ox + 40, 7, 2.0),
        (ox + 100, 8, 2.0),
    ]
    d = spectrum_points(ox + 14, ox + 114, oy + 44, lines)
    out.append(path(d, stroke="#FFFFFF", sw=4))
    out.append(
        f'<circle cx="{ox + 104}" cy="{oy + 22}" r="11" fill="none" '
        f'stroke="#FFFFFF" stroke-width="2.4"/>'
    )
    out.append(
        text(
            ox + 104,
            oy + 28.5,
            "∂",
            size=18,
            fill="#FFFFFF",
            weight=700,
            anchor="middle",
            family=SERIF,
            style="italic",
        )
    )
    return "".join(out)


def tile(ox: float, oy: float, gid: str) -> str:
    return rect(ox + 4, oy + 4, 120, 120, rx=26, fill=f"url(#{gid})") + glyph_spectrum(
        ox, oy
    )


def icon() -> str:
    return svg(
        128,
        128,
        f"<defs>{gradient('rt-icon')}</defs>" + tile(0, 0, "rt-icon"),
        "radtrax",
    )


def logo(theme: str) -> str:
    t = THEMES[theme]
    gid = f"rt-{theme}"
    body = f"<defs>{gradient(gid)}</defs>" + tile(6, 6, gid)
    body += (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{t["ink"]}">radtra'
        f'<tspan fill="{t["accent"]}">x</tspan></text>'
    )
    return svg(520, 140, body, "radtrax")


# ── the forward-model figure ─────────────────────────────────────────────────


def forward_model(theme: str) -> str:
    t = THEMES[theme]
    W, H = 1200, 470
    gid = f"fm-{theme}"
    b = [
        f"<defs>{gradient(gid)}{arrowhead('fa', t['edge'])}"
        f"{arrowhead('fb', t['accent'])}</defs>"
    ]

    # The scene: sun, L layers with a plume in the lowest, surface, sensor.
    sx, sy, sw_, sh = 20, 20, 360, 430
    b.append(rect(sx, sy, sw_, sh, rx=18, fill=t["sky"], stroke=t["panel_stroke"]))
    b.append(text(sx + 20, sy + 32, "the scene", size=13, fill=t["muted"], weight=600))
    b.append(
        text(
            sx + 20,
            sy + 52,
            "x: L layers of p, T and VMR",
            size=12,
            fill=t["muted"],
            family=MONO,
        )
    )
    top, n, dz = 150, 6, 32
    for i in range(n):
        y = top + i * dz
        b.append(
            rect(
                sx + 16,
                y,
                sw_ - 32,
                dz - 4,
                rx=5,
                fill=t["accent"],
                op=0.05 + 0.035 * i,
            )
        )
    ground = top + n * dz + 4
    b.append(
        path(f"M{sx + 16} {ground} L{sx + sw_ - 16} {ground}", stroke=t["ground"], sw=4)
    )
    for k in range(14):
        x = sx + 26 + k * 23
        b.append(
            path(
                f"M{x} {ground + 4} L{x - 10} {ground + 16}", stroke=t["ground"], sw=1.5
            )
        )
    for cx, cy, r in ((170, -14, 15), (190, -22, 17), (212, -16, 14), (230, -10, 10)):
        b.append(
            f'<circle cx="{sx + cx}" cy="{ground + cy}" r="{r}" '
            f'fill="{t["muted"]}" fill-opacity="0.45"/>'
        )
    b.append(text(sx + 250, ground - 30, "ΔX CH₄", size=12, fill=t["ink"], weight=600))
    b.append(
        text(
            sx + 20,
            ground + 34,
            "surface R · Lambertian, RossLi",
            size=12,
            fill=t["muted"],
            family=MONO,
        )
    )
    # Sun and sensor.
    sun = (sx + 58, sy + 104)
    b.append(f'<circle cx="{sun[0]}" cy="{sun[1]}" r="16" fill="{GRAD[0]}"/>')
    for k in range(8):
        a = k * math.pi / 4
        b.append(
            path(
                f"M{sun[0] + 22 * math.cos(a):.1f} {sun[1] + 22 * math.sin(a):.1f} "
                f"L{sun[0] + 29 * math.cos(a):.1f} {sun[1] + 29 * math.sin(a):.1f}",
                stroke=GRAD[0],
                sw=2.5,
            )
        )
    sat = (sx + 292, sy + 70)
    b.append(rect(sat[0] - 11, sat[1] - 9, 22, 18, rx=3, fill=t["ink"]))
    b.append(rect(sat[0] - 37, sat[1] - 5, 22, 10, rx=2, fill=t["muted"]))
    b.append(rect(sat[0] + 15, sat[1] - 5, 22, 10, rx=2, fill=t["muted"]))
    hit = (sx + 196, ground - 2)
    b.append(
        path(
            f"M{sun[0] + 18} {sun[1] + 16} L{hit[0]} {hit[1]}",
            stroke=t["accent"],
            sw=3,
            marker="fb",
        )
    )
    b.append(
        path(
            f"M{hit[0]} {hit[1]} L{sat[0] - 4} {sat[1] + 14}",
            stroke=t["accent"],
            sw=3,
            marker="fb",
        )
    )
    b.append(
        text(
            sx + 96,
            sy + 196,
            "μ₀",
            size=16,
            fill=t["accent_ink"],
            weight=700,
            style="italic",
            family=SERIF,
        )
    )
    b.append(
        text(
            sx + 272,
            sy + 196,
            "μ",
            size=16,
            fill=t["accent_ink"],
            weight=700,
            style="italic",
            family=SERIF,
        )
    )
    b.append(
        text(
            sx + 20,
            sy + sh - 14,
            "geometry g · ViewingGeometry",
            size=12,
            fill=t["muted"],
            family=MONO,
        )
    )

    # The stages.
    b.append(
        text(
            800,
            52,
            "y = I( S( O(x), R, g ) )",
            size=24,
            fill=t["ink"],
            weight=700,
            anchor="middle",
            family=SERIF,
            style="italic",
        )
    )
    cards = [
        ("state", "x", "AtmosphericState", "(L,) per gas"),
        ("opacity", "O", "TableOpacity", "τ  (L, G)"),
        ("solver", "S", "BeerLambert", "radiance  (G,)"),
        ("instrument", "I", "BandInstrument", "y  (B,)"),
    ]
    cx0, cw, cg, cy, ch = 420, 172, 22, 84, 196
    for i, (role, sym, name, shape) in enumerate(cards):
        x = cx0 + i * (cw + cg)
        b.append(
            rect(
                x,
                cy,
                cw,
                ch,
                rx=14,
                fill=t["panel"],
                stroke=t["accent"] if i in (1, 2) else t["panel_stroke"],
                sw=1.6,
            )
        )
        b.append(
            text(
                x + 16, cy + 26, role.upper(), size=11, fill=t["accent_ink"], weight=700
            )
        )
        b.append(
            text(
                x + cw - 16,
                cy + 28,
                sym,
                size=20,
                fill=t["ink"],
                weight=700,
                anchor="end",
                family=SERIF,
                style="italic",
            )
        )
        b.append(
            text(x + 16, cy + 50, name, size=13, fill=t["ink"], weight=600, family=MONO)
        )
        b.append(
            text(x + 16, cy + ch - 16, shape, size=12, fill=t["muted"], family=MONO)
        )
        px, py, pw, ph = x + 16, cy + 66, cw - 32, 88  # mini plot box
        if i == 0:  # a profile: bars shrinking with height
            for k in range(6):
                w = pw * (0.95 - 0.13 * k)
                b.append(
                    rect(
                        px,
                        py + ph - (k + 1) * 14,
                        w,
                        10,
                        rx=2,
                        fill=t["accent"],
                        op=0.25 + 0.1 * (5 - k),
                    )
                )
        elif i == 1:  # τ (L, G): rows of lines, stronger low down
            for k in range(6):
                yy = py + 4 + k * 14
                b.append(rect(px, yy, pw, 10, rx=2, fill=t["chip_stroke"], op=0.35))
                for c in (0.22, 0.27, 0.48, 0.55, 0.78):
                    b.append(
                        rect(
                            px + c * pw,
                            yy,
                            3 + k * 0.6,
                            10,
                            rx=1,
                            fill=t["accent"],
                            op=0.2 + 0.13 * k,
                        )
                    )
        elif i == 2:  # radiance with absorption dips
            lines = [
                (px + c * pw, d, 2.4)
                for c, d in ((0.23, 48), (0.28, 30), (0.49, 56), (0.56, 34), (0.79, 40))
            ]
            b.append(
                path(
                    spectrum_points(px, px + pw, py + 12, lines, 200),
                    stroke=t["ink"],
                    sw=1.6,
                )
            )
        else:  # band values
            vals = [0.92, 0.6, 0.85, 0.42, 0.78, 0.88]
            bw = pw / len(vals)
            for k, v in enumerate(vals):
                b.append(
                    rect(
                        px + k * bw + 2,
                        py + ph - v * ph,
                        bw - 4,
                        v * ph,
                        rx=2,
                        fill=t["accent"],
                        op=0.75,
                    )
                )
        if i < 3:
            ax = x + cw + 2
            b.append(
                path(
                    f"M{ax} {cy + ch / 2} L{ax + cg - 4} {cy + ch / 2}",
                    stroke=t["edge"],
                    sw=2,
                    marker="fa",
                )
            )
    # R and g enter the solver.
    sxm = cx0 + 2 * (cw + cg) + cw / 2
    b.append(
        text(
            sxm,
            cy - 10,
            "+ surface R, geometry g",
            size=12,
            fill=t["muted"],
            anchor="middle",
        )
    )

    # Autodiff back along the chain.
    yb = cy + ch + 34
    x_end = cx0 + 3 * (cw + cg) + cw / 2
    b.append(
        path(
            f"M{x_end} {cy + ch + 6} L{x_end} {yb} L{cx0 + cw / 2} {yb} "
            f"L{cx0 + cw / 2} {cy + ch + 10}",
            stroke=t["accent"],
            sw=2.4,
            dash="7 5",
            marker="fb",
        )
    )
    b.append(
        rect(
            cx0 + 2 * (cw + cg) - 150,
            yb - 15,
            300,
            30,
            rx=15,
            fill=t["panel"],
            stroke=t["accent"],
            sw=1.4,
        )
    )
    b.append(
        text(
            cx0 + 2 * (cw + cg),
            yb + 5,
            "K = ∂y/∂x  ·  jax.jacrev / jacfwd",
            size=13,
            fill=t["accent_ink"],
            weight=600,
            anchor="middle",
            family=MONO,
        )
    )

    # What uses K.
    ry = yb + 44
    b.append(
        text(
            cx0,
            ry + 4,
            "retrievals on the same model",
            size=13,
            fill=t["muted"],
            weight=600,
        )
    )
    x = cx0
    for s, dashed in (
        ("unit_absorption_target", False),
        ("matched filter", False),
        ("BandTable", False),
        ("3D-Var", False),
        ("optimal estimation · vardax", True),
    ):
        c, w = chip(x, ry + 16, s, t, size=12, dashed=dashed)
        if x + w > W - 20:
            break
        b.append(c)
        x += w + 10
    return svg(
        W,
        H,
        "".join(b),
        "The radtrax forward model: sunlight crosses L atmospheric layers, "
        "reflects off the surface and reaches the sensor; in code, state x → "
        "opacity O → solver S → instrument I gives band values y, and autodiff "
        "gives the Jacobian K that every retrieval uses",
    )


# ── the layer stack ──────────────────────────────────────────────────────────


def architecture(theme: str) -> str:
    t = THEMES[theme]
    k = 0 if theme == "light" else 1
    W = 960
    rows = [
        (
            "L3",
            "Retrievals",
            "matched filter · multi-pass · band tables · 3D-Var",
            [
                ("unit_absorption_target", False),
                ("apply_image", False),
                ("mbmp_signal", False),
                ("BandTable", False),
                ("run_gauss_newton", False),
            ],
        ),
        (
            "L2",
            "Forward model",
            "y = I(S(O(x), R, g)), one pixel; vmap for a swath",
            [
                ("ForwardModel", False),
                ("ForwardModel.radiance", False),
                ("ForwardModel.optics", False),
            ],
        ),
        (
            "L1",
            "Components",
            "one protocol per seam, swappable",
            [
                ("TableOpacity", False),
                ("BeerLambert", False),
                ("RossLi", False),
                ("PolynomialAlbedo", False),
                ("GaussianInstrument", False),
                ("SNRNoise", False),
            ],
        ),
        (
            "L0",
            "State and tables",
            "the atmosphere, its geometry, offline spectroscopy",
            [
                ("AtmosphericState", False),
                ("afgl_us_standard", False),
                ("ViewingGeometry", False),
                ("load_opacity", False),
                ("SpectralResponseFunction", False),
            ],
        ),
    ]
    main_w, x0, y, gap = 660, 20, 20, 14
    b = [f"<defs>{arrowhead('aa', t['edge'])}</defs>"]
    for lvl, title, sub, chips in rows:
        col = LAYERS[lvl][k]
        # lay the chips out first, wrapping at the layer's right edge
        placed, cx, cy = [], x0 + 70, 54
        for s, dashed in chips:
            w = mono_width(s, 12) + 22
            if cx + w > x0 + main_w - 16:
                cx, cy = x0 + 70, cy + 34
            placed.append((cx, cy, s, dashed))
            cx += w + 8
        rh = cy + 26 + 18
        b.append(rect(x0, y, main_w, rh, rx=14, fill=col, op=0.06, stroke=col, sw=1.6))
        b.append(rect(x0 + 16, y + 14, 40, 22, rx=11, fill=col))
        b.append(
            text(
                x0 + 36,
                y + 30,
                lvl,
                size=12,
                fill="#FFFFFF",
                weight=700,
                anchor="middle",
            )
        )
        b.append(text(x0 + 70, y + 32, title, size=18, fill=col, weight=700))
        b.append(
            text(
                x0 + main_w - 16,
                y + 32,
                escape(sub),
                size=12.5,
                fill=t["muted"],
                anchor="end",
            )
        )
        for cx, cy, s, dashed in placed:
            b.append(chip(cx, y + cy, s, t, size=12, dashed=dashed)[0])
        y += rh + gap
    b.append(rect(x0, y, main_w, 44, rx=12, fill=t["panel"], stroke=t["panel_stroke"]))
    b.append(
        text(
            x0 + main_w / 2,
            y + 28,
            'built on <tspan font-weight="700" fill="' + t["ink"] + '">JAX · Equinox · '
            "lineax · optimistix · gaussx · geonnax</tspan>",
            size=14,
            fill=t["muted"],
            anchor="middle",
        )
    )
    bottom = y + 44

    # Side panels: the extras.
    def side(yy: float, title: str, sub: str, items: list[str]) -> float:
        sx, sw_ = x0 + main_w + 40, W - (x0 + main_w + 40) - 20
        h = 64 + 34 * len(items)
        b.append(
            rect(
                sx,
                yy,
                sw_,
                h,
                rx=14,
                fill=t["panel"],
                stroke=t["edge"],
                sw=1.4,
                dash="6 5",
            )
        )
        b.append(
            text(sx + 16, yy + 28, escape(title), size=16, fill=t["ink"], weight=700)
        )
        b.append(text(sx + 16, yy + 48, escape(sub), size=12, fill=t["muted"]))
        for j, s in enumerate(items):
            c, _ = chip(sx + 16, yy + 60 + 34 * j, s, t, size=12)
            b.append(c)
        return h

    h1 = side(
        20,
        "radtrax[geo]",
        "pipekit operators on GeoTensors",
        ["MatchedFilter", "ColumnEnhancement", "MBMP · SBMP · MBSP", "sentinel2_srf"],
    )
    b.append(
        path(
            f"M{x0 + main_w + 38} {20 + h1 / 2} L{x0 + main_w + 6} {20 + h1 / 2}",
            stroke=t["edge"],
            sw=2,
            marker="aa",
        )
    )
    y2 = 20 + h1 + 24
    h2 = side(
        y2, "radtrax[hitran]", "build the tables offline", ["build_table", "load_table"]
    )
    b.append(
        path(
            f"M{x0 + main_w + 38} {y2 + h2 / 2} L{x0 + main_w + 6} {y2 + h2 / 2}",
            stroke=t["edge"],
            sw=2,
            marker="aa",
        )
    )
    H = max(bottom, y2 + h2) + 20
    return svg(
        W,
        H,
        "".join(b),
        "The radtrax layers: retrievals, the forward model, its components and "
        "the state and tables, built on JAX, Equinox, lineax, optimistix, gaussx "
        "and geonnax, with the geo and hitran extras beside them",
    )


# ── the place in the stack ───────────────────────────────────────────────────


def ecosystem(theme: str) -> str:
    t = THEMES[theme]
    W, H = 1100, 620
    gid = f"eco-{theme}"
    b = [
        f"<defs>{gradient(gid)}{arrowhead('ea', t['edge'])}"
        f"{arrowhead('eb', t['accent'])}</defs>"
    ]

    def band(
        y: float, h: float, name: str, sub: str, x: float = 20, w: float = 760
    ) -> None:
        b.append(rect(x, y, w, h, rx=18, fill=t["panel"], stroke=t["panel_stroke"]))
        b.append(text(x + 22, y + 32, name, size=17, fill=t["ink"], weight=700))
        b.append(text(x + w - 22, y + 32, sub, size=13, fill=t["muted"], anchor="end"))

    def node(
        x: float, y: float, w: float, name: str, sub: str, *, state: str = "dep"
    ) -> tuple[float, float, float, float]:
        stroke = {
            "dep": t["accent"],
            "opt": t["accent"],
            "off": t["faint"],
            "up": t["edge"],
        }[state]
        dash = "6 5" if state == "opt" else ("2 4" if state == "off" else "")
        b.append(
            rect(x, y, w, 64, rx=12, fill=t["chip"], stroke=stroke, sw=1.8, dash=dash)
        )
        ink = t["muted"] if state == "off" else t["ink"]
        b.append(
            text(
                x + w / 2,
                y + 28,
                name,
                size=16,
                fill=ink,
                weight=700,
                anchor="middle",
                family=MONO,
            )
        )
        b.append(
            text(x + w / 2, y + 49, sub, size=12.5, fill=t["muted"], anchor="middle")
        )
        return x, y, w, 64

    band(20, 150, "GeoModels", "physical models; never import each other")
    band(200, 130, "GeoML", "learning and assimilation")
    band(352, 170, "Numerics", "solvers and operators")
    band(20, 502, "GeoStack", "readers and pipelines", x=810, w=270)

    # radtrax, with its icon.
    rx, ry, rw = 330, 72, 300
    b.append(rect(rx, ry, rw, 84, rx=16, fill=t["chip"], stroke=t["accent"], sw=2.6))
    b.append(
        f'<g transform="translate({rx + 12} {ry + 10}) scale(0.5)">'
        + tile(0, 0, gid)
        + "</g>"
    )
    b.append(
        f'<text x="{rx + 90}" y="{ry + 42}" font-family="{SANS}" font-size="26" '
        f'font-weight="700" fill="{t["ink"]}">radtra<tspan fill="{t["accent"]}">x'
        f"</tspan></text>"
    )
    b.append(
        text(
            rx + 90,
            ry + 66,
            "differentiable radiative transfer",
            size=13,
            fill=t["muted"],
        )
    )
    node(60, ry + 10, 180, "plumax", "plume physics", state="up")
    b.append(
        path(
            f"M{240} {ry + 42} L{rx - 6} {ry + 42}", stroke=t["edge"], sw=2, marker="ea"
        )
    )
    b.append(text(285, ry + 34, "imports", size=12, fill=t["muted"], anchor="middle"))

    vx = node(400, 248, 160, "vardax", "3D/4D-Var · OE", state="opt")
    gx_ = node(160, 248, 160, "geonnax", "Legendre basis")
    nodes = {}
    for i, (name, sub, st) in enumerate(
        (
            ("gaussx", "structured solves", "dep"),
            ("lineax", "linear solves", "dep"),
            ("optimistix", "Gauss–Newton", "dep"),
            ("spectraldiffx", "spectral", "off"),
            ("finitevolX", "finite volume", "off"),
        )
    ):
        nodes[name] = node(40 + i * 148, 436, 136, name, sub, state=st)

    def down(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        accent: bool = True,
        dash: str = "",
        ymid: float | None = None,
    ) -> None:
        ym = (y1 + y2) / 2 if ymid is None else ymid
        b.append(
            path(
                f"M{x1} {y1} L{x1} {ym} L{x2} {ym} L{x2} {y2 - 6}",
                stroke=t["accent"] if accent else t["edge"],
                sw=2,
                dash=dash,
                marker="eb" if accent else "ea",
            )
        )

    rb = ry + 84
    down(rx + 210, rb, vx[0] + vx[2] / 2, 248, dash="6 5", ymid=rb + 22)
    b.append(
        text(
            rx + 216, rb + 40, "[var] · v1", size=12, fill=t["accent_ink"], family=MONO
        )
    )
    down(rx + 10, rb, gx_[0] + gx_[2] / 2, 248, ymid=rb + 22)
    # radtrax → gaussx, lineax, optimistix: one trunk down between the GeoML nodes.
    trunk, bus = rx + 45, 412
    b.append(path(f"M{trunk} {rb} L{trunk} {bus}", stroke=t["accent"], sw=2))
    for name in ("gaussx", "lineax", "optimistix"):
        nx, ny, nw, _ = nodes[name]
        b.append(
            path(
                f"M{trunk} {bus} L{nx + nw / 2} {bus} L{nx + nw / 2} {ny - 6}",
                stroke=t["accent"],
                sw=2,
                marker="eb",
            )
        )

    # GeoStack, through [geo].
    gs = [
        ("georeader", "GeoTensor · SRFs"),
        ("geotoolz", "carrier helpers"),
        ("pipekit", "Operator"),
    ]
    for i, (name, sub) in enumerate(gs):
        nx, ny, nw, _ = node(840, 80 + i * 96, 210, name, sub, state="opt")
        b.append(
            path(
                f"M{rx + rw} {ry + 42} L{790} {ry + 42} L{790} {ny + 32} "
                f"L{nx - 6} {ny + 32}",
                stroke=t["accent"],
                sw=2,
                dash="6 5",
                marker="eb",
            )
        )
    b.append(text(712, ry + 34, "[geo]", size=12, fill=t["accent_ink"], family=MONO))

    # Legend.
    ly = 572
    items = [
        ("dep", "radtrax depends on it"),
        ("opt", "an optional extra"),
        ("off", "in the layer, not a dependency"),
        ("up", "builds on radtrax"),
    ]
    lx = 40
    for st, label in items:
        stroke = {
            "dep": t["accent"],
            "opt": t["accent"],
            "off": t["faint"],
            "up": t["edge"],
        }[st]
        dash = "6 5" if st == "opt" else ("2 4" if st == "off" else "")
        b.append(
            rect(lx, ly, 40, 22, rx=6, fill=t["chip"], stroke=stroke, sw=1.8, dash=dash)
        )
        b.append(text(lx + 52, ly + 16, label, size=13, fill=t["muted"]))
        lx += 52 + 0.58 * 13 * len(label) + 40
    return svg(
        W,
        H,
        "".join(b),
        "Where radtrax sits: a GeoModels package that plumax builds on; it "
        "depends on gaussx, lineax and optimistix in Numerics and geonnax in "
        "GeoML, takes vardax through the var extra and georeader, geotoolz and "
        "pipekit through the geo extra",
    )


def main() -> None:
    out: dict[str, str] = {"icon.svg": icon()}
    for th in THEMES:
        out[f"logo-{th}.svg"] = logo(th)
        out[f"forward-model-{th}.svg"] = forward_model(th)
        out[f"architecture-{th}.svg"] = architecture(th)
        out[f"ecosystem-{th}.svg"] = ecosystem(th)
    for name, s in out.items():
        (HERE / name).write_text(s, encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
