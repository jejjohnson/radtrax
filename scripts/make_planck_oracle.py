# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "scipy>=1.13", "loguru>=0.7", "cyclopts>=3"]
# ///
"""Build (or check) the pyradi Planck oracle that gates radtrax's Planck (TH12).

pyradi's ``ryplanck`` is a careful NumPy reference for the Planck function
(radtrax#247, absorbing #250), but it cannot be a dependency, not even a test
one: no PyPI release since 2017, and no install metadata after commit
``c7cfe9d``. This script fetches that one module at the pinned commit into a
new empty directory, evaluates it in an isolated interpreter
(``python -I``, the directory put on ``sys.path`` explicitly, a stub for the
``pyradi.ryutils`` it imports but the evaluated functions never call), and
stores the values in ``tests/data/planck_pyradi_c7cfe9d.npz``, which
``tests/radiometry/test_planck_oracles.py`` reads. radtrax never imports
pyradi or this script.

pyradi returns **exitance** M = πL; the tests divide by π. The stored file
records the commit, the NumPy and SciPy versions, SciPy's (h, c, k) and the
licence (pyradi: MIT at the root, BSD-3 per file).

Usage::

    uv run --script scripts/make_planck_oracle.py build           # rewrite the file
    uv run --script scripts/make_planck_oracle.py build --check   # compare with it
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cyclopts  # ty: ignore[unresolved-import]  (inline script deps, PEP 723)
import numpy as np
from loguru import logger  # ty: ignore[unresolved-import]


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c7cfe9df96fdf0e6c258ee6fd624f62f19ebab06"
URL = f"https://raw.githubusercontent.com/NelisW/pyradi/{COMMIT}/pyradi/ryplanck.py"
OUT = ROOT / "tests" / "data" / "planck_pyradi_c7cfe9d.npz"
ATTRIBUTION = (
    "Values computed with pyradi ryplanck at commit "
    f"{COMMIT} (https://github.com/NelisW/pyradi; MIT licence at the root, "
    "BSD-3 per file; (c) the pyradi team). Exitance M = pi L, not radiance."
)

#: The design range: 0.4 to 15 µm and 150 to 2500 K.
WAVELENGTH_UM = np.geomspace(0.4, 15.0, 32)
TEMPERATURE_K = np.array([150.0, 200.0, 300.0, 500.0, 1000.0, 1800.0, 2500.0])
#: Bands [cm⁻¹] for pyradi's Widger–Woodall integral (planckInt, type "en").
BANDS_CM = np.array([[700.0, 1400.0], [800.0, 1250.0], [2000.0, 2500.0]])

#: pyradi imports ``pyradi.ryutils`` at module level; the functions evaluated
#: here never call it, so a stub that fails loudly keeps its dependencies out.
RYUTILS_STUB = '''"""Stub: radtrax's oracle never calls ryutils."""


def __getattr__(name):
    raise NotImplementedError(f"ryutils.{name} is stubbed out")
'''

app = cyclopts.App(help=__doc__)


@app.command
def build(*, out: Path = OUT, check: bool = False, rtol: float = 1e-12) -> None:
    """Fetch pyradi's ryplanck, evaluate it in isolation, write or check the oracle.

    Args:
        out: The ``.npz`` file to write, or to compare with when ``check``.
        check: Compare a fresh evaluation with ``out`` instead of writing it.
        rtol: Relative tolerance of the comparison. 1e-12, not 1e-14: numpy's
            ``exp`` differs between builds and CPUs by a few ulp (1.35e-14 on a
            CI runner), and the oracle tests compare at 1e-12 anyway.
    """
    with tempfile.TemporaryDirectory(prefix="pyradi-") as tmp:
        src = Path(tmp) / "src"
        pkg = src / "pyradi"
        pkg.mkdir(parents=True)
        logger.info("fetching {}", URL)
        with urllib.request.urlopen(URL, timeout=60) as response:
            (pkg / "ryplanck.py").write_bytes(response.read())
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "ryutils.py").write_text(RYUTILS_STUB, encoding="utf-8")
        fresh = Path(tmp) / "fresh.npz"
        cmd = [
            sys.executable,
            "-I",
            str(Path(__file__)),
            "evaluate",
            str(src),
            str(fresh),
        ]
        subprocess.run(cmd, check=True, cwd=tmp)
        if not check:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(fresh.read_bytes())
            logger.info("wrote {} ({} bytes)", out, out.stat().st_size)
            return
        new, old = np.load(fresh), np.load(out)
        for key in old.files:
            if old[key].dtype.kind in "fc":
                np.testing.assert_allclose(new[key], old[key], rtol=rtol, atol=0.0)
        logger.info("{} reproduces to rtol {}", out, rtol)


@app.command
def evaluate(src: Path, out: Path) -> None:
    """Evaluate pyradi on the fixed grid (run by ``build`` under ``python -I``).

    Args:
        src: The directory holding the fetched ``pyradi`` package.
        out: Where to write the ``.npz``.
    """
    import importlib

    import scipy
    import scipy.constants as const

    sys.path.insert(0, str(src))
    rp = importlib.import_module("pyradi.ryplanck")

    nu = 1e4 / WAVELENGTH_UM
    freq_hz = const.c * 1e2 * nu
    integrals = np.array(
        [
            [rp.planckInt(lo, hi, t, type="en") for t in TEMPERATURE_K]
            for lo, hi in BANDS_CM
        ]
    )
    np.savez(
        out,
        wavelength_um=WAVELENGTH_UM,
        wavenumber_cm=nu,
        frequency_hz=freq_hz,
        temperature_k=TEMPERATURE_K,
        plancken=rp.plancken(nu, TEMPERATURE_K),
        planckel=rp.planckel(WAVELENGTH_UM, TEMPERATURE_K),
        planckef=rp.planckef(freq_hz, TEMPERATURE_K),
        dplncken=rp.dplncken(nu, TEMPERATURE_K),
        dplnckel=rp.dplnckel(WAVELENGTH_UM, TEMPERATURE_K),
        dplnckef=rp.dplnckef(freq_hz, TEMPERATURE_K),
        bands_cm=BANDS_CM,
        planck_int_en=integrals,
        wien_en_cm_per_k=np.float64(rp.pconst.wen),
        wien_el_um_k=np.float64(rp.pconst.wel),
        scipy_h_k_c=np.array([const.h, const.k, const.c]),
        commit=np.str_(COMMIT),
        numpy_version=np.str_(np.__version__),
        scipy_version=np.str_(scipy.__version__),
        attribution=np.str_(ATTRIBUTION),
    )


if __name__ == "__main__":
    app()
