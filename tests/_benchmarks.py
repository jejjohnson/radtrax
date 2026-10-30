"""Korkin et al. (2025)'s gas-cell benchmarks, downloaded once and cached (radtrax#171).

Korkin, Sayer, Ibrahim & Lyapustin (2025), *A practical guide to coding
line-by-line trace gas absorption in Earth's atmosphere*, JQSRT 337, 109345
(CC BY 4.0), publish their ``gcell`` code's optical thickness of pure-gas
cells, and for the CH₄ cell a benchmark from GATS Inc.'s SpectralCalc:

.. math::
    τ(ν) = n\\, k(ν; T, p), \\qquad n = p\\, l / (k_B T)
    \\qquad \\text{(their eqs. 19–21)} .

Their repository (``korkins/aspect_gcell``) has no licence file, so nothing
is vendored: :func:`fetch` downloads each file at the pinned commit
:data:`KORKIN_COMMIT`, checks its SHA-256 and caches it under
``$RADTRAX_BENCHMARK_DIR`` (default ``$XDG_CACHE_HOME/radtrax/benchmarks``,
else ``~/.cache/radtrax/benchmarks``).

Each case also names the HITRAN line list ``gcell`` read. It is HITRAN2020,
and HITRANonline (what HAPI's ``fetch`` serves) now holds a later edition
whose CH₄ lines differ by several per cent, so a benchmark computed from
fetched lines measures the edition, not the code: :func:`write_lines` copies
the case's own lines, over the test window plus the builder's line margin,
into a HAPI table instead.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


#: ``korkins/aspect_gcell`` at the commit the hashes below were taken from.
KORKIN_REPO = "korkins/aspect_gcell"
KORKIN_COMMIT = "51c9433d39936d9fed1162eeac7e13698facfbdc"
#: Environment variable that overrides the benchmark cache directory.
BENCHMARK_DIR_ENV = "RADTRAX_BENCHMARK_DIR"
#: ``hitran/07_hitdb_one_line.par`` of korkins/aspect_gcell (HITRAN2020): one
#: ¹⁶O₂ line of the A band, ν = 13000.816219 cm⁻¹, in HITRAN's 160-character
#: format. From Korkin et al. (2025), §4.6.
ONE_LINE_PAR = (
    " 7113000.816219 2.708E-27 1.740E-02.04580.047 1814.01040.67-.007400"
    "       b      1       X      1                R 13Q 14     d54644449"
    " 5 5 3 1 1    29.0   29.0"
)
#: HITRAN's one-character local isotopologue code, indexed by ``I - 1``.
_ISO_CHARS = "1234567890AB"


@dataclass(frozen=True)
class RemoteFile:
    """A file of :data:`KORKIN_REPO` at :data:`KORKIN_COMMIT`.

    Attributes:
        path: Path in the repository.
        sha256: SHA-256 of its bytes.
        member: For a zip archive, the member to read.
    """

    path: str
    sha256: str
    member: str | None = None

    @property
    def url(self) -> str:
        host = "https://raw.githubusercontent.com"
        return f"{host}/{KORKIN_REPO}/{KORKIN_COMMIT}/{self.path}"


@dataclass(frozen=True)
class GasCellCase:
    """One of Korkin et al.'s pure-gas cells.

    Attributes:
        name: Test id.
        molecule: HITRAN molecule name (a key of ``ATMOSPHERIC_GASES``).
        molecule_id: HITRAN molecule number.
        nu_min_cm: First wavenumber of the benchmark [cm⁻¹].
        nu_max_cm: Last wavenumber of the benchmark [cm⁻¹].
        dnu_cm: Step of the benchmark file [cm⁻¹].
        length_cm: Cell length [cm].
        t_k: Temperature [K].
        p_atm: Pressure [atm]; pure gas, so self-broadening only.
        n_column_cm2: The cell's column [molecules/cm²], as published.
        window_cm: The part of the band the tests evaluate [cm⁻¹], to keep
            HAPI's line-by-line sum to seconds.
        benchmark: The benchmark file.
        lines: The HITRAN2020 line list ``gcell`` read.
    """

    name: str
    molecule: str
    molecule_id: int
    nu_min_cm: float
    nu_max_cm: float
    dnu_cm: float
    length_cm: float
    t_k: float
    p_atm: float
    n_column_cm2: float
    window_cm: tuple[float, float]
    benchmark: RemoteFile
    lines: RemoteFile


#: §5.4.2: 8 cm of pure CH₄ at 296 K and 1000 mbar. Columns of the file: ν,
#: gcell's τ, exp(−τ), the filter f, (T f) from gcell, (T f) from SpectralCalc
#: (HITRAN2016), their difference.
KORKIN2025_CH4_CELL = GasCellCase(
    name="korkin2025_ch4_cell",
    molecule="CH4",
    molecule_id=6,
    nu_min_cm=4081.901,
    nu_max_cm=4505.699,
    dnu_cm=0.002,
    length_cm=8.0,
    t_k=296.0,
    p_atm=1000.0e2 / 101325.0,
    n_column_cm2=1.9575594e20,
    window_cm=(4300.0, 4305.0),
    benchmark=RemoteFile(
        "benchmarks/Section5p4p2___test_gcell_ch4.txt",
        "9c64dc95d6f2b6858c1f102a5af59305110faf8c0b58ae41a34cb459b3984dd9",
    ),
    lines=RemoteFile(
        "hitran/06_hitdb.zip",
        "7b6ad1fdfbb48b76df55a7b66148cd81d83d8192e3c222f1ad3108e0a4f56526",
        member="06_hitdb.par",
    ),
)

#: §5.4.1: 1633.6 cm of pure O₂ at 296 K and 0.7145 atm, the A band. The
#: file holds ν and gcell's τ on a 0.02 cm⁻¹ grid.
KORKIN2025_O2A_CELL = GasCellCase(
    name="korkin2025_o2a_cell",
    molecule="O2",
    molecule_id=7,
    nu_min_cm=13006.0,
    nu_max_cm=13166.0,
    dnu_cm=0.02,
    length_cm=1633.6,
    t_k=296.0,
    p_atm=0.7145,
    n_column_cm2=2.8921135e22,
    window_cm=(13006.0, 13165.98),
    benchmark=RemoteFile(
        "benchmarks/Section5p4p1___test_gcell_o2a.txt",
        "c3c646ea753a320182e7b56324f10f8f40bd679d8fc04fedfc566e3a5f3716b4",
    ),
    lines=RemoteFile(
        "hitran/07_hitdb.par",
        "0ad0aaa9e6b6246f6c23fc27396077867f6c0980f4b03db1d000c1ff08f53d72",
    ),
)

CASES = (KORKIN2025_CH4_CELL, KORKIN2025_O2A_CELL)


def benchmark_cache_dir() -> Path:
    """``$RADTRAX_BENCHMARK_DIR``, else ``<user cache>/radtrax/benchmarks``."""
    env = os.environ.get(BENCHMARK_DIR_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "radtrax" / "benchmarks"


def fetch(remote: RemoteFile, cache_dir: Path | None = None) -> Path:
    """``remote``'s local copy, downloaded once and checked against its hash.

    Raises:
        OSError: If the download fails (``urllib.error.URLError``).
        ValueError: If the bytes do not match ``remote.sha256``.
    """
    root = (cache_dir or benchmark_cache_dir()) / "korkin2025" / KORKIN_COMMIT[:10]
    path = root / Path(remote.path).name
    if not path.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(remote.url, timeout=120) as response:
            payload = response.read()
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(payload)
        partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != remote.sha256:
        raise ValueError(f"{path}: SHA-256 {digest} is not {remote.sha256}.")
    return path


def load_case(case: GasCellCase, cache_dir: Path | None = None) -> xr.Dataset:
    """The benchmark of ``case`` on its wavenumbers.

    Returns:
        ``tau`` (gcell's optical thickness) on ``wavenumber`` [cm⁻¹]; for the
        CH₄ cell also ``filter``, ``tf_gcell`` and ``tf_benchmark``. The
        attribute ``n_column_cm2`` is the case's column.
    """
    path = fetch(case.benchmark, cache_dir)
    rows = np.loadtxt(path, comments="#")
    if rows[0, 0] == -999.0:  # gcell's header row: fill value, column
        rows = rows[1:]
    columns = {"tau": 1}
    if rows.shape[1] > 2:  # ν, τ, exp(−τ), f, (T f)_gcell, (T f)_bmrk, diff
        columns |= {"filter": 3, "tf_gcell": 4, "tf_benchmark": 5}
    return xr.Dataset(
        {name: ("wavenumber", rows[:, col]) for name, col in columns.items()},
        coords={"wavenumber": rows[:, 0]},
        attrs={"n_column_cm2": case.n_column_cm2, "name": case.name},
    )


def write_lines(
    case: GasCellCase,
    table: str,
    db_dir: Path,
    *,
    isotopologues: set[int],
    margin_cm: float,
    cache_dir: Path | None = None,
) -> Path:
    """Write ``case``'s HITRAN2020 lines as HAPI's ``<db_dir>/<table>.par``.

    Keeps the lines of the local isotopologues ``isotopologues`` (HITRAN's
    ``I``) within ``margin_cm`` of :attr:`GasCellCase.window_cm`; HAPI
    writes the header when ``db_begin`` loads the directory.
    """
    path = fetch(case.lines, cache_dir)
    if case.lines.member is None:
        text = path.read_text()
    else:
        with zipfile.ZipFile(path) as archive:
            text = io.TextIOWrapper(archive.open(case.lines.member)).read()
    lo, hi = case.window_cm[0] - margin_cm, case.window_cm[1] + margin_cm
    codes = {_ISO_CHARS[i - 1] for i in isotopologues}
    kept = [
        line.rstrip("\r")
        for line in text.splitlines()
        if len(line) >= 15 and line[2] in codes and lo <= float(line[3:15]) <= hi
    ]
    db_dir.mkdir(parents=True, exist_ok=True)
    out = db_dir / f"{table}.par"
    out.write_text("\n".join(kept) + "\n")
    return out


def shift_self_like_air(table: str) -> None:
    """Give HAPI's loaded ``table`` a self shift equal to its air shift.

    HITRAN carries no self shift, so HAPI shifts a pure gas
    (``Diluent={"self": 1}``) by nothing; gcell and SpectralCalc shift every
    line by δ_air p whatever the broadener. Without this the single line
    misses Table 4 by 4.6e-3 at 1 atm.
    """
    import hapi

    data = hapi.LOCAL_TABLE_CACHE[table]["data"]
    data["delta_self"] = list(data["delta_air"])
