# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru>=0.7", "cyclopts>=3"]
# ///
"""Build (or check) v1's validation scene catalogue, ``tests/validation/scenes.toml`` (radtrax#122).

The catalogue is metadata, not imagery: for each scene, the sensor, the product
identifier and its acquisition time, the site, and the reference (a metered
release rate with its uncertainty). Every value is copied from a source this
script reads; nothing is typed in by hand, so a rerun reproduces the file.

Sources (all public, no credentials, metadata only; no scene is downloaded):

- **Releases 2024–2025**: Reuland et al., *Large-Scale Controlled Methane
  Releases for Satellite-Based Detection and Emission Quantification of
  Point-Sources*, Stanford Digital Repository,
  https://doi.org/10.25740/qh001qt3946 (CC BY 4.0). One ``*_summary.csv`` per
  release (site, approximate overpass time, mean CH₄ rate over the 5 min
  before it and its 1σ), each checked against the deposit's SHA-1.
- **Releases 2022**: Sherwin et al. (2024), *Single-blind test of nine
  methane-sensing satellite systems from three continents*, AMT 17, 765–782,
  https://doi.org/10.5194/amt-17-765-2024; data and code at
  https://doi.org/10.5281/zenodo.10149992 (CC BY 4.0), the GitHub release
  ``publish`` (commit ``0a604d7``) that Zenodo archived. The per-overpass table
  ``Satellite_overpasses_with_release_rates_20230404.csv`` gives the rates;
  the consolidated results table gives Stanford's quality control, and an
  overpass is kept only when every team's row is ``OK``. Release-stack
  coordinates from the paper (Sect. 2.1).
- **Product identifiers**: NASA CMR (EMIT L1B radiance and its L2B CH₄
  enhancement), the Copernicus Data Space OData catalogue (Sentinel-2 L1C) and
  the USGS LandsatLook STAC (Landsat Collection 2 Level-1). A release is kept
  only when exactly one product of the named platform covers the site within
  ``MATCH_MINUTES`` of the release's overpass time.

Selection: every EMIT release in the 2024–2025 deposit that resolves to a
granule; every Sentinel-2 and Landsat 8/9 release in it at the same site
within ``NEAR_EMIT_DAYS`` of one of those; every Sentinel-2 and Landsat
overpass of the 2022 test that passed quality control (the zero releases are
the controls, and the clean reference passes of the multi-pass methods).

Usage::

    uv run --script scripts/build_scene_catalogue.py build           # rewrite the file
    uv run --script scripts/build_scene_catalogue.py build --check   # compare with it
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cyclopts  # ty: ignore[unresolved-import]  (inline script deps, PEP 723)
from loguru import logger  # ty: ignore[unresolved-import]


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "validation" / "scenes.toml"

SDR_DRUID = "qh001qt3946"
SDR_JSON = f"https://purl.stanford.edu/{SDR_DRUID}.json"
SDR_FILE = f"https://stacks.stanford.edu/file/druid:{SDR_DRUID}/"
SHERWIN_COMMIT = "0a604d7f264c977c5edc7efbd5f29ba75c81c4d4"
SHERWIN_RAW = (
    "https://raw.githubusercontent.com/sahar-elabbadi/SU-Controlled-Releases-2022/"
    f"{SHERWIN_COMMIT}/"
)
SHERWIN_OVERPASSES = (
    "00_raw_reports/Satellite_overpasses_with_release_rates_20230404.csv"
)
SHERWIN_QC = (
    "05_satellites/01_input_data/Satellite_results_consolidated_clean_20230418.csv"
)
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
USGS_STAC = "https://landsatlook.usgs.gov/stac-server/search"

#: A product must start within this many minutes of the release's overpass time.
MATCH_MINUTES = 30
#: Sentinel-2 and Landsat releases kept from the 2024–2025 deposit: within this
#: many days of an EMIT release at the same site.
NEAR_EMIT_DAYS = 3

SOURCES = {
    "reuland_sdr": {
        "citation": (
            "Reuland, F., Adams, T., Galvin, K., Kort, E. A., and Brandt, A. R.: "
            "Large-Scale Controlled Methane Releases for Satellite-Based Detection "
            "and Emission Quantification of Point-Sources, Stanford Digital Repository"
        ),
        "doi": "10.25740/qh001qt3946",
        "url": "https://purl.stanford.edu/qh001qt3946",
        "licence": "CC-BY-4.0",
    },
    "sherwin2024": {
        "citation": (
            "Sherwin, E. D., El Abbadi, S. H., Burdeau, P. M., Zhang, Z., Chen, Z., "
            "Rutherford, J. S., Chen, Y., and Brandt, A. R.: Single-blind test of "
            "nine methane-sensing satellite systems from three continents, "
            "Atmos. Meas. Tech., 17, 765-782, 2024; data: El Abbadi, S. H. and "
            "Sherwin, E. D., SU-Controlled-Releases-2022 (release publish, "
            f"commit {SHERWIN_COMMIT[:7]}), Zenodo"
        ),
        "doi": "10.5281/zenodo.10149992",
        "url": "https://doi.org/10.5194/amt-17-765-2024",
        "licence": "CC-BY-4.0",
    },
    "nasa_cmr": {
        "citation": "NASA Common Metadata Repository, EMITL1BRAD v001 and EMITL2BCH4ENH v002",
        "url": "https://cmr.earthdata.nasa.gov/search",
        "licence": "NASA Earth science data: no restrictions on use",
    },
    "cdse_odata": {
        "citation": "Copernicus Data Space Ecosystem OData catalogue, Sentinel-2 MSI L1C",
        "url": "https://catalogue.dataspace.copernicus.eu/odata/v1",
        "licence": "Copernicus Sentinel data: free and open (Copernicus data policy)",
    },
    "usgs_stac": {
        "citation": "USGS LandsatLook STAC, landsat-c2l1 (Collection 2 Level-1)",
        "url": "https://landsatlook.usgs.gov/stac-server",
        "licence": "USGS Landsat data: public domain",
    },
}

SITES: dict[str, dict[str, Any]] = {
    "casa_grande_sdr": {
        "name": "Casa Grande, Arizona: release stack, 2024-2025 campaign",
        "source": "reuland_sdr",
    },
    "casa_grande_2022": {
        "name": "Casa Grande, Arizona: release stacks, 2022 single-blind test",
        "lat": 32.8218205,
        "lon": -111.7857730,
        "source": "sherwin2024",
    },
}


@dataclass
class Scene:
    """One catalogue entry, as written."""

    id: str
    sensor: str
    product_id: str
    acquired_utc: dt.datetime
    product_source: str
    site: str
    cloud_cover_pct: float | None
    rate_kg_h: float
    rate_sigma_kg_h: float
    window_end_utc: dt.datetime
    source: str
    record: str
    record_sha1: str
    flow_stable: bool | None = None
    wind_record: str | None = None
    l2b_ch4enh: str | None = None
    orbit: str | None = None
    clean_reference: str | None = None


def _get(url: str, *, body: dict[str, Any] | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def _utc(text: str) -> dt.datetime:
    """Parse an ISO time (``Z`` or naive, read as UTC), to whole seconds."""
    t = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return t.astimezone(dt.UTC).replace(microsecond=0)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# --------------------------------------------------------------------------- sources


def sdr_summaries() -> list[dict[str, str]]:
    """Every EMIT, Sentinel-2 and Landsat release summary in the deposit, SHA-1 checked."""
    cocina = json.loads(_get(SDR_JSON))
    files: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "filename" in node:
                files.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(cocina)
    by_name = {f["filename"]: f for f in files}
    pattern = re.compile(r"/(\d{8})_(ET|S2[ABC]|LS[89])/[^/]+_summary\.csv$")
    rows = []
    for name, meta in sorted(by_name.items()):
        if not pattern.search(name):
            continue
        sha1 = next(
            d["digest"] for d in meta["hasMessageDigests"] if d["type"] == "sha1"
        )
        data = _get(SDR_FILE + urllib.parse.quote(name))
        if hashlib.sha1(data).hexdigest() != sha1:
            raise RuntimeError(f"SHA-1 mismatch for {name}")
        folder = name.rsplit("/", 1)[0]
        wind = [
            n
            for n in by_name
            if n.startswith(folder + "/") and "wind+meteorological_data.csv" in n
        ]
        for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            row["record"], row["record_sha1"] = name, sha1
            row["wind_record"] = wind[0] if len(wind) == 1 else ""
            row["code"] = folder.rsplit("_", 1)[-1]
            rows.append(row)
    logger.info("read {} release summaries from the SDR deposit", len(rows))
    return rows


def sherwin_overpasses() -> list[dict[str, str]]:
    """The 2022 Sentinel-2 and Landsat overpasses that passed Stanford's QC for every team."""
    raw = _get(SHERWIN_RAW + SHERWIN_OVERPASSES)
    sha1 = hashlib.sha1(raw).hexdigest()
    qc_rows = list(
        csv.DictReader(io.StringIO(_get(SHERWIN_RAW + SHERWIN_QC).decode("utf-8-sig")))
    )
    qc: dict[tuple[str, dt.date], set[str]] = {}
    for row in qc_rows:
        when = dt.datetime.strptime(row["DateTime (UTC)"], "%m/%d/%y %H:%M").date()
        qc.setdefault((row["Satellite"], when), set()).add(row["QC"])
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        if row["Satellite"] in ("Sentinel-2", "LandSat"):
            seen.setdefault((row["Satellite"], row["DateTime (UTC)"]), row)
    kept = []
    for (satellite, stamp), row in sorted(seen.items()):
        when = dt.datetime.strptime(stamp, "%m/%d/%Y %H:%M:%S").replace(tzinfo=dt.UTC)
        flags = qc.get((satellite, when.date()), set())
        if flags != {"OK"}:
            logger.info(
                "2022 {} {}: QC {}, skipped",
                satellite,
                stamp,
                sorted(flags) or "absent",
            )
            continue
        gas = float(row["gas_kgh_mean"])
        rate = 0.0 if gas == 0.0 else float(row["ch4_kgh_mean"])
        sigma = 0.0 if gas == 0.0 else float(row["ch4_kgh_sigma"])
        kept.append(
            {
                "satellite": satellite,
                "time": when.isoformat(),
                "rate": repr(rate),
                "sigma": repr(sigma),
                "record_sha1": sha1,
            }
        )
    logger.info("kept {} of the 2022 Sentinel-2 and Landsat overpasses", len(kept))
    return kept


# --------------------------------------------------------------------------- products


def emit_granule(
    lat: float, lon: float, t: dt.datetime
) -> tuple[str, str, dt.datetime, float, str | None] | None:
    """The one EMIT L1B granule over the site within the window, and its L2B CH₄ enhancement."""
    window = f"{_iso(t - dt.timedelta(minutes=MATCH_MINUTES))},{_iso(t + dt.timedelta(minutes=MATCH_MINUTES))}"
    found: dict[str, list[dict[str, Any]]] = {}
    for short_name in ("EMITL1BRAD", "EMITL2BCH4ENH"):
        query = urllib.parse.urlencode(
            {"short_name": short_name, "point": f"{lon},{lat}", "temporal": window}
        )
        found[short_name] = json.loads(_get(f"{CMR}?{query}"))["feed"]["entry"]
    if len(found["EMITL1BRAD"]) != 1:
        return None
    rad = found["EMITL1BRAD"][0]
    enh = [e.get("producer_granule_id", e["title"]) for e in found["EMITL2BCH4ENH"]]
    return (
        "EMIT",
        rad.get("producer_granule_id", rad["title"]),
        _utc(rad["time_start"]),
        float(rad["cloud_cover"]),
        enh[0] if len(enh) == 1 else None,
    )


def s2_product(
    lat: float, lon: float, t: dt.datetime, platform: str | None
) -> tuple[str, str, dt.datetime, float, str] | None:
    """The one Sentinel-2 L1C product over the site whose datatake starts within the window before ``t``."""
    flt = (
        "Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') "
        f"and ContentDate/Start gt {_iso(t - dt.timedelta(minutes=MATCH_MINUTES))} "
        f"and ContentDate/Start lt {_iso(t + dt.timedelta(minutes=1))}"
    )
    query = urllib.parse.urlencode(
        {"$filter": flt, "$expand": "Attributes", "$top": 20}
    )
    hits = []
    for product in json.loads(_get(f"{ODATA}?{query}"))["value"]:
        attrs = {a["Name"]: a["Value"] for a in product["Attributes"]}
        if platform is None or attrs["platformSerialIdentifier"] == platform:
            hits.append((product, attrs))
    if len(hits) != 1:
        return None
    product, attrs = hits[0]
    name = product["Name"].removesuffix(".SAFE")
    orbit = f"R{int(attrs['relativeOrbitNumber']):03d}_T{attrs['tileId']}"
    return (
        "S2" + attrs["platformSerialIdentifier"],
        name,
        _utc(product["ContentDate"]["Start"]),
        float(attrs["cloudCover"]),
        orbit,
    )


def landsat_product(
    lat: float, lon: float, t: dt.datetime, platform: str | None
) -> tuple[str, str, dt.datetime, float, str] | None:
    """The one Landsat 8/9 Collection 2 L1 product (T1 or T2) over the site within the window."""
    window = f"{_iso(t - dt.timedelta(minutes=MATCH_MINUTES))}/{_iso(t + dt.timedelta(minutes=MATCH_MINUTES))}"
    body = {
        "collections": ["landsat-c2l1"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": window,
        "limit": 20,
    }
    hits = []
    for item in json.loads(_get(USGS_STAC, body=body))["features"]:
        p = item["properties"]
        if p["platform"] not in ("LANDSAT_8", "LANDSAT_9") or p[
            "landsat:collection_category"
        ] not in ("T1", "T2"):
            continue
        if platform is None or p["platform"] == platform:
            hits.append(item)
    if len(hits) != 1:
        return None
    p = hits[0]["properties"]
    sensor = "L" + p["platform"][-1]
    orbit = f"P{p['landsat:wrs_path']}_R{p['landsat:wrs_row']}"
    return sensor, hits[0]["id"], _utc(p["datetime"]), float(p["eo:cloud_cover"]), orbit


# --------------------------------------------------------------------------- build


def collect() -> list[Scene]:
    """Every scene, in a fixed order (sensor, then time)."""
    scenes: list[Scene] = []
    sdr = sdr_summaries()
    emit_dates: dict[str, list[dt.date]] = {}
    for row in sdr:
        if row["code"] != "ET":
            continue
        t = _utc(f"{row['date']}T{row['time_UTC']}")
        hit = emit_granule(float(row["lat"]), float(row["lon"]), t)
        if hit is None:
            logger.warning(
                "EMIT release {}: no single L1B granule within {} min, skipped",
                row["release_ID"],
                MATCH_MINUTES,
            )
            continue
        sensor, product, acquired, cloud, enh = hit
        scenes.append(
            _sdr_scene(
                row, sensor, product, acquired, cloud, "nasa_cmr", t, l2b_ch4enh=enh
            )
        )
        emit_dates.setdefault(row["location"], []).append(t.date())
    for row in sdr:
        code = row["code"]
        if code == "ET":
            continue
        t = _utc(f"{row['date']}T{row['time_UTC']}")
        near = [
            d
            for d in emit_dates.get(row["location"], [])
            if abs((t.date() - d).days) <= NEAR_EMIT_DAYS
        ]
        if not near:
            continue
        lat, lon = float(row["lat"]), float(row["lon"])
        if code.startswith("S2"):
            hit, catalogue = s2_product(lat, lon, t, code[-1]), "cdse_odata"
        else:
            hit, catalogue = (
                landsat_product(lat, lon, t, f"LANDSAT_{code[-1]}"),
                "usgs_stac",
            )
        if hit is None:
            logger.warning(
                "{} release {}: no single product within {} min, skipped",
                code,
                row["release_ID"],
                MATCH_MINUTES,
            )
            continue
        sensor, product, acquired, cloud, orbit = hit
        scenes.append(
            _sdr_scene(row, sensor, product, acquired, cloud, catalogue, t, orbit=orbit)
        )
    site = SITES["casa_grande_2022"]
    for row in sherwin_overpasses():
        t = _utc(row["time"])
        if row["satellite"] == "Sentinel-2":
            hit, catalogue = s2_product(site["lat"], site["lon"], t, None), "cdse_odata"
        else:
            hit, catalogue = (
                landsat_product(site["lat"], site["lon"], t, None),
                "usgs_stac",
            )
        if hit is None:
            logger.warning(
                "2022 {} {}: no single product within {} min, skipped",
                row["satellite"],
                row["time"],
                MATCH_MINUTES,
            )
            continue
        sensor, product, acquired, cloud, orbit = hit
        scenes.append(
            Scene(
                id=f"{sensor}-{t:%Y%m%d}",
                sensor=sensor,
                product_id=product,
                acquired_utc=acquired,
                product_source=catalogue,
                site="casa_grande_2022",
                cloud_cover_pct=cloud,
                rate_kg_h=float(row["rate"]),
                rate_sigma_kg_h=float(row["sigma"]),
                window_end_utc=t,
                source="sherwin2024",
                record=SHERWIN_OVERPASSES,
                record_sha1=row["record_sha1"],
                orbit=orbit,
            )
        )
    _link_clean_references(scenes)
    order = {"EMIT": 0, "S2A": 1, "S2B": 1, "S2C": 1, "L8": 2, "L9": 2}
    scenes.sort(key=lambda s: (order[s.sensor], s.window_end_utc))
    return scenes


def _sdr_scene(
    row: dict[str, str],
    sensor: str,
    product: str,
    acquired: dt.datetime,
    cloud: float,
    catalogue: str,
    t: dt.datetime,
    *,
    l2b_ch4enh: str | None = None,
    orbit: str | None = None,
) -> Scene:
    return Scene(
        id=f"{sensor}-{t:%Y%m%d}",
        sensor=sensor,
        product_id=product,
        acquired_utc=acquired,
        product_source=catalogue,
        site=_sdr_site(row),
        cloud_cover_pct=cloud,
        rate_kg_h=float(row["ch4_kgh_mean"]),
        rate_sigma_kg_h=float(row["ch4_kgh_sigma"]),
        window_end_utc=t,
        source="reuland_sdr",
        record=row["record"],
        record_sha1=row["record_sha1"],
        flow_stable={"True": True, "False": False}.get(row["PI_within_10pct_of_mean"]),
        wind_record=row["wind_record"] or None,
        l2b_ch4enh=l2b_ch4enh,
        orbit=orbit,
    )


def _sdr_site(row: dict[str, str]) -> str:
    key = "casa_grande_sdr"
    if row["location"] != "Casa Grande":
        raise RuntimeError(f"no site key for {row['location']!r}; add one to SITES")
    lat, lon = float(row["lat"]), float(row["lon"])
    site = SITES[key]
    if site.setdefault("lat", lat) != lat or site.setdefault("lon", lon) != lon:
        raise RuntimeError(f"{row['record']}: the site moved ({lat}, {lon})")
    return key


def _link_clean_references(scenes: list[Scene]) -> None:
    """Point each release at the nearest zero release on the same orbit (same campaign and site)."""
    for scene in scenes:
        if scene.rate_kg_h == 0.0 or scene.orbit is None:
            continue
        zeros = [
            z
            for z in scenes
            if z.rate_kg_h == 0.0
            and z.orbit == scene.orbit
            and z.site == scene.site
            and z.sensor[0] == scene.sensor[0]
        ]
        if zeros:
            scene.clean_reference = min(
                zeros, key=lambda z: abs(z.window_end_utc - scene.window_end_utc)
            ).id


def _value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.dumps(v, ensure_ascii=False)


def render(scenes: list[Scene]) -> str:
    """The catalogue as TOML text (stable order, one table per scene)."""
    lines = [
        "# v1's validation scene catalogue (radtrax#122). Generated by",
        "# scripts/build_scene_catalogue.py from the sources below; do not edit by hand.",
        "# Loaded and validated by tests/_scene_catalogue.py.",
        "",
        "schema_version = 1",
        f"match_minutes = {MATCH_MINUTES}",
        "",
    ]
    for key, source in SOURCES.items():
        lines.append(f"[sources.{key}]")
        lines += [f"{k} = {_value(v)}" for k, v in source.items()]
        lines.append("")
    for key, site in SITES.items():
        lines.append(f"[sites.{key}]")
        lines += [f"{k} = {_value(site[k])}" for k in ("name", "lat", "lon", "source")]
        lines.append("")
    for s in scenes:
        lines.append("[[scene]]")
        for key in (
            "id",
            "sensor",
            "product_id",
            "acquired_utc",
            "product_source",
            "site",
            "cloud_cover_pct",
            "orbit",
            "l2b_ch4enh",
            "clean_reference",
        ):
            value = getattr(s, key)
            if value is not None:
                lines.append(f"{key} = {_value(value)}")
        lines.append("[scene.reference]")
        lines.append('kind = "metered_release"')
        for key in (
            "rate_kg_h",
            "rate_sigma_kg_h",
            "window_end_utc",
            "flow_stable",
            "source",
            "record",
            "record_sha1",
            "wind_record",
        ):
            value = getattr(s, key)
            if value is not None:
                lines.append(f"{key} = {_value(value)}")
        lines.append("")
    return "\n".join(lines)


app = cyclopts.App(help=__doc__)


@app.command
def build(*, out: Path = OUT, check: bool = False) -> None:
    """Query the sources, write the catalogue (or compare with it).

    Args:
        out: The TOML file to write, or to compare with when ``check``.
        check: Compare a fresh build with ``out`` instead of writing it.
    """
    text = render(collect())
    if check:
        if out.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{out} differs from a fresh build; rerun without --check")
        logger.info("{} reproduces", out)
        return
    out.write_text(text, encoding="utf-8")
    logger.info("wrote {}", out)


if __name__ == "__main__":
    app()
