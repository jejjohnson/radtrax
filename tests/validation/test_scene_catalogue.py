"""v1's validation scene catalogue (radtrax#122): the committed file, its schema and the resolver.

The unmarked tests read only ``tests/validation/scenes.toml`` and need no
network. ``test_every_scene_resolves`` (integration, slow) checks that each
product exists where georeader reads it from; it skips, naming them, without
the sensor's credentials.
"""

from __future__ import annotations

import copy
import datetime as dt
import sys
import tomllib
from collections import Counter
from typing import Any

import pytest
from _scene_catalogue import (
    CATALOGUE,
    load_scene_catalogue,
    missing_credentials,
    parse_scene_catalogue,
    product_location,
)

from radtrax.exceptions import MissingExtraError, ParameterError


SCENES = load_scene_catalogue()


def test_catalogue_has_a_handful_of_scenes_per_sensor() -> None:
    families = Counter(s.family for s in SCENES)
    assert min(families[f] for f in ("EMIT", "S2", "Landsat")) >= 5, families


def test_multispectral_sensors_have_zero_release_controls_and_clean_references() -> (
    None
):
    for family in ("S2", "Landsat"):
        scenes = [s for s in SCENES if s.family == family]
        assert any(s.reference.rate_kg_h == 0.0 for s in scenes), family
        assert any(s.clean_reference is not None for s in scenes), family


def test_every_entry_names_its_record_and_licence() -> None:
    for scene in SCENES:
        assert scene.reference.record, scene.id
        assert scene.reference.record_sha1, scene.id
        assert scene.licence and scene.product_source.licence, scene.id
        assert scene.reference.source.doi, scene.id


def test_metered_rates_sit_inside_their_published_range() -> None:
    # The 2024-2025 releases are 72-1184 kg/h (Reuland et al.); the 2022
    # releases up to ~1.5 t/h (Sherwin et al. 2024). A rate outside means a
    # unit slip (t/h, kg/s) in a rebuild.
    for scene in SCENES:
        assert scene.reference.rate_kg_h is not None
        assert 0.0 <= scene.reference.rate_kg_h < 2000.0, scene.id


# --------------------------------------------------------------------------- schema


def _raw() -> dict[str, Any]:
    return tomllib.loads(CATALOGUE.read_text(encoding="utf-8"))


def test_parse_accepts_the_committed_file() -> None:
    assert parse_scene_catalogue(_raw()) == SCENES


def _emit(raw):
    return next(s for s in raw["scene"] if s["sensor"] == "EMIT")


def _released(raw):
    return next(s for s in raw["scene"] if s.get("clean_reference"))


def _set(path: str, value: Any):
    """A mutation: set ``path`` (``which.key`` or ``which.reference.key``) on a scene."""

    def mutate(raw: dict[str, Any]) -> None:
        which, *keys = path.split(".")
        node = {"emit": _emit, "released": _released}[which](raw)
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value

    return mutate


def _top(key: str, value: Any):
    def mutate(raw: dict[str, Any]) -> None:
        raw[key] = value

    return mutate


def _duplicate(raw: dict[str, Any]) -> None:
    raw["scene"].append(copy.deepcopy(raw["scene"][0]))


def _clean_is_a_release(raw: dict[str, Any]) -> None:
    released = _released(raw)
    other = next(
        s
        for s in raw["scene"]
        if s.get("orbit") == released["orbit"]
        and s["reference"]["rate_kg_h"] > 0
        and s is not released
    )
    released["clean_reference"] = other["id"]


def _clean_other_orbit(raw: dict[str, Any]) -> None:
    released = _released(raw)
    other = next(
        s
        for s in raw["scene"]
        if s["reference"]["rate_kg_h"] == 0.0
        and s.get("orbit") not in (None, released["orbit"])
    )
    released["clean_reference"] = other["id"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_top("schema_version", 2), "schema_version"),
        (_top("scene", []), "no \\[\\[scene\\]\\]"),
        (_set("emit.sensor", "TROPOMI"), "sensor"),
        (
            _set("emit.product_id", "EMIT_L2A_RFL_001_20241202T180154_2433712_009"),
            "not a EMIT product id",
        ),
        (
            _set(
                "emit.acquired_utc", dt.datetime(2024, 12, 3, 18, 1, 54, tzinfo=dt.UTC)
            ),
            "differs from acquired_utc",
        ),
        (_set("emit.acquired_utc", dt.datetime(2024, 12, 2, 18, 1, 54)), "in UTC"),
        (
            _set("emit.l2b_ch4enh", "EMIT_L2B_CH4ENH_002_20240101T000000_0000000_000"),
            "l2b_ch4enh",
        ),
        (_set("emit.site", "nowhere"), "'nowhere' is not defined"),
        (_set("emit.colour", "red"), "unknown \\['colour'\\]"),
        (_set("emit.cloud_cover_pct", 101.0), "cloud_cover_pct"),
        (_set("emit.reference.kind", "guess"), "kind"),
        (_set("emit.reference.rate_kg_h", -1.0), "non-negative"),
        (_set("emit.reference.source", "wikipedia"), "'wikipedia' is not defined"),
        (
            _set(
                "emit.reference.window_end_utc",
                dt.datetime(2024, 12, 2, 12, 0, tzinfo=dt.UTC),
            ),
            "more than 30 min",
        ),
        (_set("released.orbit", "R999_T12SVB"), "orbit"),
        (_set("released.clean_reference", "nope"), "'nope' is not defined"),
        (_clean_is_a_release, "not a metered zero release"),
        (_clean_other_orbit, "not on the same orbit"),
        (_duplicate, "appears twice"),
    ],
)
def test_parse_rejects(mutate, message: str) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(ParameterError, match=message):
        parse_scene_catalogue(raw)


def _coincident(raw: dict[str, Any], minutes: float) -> None:
    released, emit = _released(raw), _emit(raw)
    released["reference"] = {
        "kind": "coincident_emit",
        "source": "nasa_cmr",
        "record": emit["product_id"],
        "emit_scene": emit["id"],
        "max_dt_minutes": minutes,
    }


def test_coincident_reference_pairs_with_an_emit_scene_within_its_window() -> None:
    raw = _raw()
    released, emit = _released(raw), _emit(raw)
    gap = abs(released["acquired_utc"] - emit["acquired_utc"]) / dt.timedelta(minutes=1)
    _coincident(raw, gap + 1.0)
    scene = next(s for s in parse_scene_catalogue(raw) if s.id == released["id"])
    assert scene.reference.emit_scene == emit["id"]
    raw = _raw()
    _coincident(raw, gap - 1.0)
    with pytest.raises(ParameterError, match="min away"):
        parse_scene_catalogue(raw)


# --------------------------------------------------------------------------- resolver


def _first(family: str):
    return next(s for s in SCENES if s.family == family)


def test_product_location_points_where_georeader_reads() -> None:
    pytest.importorskip("georeader")
    s2 = _first("S2")
    tile = s2.orbit.split("_T")[1]
    assert product_location(s2).endswith(
        f"/tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/{s2.product_id}.SAFE"
    )
    with pytest.raises(ParameterError, match="georeader#99"):
        product_location(_first("Landsat"))


def test_product_location_of_emit_is_the_l1b_radiance_link() -> None:
    pytest.importorskip("requests")  # georeader's EMIT reader imports it
    emit = _first("EMIT")
    assert product_location(emit).endswith(
        f"/EMITL1BRAD.001/{emit.product_id}/{emit.product_id}.nc"
    )


def test_product_location_names_the_geo_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "georeader.readers", None)
    with pytest.raises(MissingExtraError, match=r"\[geo\]"):
        product_location(_first("EMIT"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("scene", SCENES, ids=lambda s: s.id)
def test_every_scene_resolves(scene) -> None:
    pytest.importorskip("georeader")
    missing = missing_credentials(scene)
    if missing:
        pytest.skip(f"{scene.id} needs: {'; '.join(missing)}")
    if scene.family == "S2":
        from georeader.readers.S2_SAFE_reader import s2_public_bucket_path

        s2_public_bucket_path(scene.product_id + ".SAFE", check_exists=True)
        return
    requests = pytest.importorskip("requests")
    from georeader.readers.emit import get_auth

    response = requests.head(
        product_location(scene), auth=get_auth(), allow_redirects=True, timeout=60
    )
    assert response.status_code == 200, (scene.id, response.status_code)
