"""Tests für world_to_beamng.geometry.road_structures: Klassifizierung von Brücken/Tunneln/Galerien anhand OSM-Tags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.road_structures import classify_structure, extend_gallery_centerline_ends, split_by_structure_type


def test_bridge_tag_is_classified_as_bridge():
    assert classify_structure({"highway": "primary", "bridge": "yes"}) == "bridge"
    assert classify_structure({"highway": "primary", "bridge": "viaduct"}) == "bridge"


def test_bridge_no_is_not_a_bridge():
    assert classify_structure({"highway": "primary", "bridge": "no"}) == "surface"


def test_avalanche_protector_is_a_gallery_not_a_tunnel():
    assert classify_structure({"highway": "primary", "tunnel": "avalanche_protector"}) == "gallery"


def test_other_tunnel_values_are_classified_as_tunnel():
    assert classify_structure({"highway": "trunk", "tunnel": "yes"}) == "tunnel"
    assert classify_structure({"waterway": "stream", "tunnel": "culvert"}) == "tunnel"
    assert classify_structure({"highway": "path", "tunnel": "building_passage"}) == "tunnel"


def test_tunnel_no_is_not_a_tunnel():
    assert classify_structure({"highway": "primary", "tunnel": "no"}) == "surface"


def test_missing_tags_are_surface():
    assert classify_structure({}) == "surface"
    assert classify_structure(None) == "surface"
    assert classify_structure({"highway": "residential"}) == "surface"


def test_bridge_takes_priority_over_tunnel_if_both_are_present():
    assert classify_structure({"bridge": "yes", "tunnel": "yes"}) == "bridge"


def test_split_by_structure_type_separates_surface_from_structures():
    roads = [
        {"road_id": 1, "structure_type": "surface"},
        {"road_id": 2, "structure_type": "bridge"},
        {"road_id": 3, "structure_type": "tunnel"},
        {"road_id": 4, "structure_type": "gallery"},
        {"road_id": 5},  # fehlendes Feld -> gilt als surface
    ]

    surface, structures = split_by_structure_type(roads)

    assert [r["road_id"] for r in surface] == [1, 5]
    assert [r["road_id"] for r in structures] == [2, 3, 4]


# --- extend_gallery_centerline_ends ---------------------------------------------------------------------


def _gallery(centerline):
    return {"road_id": 1, "structure_type": "gallery", "trimmed_centerline": np.array(centerline, dtype=float)}


def test_both_ends_are_extrapolated_by_the_given_distance():
    road = _gallery([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (20.0, 0.0, 110.0)])

    result = extend_gallery_centerline_ends([road], extension_m=2.0)

    coords = result[0]["trimmed_centerline"]
    assert coords[0] == pytest.approx([-2.0, 0.0, 100.0])  # exakt entgegen der Richtung zu Punkt 1
    assert coords[-1] == pytest.approx([22.0, 0.0, 112.0])  # Steigung des letzten Segments (1 m Höhe je 10 m) mit extrapoliert
    assert coords[1] == pytest.approx([10.0, 0.0, 100.0])  # innere Punkte unverändert


def test_only_gallery_entries_are_touched():
    surface_road = {"road_id": 2, "structure_type": "surface", "trimmed_centerline": np.array([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)])}
    tunnel = {"road_id": 3, "structure_type": "tunnel", "trimmed_centerline": np.array([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)])}

    result = extend_gallery_centerline_ends([surface_road, tunnel], extension_m=2.0)

    assert result[0]["trimmed_centerline"][0] == pytest.approx([0.0, 0.0, 100.0])
    assert result[1]["trimmed_centerline"][0] == pytest.approx([0.0, 0.0, 100.0])


def test_zero_extension_is_a_noop_and_returns_the_same_list():
    roads = [_gallery([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)])]

    result = extend_gallery_centerline_ends(roads, extension_m=0.0)

    assert result is roads  # unveraendert durchgereicht (dieselbe Liste)


def test_short_or_missing_centerline_is_left_alone():
    degenerate = {"road_id": 4, "structure_type": "gallery", "trimmed_centerline": np.array([(0.0, 0.0, 100.0)])}
    missing = {"road_id": 5, "structure_type": "gallery"}

    result = extend_gallery_centerline_ends([degenerate, missing], extension_m=2.0)

    assert result[0] is degenerate
    assert result[1] is missing


def test_without_an_osm_mapper_road_polygon_is_left_untouched():
    """Reiner Centerline-Test (wie die Tests oben) - kein osm_mapper -> road_polygon unverändert/fehlt."""
    road = _gallery([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)])
    road["road_polygon"] = np.array([[0.0, -3.0], [10.0, -3.0], [10.0, 3.0], [0.0, 3.0]])

    result = extend_gallery_centerline_ends([road], extension_m=2.0)

    assert result[0]["road_polygon"] is road["road_polygon"]


class _FakeMapper:
    def get_road_properties(self, tags):
        return {"width": 6.0}


def test_with_an_osm_mapper_road_polygon_is_rebuilt_to_cover_the_extended_centerline():
    """Regression: road_polygon wurde bisher NIE mitverlängert (nur trimmed_centerline) - embed_roads_
    into_heightmap()/apply_embankment_blend() blieben dadurch an den alten, unverlängerten Enden stehen,
    obwohl das Galerie-Mesh (aus der verlängerten Centerline gebaut) bereits darüber sitzt. Siehe
    tests/terrain/test_road_embedding.py::test_gallery_extension_zone_is_now_embedded_too für den
    direkten Beleg am Heightmap."""
    road = _gallery([(0.0, 10.0, 100.0), (20.0, 10.0, 100.0)])

    result = extend_gallery_centerline_ends([road], extension_m=2.0, osm_mapper=_FakeMapper())

    polygon = result[0]["road_polygon"]
    # width=6.0 -> half_width=3.0; verlängerte Centerline reicht jetzt von x=-2 bis x=22 (siehe
    # test_both_ends_are_extrapolated_by_the_given_distance) - das neu gebufferte Polygon muss diesen
    # ganzen Bereich abdecken, nicht nur das ursprüngliche x=[0, 20].
    assert polygon[:, 0].min() == pytest.approx(-2.0, abs=1e-6)
    assert polygon[:, 0].max() == pytest.approx(22.0, abs=1e-6)
    assert polygon[:, 1].min() == pytest.approx(10.0 - 3.0, abs=1e-6)
    assert polygon[:, 1].max() == pytest.approx(10.0 + 3.0, abs=1e-6)
