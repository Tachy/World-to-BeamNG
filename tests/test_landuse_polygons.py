"""Tests for world_to_beamng.osm.landuse_polygons.build_landuse_polygons().

Background: The earlier code built land use polygons only from ways. A
multipolygon relation (e.g. large forest areas), however, has its geometry in
the members - they were skipped entirely (forest: ~35 % of the area, but
only 0.1 % of the terrain painted as forest).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.osm.landuse_polygons import build_landuse_polygons


def _pt(x, y):
    # Identity instead of WGS84: to_local() in the tests is (lon, lat) -> (x, y)
    return {"lon": float(x), "lat": float(y)}


def _ring(*xy):
    return [_pt(x, y) for x, y in xy]


def _to_local(points):
    return [(p["lon"], p["lat"]) for p in points]


SQUARE_10 = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
SQUARE_4 = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]


def _way(tags, coords):
    return {"type": "way", "id": 1, "tags": tags, "geometry": _ring(*coords)}


def _relation(tags, members):
    return {"type": "relation", "id": 2, "tags": {"type": "multipolygon", **tags}, "members": members}


def _member(role, coords):
    return {"type": "way", "role": role, "geometry": _ring(*coords)}


def test_closed_way_becomes_polygon():
    result = build_landuse_polygons([_way({"landuse": "meadow"}, SQUARE_10)], _to_local)

    assert len(result) == 1
    assert result[0]["osm_tags"] == {"landuse": "meadow"}
    assert result[0]["geometry"].area == pytest.approx(100.0)


def test_unclosed_way_is_skipped():
    # An open polyline with a landuse tag is not an area (e.g. member of a relation)
    result = build_landuse_polygons([_way({"landuse": "meadow"}, [(0, 0), (10, 0), (10, 10)])], _to_local)

    assert result == []


def test_elements_without_area_tags_are_ignored():
    elements = [
        _way({"building": "yes"}, SQUARE_10),
        _way({"highway": "residential"}, SQUARE_10),
        {"type": "node", "id": 5, "tags": {"natural": "tree"}, "lat": 1.0, "lon": 1.0},
    ]

    assert build_landuse_polygons(elements, _to_local) == []


def test_custom_tag_keys_select_other_areas_such_as_buildings():
    elements = [_way({"building": "yes"}, SQUARE_10), _way({"landuse": "meadow"}, SQUARE_10)]

    result = build_landuse_polygons(elements, _to_local, tag_keys=("building",))

    assert [r["osm_tags"] for r in result] == [{"building": "yes"}]


@pytest.mark.parametrize("key,value", [("landuse", "forest"), ("natural", "wood"), ("leisure", "park")])
def test_landuse_natural_and_leisure_tags_are_used(key, value):
    result = build_landuse_polygons([_way({key: value}, SQUARE_10)], _to_local)

    assert len(result) == 1


def test_multipolygon_relation_with_single_outer_ring():
    relation = _relation({"landuse": "forest"}, [_member("outer", SQUARE_10)])

    result = build_landuse_polygons([relation], _to_local)

    assert len(result) == 1
    assert result[0]["geometry"].area == pytest.approx(100.0)
    assert result[0]["osm_tags"]["landuse"] == "forest"


def test_multipolygon_relation_inner_ring_becomes_hole():
    relation = _relation({"landuse": "forest"}, [_member("outer", SQUARE_10), _member("inner", SQUARE_4)])

    geometry = build_landuse_polygons([relation], _to_local)[0]["geometry"]

    assert geometry.area == pytest.approx(100.0 - 16.0)
    assert not geometry.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(5, 5))


def test_multipolygon_relation_outer_ring_assembled_from_several_ways():
    # In OSM the outer ring is often split across several ways
    members = [
        _member("outer", [(0, 0), (10, 0), (10, 10)]),
        _member("outer", [(10, 10), (0, 10), (0, 0)]),
    ]

    result = build_landuse_polygons([_relation({"landuse": "forest"}, members)], _to_local)

    assert len(result) == 1
    assert result[0]["geometry"].area == pytest.approx(100.0)


def test_multipolygon_relation_with_two_separate_outer_rings():
    members = [_member("outer", SQUARE_10), _member("outer", [(20, 0), (30, 0), (30, 10), (20, 10), (20, 0)])]

    geometry = build_landuse_polygons([_relation({"landuse": "forest"}, members)], _to_local)[0]["geometry"]

    assert geometry.area == pytest.approx(200.0)


def test_non_multipolygon_relation_is_ignored():
    relation = {
        "type": "relation",
        "id": 3,
        "tags": {"type": "route", "natural": "mountain_range"},
        "members": [_member("outer", SQUARE_10)],
    }

    assert build_landuse_polygons([relation], _to_local) == []


def test_relation_with_unassemblable_members_is_skipped_without_error():
    relation = _relation({"landuse": "forest"}, [_member("outer", [(0, 0), (5, 5)])])

    assert build_landuse_polygons([relation], _to_local) == []


def test_relation_members_without_geometry_are_ignored():
    relation = _relation({"landuse": "forest"}, [{"type": "node", "role": "outer", "ref": 1}, _member("outer", SQUARE_10)])

    result = build_landuse_polygons([relation], _to_local)

    assert result[0]["geometry"].area == pytest.approx(100.0)


def test_self_intersecting_way_is_repaired_or_skipped_not_raised():
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)]

    result = build_landuse_polygons([_way({"landuse": "meadow"}, bowtie)], _to_local)

    for item in result:
        assert item["geometry"].is_valid


def test_detention_basin_relation_is_filled_so_the_pond_lies_inside_the_meadow():
    # dry retention basin: the small water body is entered as an inner ring, but the meadow should continue through
    relation = _relation({"landuse": "basin", "basin": "detention"}, [_member("outer", SQUARE_10), _member("inner", SQUARE_4)])

    result = build_landuse_polygons([relation], _to_local)

    assert len(result) == 1
    assert list(result[0]["geometry"].interiors) == []
    assert result[0]["geometry"].area == pytest.approx(100.0)


def test_other_relations_keep_their_holes():
    relation = _relation({"landuse": "forest"}, [_member("outer", SQUARE_10), _member("inner", SQUARE_4)])

    result = build_landuse_polygons([relation], _to_local)

    assert result[0]["geometry"].area == pytest.approx(100.0 - 16.0)
