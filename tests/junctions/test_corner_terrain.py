"""Junction corner polygons for the terrain: fill + sidewalk corner, in the format of embed_roads_into_heightmap()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.junctions.corners import corner_embed_roads, find_junction_corners, junction_roads
from world_to_beamng.terrain.road_embedding import embed_roads_into_heightmap

TABLE = {"default_radius": 6.0, "radius_by_highway": {}}


def _dict(road_id, start, end, width, highway="residential", **extra):
    xs = np.linspace(start[0], end[0], 13)
    ys = np.linspace(start[1], end[1], 13)
    return {"road_id": road_id, "trimmed_centerline": np.column_stack([xs, ys, np.full(13, 100.0)]),
            "osm_tags": {"highway": highway}, "structure_type": "surface", "_width": width, **extra}


def _props(poly):
    internal = "concrete" if poly["osm_tags"].get("surface") == "concrete" else "asphalt_road_standard"
    return {"width": poly["_width"], "internal_name": internal}


ROADS = [_dict("east", (50, 50), (110, 50), 6.0), _dict("west", (-10, 50), (50, 50), 6.0), _dict("north", (50, 50), (50, 110), 5.0)]


def _corners(roads):
    return find_junction_corners(junction_roads(roads, _props, frozenset({"footway"})),
                                 TABLE, 0.5, 160.0, rank={})


def test_junction_roads_skips_footways_and_structures_but_keeps_roads_without_decal():
    roads = ROADS + [_dict("path", (50, 50), (50, -10), 2.0, highway="footway"),
                     _dict("bridge", (50, 50), (0, 0), 6.0, structure_type="bridge"),
                     _dict("plain", (50, 50), (100, 100), 6.0, osm_tags={"highway": "residential", "surface": "concrete"})]
    # a road without DecalRoad still shapes the terrain at its corners; export_junctions() just does not draw them
    assert [r["road_id"] for r in junction_roads(roads, _props, frozenset({"footway"}))] == ["east", "west", "north", "plain"]


def test_cross_with_a_footway_arm_behaves_like_a_t():
    roads = ROADS + [_dict("path", (50, 50), (50, -10), 2.0, highway="footway")]
    assert len(_corners(roads)) == 2


def test_junction_roads_uses_blended_width_nodes():
    nodes = np.array([[50.0, 50.0, 100.0, 8.0], [110.0, 50.0, 100.0, 6.0]])
    roads = [dict(ROADS[0], width_nodes=nodes)]
    arm = junction_roads(roads, _props, frozenset())[0]
    assert arm["half_widths"][0] == pytest.approx(4.0)


def test_fill_polygon_is_embedded_at_road_height():
    heights = np.full((128, 128), 90.0)
    embedded = embed_roads_into_heightmap(heights, 0.0, 0.0, 1.0, corner_embed_roads(_corners(ROADS), {}, 1.15))
    assert embedded[54, 54] == pytest.approx(100.0)  # (54, 54) lies in the NE fill: x 52.5..58.5, y 53..59, near the edges
    assert embedded[80, 80] == pytest.approx(90.0)


def test_sidewalk_corner_is_added_only_when_both_arms_have_one_facing_the_corner():
    corners = _corners(ROADS)
    plain = corner_embed_roads(corners, {}, 1.15)
    both = corner_embed_roads(corners, {"east": {"left": "a"}, "north": {"right": "a"}}, 1.15)
    one = corner_embed_roads(corners, {"east": {"left": "a"}}, 1.15)
    assert len(plain) == len(one) == 2 and len(both) == 3


def test_no_corners_no_polygons():
    assert corner_embed_roads([], {}, 1.15) == []


def test_embedding_reaches_past_the_arc_so_no_terrain_shows_through():
    heights = np.full((128, 128), 90.0)
    embedded = embed_roads_into_heightmap(heights, 0.0, 0.0, 1.0, corner_embed_roads(_corners(ROADS), {}, 1.15, margin=1.5))
    # NE fillet centre (58.5, 59), r = 6: cell (x 55, y 55) lies 0.7 m beyond the arc
    assert embedded[55, 55] == pytest.approx(100.0)
    assert embedded[80, 80] == pytest.approx(90.0)


def test_gravel_and_dirt_tracks_both_form_corners():
    from world_to_beamng import config

    assert "track" not in config.JUNCTION_EXCLUDED_HIGHWAYS and "footway" in config.JUNCTION_EXCLUDED_HIGHWAYS

    def props(poly):
        surface = poly["osm_tags"].get("surface")
        return {"width": poly["_width"], "internal_name": {"gravel": "gravel_road", "dirt": "dirt_road"}.get(surface, "asphalt_road_standard")}

    roads = ROADS + [_dict("gravel", (50, 50), (50, -10), 3.0, osm_tags={"highway": "track", "surface": "gravel"}),
                     _dict("dirt", (50, 50), (0, 0), 3.0, osm_tags={"highway": "track", "surface": "dirt"})]
    ids = [r["road_id"] for r in junction_roads(roads, props, config.JUNCTION_EXCLUDED_HIGHWAYS)]
    assert "gravel" in ids and "dirt" in ids
