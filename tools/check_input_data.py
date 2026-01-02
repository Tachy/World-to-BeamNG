"""Check if road_polygons have junction_indices set"""

import sys

sys.path.insert(0, "C:\\Eigene_Programme\\World-to-BeamNG")

from world_to_beamng import config
from world_to_beamng.terrain.elevation import load_height_data
from world_to_beamng.terrain.grid import create_terrain_grid
from world_to_beamng.osm.parser import (
    calculate_bbox_from_height_data,
    extract_roads_from_osm,
)
from world_to_beamng.osm.downloader import get_osm_data
from world_to_beamng.geometry.polygon import get_road_polygons, clip_road_polygons
from world_to_beamng.geometry.junctions import (
    detect_junctions_in_centerlines,
    mark_junction_endpoints,
    split_roads_at_mid_junctions,
)
import numpy as np

# Lade Daten (kopiert aus main())
config.LOCAL_OFFSET = None
config.BBOX = None
config.GRID_BOUNDS_LOCAL = None

height_points, height_elevations = load_height_data()
config.BBOX = calculate_bbox_from_height_data(height_points)

if config.LOCAL_OFFSET is None:
    config.LOCAL_OFFSET = (
        height_points[0, 0],
        height_points[0, 1],
        height_elevations[0],
    )

ox, oy, oz = config.LOCAL_OFFSET
height_points[:, 0] -= ox
height_points[:, 1] -= oy
height_elevations = height_elevations - oz

osm_elements = get_osm_data(config.BBOX)
roads = extract_roads_from_osm(osm_elements)

grid_points, grid_elevations, nx, ny = create_terrain_grid(
    height_points, height_elevations, grid_spacing=config.GRID_SPACING
)
config.GRID_BOUNDS_LOCAL = (
    grid_points[:, 0].min(),
    grid_points[:, 0].max(),
    grid_points[:, 1].min(),
    grid_points[:, 1].max(),
)

road_polygons = get_road_polygons(roads, config.BBOX, height_points, height_elevations)
if config.ROAD_CLIP_MARGIN > 0:
    road_polygons = clip_road_polygons(
        road_polygons, config.GRID_BOUNDS_LOCAL, margin=config.ROAD_CLIP_MARGIN
    )

junctions = detect_junctions_in_centerlines(road_polygons)
road_polygons, junctions = split_roads_at_mid_junctions(road_polygons, junctions)
road_polygons = mark_junction_endpoints(road_polygons, junctions)

# Prüfe Straßen 1846 und 1847
for idx in [1846, 1847]:
    if idx < len(road_polygons):
        road = road_polygons[idx]
        road_id = road.get("id")
        junction_indices = road.get("junction_indices", {})

        print(f"\nRoad {road_id} (index {idx}):")
        print(f"  junction_indices: {junction_indices}")

        if junction_indices:
            start_idx = junction_indices.get("start")
            end_idx = junction_indices.get("end")

            if start_idx is not None and start_idx < len(junctions):
                print(
                    f"  Start Junction {start_idx}: {junctions[start_idx]['position']}"
                )

            if end_idx is not None and end_idx < len(junctions):
                print(f"  End Junction {end_idx}: {junctions[end_idx]['position']}")

print(f"\n\nJunction 381: {junctions[381]['position']}")
print(f"Junction 381 road_indices: {junctions[381]['road_indices']}")
