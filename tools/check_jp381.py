"""Quick check for Junction Point 381"""

import json
import math
import os


def calculate_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
json_path = os.path.join(project_root, "cache", "debug_trimmed_roads.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

roads = data["roads"]
junctions = data["junctions"]

# Junction 381
if 381 < len(junctions):
    jp381 = junctions[381]
    jp_pos = jp381["position"]
    road_indices = jp381["road_indices"]

    print(f"Junction 381 at {jp_pos}")
    print(f"Connected roads (via road_indices): {road_indices}")
    print(f"Num roads: {len(road_indices)}")
    print()

    # Finde Straßen mit original_idx in road_indices
    for road in roads:
        if road.get("original_idx") in road_indices:
            coords = road["coords"]
            road_id = road.get("road_id")
            orig_idx = road.get("original_idx")

            if len(coords) > 0:
                dist_first = calculate_distance(coords[0], jp_pos)
                dist_last = calculate_distance(coords[-1], jp_pos)

                print(f"Road {road_id} (original_idx={orig_idx}):")
                print(f"  {len(coords)} points")
                print(f"  First point: {dist_first:.2f}m from JP381")
                print(f"  Last point: {dist_last:.2f}m from JP381")
                print(f"  junction_start_id: {road.get('junction_start_id')}")
                print(f"  junction_end_id: {road.get('junction_end_id')}")
                print()
