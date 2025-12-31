#!/usr/bin/env python3
"""Test script to validate Junction geometry without full generator run"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from world_to_beamng.geometry.junction_vertices import build_junction_vertices
from world_to_beamng.geometry.junctions import find_junctions
from world_to_beamng.io.cache import load_osm_data
import numpy as np

# Load minimal data
print("Loading OSM data for junction test...")
roads, ways = load_osm_data()
print(f"  Loaded {len(roads)} roads, {len(ways)} ways")

# Find junctions
print("\nFinding junctions...")
junctions = find_junctions(roads)
print(f"  Found {len(junctions)} junctions")

# Build junction vertices for first 3 junctions
print(f"\nBuilding junction geometry for first 3 junctions...")
debug_junction_idx = 0  # Print debug info for junction 0

if len(junctions) > 0:
    # Simulate what the main generator does
    junction_dict = {}
    for junction_idx, junction_pos in enumerate(junctions[:3]):
        print(
            f"\n=== Junction {junction_idx} @ ({junction_pos[0]:.1f}, {junction_pos[1]:.1f}) ==="
        )

        # Get roads at this junction
        roads_at_junction = [r for r in roads if len(r.junctions) > 0]
        matching_roads = []
        for road in roads_at_junction:
            for j_idx in road.junctions:
                if j_idx == junction_idx:
                    matching_roads.append(road)
                    break

        print(f"  Roads at junction: {len(matching_roads)}")
        if len(matching_roads) >= 2:
            for i, road in enumerate(matching_roads[:4]):
                print(f"    Road {i}: {road.osm_id}, points: {len(road.points)}")

print("\nDone!")
