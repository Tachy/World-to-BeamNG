#!/usr/bin/env python3
"""Debug script - Test junction geometry directly"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

print("DEBUG: Starting junction geometry test...")
sys.stdout.flush()

# Load OSM data from cache
cache_file = "cache/osm_all_392ceb2f01cd.json"
if os.path.exists(cache_file):
    print(f"Loading OSM data from {cache_file}...")
    with open(cache_file, "r") as f:
        osm_data = json.load(f)
    print(f"  Loaded {len(osm_data)} elements")

    # Count nodes and ways
    nodes = [e for e in osm_data if e.get("type") == "node"]
    ways = [e for e in osm_data if e.get("type") == "way"]
    print(f"  Nodes: {len(nodes)}, Ways: {len(ways)}")
else:
    print(f"ERROR: {cache_file} not found!")
    sys.exit(1)

print("\nTest complete!")
