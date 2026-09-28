"""
Junction corners with a fillet.

DecalRoads always end flat, so at a junction the corners between two arms are angular. For every pair of angularly
neighbouring arms at a node with three or more arms, a circle of the corner radius is fitted tangent to both carriageway
edges on the corner side; the area between the edge intersection, the two tangent points and the arc is the fill. OSM
has no corner radii, so they come from a table per highway class (the smaller radius of both arms wins).
"""

import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


def corner_radius(highway_a: str, highway_b: str, table: Mapping) -> float:
    """Fillet radius of a corner between two arms: the smaller radius of both highway classes (table: the
    "junction_corners" section of data/osm_to_beamng.json)."""
    radii = table.get("radius_by_highway", {})
    default = float(table.get("default_radius", 6.0))
    return min(float(radii.get(highway_a, default)), float(radii.get(highway_b, default)))
