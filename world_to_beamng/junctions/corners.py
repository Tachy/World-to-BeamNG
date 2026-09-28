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


def _arm(road: Dict, end: str) -> Dict:
    coords = np.asarray(road["coords"], dtype=float)
    half = np.asarray(road["half_widths"], dtype=float)
    points = coords if end == "start" else coords[::-1]
    return {"road": road, "end": end, "points": points, "half": float(half[0] if end == "start" else half[-1])}


def _direction(points: np.ndarray, length: float) -> Optional[np.ndarray]:
    from ..geometry.polyline import arc_lengths

    cum = arc_lengths(points[:, :2])
    s = min(length, float(cum[-1]))
    target = np.array([np.interp(s, cum, points[:, 0]), np.interp(s, cum, points[:, 1])])
    d = target - points[0, :2]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else None


def _nodes(roads: Sequence[Dict], tol: float) -> List[List[Dict]]:
    """Arms (road ends) grouped by position; only groups with three or more arms."""
    from scipy.spatial import cKDTree

    arms = [_arm(r, end) for r in roads if len(r["coords"]) >= 2 for end in ("start", "end")]
    if not arms:
        return []
    xy = np.array([a["points"][0, :2] for a in arms])
    tree, seen, groups = cKDTree(xy), set(), []
    for i in range(len(arms)):
        if i in seen:
            continue
        members = [m for m in sorted(tree.query_ball_point(xy[i], tol)) if m not in seen]
        seen.update(members)
        if len(members) >= 3:
            groups.append([arms[m] for m in members])
    return groups


def _fillet(node: np.ndarray, a: Dict, b: Dict, table: Mapping, max_angle_deg: float, rank: Mapping, arc_step: float) -> Optional[Dict]:
    """Fillet between arm `a` and the next arm `b` counter-clockwise; None if it does not fit."""
    from ..geometry.polyline import arc_lengths
    from ..terrain.road_embedding import _project_onto_polyline

    ua, ub = a["u"], b["u"]
    theta = (math.atan2(ub[1], ub[0]) - math.atan2(ua[1], ua[0])) % (2.0 * math.pi)
    if theta < 1e-3 or theta > math.radians(max_angle_deg):
        return None
    na = np.array([-ua[1], ua[0]])  # left of a = towards the corner
    nb = np.array([ub[1], -ub[0]])  # right of b = towards the corner
    pa, pb = node[:2] + na * a["half"], node[:2] + nb * b["half"]
    s, q = np.linalg.solve(np.column_stack([ua, -ub]), pb - pa)
    corner_xy = pa + s * ua
    r = corner_radius(a["road"]["highway"], b["road"]["highway"], table)
    t = r / math.tan(theta / 2.0)
    trim_a, trim_b = float(s + t), float(q + t)
    if min(trim_a, trim_b) < 0.0 or trim_a > arc_lengths(a["points"][:, :2])[-1] or trim_b > arc_lengths(b["points"][:, :2])[-1]:
        return None
    tangent_a, tangent_b = corner_xy + ua * t, corner_xy + ub * t
    center = tangent_a + na * r
    a0 = math.atan2(*(tangent_a - center)[::-1])
    a1 = math.atan2(*(tangent_b - center)[::-1])
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi  # the short way: the arc facing the corner point
    count = max(2, int(math.ceil(abs(sweep) * r / arc_step)) + 1)
    angles = a0 + sweep * np.linspace(0.0, 1.0, count)
    arc_xy = center + r * np.column_stack([np.cos(angles), np.sin(angles)])
    arc_xy[0], arc_xy[-1] = tangent_a, tangent_b

    centerline = np.vstack([a["points"][::-1], b["points"][1:]])
    z = lambda xy: _project_onto_polyline(xy[:, 0], xy[:, 1], centerline[:, 0], centerline[:, 1], centerline[:, 2])
    better = a if (rank.get(a["road"]["highway"], 0), a["half"]) >= (rank.get(b["road"]["highway"], 0), b["half"]) else b
    return {
        "node": node,
        "radius": r,
        "center": center,
        "corner_point": np.array([*corner_xy, float(z(corner_xy[None])[0])]),
        "arc": np.column_stack([arc_xy, z(arc_xy)]),
        "centerline": centerline,
        "surface": better["road"]["surface"],
        "arms": [
            {"road_id": a["road"]["road_id"], "end": a["end"], "side": "left" if a["end"] == "start" else "right", "trim": trim_a},
            {"road_id": b["road"]["road_id"], "end": b["end"], "side": "right" if b["end"] == "start" else "left", "trim": trim_b},
        ],
    }


def find_junction_corners(
    roads: Sequence[Dict],
    table: Mapping,
    endpoint_tol: float,
    max_angle_deg: float,
    rank: Mapping,
    direction_length: float = 5.0,
    arc_step: float = 0.5,
) -> List[Dict]:
    """
    Fillet corners of all junction nodes (three or more arms) of `roads`.

    Args:
        roads: [{"road_id", "coords" (N, 3), "half_widths" (N,), "highway", "surface"}] - only roads whose ends may form
            junction corners (see junction_roads())
        table: "junction_corners" section of data/osm_to_beamng.json
        endpoint_tol: road ends closer than this form one node, in meters
        max_angle_deg: corners with a wider opening angle get no fillet
        rank: highway -> rank; the corner is filled with the surface of the higher-ranked arm (tie: wider arm)

    Returns:
        Corner dicts {"node", "radius", "center", "corner_point", "arc" (M, 3) from tangent A to tangent B,
        "centerline" (both arm centerlines through the node), "surface", "arms": [{"road_id", "end", "side", "trim"}] x 2}
        - "side" is the road side (drawing direction) facing the corner, "trim" the distance along that kerb line from
        the road end to the tangent point
    """
    corners = []
    for arms in _nodes(roads, endpoint_tol):
        for arm in arms:
            arm["u"] = _direction(arm["points"], direction_length)
        arms = sorted((a for a in arms if a["u"] is not None), key=lambda a: math.atan2(a["u"][1], a["u"][0]))
        if len(arms) < 3:
            continue
        node = np.mean([a["points"][0] for a in arms], axis=0)
        for k, a in enumerate(arms):
            corner = _fillet(node, a, arms[(k + 1) % len(arms)], table, max_angle_deg, rank, arc_step)
            if corner is not None:
                corners.append(corner)
    return corners


def junction_roads(road_dicts: Sequence[Dict], road_props, excluded_highways, excluded_surfaces) -> List[Dict]:
    """
    Input of find_junction_corners() from the road dicts (road_slope_polygons_2d): surface roads with a visible
    DecalRoad only - footways/paths/tracks, roads without a DecalRoad (the aerial photo shows them) and structures form no
    corners. Widths follow the blended "width_nodes" where present, else the OSM mapper width.
    """
    result = []
    for poly in road_dicts:
        tags = poly.get("osm_tags") or {}
        if poly.get("structure_type", "surface") != "surface" or tags.get("highway") in excluded_highways:
            continue
        props = road_props(poly)
        if props.get("internal_name") in excluded_surfaces:
            continue
        nodes = poly.get("width_nodes")
        if nodes is not None:
            nodes = np.asarray(nodes, dtype=float)
            coords, half = nodes[:, :3], nodes[:, 3] / 2.0
        else:
            coords = np.asarray(poly["trimmed_centerline"], dtype=float)[:, :3]
            half = np.full(len(coords), float(props["width"]) / 2.0)
        result.append({"road_id": poly.get("road_id"), "coords": coords, "half_widths": half,
                       "highway": tags.get("highway", ""), "surface": props.get("internal_name", "asphalt_road_standard")})
    return result


def _sector(corner: Dict, width: float) -> np.ndarray:
    """Sidewalk band behind the arc: between the arc (radius r) and radius r - width, towards the fillet centre."""
    arc = corner["arc"][:, :2]
    center = np.asarray(corner["center"], dtype=float)
    inner = center + (arc - center) * (corner["radius"] - width) / corner["radius"]
    return np.vstack([arc, inner[::-1]])


def corner_has_sidewalks(corner: Dict, sidewalk_sides_by_id: Mapping) -> bool:
    """Both arms have a sidewalk on the side facing the corner."""
    return all(arm["side"] in (sidewalk_sides_by_id.get(arm["road_id"]) or {}) for arm in corner["arms"])


def corner_embed_roads(corners: Sequence[Dict], sidewalk_sides_by_id: Mapping, sidewalk_width: float) -> List[Dict]:
    """
    Fill polygons (and the sidewalk band of corners with a sidewalk on both arms) as road dicts for
    embed_roads_into_heightmap() and union_road_surfaces(): "road_polygon" + "trimmed_centerline" (both arm centerlines,
    so every cell takes the height of the nearest arm).
    """
    result = []
    for corner in corners:
        fill = np.vstack([corner["corner_point"][None, :2], corner["arc"][:, :2]])
        result.append({"road_polygon": fill, "trimmed_centerline": corner["centerline"]})
        if corner_has_sidewalks(corner, sidewalk_sides_by_id):
            result.append({"road_polygon": _sector(corner, sidewalk_width), "trimmed_centerline": corner["centerline"]})
    return result
