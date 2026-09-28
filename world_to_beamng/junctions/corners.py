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


def _local(points: np.ndarray, length: float) -> np.ndarray:
    """The first `length` meters of an arm polyline (from the node)."""
    from shapely.geometry import LineString
    from shapely.ops import substring

    from ..geometry.polyline import arc_lengths

    cum = arc_lengths(points[:, :2])
    if cum[-1] <= length:
        return points
    xy = np.asarray(substring(LineString(points[:, :2]), 0.0, length).coords)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return np.column_stack([xy, np.interp(s, cum, points[:, 2])])


def _offset(line, distance: float):
    """Offset line (positive = left); None if it is not one line. GEOS sometimes returns a straight offset as several
    pieces that merely touch - those are merged back into one line."""
    from shapely import line_merge

    offset = line.offset_curve(distance)
    if offset.geom_type == "MultiLineString":
        offset = line_merge(offset)
    return offset if offset.geom_type == "LineString" and not offset.is_empty else None


def corner_height(corner: Dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Height inside a corner: the heights of both arms (projected onto their nearby centerline) blended by the angle
    around the corner point - arm A's height along its edge, arm B's along its edge, a smooth transition in between
    (the nearest-arm height would jump on the bisector where the arms have different grades).
    """
    from ..terrain.road_embedding import _project_onto_polyline

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    line_a, line_b = corner["arm_lines"]
    za = _project_onto_polyline(x, y, line_a[:, 0], line_a[:, 1], line_a[:, 2])
    zb = _project_onto_polyline(x, y, line_b[:, 0], line_b[:, 1], line_b[:, 2])
    ua, p = corner["u_a"], corner["corner_point"]
    dx, dy = x - p[0], y - p[1]
    phi = np.arctan2(ua[0] * dy - ua[1] * dx, ua[0] * dx + ua[1] * dy)  # angle from arm A's direction, counter-clockwise
    w = np.where(dx * dx + dy * dy < 1e-12, 0.5, np.clip(phi / corner["theta"], 0.0, 1.0))
    return (1.0 - w) * za + w * zb


def _fillet(node: np.ndarray, a: Dict, b: Dict, r: float, max_angle_deg: float, rank: Mapping, arc_step: float) -> Optional[Dict]:
    """
    Fillet of radius `r` between arm `a` and the next arm `b` counter-clockwise; None if it does not fit. The circle is fitted to the
    real kerb lines (the arm centerlines offset by their half width), so it also sits right on curved arms: its centre is
    where both centerlines offset by half width + r intersect, the tangent points are its feet on the kerb lines.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import substring

    ua, ub = a["u"], b["u"]
    theta = (math.atan2(ub[1], ub[0]) - math.atan2(ua[1], ua[0])) % (2.0 * math.pi)
    if theta < 1e-3 or theta > math.radians(max_angle_deg):
        return None
    # Corner point (fan apex of the fill): intersection of both kerb lines at the node, as straight edges - at an acute
    # corner it lies far out, where the two carriageways part
    na, nb = np.array([-ua[1], ua[0]]), np.array([ub[1], -ub[0]])
    pa, pb = node[:2] + na * a["half"], node[:2] + nb * b["half"]
    s, q = np.linalg.solve(np.column_stack([ua, -ub]), pb - pa)
    corner_xy = pa + s * ua

    # only the part of the arms near the node: a tight bend far away must not spoil the offset lines
    near = max(s, q, 0.0) + r / math.tan(theta / 2.0) + 2.0 * (a["half"] + b["half"] + r) + 10.0
    line_a, line_b = LineString(_local(a["points"], near)[:, :2]), LineString(_local(b["points"], near)[:, :2])
    kerb_a, kerb_b = _offset(line_a, a["half"]), _offset(line_b, -b["half"])  # corner: left of a, right of b
    centre_a, centre_b = _offset(line_a, a["half"] + r), _offset(line_b, -(b["half"] + r))
    if kerb_a is None or kerb_b is None or centre_a is None or centre_b is None:
        return None  # the offset folds over itself (tight bend right at the node): no clean kerb to fit to
    hits = centre_a.intersection(centre_b)
    hits = [g for g in getattr(hits, "geoms", [hits]) if g.geom_type == "Point"]
    if not hits:
        return None  # an arm is too short for the arc
    center_pt = min(hits, key=lambda g: g.distance(Point(node[:2])))
    trim_a, trim_b = kerb_a.project(center_pt), kerb_b.project(center_pt)
    if not (0.0 < trim_a < kerb_a.length and 0.0 < trim_b < kerb_b.length):
        return None
    center = np.array([center_pt.x, center_pt.y])
    tangent_a = np.asarray(kerb_a.interpolate(trim_a).coords[0])
    tangent_b = np.asarray(kerb_b.interpolate(trim_b).coords[0])

    a0 = math.atan2(*(tangent_a - center)[::-1])
    a1 = math.atan2(*(tangent_b - center)[::-1])
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi  # the short way: the arc facing the corner point
    count = max(2, int(math.ceil(abs(sweep) * r / arc_step)) + 1)
    angles = a0 + sweep * np.linspace(0.0, 1.0, count)
    arc_xy = center + r * np.column_stack([np.cos(angles), np.sin(angles)])
    arc_xy[0], arc_xy[-1] = tangent_a, tangent_b

    # Fill outline from the corner point along kerb A to the arc, around it and back along kerb B
    edge_a = np.asarray(substring(kerb_a, kerb_a.project(Point(corner_xy)), trim_a).coords)
    edge_b = np.asarray(substring(kerb_b, kerb_b.project(Point(corner_xy)), trim_b).coords)[::-1]
    rim_xy = np.vstack([edge_a[:-1], arc_xy, edge_b[1:]])

    reach = max(trim_a, trim_b) + 2.0 * max(a["half"], b["half"])
    corner = {
        "node": node,
        "radius": r,
        "center": center,
        "u_a": ua,
        "theta": theta,
        "corner_point": corner_xy,
        "arm_lines": (_local(a["points"], reach), _local(b["points"], reach)),
    }
    height = lambda xy: corner_height(corner, xy[:, 0], xy[:, 1])
    # The joining road (lower rank, then the narrower arm) gives the fill its surface: the flare belongs to it
    joining = min((a, b), key=lambda arm: (rank.get(arm["road"]["highway"], 0), arm["half"], str(arm["road"]["road_id"])))
    corner.update(
        corner_point=np.array([*corner_xy, float(height(corner_xy[None])[0])]),
        arc=np.column_stack([arc_xy, height(arc_xy)]),
        rim=np.column_stack([rim_xy, height(rim_xy)]),
        centerline=np.vstack([corner["arm_lines"][0][::-1], corner["arm_lines"][1][1:]]),
        surface=joining["road"]["surface"],
        arms=[
            {"road_id": a["road"]["road_id"], "end": a["end"], "side": "left" if a["end"] == "start" else "right", "trim": float(trim_a)},
            {"road_id": b["road"]["road_id"], "end": b["end"], "side": "right" if b["end"] == "start" else "left", "trim": float(trim_b)},
        ],
    )
    return corner


def _fit(candidate, table: Mapping, max_angle_deg: float, rank: Mapping, arc_step: float, factors, min_radius: float, start: int,
         acute_angle_deg: float = 0.0):
    """The corner with the first radius step from `start` on that fits: (corner, step) or (None, len(factors))."""
    node, a, b, has_kerb = candidate
    base = corner_radius(a["road"]["highway"], b["road"]["highway"], table)
    min_radius = max(min_radius, has_kerb)
    theta = (math.atan2(a["u"][1], a["u"][0]) - math.atan2(b["u"][1], b["u"][0])) % (2.0 * math.pi)
    if 2.0 * math.pi - theta < math.radians(acute_angle_deg):
        # Two roads leaving almost in parallel: the table radius would only touch both kerbs far out and pave the whole
        # wedge between them (in reality a traffic island) - only the tip is rounded
        base = 0.0
    tried = set()
    for step in range(start, len(factors)):
        radius = max(min_radius, base * factors[step])
        if radius in tried:
            continue
        tried.add(radius)
        corner = _fillet(node, a, b, radius, max_angle_deg, rank, arc_step)
        if corner is not None:
            return corner, step
    return None, len(factors)


def find_junction_corners(
    roads: Sequence[Dict],
    table: Mapping,
    endpoint_tol: float,
    max_angle_deg: float,
    rank: Mapping,
    direction_length: float = 5.0,
    arc_step: float = 0.5,
    radius_factors: Sequence[float] = (1.0,),
    min_radius: float = 0.5,
    kerb_min_radius: float = 2.0,
    sidewalk_sides_by_id: Optional[Mapping] = None,
    acute_angle_deg: float = 0.0,
) -> List[Dict]:
    """
    Fillet corners of all junction nodes (three or more arms) of `roads`.

    Where the table radius does not fit (short arm between two close junctions, acute angle), the radius is reduced step
    by step (`radius_factors` of the table radius, never below `min_radius`; where both arms have a sidewalk facing the
    corner, a kerb runs around the arc and the radius stays at least `kerb_min_radius`, so the sidewalk band behind it
    keeps its width). Where the arcs at both ends of a road side
    together need more than the road is long, the larger one is reduced further (dropped at the last step) - otherwise
    the two fills would overlap. Corners sharper than `acute_angle_deg` get the smallest radius right away: only their
    tip is rounded, the wedge between the two roads stays terrain.

    Args:
        roads: [{"road_id", "coords" (N, 3), "half_widths" (N,), "highway", "surface"}] - only roads whose ends may form
            junction corners (see junction_roads())
        table: "junction_corners" section of data/osm_to_beamng.json
        endpoint_tol: road ends closer than this form one node, in meters
        max_angle_deg: corners with a wider opening angle get no fillet
        rank: highway -> rank; the corner is filled with the surface of the joining road - the lower-ranked arm (tie:
            the narrower one)
        sidewalk_sides_by_id: road id -> {side: surface} of the roads with a sidewalk

    Returns:
        Corner dicts {"node", "radius", "center", "corner_point", "arc" (M, 3) from tangent A to tangent B, "rim" (fill
        outline from the corner point along kerb A, the arc and back along kerb B), "centerline" (the nearby parts of
        both arm centerlines through the node), "arm_lines", "u_a", "theta" (for corner_height()), "surface",
        "arms": [{"road_id", "end", "side", "trim"}] x 2}
    """
    from ..geometry.polyline import arc_lengths

    sides = sidewalk_sides_by_id or {}
    candidates = []
    for arms in _nodes(roads, endpoint_tol):
        for arm in arms:
            arm["u"] = _direction(arm["points"], direction_length)
        arms = sorted((a for a in arms if a["u"] is not None), key=lambda a: math.atan2(a["u"][1], a["u"][0]))
        if len(arms) < 3:
            continue
        node = np.mean([a["points"][0] for a in arms], axis=0)
        for k, a in enumerate(arms):
            b = arms[(k + 1) % len(arms)]
            facing = ((a, "left" if a["end"] == "start" else "right"), (b, "right" if b["end"] == "start" else "left"))
            has_kerb = all(side in (sides.get(arm["road"]["road_id"]) or {}) for arm, side in facing)
            candidates.append((node, a, b, kerb_min_radius if has_kerb else 0.0))

    fit = lambda i, start: _fit(candidates[i], table, max_angle_deg, rank, arc_step, radius_factors, min_radius, start, acute_angle_deg)
    fitted = [fit(i, 0) for i in range(len(candidates))]
    lengths = {r["road_id"]: float(arc_lengths(np.asarray(r["coords"])[:, :2])[-1]) for r in roads if len(r["coords"]) >= 2}

    # Both ends of a road side must share it: shrink the larger arc until the trims fit the road
    while True:
        by_side: Dict[tuple, List] = {}
        for i, (corner, _) in enumerate(fitted):
            for arm in corner["arms"] if corner is not None else ():
                by_side.setdefault((arm["road_id"], arm["side"]), []).append((arm["trim"], i))
        conflict = next(
            (max(entries) for key, entries in by_side.items() if len(entries) > 1 and sum(t for t, _ in entries) > lengths[key[0]]),
            None,
        )
        if conflict is None:
            break
        i = conflict[1]
        fitted[i] = fit(i, fitted[i][1] + 1)

    return [corner for corner, _ in fitted if corner is not None]


def junction_roads(road_dicts: Sequence[Dict], road_props, excluded_highways) -> List[Dict]:
    """
    Input of find_junction_corners() from the road dicts (road_slope_polygons_2d): surface roads; footways/paths and
    structures form no corners. Roads without a DecalRoad (dirt tracks - the aerial photo shows them) do take part: their
    corners shape the terrain, export_junctions() just does not draw them. Widths follow the blended "width_nodes" where
    present, else the OSM mapper width.
    """
    result = []
    for poly in road_dicts:
        tags = poly.get("osm_tags") or {}
        if poly.get("structure_type", "surface") != "surface" or tags.get("highway") in excluded_highways:
            continue
        props = road_props(poly)
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


def corner_embed_roads(corners: Sequence[Dict], sidewalk_sides_by_id: Mapping, sidewalk_width: float, margin: float = 0.0) -> List[Dict]:
    """
    Fill polygons (and the sidewalk band of corners with a sidewalk on both arms) as road dicts for
    embed_roads_into_heightmap() and union_road_surfaces(), with the blended corner height ("height_at", see
    corner_height()). The fill is widened by `margin` (at least one raster cell diagonal): the terrain triangles of the
    cells just outside the arc would otherwise rise through the slightly lifted fill mesh.
    """
    from functools import partial

    from shapely.geometry import Polygon

    result = []
    for corner in corners:
        height_at = partial(corner_height, corner)
        fill = Polygon(np.vstack([corner["corner_point"][None, :2], corner["rim"][:, :2]]))
        if margin > 0:
            fill = fill.buffer(margin)
        if fill.geom_type != "Polygon":
            fill = max(fill.geoms, key=lambda g: g.area)
        result.append({"road_polygon": np.asarray(fill.exterior.coords)[:-1], "trimmed_centerline": corner["centerline"], "height_at": height_at})
        if corner_has_sidewalks(corner, sidewalk_sides_by_id):
            result.append({"road_polygon": _sector(corner, sidewalk_width), "trimmed_centerline": corner["centerline"], "height_at": height_at})
    return result
