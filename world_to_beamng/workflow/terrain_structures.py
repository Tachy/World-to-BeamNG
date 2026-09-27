"""
Structure side of the terrain workflow: bridge groups/stems/footprints, tunnel and gallery plans with their terrain
embedding inputs, tunnel zones/lights and the roadblocks at tunnels leaving the map.
"""

from typing import Dict, List, Tuple

import numpy as np

from .. import config


def _dropped_tunnel_road_ids(all_piece_ids: set, kept_plans: List[Dict]) -> frozenset:
    """Original (pre-chaining) tunnel road ids that belonged to a chain shape_terrain_for_tunnels() dropped (no
    reachable portal at all): `all_piece_ids` from every plan.piece_ids BEFORE that filtering ran, `kept_plans` the
    same list AFTER (it filters in place - capture all_piece_ids before calling it)."""
    kept_ids = {piece_id for plan in kept_plans for piece_id in plan.get("piece_ids", [plan["id"]])}
    return frozenset(all_piece_ids - kept_ids)


def _bridge_footprints(bridge_roads: List[Dict], extra: float) -> List[Dict]:
    """Bridge outlines for vegetation rules: {"road_polygon" (carriageway polygon widened by `extra` - curbs plus margin),
    "trimmed_centerline"} per bridge (see road_embedding.near_deck_mask())."""
    from shapely.geometry import Polygon

    footprints = []
    for road in bridge_roads:
        polygon = Polygon(np.asarray(road["road_polygon"], dtype=float)[:, :2]).buffer(extra, join_style="mitre")
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        footprints.append({"road_polygon": np.asarray(polygon.exterior.coords), "trimmed_centerline": road["trimmed_centerline"]})
    return footprints


def _bridge_photo_areas(structure_roads: List[Dict]) -> List[np.ndarray]:
    """Outlines ((N, 2) arrays) of the bridges plus curbs and config.BRIDGE_PHOTO_FILL_MARGIN - the aerial photo shows
    the deck there, see io/aerial_bridge_fill.py. Empty with config.AERIAL_BRIDGE_RETOUCH switched off."""
    if not config.AERIAL_BRIDGE_RETOUCH:
        return []
    bridges = [r for r in structure_roads if r.get("structure_type") == "bridge"]
    extra = config.BRIDGE_CURB_WIDTH + config.BRIDGE_PHOTO_FILL_MARGIN
    return [np.asarray(footprint["road_polygon"])[:, :2] for footprint in _bridge_footprints(bridges, extra)]


def _bridge_groups(bridge_roads: List[Dict], reach: float, endpoint_tol: float = 0.5) -> Dict[object, int]:
    """
    road id -> group index for bridges that form ONE structure (bridges/bridge_mesh.py build_bridge_group_mesh()): the
    trunk and the branches of a lane split on a bridge (see geometry/lane_splits.py), plus bridge pieces that continue
    them at a joint at most `reach` meters from the split node (a branch runs beside the others until there). Bridges
    outside any group keep their own deck.
    """
    def split_nodes(road):
        nodes = list(road.get("lane_split_trunk_nodes", ()))
        if road.get("lane_split_branch"):
            nodes.append(road["lane_split_branch"]["node"])
        return [(round(float(x), 2), round(float(y), 2)) for x, y in nodes]

    def ends(road):
        line = np.asarray(road["trimmed_centerline"], dtype=float)
        return [line[0, :2], line[-1, :2]]

    group_of: Dict[int, int] = {}  # road index -> group
    group_nodes: Dict[int, set] = {}
    for index, road in enumerate(bridge_roads):
        nodes = split_nodes(road)
        if not nodes:
            continue
        joined = {group_of_node for group_of_node, node_set in group_nodes.items() if node_set & set(nodes)}
        target = min(joined) if joined else len(group_nodes)
        group_nodes.setdefault(target, set()).update(nodes)
        for other in joined - {target}:
            group_nodes[target] |= group_nodes.pop(other)
            for member, group in group_of.items():
                if group == other:
                    group_of[member] = target
        group_of[index] = target

    changed = True
    while changed:
        changed = False
        for index, road in enumerate(bridge_roads):
            if index in group_of:
                continue
            for point in ends(road):
                for member, group in list(group_of.items()):
                    touches = any(np.linalg.norm(point - end) <= endpoint_tol for end in ends(bridge_roads[member]))
                    near = min(np.hypot(point[0] - x, point[1] - y) for x, y in group_nodes[group]) <= reach
                    if touches and near:
                        group_of[index] = group
                        changed = True
                        break
                if index in group_of:
                    break

    sizes: Dict[int, int] = {}
    for group in group_of.values():
        sizes[group] = sizes.get(group, 0) + 1
    return {bridge_roads[i]["road_id"]: g for i, g in group_of.items() if sizes[g] >= 2}


def _bridge_stems(bridge_roads: List[Dict]) -> Dict[object, Dict]:
    """
    road id -> stem (bridges/bridge_mesh.py build_bridge_group_mesh()) for every branch of a lane split whose trunk is a
    bridge too: the trunk's deck goes on past the node over the stem length before it is cut into the branches. A split
    on the ground has no stem - its branches that become bridges further on stay separate bridges.
    """
    def key(node):
        return (round(float(node[0]), 2), round(float(node[1]), 2))

    trunks = {}
    for road in bridge_roads:
        for node in road.get("lane_split_trunk_nodes", ()):
            trunks[key(node)] = road
    stems = {}
    for road in bridge_roads:
        mark = road.get("lane_split_branch")
        if not mark or key(mark["node"]) not in trunks or mark.get("stem_length", 0.0) < 1.0 or len(mark.get("stem_path", ())) < 2:
            continue
        trunk = trunks[key(mark["node"])]
        internal_name = config.OSM_MAPPER.get_road_properties(trunk.get("osm_tags", {})).get("internal_name", "road_default")
        stems[road["road_id"]] = {
            "path": [tuple(p) for p in mark["stem_path"]], "width": float(mark["trunk_width"]),
            "deck_material": f"{internal_name}_structure",
        }
    return stems


def _structure_items(structure_road_polygons: List[Dict], structure_type: str) -> List[Dict]:
    """Tunnel/gallery inputs for tunnels/*: {"id", "coords", "width", "floor_material", "osm_tags"} per road
    with the given structure_type."""
    items = []
    for road in structure_road_polygons:
        if road.get("structure_type") != structure_type:
            continue
        properties = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
        items.append(
            {
                "id": road["road_id"],
                "open_side": road.get("open_side"),  # Galleries: determined by _gallery_embedding()
                "coords": road["trimmed_centerline"],
                "width": properties["width"],
                "floor_material": f"{properties.get('internal_name', 'road_default')}_structure",
                "osm_tags": road.get("osm_tags", {}),
            }
        )
    return items


def _plan_tunnels(structure_road_polygons: List[Dict]) -> List[Dict]:
    """Tunnel plans (tunnels/tunnel_portal.py::plan_tunnels()) with the galleries as possible transitions. Only roads
    and cycle paths get a tunnel structure - path "tunnels" in the mountains (e.g. fortress adits) are dropped
    (config.TUNNEL_EXCLUDED_HIGHWAYS)."""
    from ..tunnels.tunnel_portal import plan_tunnels

    tunnels = [
        t for t in _structure_items(structure_road_polygons, "tunnel")
        if t["osm_tags"].get("highway") not in config.TUNNEL_EXCLUDED_HIGHWAYS
    ]
    return plan_tunnels(
        tunnels,
        segment_step=config.TUNNEL_SEGMENT_STEP,
        flat_depth=config.TUNNEL_PORTAL_FLAT_DEPTH,
        length=config.TUNNEL_PORTAL_LENGTH,
        galleries=_structure_items(structure_road_polygons, "gallery"),
        gallery_height=config.GALLERY_HEIGHT,
        gallery_roof_thickness=config.GALLERY_ROOF_THICKNESS,
        gallery_wall_thickness=config.GALLERY_WALL_THICKNESS,
        transition_tol=config.TUNNEL_TRANSITION_ENDPOINT_TOL,
        shell_ratio=config.TUNNEL_SHELL_RATIO,
        tilt_deg=config.TUNNEL_PORTAL_TILT_DEG,
        collar_ratio=config.TUNNEL_PORTAL_COLLAR_RATIO,
        collar_min_side=config.TUNNEL_PORTAL_COLLAR_MIN_SIDE,
        curb_width=config.TUNNEL_CURB_WIDTH,
        curb_height=config.TUNNEL_CURB_HEIGHT,
        edge_height=config.TUNNEL_EDGE_HEIGHT,
        max_arc_deg=config.TUNNEL_MAX_ARC_DEG,
    )


def _roadblock_items(
    tunnel_plans: List[Dict], heights: np.ndarray, origin_x: float, origin_y: float, bounds, surface_roads: List[Dict]
) -> List[Dict]:
    """Roadblocks in front of tunnel entrances whose tunnel extends beyond the map border (tunnels/roadblock.py), with
    the height of the finished terrain at each element: [{"name", "position" (x, y, z), "rotation_matrix"}, ...].
    Entrance = portal on the endpoint of a surface road (`surface_roads`, road_slope_polygons_2d dicts)."""
    from ..terrain.road_embedding import sample_heightmap_bilinear
    from ..tunnels.roadblock import plan_roadblocks

    blocks = plan_roadblocks(
        tunnel_plans,
        bounds=bounds,
        edge_margin=config.MAP_EDGE_TUNNEL_MARGIN,
        distance=config.ROADBLOCK_DISTANCE,
        side_margin=config.ROADBLOCK_SIDE_MARGIN,
        spacing=config.ROADBLOCK_SPACING,
        entrances=[
            (float(p[0]), float(p[1]))
            for road in surface_roads
            if road.get("trimmed_centerline") is not None and len(road["trimmed_centerline"]) >= 2
            for p in (road["trimmed_centerline"][0], road["trimmed_centerline"][-1])
        ],
        entrance_tol=config.TUNNEL_TRANSITION_ENDPOINT_TOL,
    )
    if not blocks:
        return []
    xy = np.array([b["xy"] for b in blocks], dtype=float)
    z = sample_heightmap_bilinear(heights, origin_x, origin_y, config.TERRAIN_SQUARE_SIZE, xy)
    return [
        {"name": b["name"], "position": (float(x), float(y), float(zz)), "rotation_matrix": b["rotation_matrix"]}
        for b, (x, y), zz in zip(blocks, xy, z)
    ]


def _gallery_embedding(road: Dict, ground_at) -> Tuple[Dict, set]:
    """
    Embankment of a gallery: fixed width GALLERY_VALLEY_SLOPE_WIDTH on the valley side, only a flat fringe
    (GALLERY_MOUNTAIN_EMBED_MARGIN) at the wall on the mountain side. The open side comes from
    tunnels/gallery_mesh.gallery_open_side() (tag, otherwise terrain) and is stored in road["open_side"] - the
    gallery mesh uses the same side (_structure_items() -> build_galleries()).

    Returns:
        (slope_width_override, flat_shoulder_sides) for build_road_embankment_profiles()
    """
    from ..tunnels.gallery_mesh import gallery_open_side

    width = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"]
    open_side = gallery_open_side(road.get("osm_tags", {}), road["trimmed_centerline"], ground_at, width)
    road["open_side"] = open_side
    mountain = "left" if open_side == "right" else "right"
    valley_widths = _gallery_valley_slope_widths(road["trimmed_centerline"], ground_at, width / 2.0, open_side)
    return {open_side: valley_widths, mountain: config.GALLERY_MOUNTAIN_EMBED_MARGIN}, {mountain}


def _gallery_embankment_cuts(surface_roads: List[Dict], gallery_roads: List[Dict], tol: float) -> None:
    """
    Lets the terrain adaptation of galleries end flush with the gallery, and that of the approach roads flush with the
    transition (road dict field "embankment_cuts", see road_embedding.build_road_embankment_profiles()). Without cuts
    the corridor of the last edge point reaches around a road end like a round cap; the gallery (blended after the
    surface roads) thus overwrote the approach road's embankment with its valley-side slope - a pit of up to 3 m beside
    the road just before the gallery (in game 2026-09-26).

    Each gallery end gets a cut; a surface road end within `tol` of it gets the same line with the opposite normal. The
    line is the bisector of both end directions, so no wedge stays open and nothing overlaps at a kink; a gallery end
    without an approach (e.g. transition into a tunnel) is cut perpendicular. In place.
    """
    def ends(road):
        points = np.asarray(road["trimmed_centerline"], dtype=float)[:, :2]
        for index, neighbour in ((0, 1), (-1, -2)):
            outward = points[index] - points[neighbour]
            yield points[index], outward / np.linalg.norm(outward)

    road_ends = [(road, point, outward) for road in surface_roads if len(road["trimmed_centerline"]) >= 2
                 for point, outward in ends(road)]
    for gallery in gallery_roads:
        if len(gallery["trimmed_centerline"]) < 2:
            continue
        cuts = []
        for point, outward in ends(gallery):
            normal = outward
            for road, road_point, road_outward in road_ends:
                if np.hypot(*(road_point - point)) <= tol:
                    bisector = outward - road_outward
                    if np.linalg.norm(bisector) > 1e-9:
                        normal = bisector / np.linalg.norm(bisector)
                    road.setdefault("embankment_cuts", []).append(
                        ((float(point[0]), float(point[1])), (float(-normal[0]), float(-normal[1])))
                    )
            cuts.append(((float(point[0]), float(point[1])), (float(normal[0]), float(normal[1]))))
        gallery["embankment_cuts"] = cuts


def _gallery_valley_slope_widths(centerline, ground_at, half_width: float, open_side: str) -> np.ndarray:
    """
    Valley-side embankment width per centerline point of a gallery: above the gallery the DGM shows its roof (~5 m above
    the road surface), often several meters beyond the road edge. A fixed reference (edge +
    GALLERY_VALLEY_SLOPE_WIDTH) would then still lie on the roof, the embankment would rise toward the valley and break
    off behind it (spikes at the long gallery of the Nuova strada). Therefore the search runs toward the valley; the
    first point where the terrain is no more than GALLERY_VALLEY_STRUCTURE_HEIGHT above the road surface is the
    reference - at least GALLERY_VALLEY_SLOPE_WIDTH, at most GALLERY_VALLEY_SEARCH_MAX. If the terrain is higher
    everywhere up to that point (valley side rising), it stays at GALLERY_VALLEY_SLOPE_WIDTH.
    """
    points = np.asarray(centerline, dtype=float)
    xy, z = points[:, :2], points[:, 2]
    tangents = np.gradient(xy, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    left = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    valley = left if open_side == "left" else -left

    distances = np.arange(config.GALLERY_VALLEY_SLOPE_WIDTH, config.GALLERY_VALLEY_SEARCH_MAX + 1e-9, config.GALLERY_VALLEY_SEARCH_STEP)
    widths = np.full(len(xy), config.GALLERY_VALLEY_SLOPE_WIDTH)  # nothing found (valley side higher): minimum width
    found = np.zeros(len(xy), dtype=bool)
    for d in distances:
        probe = xy + valley * (half_width + d)
        below = np.asarray(ground_at(probe[:, 0], probe[:, 1]), dtype=float) <= z + config.GALLERY_VALLEY_STRUCTURE_HEIGHT
        hit = below & ~found
        widths[hit] = d
        found |= hit
    return widths


def _tunnel_zone_items(tunnel_plans: List[Dict]) -> List[Dict]:
    """Zone boxes that darken the tunnel tubes (tunnels/tunnel_zones.py), with the values from the config."""
    from ..tunnels.tunnel_zones import plan_tunnel_zones

    return plan_tunnel_zones(
        tunnel_plans,
        max_length=config.TUNNEL_ZONE_MAX_LENGTH,
        max_deviation=config.TUNNEL_ZONE_MAX_DEVIATION,
        end_overlap=config.TUNNEL_ZONE_END_OVERLAP,
        width_margin=config.TUNNEL_ZONE_WIDTH_MARGIN,
        height_margin=config.TUNNEL_ZONE_HEIGHT_MARGIN,
        portal_inset=config.TUNNEL_ZONE_PORTAL_INSET,
        portal_depth=config.TUNNEL_ZONE_PORTAL_DEPTH,
        entrance_inset=config.TUNNEL_ZONE_ENTRANCE_INSET,
    )


def _tunnel_light_items(tunnel_plans: List[Dict]) -> List[Dict]:
    """SpotLight fixtures inside the tunnel tubes (tunnels/tunnel_lights.py), with the values from the config."""
    from ..tunnels.tunnel_lights import plan_tunnel_lights

    return plan_tunnel_lights(
        tunnel_plans,
        spacing=config.TUNNEL_LIGHT_SPACING,
        start_inset=config.TUNNEL_LIGHT_START_INSET,
        ceiling_margin=config.TUNNEL_LIGHT_CEILING_MARGIN,
        fields=config.TUNNEL_LIGHT_FIELDS,
    )
