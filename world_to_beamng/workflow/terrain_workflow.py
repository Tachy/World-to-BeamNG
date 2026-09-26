"""
Terrain export workflow.

Orchestrates the complete terrain export process.
"""

from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from .tile_processor import TileProcessor
from ..progress import PipelineTask

logger = logging.getLogger(__name__)

WATER_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "data" / "water_templates.json"


def make_height_sampler_for_water(heights, origin_x, origin_y):
    """Bilinear elevation lookup on the finished heightmap (as for the vineyard vines)."""
    from ..forest.vineyard_generator import make_height_sampler

    return make_height_sampler(heights, origin_x, origin_y, config.TERRAIN_SQUARE_SIZE)


WATER_BOUNDS_MARGIN = 2.0  # just inside the real data: beyond it the terrain is filled in


def water_bounds(grid_bounds_local):
    """(xmin, ymin, xmax, ymax) of the terrain with real elevation data in which water may be created."""
    x_min, x_max, y_min, y_max = grid_bounds_local
    m = WATER_BOUNDS_MARGIN
    return (x_min + m, y_min + m, x_max - m, y_max - m)



def _gets_decal_road(road: Dict, dropped_ids: frozenset = frozenset()) -> bool:
    """Whether a road is exported as a DecalRoad: surface roads always; bridges, galleries and tunnels only with
    config.STRUCTURE_AI_ROADS (as an invisible DecalRoad for the AI road network) - except tunnels that get no tube
    (TUNNEL_EXCLUDED_HIGHWAYS) or that belonged to a chain dropped for having no reachable portal at all (a
    pass-through tunnel like the Gotthard road tunnel, see _dropped_tunnel_road_ids()): its AI road would otherwise
    float underground, disconnected from the surface at both ends."""
    structure_type = road.get("structure_type", "surface")
    if structure_type == "surface":
        return True
    if not config.STRUCTURE_AI_ROADS:
        return False
    if structure_type == "tunnel":
        return road.get("road_id") not in dropped_ids and (road.get("osm_tags") or {}).get("highway") not in config.TUNNEL_EXCLUDED_HIGHWAYS
    return True


def _dropped_tunnel_road_ids(all_piece_ids: set, kept_plans: List[Dict]) -> frozenset:
    """Original (pre-chaining) tunnel road ids that belonged to a chain shape_terrain_for_tunnels() dropped (no
    reachable portal at all): `all_piece_ids` from every plan.piece_ids BEFORE that filtering ran, `kept_plans` the
    same list AFTER (it filters in place - capture all_piece_ids before calling it)."""
    kept_ids = {piece_id for plan in kept_plans for piece_id in plan.get("piece_ids", [plan["id"]])}
    return frozenset(all_piece_ids - kept_ids)


def _guardrail_instances(specs: List[Tuple[Dict, Dict, List]], node_lists: List[List[List[float]]], mesh_data: Dict) -> List[Dict]:
    """Guard rail forest items along the surface roads (geometry/guardrails.py), planned on the finished heightmap.
    `specs`/`node_lists` as in export_decal_roads(); structures are left out, so a rail ends at a structure."""
    from ..geometry.guardrails import place_guardrail_items, plan_guardrail_runs
    from ..io.guardrail_assets import GUARDRAIL_ITEMS
    from ..terrain.road_embedding import sample_heightmap_bilinear

    heights = mesh_data["heightmap"]
    origin_x, origin_y = mesh_data["terrain_origin_x"], mesh_data["terrain_origin_y"]

    def height_at(x, y):
        xy = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        return sample_heightmap_bilinear(heights, origin_x, origin_y, config.TERRAIN_SQUARE_SIZE, xy)

    surface = [(poly, nodes) for (poly, _, _), nodes in zip(specs, node_lists) if poly.get("structure_type", "surface") == "surface"]
    runs = plan_guardrail_runs(
        [nodes for _, nodes in surface],
        [(poly.get("osm_tags") or {}).get("highway") not in config.GUARDRAIL_EXCLUDED_HIGHWAYS for poly, _ in surface],
        height_at,
        probe_offset=config.GUARDRAIL_PROBE_OFFSET,
        min_drop=config.GUARDRAIL_MIN_DROP,
        edge_gap=config.GUARDRAIL_EDGE_GAP,
        extension=config.GUARDRAIL_EXTENSION,
        junction_clearance=config.GUARDRAIL_JUNCTION_CLEARANCE,
        endpoint_tol=config.ROAD_CONTINUATION_ENDPOINT_TOL,
        max_angle_deg=config.ROAD_CONTINUATION_MAX_ANGLE_DEG,
        min_length=config.GUARDRAIL_SEGMENT_LENGTH,
    )
    items = place_guardrail_items(
        runs, config.GUARDRAIL_SEGMENT_LENGTH, config.GUARDRAIL_BEAM_OFFSET,
        GUARDRAIL_ITEMS["segment"], GUARDRAIL_ITEMS["start"], GUARDRAIL_ITEMS["end"],
    )
    logger.debug(f"  [OK] {len(runs)} guard rail run(s), {len(items)} forest item(s)")
    return items


def _widths_along(coords, nodes) -> Optional[np.ndarray]:
    """Width per coordinate of `coords` from DecalRoad `nodes` [x, y, z, width], by arc length (None without nodes). The
    node list is the same polyline with some nodes dropped or inserted, so the arc lengths agree closely."""
    if nodes is None or len(nodes) < 2:
        return None
    points, node_array = np.asarray(coords, dtype=float), np.asarray(nodes, dtype=float)

    def arc(xy):
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy[:, :2], axis=0), axis=1))])

    node_arc, point_arc = arc(node_array), arc(points)
    scale = node_arc[-1] / point_arc[-1] if point_arc[-1] > 0.0 else 1.0
    return np.interp(point_arc * scale, node_arc, node_array[:, 3])


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


def _invisible_road_material(name: str) -> Dict:
    """materials.json entry of the invisible DecalRoad on structures: alpha-tested, fully transparent texture
    (TerrainWorkflow._export_structure_road_assets() writes it) - schema like vanilla "road_invisible"
    (art/shapes/common/decalroads/main.materials.json), which itself points into west_coast_usa. Deliberately a
    version-1 material (no "version" field): only those read "colorMap" - as a 1.5 (PBR) material the texture was ignored,
    there was no alpha to test and the decal rendered black on the terrain above the tunnel."""
    return {
        "name": name,
        "mapTo": name,
        "class": "Material",
        "Stages": [{"colorMap": str(config.RELATIVE_DIR_TEXTURES / f"{name}.png")}, {}, {}, {}],
        "alphaRef": 127,
        "alphaTest": True,
        "annotation": "STREET",
        "castShadows": False,
        "materialTag0": "RoadAndPath",
        "materialTag1": "beamng",
    }


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


def _road_width_specs(road_slope_polygons_2d: List[Dict], dropped_tunnel_road_ids: frozenset = frozenset()):
    """
    (specs, node_lists) of all roads that get a DecalRoad: specs = [(road_slope_polygon, road_props, nodes)], node_lists =
    their DecalRoad nodes [x, y, z, width] with the widths blended along the width transitions (lane changes, structures).
    Shared by the terrain (embankment/embedding follow the blended width) and export_decal_roads().
    """
    from ..geometry.polygon import drop_close_nodes
    from ..geometry.road_markings import lane_count
    from ..geometry.road_width_transitions import apply_width_transitions

    specs = []  # (road_slope_polygon, road_props, nodes) per exportable DecalRoad

    for poly in road_slope_polygons_2d:
        if not _gets_decal_road(poly, dropped_tunnel_road_ids):
            continue
        road_id = poly.get("road_id")
        centerline = poly.get("trimmed_centerline")
        if road_id is None or centerline is None or len(centerline) < 2:
            continue

        # Skip degenerate (zero-length) roads: clipping/junction
        # split can occasionally leave a "remainder" with 2 identical points.
        # A DecalRoad of length 0 is a degenerate
        # spline (in the old mesh approach this was an invisible
        # zero triangle, here it would produce a broken decal item).
        xy_unique = {(round(float(x), 3), round(float(y), 3)) for x, y, _ in centerline}
        if len(xy_unique) < 2:
            continue

        props = config.OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
        width = float(props.get("width", 4.0))
        nodes = [[float(x), float(y), float(z), width] for x, y, z in centerline]

        # Remove segments that are too short: BeamNG does not draw a DecalRoad with
        # a too-short segment (e.g. 0.10 m from the junction cut next to
        # a resample point) at all - the whole piece is then missing.
        nodes = drop_close_nodes(nodes, config.DECAL_ROAD_MIN_NODE_SPACING)
        if len(nodes) < 2:
            continue
        specs.append((poly, props, nodes))

    # Smooth width transitions at straight-through joints (5 m before/after each, spline) - see
    # geometry/road_width_transitions.py. Inserts nodes only with >= DECAL_ROAD_MIN_NODE_SPACING spacing.
    # Lane-count changes to 3+ lanes blend over ROAD_LANE_CHANGE_TRANSITION_LENGTH; structures (bridges, tunnels,
    # galleries) keep their width - there the whole transition lies on the road (ROAD_STRUCTURE_TRANSITION_LENGTH).
    node_lists = apply_width_transitions(
        [nodes for _, _, nodes in specs],
        transition_length=config.ROAD_WIDTH_TRANSITION_LENGTH,
        step=config.ROAD_WIDTH_TRANSITION_STEP,
        endpoint_tol=config.ROAD_CONTINUATION_ENDPOINT_TOL,
        max_angle_deg=config.ROAD_CONTINUATION_MAX_ANGLE_DEG,
        min_delta=config.ROAD_WIDTH_TRANSITION_MIN_DELTA,
        min_spacing=config.DECAL_ROAD_MIN_NODE_SPACING,
        lanes=[lane_count(poly.get("osm_tags", {}), float(props.get("width", 4.0)), config.ROAD_MARKING_MIN_TWO_LANE_WIDTH)
               for poly, props, _ in specs],
        lane_change_length=config.ROAD_LANE_CHANGE_TRANSITION_LENGTH,
        fixed=[poly.get("structure_type") in config.ROAD_FIXED_WIDTH_STRUCTURES for poly, _, _ in specs],
        fixed_transition_length=config.ROAD_STRUCTURE_TRANSITION_LENGTH,
    )
    return specs, node_lists


def _attach_width_nodes(road_slope_polygons_2d: List[Dict]) -> int:
    """
    Gives every road whose width changes along a width transition the blended nodes ("width_nodes") and an outline
    ("road_polygon") that follow them, so that the embankment and the embedding of the terrain fit the DecalRoad that is
    exported later. Roads of constant width and structures with a fixed width stay untouched. Returns the number of roads.
    """
    from ..geometry.road_width_transitions import variable_width_polygon

    specs, node_lists = _road_width_specs(road_slope_polygons_2d)
    count = 0
    for (poly, _, _), nodes in zip(specs, node_lists):
        arr = np.asarray(nodes, dtype=float)
        if np.ptp(arr[:, 3]) < 1e-6 or poly.get("structure_type") in config.ROAD_FIXED_WIDTH_STRUCTURES:
            continue
        poly["width_nodes"] = arr
        poly["road_polygon"] = variable_width_polygon(arr)
        count += 1
    return count


def _road_marking_lines(specs: List[Tuple[Dict, Dict, List]], node_lists: List[List[List[float]]]) -> List[Dict]:
    """
    Marking lines (edge and center lines) of all marked DecalRoads as {"name", "nodes", "material"} - see
    geometry/road_markings.py. `specs`: (road_slope_polygon, road_props, _) per DecalRoad, `node_lists`: their finished
    nodes (with width transitions), in the same order.

    At T-junctions and junctions the lines are interrupted: they are clipped with the road surfaces of all
    touching roads except the straight-through partners (there the line continues) and except field/foot paths
    (ROAD_MARKING_NO_GAP_HIGHWAYS).
    """
    from shapely import STRtree

    from ..geometry.polygon import drop_close_nodes
    from ..geometry.road_markings import (
        BLOCK,
        CENTER,
        EDGE,
        block_inputs,
        structure_boundary_shifts,
        taper_zones,
        zone_boundary_shifts,
        zone_divider_masks,
        boundary_shifts,
        build_marking_lines,
        clip_line,
        joint_normals,
        junction_obstacles,
        marking_layout,
        road_surface_polygon,
    )
    from ..geometry.road_width_transitions import continuation_partners, find_continuations

    if not specs:
        return []
    pairs = find_continuations(node_lists, config.ROAD_CONTINUATION_ENDPOINT_TOL, config.ROAD_CONTINUATION_MAX_ANGLE_DEG)
    partners = continuation_partners(pairs)
    normals = joint_normals(node_lists, pairs)  # Lines at kinked straight-through joints meet exactly
    centerlines = [np.asarray(nodes, dtype=float)[:, :2] for nodes in node_lists]
    polygons = [road_surface_polygon(nodes, config.ROAD_MARKING_JUNCTION_CLEARANCE) for nodes in node_lists]
    tree = STRtree(polygons)
    no_gap = {
        i for i, (poly, _, _) in enumerate(specs)
        if poly.get("osm_tags", {}).get("highway") in config.ROAD_MARKING_NO_GAP_HIGHWAYS
    }

    layouts = [
        marking_layout(
            poly.get("osm_tags", {}),
            float(props.get("width", 4.0)),
            props.get("internal_name", ""),
            config.ROAD_MARKING_HIGHWAYS,
            config.ROAD_MARKING_SURFACE,
            config.ROAD_MARKING_MIN_TWO_LANE_WIDTH,
            double_center_min_lanes=config.ROAD_MARKING_CENTER_MIN_LANES,
            force_double_center=poly.get("structure_type") in config.ROAD_MARKING_CENTER_STRUCTURES,
        )
        for poly, props, _ in specs
    ]
    fixed = [poly.get("structure_type") in config.ROAD_FIXED_WIDTH_STRUCTURES for poly, _, _ in specs]
    own_widths = [float(props.get("width", 4.0)) for _, props, _ in specs]
    # A lane that is dropped or added along a 100 m taper zone: block stripes replace its dashed divider over the whole zone,
    # and the lane of the other direction keeps its width - the line between the directions follows its edge
    zones = taper_zones(node_lists, layouts, own_widths, fixed, pairs)
    blocks, masks = block_inputs(zones), zone_divider_masks(zones)
    shifts = zone_boundary_shifts(zones)
    # Other joints: the centre line of a narrower road (2 lanes) runs onto the double line of a wider one (3+ lanes), and at a
    # structure the road's lines are aligned with the structure's 50 m before it (the width still changes up to it)
    plain_pairs = [pair for pair in pairs if pair not in {zone["pair"] for zone in zones}]
    for road_index, shift in boundary_shifts(node_lists, layouts, own_widths, plain_pairs, fixed).items():
        shifts[road_index] = shifts.get(road_index, 0.0) + shift
    for road_index, shift in structure_boundary_shifts(
        node_lists, layouts, fixed, plain_pairs, config.ROAD_STRUCTURE_TRANSITION_LENGTH, config.ROAD_STRUCTURE_LINES_DONE_AT
    ).items():
        shifts[road_index] = shifts.get(road_index, 0.0) + shift

    lines = []
    for index, ((poly, props, _), nodes) in enumerate(zip(specs, node_lists)):
        layout = layouts[index]
        if layout is None:
            continue
        obstacles = junction_obstacles(
            index,
            polygons,
            tree,
            excluded=partners.get(index, set()) | no_gap,
            centerlines=centerlines,
            endpoint_tol=config.ROAD_CONTINUATION_ENDPOINT_TOL,
        )
        marking_lines = build_marking_lines(
            nodes,
            layout,
            config.ROAD_MARKING_EDGE_INSET,
            start_normal=normals.get((index, "start")),
            end_normal=normals.get((index, "end")),
            center_gap=config.ROAD_MARKING_CENTER_GAP,
            line_width=config.ROAD_MARKING_LINE_WIDTH,
            boundary_shift=shifts.get(index),
            divider_keep=masks.get(index),
            blocks=blocks.get(index),
        )
        materials = {EDGE: config.ROAD_MARKING_EDGE_MATERIAL, CENTER: config.ROAD_MARKING_CENTER_MATERIAL,
                     BLOCK: config.ROAD_MARKING_BLOCK_MATERIAL}
        widths = {BLOCK: config.ROAD_MARKING_BLOCK_WIDTH}
        for line_idx, (kind, line) in enumerate(marking_lines):
            material = materials.get(kind, config.ROAD_MARKING_DIVIDER_MATERIAL)
            for piece_idx, piece in enumerate(clip_line(line, obstacles, config.ROAD_MARKING_MIN_PIECE_LENGTH)):
                line_width = widths.get(kind, config.ROAD_MARKING_LINE_WIDTH)
                line_nodes = [[x, y, z, line_width] for x, y, z in piece.tolist()]
                # Inside curves the line nodes bunch up - same minimum segment length as for the road surface
                line_nodes = drop_close_nodes(line_nodes, config.DECAL_ROAD_MIN_NODE_SPACING)
                if len(line_nodes) >= 2:
                    lines.append({
                        "name": f"marking_{poly['road_id']}_{line_idx}_{piece_idx}",
                        "nodes": line_nodes,
                        "material": material,
                        "structure": poly.get("structure_type", "surface") != "surface",
                    })
    return lines


class TerrainWorkflow:
    """
    Orchestrates the terrain export workflow.

    Responsible for:
    - Mesh generation
    - Road integration
    - DAE export
    - Material/item management
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = MaterialManager.get_instance()  # Singleton
        self.items = ItemManager.get_instance()  # Singleton
        self.dae = dae_exporter
        self.tile_processor = TileProcessor(cache_manager)

    def process_tile(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float],
        task: PipelineTask,
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Process all given tiles as ONE contiguous area.

        The elevation data of all tiles is combined into a single point cloud
        (assumes that they form a gapless, rectangular
        area - user responsibility, see utils.tile_scanner).
        From here on, all remaining processing (BBox, OSM query,
        grid, road mesh, junction detection, embankment, heightmap) runs ONCE
        over the whole area instead of once per tile - clipping only happens
        at the outer edge of the whole area, no longer at the former
        tile borders. A single tile is simply the special case
        len(tiles) == 1 of the same code path.

        Args:
            tiles: List of tile metadata (typically all DGM1 tiles
                of an export)
            global_offset: Global offset (origin_x, origin_y)
            task: PipelineTask for the progress display of the subtasks
            bbox_margin: BBox expansion in meters
            buildings_data: Optional - LoD2 building data

        Returns:
            Dict with processing results
        """
        from ..osm.parser import calculate_bbox_from_height_data, extract_roads_from_osm
        from ..osm.downloader import get_osm_data
        from ..geometry.polygon import get_road_polygons, clip_road_polygons
        from ..geometry.junctions import build_junction_network
        from ..io.cache import calculate_global_tiles_hash

        sub = task.begin_subtask("Load OSM data")

        # 1. Combine the elevation data of all tiles into one point cloud
        height_points, height_elevations = self.tile_processor.load_height_data_multi(tiles)
        if height_points is None:
            sub.fail("no height data")
            return {"status": "failed", "reason": "no_height_data"}

        # Combined hash over all tiles - cache identity for OSM/
        # elevation/grid (replaces the former per-tile file hash)
        tile_hash = calculate_global_tiles_hash(tiles) if tiles else "unknown"

        # 2. Compute the BBox (BEFORE the local transformation!)
        # Compute the BBox with margin directly in UTM (meters), then transform to WGS84
        osm_bbox = calculate_bbox_from_height_data(height_points, margin=bbox_margin)

        # 3. Transform to local coordinates
        local_points, elevations = self.tile_processor.ensure_local_offset(
            global_offset, height_points, height_elevations
        )

        # 4. Load OSM data (with tile_hash for the tile-specific cache)
        osm_data = get_osm_data(osm_bbox, height_hash=tile_hash)

        if not osm_data:
            logger.warning("  [!] No OSM data")
            sub.fail("no OSM data")
            return {"status": "failed", "reason": "no_osm_data"}

        # 5. Extract roads
        roads = extract_roads_from_osm(osm_data)

        # 6. Road polygons (converts OSM data to coords)
        # IMPORTANT: Pass LOCAL coordinates! All internal calculations in local!

        road_polygons = get_road_polygons(roads, osm_bbox, local_points, elevations, global_offset, tile_hash=tile_hash)
        sub.finish()  # covers the OSM query AND road extraction - both part of "Load OSM data"

        # 6a. Aerial photos: are NO longer processed per tile here - with
        # several tiles the file existence check ("is there already
        # any .dds?") would skip the export for all tiles except the first.
        # Instead, BeamNGExporter.export_complete_level() calls
        # process_aerial_images() once for the total BBox of all tiles,
        # before the tile loop begins.

        # 6b. Load LoD2 buildings (if enabled and not yet passed in)
        sub = task.begin_subtask("Normalize buildings")
        if buildings_data is None and config.LOD2_ENABLED:
            from ..io.lod2 import cache_lod2_buildings, load_buildings_from_cache

            # Compute Z-min from the elevation data for full 3D normalization
            # IMPORTANT: Do NOT normalize building Z coordinates!
            # The terrain itself has absolute elevations (263-580m), not normalized.
            # The buildings in CityGML also have absolute elevations above sea level.
            # Therefore: z_offset = 0 (no Z normalization for buildings!)
            z_offset = 0.0
            # Extend global_offset to a 3D offset
            local_offset_3d = (global_offset[0], global_offset[1], z_offset)

            # Try to load normalized buildings directly from the cache
            # (they were already normalized by cache_lod2_buildings)
            buildings_cache_path = cache_lod2_buildings(
                lod2_dir=config.LOD2_DATA_DIR,
                bbox=osm_bbox,  # WGS84-BBox
                local_offset=local_offset_3d,  # 3D offset with Z-min!
                cache_dir=config.CACHE_DIR,
                height_hash=tile_hash,
            )
            if buildings_cache_path:
                buildings_data = load_buildings_from_cache(buildings_cache_path)
                if buildings_data:
                    logger.info(f"  [OK] {len(buildings_data)} normalized buildings loaded from cache")

            if not buildings_data:
                logger.info("  [i] No LoD2 buildings found")

        # Church towers: no windows, a tower clock instead (church from OSM, tower walls from the geometry)
        if buildings_data:
            from ..facade.church_towers import ChurchTowerFinder
            from ..osm.landuse_polygons import make_local_transform

            towers = ChurchTowerFinder.from_osm(osm_data, make_local_transform(global_offset)).mark(buildings_data)
            logger.info(f"  [OK] {towers} churches with a tower detected (tower clock instead of windows)")

        sub.finish(f"{len(buildings_data)} buildings" if buildings_data else "no LoD2 buildings")

        sub = task.begin_subtask("Road network + infrastructure")

        # Compute grid bounds from local points for clipping
        grid_bounds_local = (
            float(local_points[:, 0].min()),
            float(local_points[:, 0].max()),
            float(local_points[:, 1].min()),
            float(local_points[:, 1].max()),
        )

        # Use ROAD_CLIP_MARGIN from the config (negative = expand!)
        road_polygons = clip_road_polygons(road_polygons, grid_bounds_local, margin=config.ROAD_CLIP_MARGIN)

        # 7. Junction detection (needs road_polygons with coords): detect → split → mark, tunnels excluded
        # (they lie on a different level, never form junctions - see build_junction_network())
        road_polygons, junctions = build_junction_network(road_polygons)

        # Roads that pass under a bridge: the terrain model shows the deck there, so their sampled height climbs to it -
        # interpolate from before to behind the bridge (the embedding below then cuts them in with slopes on both sides)
        from ..geometry.road_structures import fix_underpass_elevations

        underpasses = fix_underpass_elevations(
            road_polygons,
            lambda road: config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"] / 2.0,
            max_search=config.UNDERPASS_MAX_SEARCH,
            stable_length=config.UNDERPASS_STABLE_LENGTH,
            max_grade=config.UNDERPASS_STABLE_GRADE,
            min_rise=config.UNDERPASS_MIN_RISE,
        )
        if underpasses:
            logger.debug(f"  [OK] {underpasses} road(s) under bridges: height interpolated")

        # Convert road_polygons into road_slope_polygons_2d (for classification)
        # IMPORTANT: AFTER junction detection, so that the split roads are used!
        # IMPORTANT: Create actual road polygons (buffer around the centerline)
        from shapely.geometry import LineString
        from ..config import OSM_MAPPER
        from ..geometry.road_structures import classify_structure, split_by_structure_type
        from ..utils.debug_exporter import DebugNetworkExporter

        road_slope_polygons_2d = []
        debug_exporter = DebugNetworkExporter.get_instance()

        for road in road_polygons:
            coords = np.asarray(road.get("coords", []), dtype=float)
            if len(coords) < 2:
                continue

            # Compute the road width from OSM tags
            osm_tags = road.get("osm_tags", {})
            road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]

            # Create the polygon by buffering the centerline
            centerline_2d = coords[:, :2]
            try:
                line = LineString(centerline_2d)
                road_poly = line.buffer(road_width / 2.0, cap_style=2)  # cap_style=2 = flat
                road_polygon_2d = np.array(road_poly.exterior.coords[:-1])  # without duplicate
            except Exception:
                # Fallback: use the centerline directly
                road_polygon_2d = centerline_2d

            road_id = road.get("id")
            road_slope_polygons_2d.append(
                {
                    "road_id": road_id,  # Important for material mapping
                    "road_polygon": road_polygon_2d,
                    "trimmed_centerline": coords,
                    "osm_tags": osm_tags,
                    "structure_type": classify_structure(osm_tags),
                    "daylight_slopes": bool(road.get("underpass")),  # road under a bridge: slopes up to the terrain
                }
            )

            # Export the road for debug visualization
            debug_exporter.add_line(
                coords,
                color=[0.0, 0.0, 1.0],
                width=2.0,
                label=f"Road_{road_id}",
            )

        # The terrain follows the widths that the DecalRoads get along the width transitions (lane changes, structures)
        widened = _attach_width_nodes(road_slope_polygons_2d)
        if widened:
            logger.debug(f"  [OK] {widened} road(s) with a width transition: embankment follows the blended width")

        # 8. Create the grid (with builder)
        from ..builders import GridBuilder

        grid_builder = GridBuilder()
        grid = (
            grid_builder.with_points(local_points)
            .with_elevations(elevations)
            .with_spacing(config.GRID_SPACING)
            .with_cache_key(f"grid_{tile_hash}")
            .build()
        )

        # 9. Extract grid dimensions (vertex classification no longer applies -
        # the terrain is no longer triangulated, see task 9)
        grid_points, grid_elevations, nx, ny = grid

        # 10. Terrain heightmap instead of mesh triangulation.
        # Since the switch to DecalRoad, roads are no longer built as a mesh
        # (no RoadMeshBuilder/junction fan material majority vote
        # needed anymore) - see export_decal_roads().
        from ..config import OSM_MAPPER
        from ..terrain.heightmap import build_heightmap
        from ..terrain.road_embedding import (
            embed_roads_into_heightmap,
            build_road_embankment_profiles,
            apply_embankment_blend,
            sample_heightmap_bilinear,
        )
        from ..terrain.terrain_materials import (
            DEFAULT_LANDUSE_CATEGORY,
            build_photo_fallback_layer,
            mark_padding_as_holes,
            mask_layer_map_with_photo,
            paint_landuse_materials,
        )

        heightmap_result = build_heightmap(
            grid_points, grid_elevations, nx, ny, config.TERRAIN_SQUARE_SIZE
        )
        heights = heightmap_result["heights"]
        terrain_size = heightmap_result["size"]
        terrain_origin_x = heightmap_result["origin_x"]
        terrain_origin_y = heightmap_result["origin_y"]

        # Embankment: create the transition from the road edge to the natural surroundings directly
        # in the heightmap (the mesh no longer generates embankment geometry - see spec section 4b).
        # IMPORTANT: must run on the still UNMODIFIED heights, so that
        # "natural height" is really natural (before embed_roads_into_heightmap).
        # Bridges/tunnels are NOT embedded into the terrain and get no embankment
        # (the terrain below/beside them stays completely natural). Galleries,
        # on the other hand, ARE embedded like normal roads (see below) - no separate terrain hole
        # needed anymore, since floor/wall/roof are solid boxes (tunnels/gallery_mesh.py).
        surface_road_polygons, structure_road_polygons = split_by_structure_type(road_slope_polygons_2d)

        # Embed galleries like normal roads (same embankment/embedding parameters), but with
        # fixed instead of computed embankment widths on both sides (slope_width_override, see
        # build_road_embankment_profiles() docstring):
        # - Mountain side: GALLERY_MOUNTAIN_EMBED_MARGIN (1 m) beyond the inner edge of the (solid) wall,
        #   FLAT at road surface height (flat_shoulder_sides, no interpolation to the terrain - the wall reaches
        #   into the slope anyway, this narrow fringe only ensures a clean wall-floor transition).
        # - Valley side: GALLERY_VALLEY_SLOPE_WIDTH, real downward interpolation to the natural terrain (the
        #   DGM shows the real valley-side structure right at the road edge instead of natural terrain, see
        #   build_road_embankment_profiles() docstring - a computed width would be noisy/faceted).
        # Without an avalanche_protector:left/right tag the valley side is determined from the (still natural) terrain -
        # the gallery mesh later uses the same side (see _gallery_embedding()).
        def _natural_ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        def _gallery_road(r):
            override, flat_sides = _gallery_embedding(r, _natural_ground_at)
            return {**r, "slope_width_override": override, "flat_shoulder_sides": flat_sides}

        gallery_roads = [_gallery_road(r) for r in structure_road_polygons if r.get("structure_type") == "gallery"]
        # Gallery embankments end flush with the gallery, the approach roads' flush with the transition
        _gallery_embankment_cuts(surface_road_polygons, gallery_roads, config.ROAD_CONTINUATION_ENDPOINT_TOL)
        embeddable_roads = surface_road_polygons + gallery_roads

        embankment_profiles = build_road_embankment_profiles(
            embeddable_roads,
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            OSM_MAPPER,
            config.SLOPE_ANGLE,
            config.MIN_SLOPE_WIDTH,
            max_slope_width=config.MAX_SLOPE_WIDTH,
        )
        heights = apply_embankment_blend(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, embankment_profiles)

        # Road embedding: set the terrain on the pure road surface exactly to
        # centerline height (the embankment is already covered by
        # apply_embankment_blend) - see the road_embedding.py
        # module docstring for the DecalRoad rationale.
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            embeddable_roads,
        )

        # Bridges: cap terrain that lies HIGHER than the deck within the bridge width to deck level
        # (not necessarily set it as above) - this practically only affects the bridge ends (abutments), where the
        # road surface merges into the natural terrain and is not necessarily flat across the driving direction;
        # without capping, the terrain could poke through the (flat) deck in places. The valley floor that
        # the bridge spans stays visible unchanged (well below deck level, clamp_to_max
        # does not apply there).
        bridge_roads = [r for r in structure_road_polygons if r.get("structure_type") == "bridge"]
        if bridge_roads:
            heights = embed_roads_into_heightmap(
                heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, bridge_roads,
                clamp_to_max=True,
            )

        # Tunnel: join pieces into chains, define portals and adapt the terrain to them - cover
        # above the tube, portal zone at floor height, hole cells behind the portal (see terrain/tunnel_terrain.py).
        # After the road embedding; the surface roads themselves are left untouched.
        from ..geometry.road_surfaces import union_road_surfaces

        tunnel_plans = []
        tunnel_holes = None
        dropped_tunnel_road_ids = frozenset()
        if config.TUNNELS_ENABLED:
            from ..terrain.tunnel_terrain import shape_terrain_for_tunnels

            tunnel_plans = _plan_tunnels(structure_road_polygons)
            if tunnel_plans:
                import shapely

                all_tunnel_piece_ids = {pid for plan in tunnel_plans for pid in plan.get("piece_ids", [plan["id"]])}

                protected = union_road_surfaces(surface_road_polygons + gallery_roads)
                if protected is not None:
                    shapely.prepare(protected)
                heights, tunnel_holes = shape_terrain_for_tunnels(
                    heights,
                    terrain_origin_x,
                    terrain_origin_y,
                    config.TERRAIN_SQUARE_SIZE,
                    tunnel_plans,
                    cover=config.TUNNEL_COVER,
                    protected=protected,
                    bounds=grid_bounds_local,
                    edge_margin=config.MAP_EDGE_TUNNEL_MARGIN,
                )
                # tunnel_plans was filtered in place: pass-through chains without a reachable portal are gone
                dropped_tunnel_road_ids = _dropped_tunnel_road_ids(all_tunnel_piece_ids, tunnel_plans)

        # Layer map: ONE aerial photo material for the whole area, then OSM
        # land use on top (see build_photo_fallback_layer()).
        layer_map, photo_tile_names = build_photo_fallback_layer(terrain_size)

        from ..osm.landuse_polygons import build_landuse_polygons, make_local_transform

        # At this point osm_data still contains RAW Overpass geometry (lat/lon).
        # Land use polygons are built from ways AND multipolygon relations
        # (large forest/vineyard/residential areas are usually relations in OSM).
        landuse_polygons = build_landuse_polygons(osm_data, make_local_transform(global_offset))

        # Pond basins: lower the terrain within the water areas (before everything that reuses the elevations:
        # streams, trees, vines). The water level comes from the natural edge, see _build_water().
        natural_heights = heights
        if config.WATER_ENABLED:
            from ..terrain.water import carve_pond_basins, select_pond_areas

            pond_areas = select_pond_areas(landuse_polygons, water_bounds(grid_bounds_local))
            if pond_areas:
                heights = carve_pond_basins(
                    heights,
                    terrain_origin_x,
                    terrain_origin_y,
                    config.TERRAIN_SQUARE_SIZE,
                    pond_areas,
                    depth=config.WATER_POND_BANK_DEPTH,
                    slope_deg=config.WATER_POND_BANK_SLOPE_DEG,
                )
                logger.info(
                    f"  [OK] Pond basins: {len(pond_areas)} water area(s), terrain {config.WATER_POND_BANK_DEPTH * 100:.0f} cm lower "
                    f"(bank {config.WATER_POND_BANK_SLOPE_DEG:.0f} degrees)"
                )

        # background_category: areas without any land use polygon (no OSM element covers them)
        # still get meadow this way instead of staying on the photo fallback forever - closes the gap that
        # get_landuse_category()'s DEFAULT_LANDUSE_CATEGORY fallback leaves open (it only applies to an
        # EXISTING but unknown landuse tag value, see paint_landuse_materials() docstring).
        layer_map, terrain_material_names = paint_landuse_materials(
            layer_map,
            photo_tile_names,
            terrain_size,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            landuse_polygons,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            background_category=DEFAULT_LANDUSE_CATEGORY,
        )

        # Road and building areas: (1) ground cover grows on the layer - there it goes
        # back to the aerial photo, otherwise grass grows through decals and houses; (2) exclusion zone
        # for the vineyard vines.
        # All road surfaces unioned ONCE (simplified): serves the mask, vine exclusion and the forest.
        # Surface roads AND galleries (now embedded into the terrain like normal roads, see above) -
        # only bridges/tunnels are left out, they should not block vegetation, otherwise e.g. a tunnel
        # would leave a bare strip over the whole ridge. The tunnel portal blocks, however,
        # do count (otherwise trees/grass would grow through the block).
        from ..tunnels.tunnel_portal import portal_footprint

        portal_footprints = [
            {"road_polygon": np.array(portal_footprint(portal))} for plan in tunnel_plans for portal in plan["portals"] if portal["open"]
        ]
        road_surface_union = union_road_surfaces(surface_road_polygons + gallery_roads + portal_footprints)
        road_shapes = [road_surface_union] if road_surface_union is not None else []
        building_shapes = [
            p["geometry"]
            for p in build_landuse_polygons(osm_data, make_local_transform(global_offset), tag_keys=("building",))
        ]

        if config.GROUND_COVER_ENABLED:
            layer_map = mask_layer_map_with_photo(
                layer_map,
                terrain_size,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
                road_shapes,
                buffer=config.GROUND_COVER_ROAD_MARGIN,
            )
            layer_map = mask_layer_map_with_photo(
                layer_map,
                terrain_size,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
                building_shapes,
                buffer=config.GROUND_COVER_BUILDING_MARGIN,
            )

        # Bridges: no grass where the terrain lies so close below the deck that it would grow through it (hillside
        # bridges); under bridges spanning a valley the grass stays. No trees anywhere under a bridge.
        bridge_footprints = _bridge_footprints(bridge_roads, config.BRIDGE_CURB_WIDTH + config.BRIDGE_UNDERGROWTH_MARGIN)
        if config.GROUND_COVER_ENABLED and bridge_footprints:
            from ..terrain.road_embedding import near_deck_mask

            near_deck = near_deck_mask(
                heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, bridge_footprints,
                clearance=config.BRIDGE_UNDERGROWTH_CLEARANCE,
            )
            layer_map = layer_map.copy()
            layer_map[near_deck] = 0  # aerial photo layer: nothing grows on it
        tree_exclusion = road_surface_union
        bridge_tree_areas = _bridge_footprints(bridge_roads, config.BRIDGE_CURB_WIDTH + config.BRIDGE_TREE_MARGIN)
        if bridge_tree_areas:
            from shapely import union_all
            from shapely.geometry import Polygon

            parts = [Polygon(b["road_polygon"]) for b in bridge_tree_areas]
            tree_exclusion = union_all(parts + ([road_surface_union] if road_surface_union is not None else []))

        # Excess border of the power-of-two heightmap (extrapolation only) as a hole: the visible
        # terrain ends exactly at the data edge, the horizon covers the strip behind it.
        # Last, so that painting/masking above run unchanged on the full layer map.
        if config.TERRAIN_PADDING_AS_HOLES:
            layer_map = mark_padding_as_holes(layer_map, data_cols=nx, data_rows=ny)
        # Tunnel portals: hole cells at the portal level (hidden by the tube shell or collar)
        if tunnel_holes is not None and np.any(tunnel_holes):
            from ..terrain.ter_writer import EMPTY_LAYER_VALUE

            layer_map = layer_map.copy()
            layer_map[tunnel_holes] = EMPTY_LAYER_VALUE

        # Four-photo mode: only now (painting, masks and holes are done) is the layer map split per tile into
        # physical materials - each tile gets its own photo. The photo tiling is a
        # FIXED grid over the whole area (config.PHOTO_TILE_SIZE_M), independent of the size/number of the
        # raw elevation data tiles (see terrain/photo_tiles.py module docstring) - export/beamng_exporter.py
        # builds the same grid from the same inputs (deterministic, without having to share data).
        from ..terrain.photo_tiles import build_processing_tile_grid
        from ..utils.tile_scanner import compute_global_bbox

        processing_tiles = build_processing_tile_grid(compute_global_bbox(tiles), config.PHOTO_TILE_SIZE_M)

        photo_tiles = None
        if config.AERIAL_PHOTO_PER_TILE and len(processing_tiles) > 1:
            from ..terrain.photo_tiles import split_layers_by_tile

            photo_tiles = split_layers_by_tile(
                layer_map,
                terrain_material_names,
                processing_tiles,
                global_offset,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
            )
            layer_map = photo_tiles["layer_map"]
            terrain_material_names = photo_tiles["material_names"]
            photo_tile_names = photo_tiles["photo_tile_names"]
            logger.info(
                f"  [OK] Four-photo mode: {len(photo_tile_names)} aerial photos, {len(terrain_material_names)} terrain materials"
            )

        # Vineyard vines (forest items) along the fall line, on the finished heightmap
        vineyard_instances = []
        if config.VINEYARDS_ENABLED and config.FORESTS_ENABLED:
            from ..forest.vineyard_generator import build_exclusion_geometry, generate_vineyards, make_height_sampler
            from shapely.geometry import box

            vineyard_instances = generate_vineyards(
                landuse_polygons,
                config.OSM_MAPPER.config.get("landuse_mappings", {}),
                make_height_sampler(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE),
                exclusion=build_exclusion_geometry(road_shapes + building_shapes, config.VINEYARD_EXCLUSION_MARGIN),
                # Only over real elevation data: the OSM query extends beyond the terrain,
                # and the terrain edge is filled in (there are no real elevations there)
                bounds=box(
                    grid_bounds_local[0] + config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[2] + config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[1] - config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[3] - config.VINEYARD_EXCLUSION_MARGIN,
                ),
            )
            logger.debug(f"  [OK] {len(vineyard_instances)} vine row segments generated")

        # Real water: streams as River splines, water areas as WaterBlocks (on the finished heightmap)
        water = {"rivers": [], "ponds": []}
        if config.WATER_ENABLED:
            water = self._build_water(
                osm_data,
                landuse_polygons,
                global_offset,
                make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y),
                grid_bounds_local,
                rim_height_at=make_height_sampler_for_water(natural_heights, terrain_origin_x, terrain_origin_y),
            )

        # Rubble stone walls (OSM barrier=wall with height) on the finished heightmap
        wall_meshes = []
        if config.WALLS_ENABLED:
            wall_meshes, _ = self._build_wall_meshes(
                osm_data,
                global_offset,
                make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y),
                road_slope_polygons_2d,
            )

        # Bridges (deck + piers) are built in export_bridges(): their width follows the blended widths of the transitions,
        # which export_decal_roads() computes first - see bridges/bridge_mesh.py

        # Tunnels (tube + portals) and galleries (roof + supports) on the finished heightmap - see tunnels/
        tunnel_meshes = []
        roadblocks = []
        tunnel_zones = []
        tunnel_lights = []
        if config.TUNNELS_ENABLED:
            from ..tunnels.tunnel_lights import build_lamp_mesh

            tunnel_zones = _tunnel_zone_items(tunnel_plans)
            tunnel_lights = _tunnel_light_items(tunnel_plans)
            tunnel_meshes = self._build_tunnels(structure_road_polygons, tunnel_plans, heights, terrain_origin_x, terrain_origin_y)
            lamps = build_lamp_mesh(
                tunnel_lights, config.TUNNEL_LAMP_MATERIAL_NAME, config.TUNNEL_LAMP_LENGTH, config.TUNNEL_LAMP_WIDTH,
                config.TUNNEL_LAMP_HEIGHT, config.TUNNEL_LIGHT_CEILING_MARGIN,
            )
            if lamps is not None:
                tunnel_meshes.append({"id": "tunnel_lamps", **lamps})
            roadblocks = _roadblock_items(
                tunnel_plans, heights, terrain_origin_x, terrain_origin_y, grid_bounds_local, surface_road_polygons
            )

        # POI candidates (villages/towns, large parking lots) for additional spawn points selectable in the
        # vehicle selection - see osm/poi_points.py and ItemManager._compute_poi_spawn_points(). Height sampled on
        # the FINISHED heightmap (the positions come as pure XY points from OSM, not from
        # a road centerline).
        poi_points = self._collect_poi_points(osm_data, global_offset, heights, terrain_origin_x, terrain_origin_y, grid_bounds_local)

        # Selectable spawn points in front of tunnel chain entrances, facing into the tunnel (tunnels/entrance_spawns.py)
        from ..tunnels.entrance_spawns import plan_entrance_spawns

        tunnel_spawns = plan_entrance_spawns(road_slope_polygons_2d, config.TUNNEL_SPAWN_DISTANCE, config.POI_SPAWN_EXCLUDED_HIGHWAYS)

        z_min = float(heights.min())
        z_max = float(heights.max())
        max_height = (z_max - z_min) + config.TERRAIN_MAX_HEIGHT_BUFFER

        sub.finish(f"{len(road_slope_polygons_2d)} road segments")

        return {
            "status": "success",
            "heightmap": heights,
            "terrain_size": terrain_size,
            "terrain_origin_x": terrain_origin_x,
            "terrain_origin_y": terrain_origin_y,
            "z_min": z_min,
            "max_height": max_height,
            "layer_map": layer_map,
            "terrain_material_names": terrain_material_names,
            "photo_tile_names": photo_tile_names,
            # Four-photo mode (otherwise None): tile variants of the layers, their photo and the photo size per tile
            "layer_variants": photo_tiles["layer_variants"] if photo_tiles else None,
            "variant_parents": photo_tiles["variant_parents"] if photo_tiles else None,
            "photo_extents": photo_tiles["photo_extents"] if photo_tiles else None,
            "poi_points": poi_points,  # Villages/towns and large parking lots for ItemManager._compute_poi_spawn_points()
            "vineyard_instances": vineyard_instances,  # Forest-Items (grape_vine)
            "water": water,  # {"rivers": [...], "ponds": [...]} for export_water()
            "wall_meshes": wall_meshes,  # Mesh dicts of the rubble stone walls for export_walls()
            "tunnel_meshes": tunnel_meshes,  # Tunnel/gallery mesh dicts for export_tunnels()
            "tunnel_spawns": tunnel_spawns,  # Spawn points in front of tunnel entrances for ItemManager.save(fixed_spawns=...)
            "roadblocks": roadblocks,  # Roadblocks in front of entrances of tunnels beyond the map border, for export_roadblocks()
            "tunnel_zones": tunnel_zones,  # Zone boxes for dark tunnel tubes, for export_tunnel_zones()
            "tunnel_lights": tunnel_lights,  # SpotLight fixtures inside the tubes, for export_tunnel_lights()
            "dropped_tunnel_road_ids": dropped_tunnel_road_ids,  # pieces of pass-through tunnels: no AI DecalRoad
            "grid": grid,
            "road_polygons": road_polygons,
            "road_slope_polygons_2d": road_slope_polygons_2d,  # For DecalRoad export
            "structure_road_polygons": structure_road_polygons,  # Bridges/tunnels/galleries - for export_bridges()/export_tunnels()
            "road_surface_union": road_surface_union,  # unioned road surface for exclusion zones (or None)
            "tree_exclusion": tree_exclusion,  # road surfaces plus the areas under bridges: no trees there (or None)
            "grid_bounds_local": grid_bounds_local,
            "global_offset": global_offset,
            "buildings_data": buildings_data,  # Pass on the building data
            "height_points": local_points,  # For spawn point calculation
            "height_elevations": elevations,  # For spawn point calculation
            "height_hash": tile_hash,  # For cache consistency in the forest workflow
        }

    def _build_water(self, osm_data, landuse_polygons, global_offset, height_at, grid_bounds_local, rim_height_at=None) -> Dict:
        """
        Computes stream nodes and pond blocks (see terrain/water.py). The water heights are derived from the finished
        heightmap; the pond water level from the edge of the NATURAL heightmap (`rim_height_at`, without the
        pond basin - otherwise it would be too low by the bank depth), otherwise from `height_at`.

        Returns:
            {"rivers": [{"name", "waterway", "nodes"}], "ponds": [{"name", "blocks"}]}
        """
        from shapely.ops import unary_union

        from ..osm.landuse_polygons import make_local_transform
        from ..terrain.water import (
            build_pond_blocks,
            build_river_nodes,
            clip_line_to_bounds,
            cut_line_by_area,
            select_pond_areas,
            select_waterways,
            split_nodes,
        )

        bounds = water_bounds(grid_bounds_local)

        # Ponds first: the streams end at their bank
        ponds = []
        pond_areas = select_pond_areas(landuse_polygons, bounds)
        for geometry in pond_areas:
            blocks = build_pond_blocks(
                geometry,
                rim_height_at or height_at,
                depth=config.WATER_POND_DEPTH,
                cell=config.WATER_POND_CELL,
                margin=config.WATER_POND_MARGIN,
            )
            if blocks:
                ponds.append({"name": f"pond_{len(ponds)}", "blocks": blocks})
        pond_area = unary_union(pond_areas) if pond_areas else None

        rivers = []
        for way in select_waterways(osm_data, make_local_transform(global_offset), config.WATERWAY_WIDTHS):
            for clipped in clip_line_to_bounds(way["coords"], bounds):
                for part in cut_line_by_area(clipped, pond_area):
                    if len(part) < 2:
                        continue
                    nodes = build_river_nodes(
                        part,
                        height_at,
                        width=way["width"],
                        depth=config.WATER_RIVER_DEPTH,
                        spacing=config.WATER_NODE_SPACING,
                        lift=config.WATER_STREAM_LIFT,
                    )
                    for chunk in split_nodes(nodes, config.WATER_MAX_RIVER_NODES):
                        if len(chunk) >= 2:
                            rivers.append({"name": f"river_{len(rivers)}", "waterway": way["waterway"], "nodes": chunk})

        length = sum(sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(r["nodes"], r["nodes"][1:])) for r in rivers)
        logger.debug(
            f"  [OK] Water: {len(rivers)} river object(s) ({length:.0f} m of stream), "
            f"{len(ponds)} water area(s) with {sum(len(p['blocks']) for p in ponds)} WaterBlocks"
        )
        return {"rivers": rivers, "ponds": ponds}

    def _build_wall_meshes(self, osm_data, global_offset, height_at, road_polygons=None):
        """
        Rubble stone walls from OSM (barrier=wall/retaining_wall, only with a height tag), following the terrain (see
        walls/wall_mesh.py); at most config.WALL_ROAD_SNAP_M beside a road centerline at its height (road_base.py).

        Args:
            road_polygons: Road dicts with "trimmed_centerline" (see road_slope_polygons_2d)

        Returns:
            (mesh dicts for the DAE export, statistics {"built", "length", "without_height"})
        """
        from ..osm.landuse_polygons import make_local_transform
        from ..textures import library
        from ..walls.road_base import RoadBaseHeight, centerlines_from_roads
        from ..walls.wall_mesh import build_walls

        meshes, stats = build_walls(
            osm_data,
            make_local_transform(global_offset),
            height_at,
            config.WALL_MATERIAL_NAME,
            thickness=config.WALL_THICKNESS,
            sink=config.WALL_SINK,
            max_step=config.WALL_MAX_SEGMENT,
            tile_m=library.texture_tile_m(config.WALL_TEXTURE_NAME, config.WALL_TEXTURE_TILE_M),
            road_base_at=RoadBaseHeight(centerlines_from_roads(road_polygons or []), config.WALL_ROAD_SNAP_M),
            cap_thickness=config.WALL_CAP_THICKNESS,
            cap_overhang=config.WALL_CAP_OVERHANG,
            cap_plate_length=config.WALL_CAP_PLATE_LENGTH,
            cap_joint=config.WALL_CAP_JOINT,
        )
        logger.debug(
            f"  [OK] Walls: {stats['built']} rubble wall(s) with height ({stats['length']:.0f} m), "
            f"{stats['without_height']} without height skipped"
        )
        return meshes, stats

    def _build_bridges(
        self, structure_road_polygons: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float,
        bridge_widths: Optional[Dict] = None,
    ) -> List[Dict]:
        """Bridge meshes (deck + piers) for all roads with structure_type == "bridge" (see bridges/bridge_mesh.py).
        `bridge_widths`: road id -> DecalRoad nodes [x, y, z, width] with the blended widths (export_decal_roads()); the
        deck follows them."""
        from ..bridges.bridge_mesh import build_bridges
        from ..terrain.road_embedding import sample_heightmap_bilinear

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        bridges = [
            {
                "id": road["road_id"],
                "coords": road["trimmed_centerline"],
                "width": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"],
                "deck_material": f"{config.OSM_MAPPER.get_road_properties(road.get('osm_tags', {})).get('internal_name', 'road_default')}_structure",
                "widths": _widths_along(road["trimmed_centerline"], (bridge_widths or {}).get(road["road_id"])),
            }
            for road in structure_road_polygons
            if road.get("structure_type") == "bridge"
        ]
        return build_bridges(
            bridges,
            ground_at,
            pier_material=config.BRIDGE_MATERIAL_NAME,
            railing_material=config.BRIDGE_RAILING_MATERIAL_NAME,
            deck_thickness=config.BRIDGE_DECK_THICKNESS,
            pier_spacing=config.BRIDGE_PIER_SPACING,
            pier_width_fraction=config.BRIDGE_PIER_WIDTH_FRACTION,
            pier_depth_fraction=config.BRIDGE_PIER_DEPTH_FRACTION,
            pier_burial=config.BRIDGE_PIER_BURIAL,
            min_pier_clearance=config.BRIDGE_MIN_PIER_CLEARANCE,
            curb_width=config.BRIDGE_CURB_WIDTH,
            curb_height=config.BRIDGE_CURB_HEIGHT,
            railing_height=config.BRIDGE_RAILING_HEIGHT,
            railing_post_spacing=config.BRIDGE_RAILING_POST_SPACING,
            railing_post_size=config.BRIDGE_RAILING_POST_SIZE,
            road_texture_length=config.ROAD_DECAL_TEXTURE_LENGTH,
        )

    def export_bridges(self, mesh_data: Dict) -> int:
        """
        Exports bridges as ONE DAE (deck + piers per bridge) with ONE TSStatic and registers the road surface
        and concrete material. Without bridges, leftovers of a previous export are removed.

        Returns:
            Number of exported bridges
        """
        bridges_dir = config.BEAMNG_DIR_SHAPES / "bridges"
        meshes = mesh_data.get("bridge_meshes")  # ready-made mesh dicts (tests, callers that build them themselves)
        if meshes is None:
            # Built once, with the widths of the transitions from export_decal_roads() (the deck follows them)
            meshes = []
            if config.BRIDGES_ENABLED and mesh_data.get("heightmap") is not None:
                meshes = self._build_bridges(
                    mesh_data["structure_road_polygons"], mesh_data["heightmap"], mesh_data["terrain_origin_x"],
                    mesh_data["terrain_origin_y"], mesh_data.get("bridge_widths"),
                )
        if not config.BRIDGES_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (bridges_dir / f"bridges{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )
        railing = registry.prepared_textures()[config.RAILING_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_RAILING_MATERIAL_NAME,
            textures={**railing, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        unique_deck_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") != "bridge":
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            mat_name = f"{props.get('internal_name', 'road_default')}_structure"
            unique_deck_materials[mat_name] = props

        for mat_name, props in unique_deck_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=bridges_dir / "bridges.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "bridges",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "bridges" / "bridges.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.debug(f"  [OK] {len(meshes)} bridges exported (bridges.dae)")
        return len(meshes)

    def _build_tunnels(
        self, structure_road_polygons: List[Dict], tunnel_plans: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float
    ) -> List[Dict]:
        """Tunnel meshes (tube + portal blocks, from the tunnel plans, see tunnels/tunnel_portal.py) and gallery meshes
        (roof + supports) for all roads with structure_type "gallery" - see tunnels/gallery_mesh.py."""
        from ..terrain.road_embedding import sample_heightmap_bilinear
        from ..tunnels.gallery_mesh import build_galleries
        from ..tunnels.tunnel_mesh import build_tunnels

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        tunnel_meshes = build_tunnels(
            tunnel_plans,
            wall_material=config.TUNNEL_MATERIAL_NAME,
            portal_material=config.TUNNEL_MATERIAL_NAME,
            arc_segments=config.TUNNEL_ARC_SEGMENTS,
            transition_cover=config.TUNNEL_TRANSITION_COVER_THICKNESS,
            road_texture_length=config.ROAD_DECAL_TEXTURE_LENGTH,
        )
        gallery_meshes = build_galleries(
            _structure_items(structure_road_polygons, "gallery"),
            ground_at,
            roof_material=config.TUNNEL_MATERIAL_NAME,
            height=config.GALLERY_HEIGHT,
            column_spacing=config.GALLERY_COLUMN_SPACING,
            roof_thickness=config.GALLERY_ROOF_THICKNESS,
            floor_thickness=config.GALLERY_FLOOR_THICKNESS,
            wall_thickness=config.GALLERY_WALL_THICKNESS,
            column_size=config.GALLERY_COLUMN_SIZE,
            curb_height=config.GALLERY_CURB_HEIGHT,
            curb_width=config.GALLERY_CURB_WIDTH,
            road_texture_length=config.ROAD_DECAL_TEXTURE_LENGTH,
        )
        return tunnel_meshes + gallery_meshes

    def _collect_poi_points(
        self,
        osm_data: List[Dict],
        global_offset: Tuple[float, float],
        heights: np.ndarray,
        terrain_origin_x: float,
        terrain_origin_y: float,
        grid_bounds_local: Tuple[float, float, float, float],
    ) -> List[Dict]:
        """POI candidates (villages/towns, large parking lots) with their height on the finished heightmap - see
        osm/poi_points.py. Only within the real terrain area (the OSM query extends beyond the
        terrain, see the generate_vineyards() caller)."""
        if not config.POI_SPAWN_POINTS_ENABLED:
            return []

        from ..osm.landuse_polygons import make_local_transform
        from ..osm.poi_points import extract_parking_points, extract_place_points

        to_local = make_local_transform(global_offset)
        candidates = extract_place_points(osm_data, to_local) + extract_parking_points(
            osm_data, to_local, min_area_m2=config.POI_MIN_PARKING_AREA_M2
        )
        if not candidates:
            return []

        margin = config.POI_SPAWN_BOUNDS_MARGIN
        x_min, x_max, y_min, y_max = grid_bounds_local
        in_bounds = [
            c for c in candidates
            if x_min + margin <= c["position_xy"][0] <= x_max - margin
            and y_min + margin <= c["position_xy"][1] <= y_max - margin
        ]
        if not in_bounds:
            return []

        height_at = make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y)
        xs = np.array([c["position_xy"][0] for c in in_bounds])
        ys = np.array([c["position_xy"][1] for c in in_bounds])
        zs = height_at(xs, ys)
        return [
            {**candidate, "position": [float(x), float(y), float(z)]}
            for candidate, x, y, z in zip(in_bounds, xs, ys, zs)
        ]

    def export_tunnel_zones(self, mesh_data: Dict) -> int:
        """Zone and portal objects that darken the tunnel tubes (see _tunnel_zone_items()).

        Returns:
            Number of zones
        """
        zones = mesh_data.get("tunnel_zones") or []
        for zone in zones:
            self.items.add_item(
                zone["name"],
                item_class=zone["class"],
                position=zone["position"],
                rotation_matrix=zone["rotation_matrix"],
                scale=zone["scale"],
                overwrite=True,
                **zone["fields"],
            )
        return len(zones)

    def export_tunnel_lights(self, mesh_data: Dict) -> int:
        """SpotLight fixtures inside tunnel tubes (see _tunnel_light_items()).

        Returns:
            Number of lights
        """
        lights = mesh_data.get("tunnel_lights") or []
        for light in lights:
            self.items.add_item(
                light["name"],
                item_class=light["class"],
                position=light["position"],
                rotation_matrix=light["rotation_matrix"],
                overwrite=True,
                **light["fields"],
            )
        return len(lights)

    def export_roadblocks(self, mesh_data: Dict) -> int:
        """Roadblocks (see _roadblock_items()) as TSStatic with the BeamNG default asset config.ROADBLOCK_SHAPE.

        Returns:
            Number of barrier elements
        """
        blocks = mesh_data.get("roadblocks") or []
        for block in blocks:
            self.items.add_item(
                block["name"],
                item_class="TSStatic",
                shape_name=config.ROADBLOCK_SHAPE,
                position=block["position"],
                rotation_matrix=block["rotation_matrix"],
                overwrite=True,
            )
        return len(blocks)

    def export_tunnels(self, mesh_data: Dict) -> int:
        """
        Exports tunnels (tube + 2 portal frames per tunnel) and galleries (roof + supports) as ONE DAE with
        ONE TSStatic and registers the road surface and concrete material. Without tunnels/galleries, leftovers of a
        previous export are removed.

        Returns:
            Number of exported tunnel/portal/gallery meshes
        """
        tunnels_dir = config.BEAMNG_DIR_SHAPES / "tunnels"
        meshes = mesh_data.get("tunnel_meshes") or []
        if not config.TUNNELS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (tunnels_dir / f"tunnels{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.TUNNEL_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        # Lamp bodies: glowing without a texture (PBR stage like vanilla's emissive materials)
        self.materials.materials[config.TUNNEL_LAMP_MATERIAL_NAME] = {
            "name": config.TUNNEL_LAMP_MATERIAL_NAME,
            "mapTo": config.TUNNEL_LAMP_MATERIAL_NAME,
            "class": "Material",
            "version": 1.5,
            "Stages": [{
                "baseColorFactor": [*config.TUNNEL_LAMP_COLOR, 1.0],
                "emissive": True,
                "emissiveFactor": list(config.TUNNEL_LAMP_COLOR),
                "emissiveIntensityNits": config.TUNNEL_LAMP_EMISSIVE_NITS,
                "roughnessFactor": 0.4,
            }, {}, {}, {}],
            "castShadows": False,
        }

        unique_floor_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") not in ("tunnel", "gallery"):
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            mat_name = f"{props.get('internal_name', 'road_default')}_structure"
            unique_floor_materials[mat_name] = props

        for mat_name, props in unique_floor_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=tunnels_dir / "tunnels.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "tunnels",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "tunnels" / "tunnels.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.debug(f"  [OK] {len(meshes)} tunnel/gallery mesh(es) exported (tunnels.dae)")
        return len(meshes)

    def _set_fog_height(self, heights: np.ndarray) -> None:
        """
        fogAtmosphereHeight (height above which the height fog thins out) = highest terrain point + margin. All original
        levels set a value on the order of their terrain height; a fixed value would be wrong for our terrain (here
        236-689 m absolute).
        """
        height = float(np.max(heights)) + float(config.ENV_FOG_HEIGHT_MARGIN)
        self.items.set_base_line_fields("theLevelInfo", fogAtmosphereHeight=round(height, 1))

    def export_water(self, mesh_data: Dict) -> int:
        """
        Registers streams (`River`) and ponds/lakes (`WaterBlock`) as BeamNG objects. The render
        parameters come from BeamNG's own east_coast_usa level (data/water_templates.json,
        core textures only), see tools/extract_water_templates.py.

        Returns:
            Number of created water objects
        """
        water = mesh_data.get("water") or {}
        if not config.WATER_ENABLED or not (water.get("rivers") or water.get("ponds")):
            return 0

        import copy

        templates = json.loads(WATER_TEMPLATES_PATH.read_text(encoding="utf-8"))
        count = 0

        for river in water.get("rivers", []):
            fields = copy.deepcopy(templates["stream"]["fields"])
            fields.pop("class", None)
            nodes = river["nodes"]
            self.items.add_item(
                river["name"],
                item_class="River",
                position=tuple(nodes[0][:3]),
                overwrite=True,
                nodes=nodes,
                **fields,
            )
            count += 1

        for pond in water.get("ponds", []):
            for index, block in enumerate(pond["blocks"]):
                fields = copy.deepcopy(templates["pond"]["fields"])
                fields.pop("class", None)
                fields["cubemap"] = config.WATER_POND_CUBEMAP
                # The water grid must not be larger than the block (otherwise BeamNG warns and shortens it itself)
                fields["gridElementSize"] = float(min(fields.get("gridElementSize", 5.0), block["scale"][0], block["scale"][1]))
                self.items.add_item(
                    f"{pond['name']}_{index}",
                    item_class="WaterBlock",
                    position=tuple(block["position"]),
                    scale=tuple(block["scale"]),
                    overwrite=True,
                    **fields,
                )
                count += 1

        logger.debug(f"  [OK] {count} water object(s) exported")
        return count

    def export_walls(self, mesh_data: Dict) -> int:
        """
        Exports the rubble stone walls as ONE DAE (each wall a node) with ONE TSStatic and registers the
        stone material. Without walls, leftovers of a previous export are removed.

        Returns:
            Number of exported walls
        """
        walls_dir = config.BEAMNG_DIR_SHAPES / "walls"
        meshes = mesh_data.get("wall_meshes") or []
        if not config.WALLS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (walls_dir / f"walls{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        stone = registry.prepared_textures()[config.WALL_TEXTURE_NAME]  # if the photo is missing, the export was aborted long ago
        self.materials.add_building_material(
            config.WALL_MATERIAL_NAME,
            textures={**stone, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )
        self.dae.export_multi_mesh(output_path=walls_dir / "walls.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "walls",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "walls" / "walls.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.debug(f"  [OK] {len(meshes)} wall(s) exported (walls.dae)")
        return len(meshes)

    def _export_structure_road_assets(self, marking_lines: List[Dict]) -> None:
        """
        Files for the roads on structures (see config.STRUCTURE_AI_ROADS): the fully transparent texture of the invisible
        AI DecalRoad and the marking lines as mesh strips (geometry/marking_mesh.py) in ONE DAE with ONE TSStatic
        without collision (the strips must not make bumps). Without lines, leftovers of a previous export are removed.
        """
        from PIL import Image

        from ..geometry.marking_mesh import build_marking_meshes

        if config.STRUCTURE_AI_ROADS:
            config.BEAMNG_DIR_TEXTURES.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(config.BEAMNG_DIR_TEXTURES / f"{config.STRUCTURE_AI_ROAD_MATERIAL}.png")

        markings_dir = config.BEAMNG_DIR_SHAPES / "structure_markings"
        texture_lengths = {name: entry["textureLength"] for name, entry in config.OSM_MAPPER.road_markings.items()}
        meshes = build_marking_meshes(marking_lines, texture_lengths, config.STRUCTURE_MARKING_LIFT)
        if not meshes:
            for suffix in (".dae", ".cdae"):
                (markings_dir / f"structure_markings{suffix}").unlink(missing_ok=True)
            return
        self.dae.export_multi_mesh(output_path=markings_dir / "structure_markings.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "structure_markings",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "structure_markings" / "structure_markings.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="None",
        )
        logger.debug(f"  [OK] {len(meshes)} marking strip(s) on structures exported (structure_markings.dae)")

    def export_decal_roads(self, mesh_data: Dict) -> int:
        """
        Exports each road as its own BeamNG `DecalRoad` item - a
        spline decal that is projected directly onto the terrain surface
        at runtime (see the road_embedding.py module docstring for the
        rationale). Completely replaces the former mesh construction (prepare_road_export()/
        export_merged_roads()/RoadMeshBuilder/DAE export) - no
        road mesh, no junction fan geometry, no
        face material majority vote needed anymore: each road gets
        its own DecalRoad item with its own (OSM-derived)
        material, BeamNG's `autoJunction` connects adjacent roads
        automatically.

        Args:
            mesh_data: Result of process_tile() (needs
                "road_slope_polygons_2d")

        Returns:
            Number of created DecalRoad items
        """
        from ..config import OSM_MAPPER
        from ..geometry.decal_chunks import split_decal_nodes
        from ..geometry.road_width_transitions import close_continuation_gaps

        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        unique_materials: Dict[str, Dict] = {}
        specs, node_lists = _road_width_specs(road_slope_polygons_2d, mesh_data.get("dropped_tunnel_road_ids") or frozenset())

        if config.GUARDRAILS_ENABLED and mesh_data.get("heightmap") is not None:
            mesh_data["guardrail_instances"] = _guardrail_instances(specs, node_lists, mesh_data)

        # Extend the carriageway decals at kinked straight-through joints past the joint point (otherwise a wedge gap
        # on the outside, see close_continuation_gaps()). Only for the carriageway - the markings keep using node_lists.
        decal_node_lists = close_continuation_gaps(
            node_lists, config.ROAD_CONTINUATION_ENDPOINT_TOL, config.ROAD_CONTINUATION_MAX_ANGLE_DEG
        )

        # Bridges follow the blended widths of the transitions: export_bridges() rebuilds the deck from them
        mesh_data["bridge_widths"] = {
            poly["road_id"]: nodes
            for (poly, _, _), nodes in zip(specs, node_lists)
            if poly.get("structure_type") == "bridge"
        }

        count = 0
        for (poly, props, _), nodes in zip(specs, decal_node_lists):
            if poly.get("structure_type", "surface") != "surface":
                # Structure: only the AI road network - the visible carriageway is part of the structure mesh
                mat_name = config.STRUCTURE_AI_ROAD_MATERIAL
                self.materials.materials[mat_name] = _invisible_road_material(mat_name)
            else:
                mat_name = props.get("internal_name", "road_default")
                unique_materials[mat_name] = props

            # Derive renderPriority from the existing "priority" field
            # (surface_types in data/osm_to_beamng.json): at junctions the
            # ends (which always have full width) of several DecalRoad
            # objects overlap - without an explicit, consistent
            # drawing order BeamNG sorts them arbitrarily, which looks like a
            # "patchwork" at junctions. BeamNG draws
            # in DESCENDING renderPriority (smallest value on top, see
            # config.ROAD_RENDER_PRIORITY_BASE) - higher-grade roads
            # (asphalt) therefore get the smaller value and lie above
            # lower-grade ones (dirt/concrete).
            render_priority = config.ROAD_RENDER_PRIORITY_BASE - int(props.get("priority", 0))

            # Split long carriageways into pieces - BeamNG only draws a limited amount of geometry per DecalRoad (see
            # geometry/decal_chunks.py). The marking lines stay unsplit: narrow, far below the budget.
            chunks = split_decal_nodes(nodes, config.ROAD_DECAL_MAX_AREA, config.ROAD_DECAL_MIN_TAIL_LENGTH)
            for chunk_idx, chunk in enumerate(chunks):
                name = f"road_{poly.get('road_id')}" if len(chunks) == 1 else f"road_{poly.get('road_id')}_{chunk_idx}"
                self.items.add_decal_road(
                    name=name,
                    nodes=chunk,
                    material=mat_name,
                    drivability=props.get("drivability", 1.0),
                    overwrite=True,
                    autoLanes=True,
                    autoJunction=True,
                    improvedSpline=True,
                    renderPriority=render_priority,
                )
                count += 1

        road_material_entries = [
            OSM_MAPPER.generate_materials_json_entry(mat_name, props) for mat_name, props in unique_materials.items()
        ]
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        # Road markings as their own narrow DecalRoads on top (geometry/road_markings.py). drivability=-1:
        # BeamNG's AI road network (lua/ge/map.lua) only picks up DecalRoads with drivability > 0.
        marking_count = 0
        structure_lines = []
        if config.ROAD_MARKINGS_ENABLED:
            used_markings = set()
            for line in _road_marking_lines(specs, node_lists):
                used_markings.add(line["material"])
                if line["structure"]:
                    structure_lines.append(line)  # mesh strips on the structure floor, see _export_structure_road_assets()
                    continue
                marking = OSM_MAPPER.road_markings[line["material"]]
                self.items.add_decal_road(
                    name=line["name"],
                    nodes=line["nodes"],
                    material=line["material"],
                    drivability=-1,
                    overwrite=True,
                    improvedSpline=True,
                    textureLength=marking["textureLength"],
                    renderPriority=config.ROAD_MARKING_RENDER_PRIORITY,
                )
                marking_count += 1
            for mat_name in sorted(used_markings):
                self.materials.materials[mat_name] = OSM_MAPPER.generate_marking_material_entry(
                    mat_name, OSM_MAPPER.road_markings[mat_name]
                )
        self._export_structure_road_assets(structure_lines)

        logger.debug(
            f"  [OK] {count} DecalRoad item(s) exported ({len(unique_materials)} materials), "
            f"{marking_count} marking line(s)"
        )
        return count

    def export_ground_cover(
        self, layer_map: np.ndarray, terrain_material_names: List[str], layer_variants: Optional[Dict] = None
    ) -> int:
        """
        Registers ground cover (grass, flowers, fern, weeds) as GroundCover
        objects for each terrain layer that actually occurs in the layer map,
        together with the billboard materials (shared BeamNG assets).

        Args:
            layer_map: finished global layer map (index into terrain_material_names)
            terrain_material_names: Layer names in index order
            layer_variants: Four-photo mode: layer -> tile variants (names in terrain_material_names)

        Returns:
            Number of created GroundCover objects
        """
        if not config.GROUND_COVER_ENABLED:
            return 0

        from ..terrain.ground_cover import (
            build_billboard_material_entries,
            build_ground_cover_items,
            load_ground_cover_templates,
        )

        used_physical = [terrain_material_names[i] for i in np.unique(layer_map) if i < len(terrain_material_names)]
        # In four-photo mode the layer map only contains variants (mat_grass_t0 ...): map back to the layer
        logical_of = {variant: layer for layer, variants in (layer_variants or {}).items() for variant in variants}
        used_layers = list(dict.fromkeys(logical_of.get(name, name) for name in used_physical))
        templates_data = load_ground_cover_templates()
        items = build_ground_cover_items(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            used_layers,
            templates_data,
            max_elements=config.GROUND_COVER_MAX_ELEMENTS,
            max_radius=config.GROUND_COVER_MAX_RADIUS,
            layer_variants=layer_variants,
        )
        for item in items:
            fields = dict(item)
            self.items.add_ground_cover(fields.pop("name"), fields.pop("material"), fields.pop("Types"), **fields)

        self.materials.materials.update(build_billboard_material_entries(items, templates_data))
        logger.info(f"  [OK] {len(items)} GroundCover object(s) for {len(used_layers) - 1} land use layers")
        return len(items)

    def export_merged_terrain(
        self,
        heights: np.ndarray,
        layer_map: np.ndarray,
        terrain_material_names: List[str],
        terrain_origin_x: float,
        terrain_origin_y: float,
        terrain_size: int,
        z_min: float,
        max_height: float,
        photo_tile_names: List[str],
        layer_variants: Optional[Dict] = None,
        variant_parents: Optional[Dict] = None,
        photo_extents: Optional[Dict] = None,
    ) -> None:
        """
        Writes ONE .ter and registers TerrainBlock + TerrainMaterials.

        Args:
            heights, layer_map: finished (already padded) global arrays,
                shape (terrain_size, terrain_size)
            terrain_material_names: global, deduplicated material list
                (index corresponds to layer_map values)
            terrain_origin_x, terrain_origin_y: World coordinates of cell [0, 0]
            z_min, max_height: see ter_writer.encode_heights_to_u16()
            photo_tile_names: Subset of terrain_material_names that refer to
                the composed aerial photo (see
                terrain_materials.build_terrain_material_entries)
        """
        from ..terrain.ter_writer import write_ter, encode_heights_to_u16
        from ..terrain.terrain_materials import (
            build_terrain_material_entries,
            build_terrain_material_texture_set,
            ensure_flat_pbr_placeholders,
            ensure_landuse_detail_textures_sized,
            DETAIL_TEX_SIZE,
        )

        self._set_fog_height(heights)
        heightmap_u16 = encode_heights_to_u16(heights, z_min, max_height)
        ter_filename = f"{config.LEVEL_NAME}.ter"
        ter_path = config.BEAMNG_DIR / ter_filename
        write_ter(ter_path, heightmap_u16, layer_map.astype("uint8"), terrain_material_names)
        logger.debug(f"  [OK] Terrain exported: {ter_filename} ({terrain_size}x{terrain_size})")

        placeholders = ensure_flat_pbr_placeholders(
            config.BEAMNG_DIR_TEXTURES, config.LEVEL_NAME, config.TERRAIN_BASE_TEX_PIXEL_SIZE
        )
        sized_landuse_mappings = ensure_landuse_detail_textures_sized(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            DETAIL_TEX_SIZE,
            config.BEAMNG_DIR,
            config.BEAMNG_DIR_TEXTURES,
            config.LEVEL_NAME,
        )
        # photo_extent_size: the value we report to BeamNG as baseColorBaseTexSize
        # for the aerial photo material.
        #
        # DIAGNOSIS 2026-09-18 (three measurement points with a real user test, see
        # the [[project_road_embed_slope_margin]] neighbor memory for the context
        # of this session):
        #   - Grid @ 2m/cell: terrain_size=1024, TERRAIN_SQUARE_SIZE=2.0,
        #     physical size 2048m -> declared value 1024 was correct.
        #   - Grid @ 1m/cell: terrain_size=2048, TERRAIN_SQUARE_SIZE=1.0,
        #     physical size UNCHANGED 2048m -> the same value 1024 (from the
        #     old "physical_size * 0.5" formula) was now WRONG, the correct one
        #     is 2048.
        # The physical map size stayed identical in both cases (2048m),
        # only the grid resolution changed - nevertheless the
        # correct value had to double with the grid resolution. That means
        # BeamNG apparently interprets baseColorBaseTexSize in heightmap
        # grid cells, NOT in world meters (contrary to the
        # documented formula "world_size = size * squareSize"): the correct value
        # is simply terrain_size, the plain .ter grid point count, completely
        # independent of TERRAIN_SQUARE_SIZE.
        photo_extent_size = float(terrain_size)
        terrain_material_entries = build_terrain_material_entries(
            terrain_material_names,
            photo_tile_names,
            sized_landuse_mappings,
            config.LEVEL_NAME,
            photo_extent_size,
            placeholders,
            variant_parents=variant_parents,
            photo_extents=photo_extents,
        )
        texture_set_name = f"{config.LEVEL_NAME}TerrainMaterialTextureSet"
        terrain_material_entries.update(
            build_terrain_material_texture_set(
                texture_set_name, base_tex_size=config.TERRAIN_BASE_TEX_PIXEL_SIZE
            )
        )
        self.materials.add_terrain_materials(terrain_material_entries)

        self.export_ground_cover(layer_map, terrain_material_names, layer_variants=layer_variants)

        self.items.add_terrain_block(
            name="theTerrain",
            terrain_filename=ter_filename,
            material_texture_set=texture_set_name,
            max_height=max_height,
            z_min=z_min,
            origin_x=terrain_origin_x,
            origin_y=terrain_origin_y,
            square_size=config.TERRAIN_SQUARE_SIZE,
            overwrite=True,
        )

    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict, task: PipelineTask) -> int:
        """
        Export ONE single tile completely (DecalRoad roads + its own .ter).

        Convenience wrapper around export_decal_roads()/export_merged_terrain()
        for callers that only export a single tile
        (export_single_tile()/export_terrain_only()).

        Args:
            tile_x, tile_y: unused, only for caller compatibility
            mesh_data: Mesh data from process_tile()
            task: PipelineTask for the progress display of the subtasks

        Returns:
            Number of created DecalRoad items
        """
        with task.subtask("DecalRoads") as sub:
            road_count = self.export_decal_roads(mesh_data)
            sub.finish(f"{road_count} roads" if road_count else "no roads")

        with task.subtask("Water") as sub:
            count = self.export_water(mesh_data)
            sub.finish(f"{count} objects" if count else "no water areas")

        with task.subtask("Walls") as sub:
            count = self.export_walls(mesh_data)
            sub.finish(f"{count} walls" if count else "no walls")

        with task.subtask("Bridges") as sub:
            count = self.export_bridges(mesh_data)
            sub.finish(f"{count} bridges" if count else "no bridges")

        with task.subtask("Tunnels/galleries") as sub:
            count = self.export_tunnels(mesh_data)
            blocked = self.export_roadblocks(mesh_data)
            zones = self.export_tunnel_zones(mesh_data)
            lights = self.export_tunnel_lights(mesh_data)
            sub.finish(
                (f"{count} mesh(es)" if count else "no tunnels/galleries")
                + (f", {zones} darkness zones" if zones else "")
                + (f", {lights} lights" if lights else "")
                + (f", {blocked} barrier elements" if blocked else "")
            )

        with task.subtask("Terrain export") as sub:
            self.export_merged_terrain(
                heights=mesh_data["heightmap"],
                layer_map=mesh_data["layer_map"],
                terrain_material_names=list(mesh_data["terrain_material_names"]),
                terrain_origin_x=mesh_data["terrain_origin_x"],
                terrain_origin_y=mesh_data["terrain_origin_y"],
                terrain_size=mesh_data["terrain_size"],
                z_min=mesh_data["z_min"],
                max_height=mesh_data["max_height"],
                photo_tile_names=mesh_data["photo_tile_names"],
                layer_variants=mesh_data.get("layer_variants"),
                variant_parents=mesh_data.get("variant_parents"),
                photo_extents=mesh_data.get("photo_extents"),
            )
            sub.finish()

        return road_count
