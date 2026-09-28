"""
DecalRoad side of the terrain workflow: which roads get a DecalRoad, their blended width nodes, the marking lines
and the guard rails along them.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..geometry.polyline import arc_lengths

logger = logging.getLogger(__name__)


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
        [(poly.get("osm_tags") or {}).get("highway") not in config.GUARDRAIL_EXCLUDED_HIGHWAYS and not poly.get("sidewalk_sides")
         for poly, _ in surface],
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


def _sidewalk_meshes(specs: List[Tuple[Dict, Dict, List]], node_lists: List[List[List[float]]], corners=()) -> List[Dict]:
    """Kerb + sidewalk mesh dicts (sidewalks/) along the surface roads with "sidewalk_sides", from the finished DecalRoad
    nodes - the kerb stands exactly at the carriageway edge the decal is drawn to. `specs`/`node_lists` as in
    export_decal_roads(); `corners`: junction corner dicts - runs end at their tangent points and are joined through the
    arc where both arms have a sidewalk on the corner side."""
    from ..sidewalks.runs import plan_sidewalk_runs
    from ..sidewalks.sidewalk_mesh import build_sidewalk_mesh

    surface = [(poly, nodes) for (poly, _, _), nodes in zip(specs, node_lists) if poly.get("structure_type", "surface") == "surface"]
    runs = plan_sidewalk_runs(
        [nodes for _, nodes in surface],
        [poly.get("sidewalk_sides") or {} for poly, _ in surface],
        [(poly.get("osm_tags") or {}).get("highway") not in config.SIDEWALK_EXCLUDED_HIGHWAYS for poly, _ in surface],
        clearance=config.SIDEWALK_KERB_WIDTH + config.SIDEWALK_WIDTH,
        min_length=config.SIDEWALK_MIN_LENGTH,
        endpoint_tol=config.ROAD_CONTINUATION_ENDPOINT_TOL,
        max_angle_deg=config.ROAD_CONTINUATION_MAX_ANGLE_DEG,
        corners=corners,
        road_ids=[poly.get("road_id") for poly, _ in surface],
    )
    meshes = []
    for number, run in enumerate(runs):
        mesh = build_sidewalk_mesh(
            run["points"], config.SIDEWALK_KERB_WIDTH, config.SIDEWALK_WIDTH, config.SIDEWALK_KERB_HEIGHT,
            config.SIDEWALK_SKIRT_DEPTH, config.SIDEWALK_MAX_SEGMENT, config.SIDEWALK_TEXTURE_TILE_M,
            config.BRIDGE_MATERIAL_NAME, f"{run['surface']}_structure",
        )
        meshes.append({"id": f"sidewalk_{number}", **mesh})
    length = sum(float(arc_lengths(run["points"][:, :2])[-1]) for run in runs)
    logger.info(f"  [OK] {len(runs)} sidewalk run(s), {length:.0f} m")
    return meshes


def _widths_along(coords, nodes) -> Optional[np.ndarray]:
    """Width per coordinate of `coords` from DecalRoad `nodes` [x, y, z, width], by arc length (None without nodes). The
    node list is the same polyline with some nodes dropped or inserted, so the arc lengths agree closely."""
    if nodes is None or len(nodes) < 2:
        return None
    points, node_array = np.asarray(coords, dtype=float), np.asarray(nodes, dtype=float)

    node_arc, point_arc = arc_lengths(node_array[:, :2]), arc_lengths(points[:, :2])
    scale = node_arc[-1] / point_arc[-1] if point_arc[-1] > 0.0 else 1.0
    return np.interp(point_arc * scale, node_arc, node_array[:, 3])


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
        # Lane splits (geometry/lane_splits.py): the branches start in their lanes of the trunk, the trunk keeps its width
        split_trunk_ends={(i, end) for i, (poly, _, _) in enumerate(specs) for end in poly.get("lane_split_trunk", ())},
        split_branches={
            i: tuple(poly["lane_split_branch"][key] for key in ("end", "slot_width", "hold", "length"))
            for i, (poly, _, _) in enumerate(specs) if poly.get("lane_split_branch")
        },
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
        no_overtaking_masks,
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
    # The double line of the wider road (no overtaking) continues on the two-lane road beyond the transition
    double_lines = no_overtaking_masks(node_lists, layouts, own_widths, fixed, pairs, config.ROAD_MARKING_NO_OVERTAKING_EXTRA)
    # Other joints: the centre line of a narrower road (2 lanes) runs onto the double line of a wider one (3+ lanes), and at a
    # structure the road's lines are aligned with the structure's 50 m before it (the width still changes up to it)
    plain_pairs = [pair for pair in pairs if pair not in {zone["pair"] for zone in zones}]
    for road_index, shift in boundary_shifts(node_lists, layouts, own_widths, plain_pairs, fixed).items():
        shifts[road_index] = shifts.get(road_index, 0.0) + shift
    for road_index, shift in structure_boundary_shifts(
        node_lists, layouts, fixed, plain_pairs, config.ROAD_STRUCTURE_TRANSITION_LENGTH, config.ROAD_STRUCTURE_LINES_DONE_AT
    ).items():
        shifts[road_index] = shifts.get(road_index, 0.0) + shift

    # Lane splits: on the stem the boundary between two branches is one block marking, behind it their edge lines move
    # apart with the carriageways; the roads of a split do not cut each other's lines
    from ..geometry.lane_splits import stem_marking_masks

    stems = {}
    for poly, _, _ in specs:
        mark = poly.get("lane_split_branch")
        if mark and mark.get("stem_path"):
            stems[(round(mark["node"][0], 2), round(mark["node"][1], 2))] = mark
    stem_masks = stem_marking_masks(node_lists, list(stems.values()))

    lines = []
    for index, ((poly, props, _), nodes) in enumerate(zip(specs, node_lists)):
        layout = layouts[index]
        if layout is None:
            continue
        stem_mask = stem_masks.get(index, {"edge_keep": {}, "blocks": [], "siblings": set()})
        obstacles = junction_obstacles(
            index,
            polygons,
            tree,
            excluded=partners.get(index, set()) | no_gap | stem_mask["siblings"],
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
            blocks=(blocks.get(index) or []) + stem_mask["blocks"] or None,
            double_keep=double_lines.get(index),
            edge_keep=stem_mask["edge_keep"] or None,
            # half the double line (gap + line) plus half the block stripe: closer and the block would paint over it
            block_clearance=(config.ROAD_MARKING_CENTER_GAP + config.ROAD_MARKING_LINE_WIDTH) / 2.0
            + config.ROAD_MARKING_LINE_WIDTH / 2.0 + config.ROAD_MARKING_BLOCK_WIDTH / 2.0,
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
