"""
Galleries from OSM lines (highway=* with tunnel=avalanche_protector): like a tunnel, but open on the valley side (roof +
supports instead of a second wall) - see design spec section 6. No portal frames: galleries are not openings cut
into the rock, but open protective structures along the road - their ends stay square.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def resolve_open_side(osm_tags: Dict) -> Optional[str]:
    """
    Reads the valley-side open wall directly from the OSM tags `avalanche_protector:left`/
    `avalanche_protector:right` (value "open"), if present - "left"/"right" follow the
    digitization direction of the way, exactly the same convention as offset_points()/valley_side().

    Much more reliable than the height comparison in valley_side(): at the structure the DGM does not capture
    the original terrain, but the already finished gallery including earth cover/roof - a "natural"
    terrain height left/right of the centerline thus does not exist at this location at all, so the gallery
    may well disappear completely into the (in reality not natural at all) "terrain".

    Returns:
        "left" | "right" | None (no tag present -> caller must fall back to valley_side())
    """
    if str(osm_tags.get("avalanche_protector:left", "")).lower() == "open":
        return "left"
    if str(osm_tags.get("avalanche_protector:right", "")).lower() == "open":
        return "right"
    return None


VALLEY_PROBE_OFFSETS = (10.0, 20.0, 40.0)  # Distances beyond the carriageway edge, in meters


def valley_score(xy: np.ndarray, ground_at: HeightAt, half_width: float) -> np.ndarray:
    """
    Per point: sum of (terrain left - terrain right) at VALLEY_PROBE_OFFSETS meters beyond the carriageway edge
    (positive = right lower = valley on the right). Deliberately measured OUTSIDE the embedding: directly next to the
    gallery the terrain is flat after the embedding (carriageway, mountain wall border, embankment) - there
    centimeters or a tie decided the side (Nuova strada 2026-09-24: both galleries open toward the mountain).
    """
    directions = np.diff(xy, axis=0)
    directions = np.vstack([directions, directions[-1:]])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    directions = directions / norms
    # same sign convention as offset_points(): left = point + (-dy, dx)
    perp = np.column_stack([-directions[:, 1], directions[:, 0]])
    score = np.zeros(len(xy))
    for offset in VALLEY_PROBE_OFFSETS:
        distance = half_width + offset
        left_xy, right_xy = xy + perp * distance, xy - perp * distance
        score += np.asarray(ground_at(left_xy[:, 0], left_xy[:, 1]), float) - np.asarray(ground_at(right_xy[:, 0], right_xy[:, 1]), float)
    return score


def valley_side(xy: np.ndarray, ground_at: HeightAt, half_width: float) -> np.ndarray:
    """
    Per point: +1.0 if the side to the RIGHT of the direction of travel lies downhill, otherwise -1.0 (see valley_score()).

    ONLY a fallback for the case without an `avalanche_protector:left`/`:right` tag (see resolve_open_side()).
    """
    return np.where(valley_score(xy, ground_at, half_width) > 0.0, 1.0, -1.0)


def gallery_open_side(osm_tags: Dict, coords, ground_at: HeightAt, width: float) -> str:
    """
    Open (valley) side of a gallery in digitization direction: from `avalanche_protector:left/right=open`, otherwise from
    the terrain comparison (sum of valley_score() over the whole gallery, >= 0 -> "right"). ONE place for the gallery
    mesh and the embankment (terrain_workflow), so that both take the same side.
    """
    tagged = resolve_open_side(osm_tags or {})
    if tagged:
        return tagged
    xy = np.asarray(coords, dtype=float)[:, :2]
    return "right" if float(valley_score(xy, ground_at, width / 2.0).sum()) >= 0.0 else "left"


def build_gallery_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    height: float,
    ground_at: HeightAt,
    floor_material: str,
    roof_material: str,
    column_spacing: float = 6.0,
    roof_thickness: float = 0.5,
    floor_thickness: float = 5.0,
    wall_thickness: float = 5.0,
    column_size: float = 0.4,
    curb_height: float = 0.5,
    curb_width: float = 0.4,
    tile_m: float = 5.0,
    open_side: Optional[str] = None,
    cap_start: bool = True,
    cap_end: bool = True,
) -> Dict:
    """
    Gallery mesh: floor, roof and mountain-side wall are real boxes (not just thin faces) - floor
    floor_thickness downward, roof roof_thickness upward, mountain-side wall wall_thickness further into
    the slope (flush with the roof top edge). In addition a continuous plinth (curb_height/curb_width,
    like the curb on bridges) on the valley-side open side (no wall there) AND supports that sit FLUSH
    on the plinth: centered on its centerline in plan (curb_width == column_size ->
    support outer edge == carriageway edge == roof edge, everything flush) and in height resting on the plinth
    top edge instead of sinking into the floor (support height shortened by curb_height accordingly, the top edge
    stays at the roof bottom edge). Both ends are closed completely (floor/roof/wall cross-section)
    - looks like a clean cut through the structure, exactly at the original OSM way boundary points (no
    artificial extension of the centerline).

    Args:
        floor_thickness, wall_thickness: see config.GALLERY_FLOOR_THICKNESS/GALLERY_WALL_THICKNESS
        curb_height, curb_width: see config.GALLERY_CURB_HEIGHT/GALLERY_CURB_WIDTH - plinth on the
            support side, curb_width offset inward from the carriageway edge; column_size should match curb_width
            so that the support sits flush on the plinth (see docstring above)
        open_side: "left" | "right" | None - if set (from resolve_open_side(), reliable OSM tag),
            this side counts as open for the ENTIRE gallery. Without a tag, ONE side also applies to the whole
            gallery: the majority of the per-point valley_side() (height comparison, fallback only).
        cap_start, cap_end: Build the end face at the start/end (default: both).

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], roof_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    floor_bottom_z = floor_z - floor_thickness
    roof_bottom_z = floor_z + height
    roof_top_z = roof_bottom_z + roof_thickness
    curb_top_z = floor_z + curb_height

    left, right = offset_points(xy, width / 2.0, closed=False)
    outer_left, outer_right = offset_points(xy, width / 2.0 + wall_thickness, closed=False)
    inner_left, inner_right = offset_points(xy, max(width / 2.0 - curb_width, 0.0), closed=False)
    # Centerline of the plinth footprint (between curb_inner and curb_edge = left/right) - the supports
    # sit centered there (flush with the plinth footprint, see support loop below). The element-wise
    # mean of two offset_points() results on the same (possibly mitered) normal is
    # exactly equivalent to a separate offset_points() call with the averaged distance.
    mid_left = (left + inner_left) / 2.0
    mid_right = (right + inner_right) / 2.0
    # +1 = open on the right (valley), -1 = open on the left - see open_side/resolve_open_side() docstring.
    if open_side == "left":
        side = np.full(len(points), -1.0)
    elif open_side == "right":
        side = np.full(len(points), 1.0)
    else:
        # Without a tag: ONE side for the whole gallery (majority of the per-point valley side) - a gallery does not
        # switch its open side midway, but the per-point terrain comparison flips easily at the structure.
        # Sum of the height differences instead of counting points: no silent tie at half/half
        total = float(valley_score(xy, ground_at, width / 2.0).sum())
        side = np.full(len(points), 1.0 if total >= 0.0 else -1.0)

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / tile_m
    across = width / tile_m
    # The wall ends flush with the roof TOP EDGE (not just the interior height) - so it reaches
    # height + roof_thickness, not just height.
    wall_h = (height + roof_thickness) / tile_m
    floor_h = floor_thickness / tile_m
    roof_h = roof_thickness / tile_m
    wall_extra = wall_thickness / tile_m
    curb_h = curb_height / tile_m
    curb_w = curb_width / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    floor_builder = MeshBuilder()
    roof_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        side_normal = [float(-direction[1]), float(direction[0]), 0.0]
        # mountain-side wall: the side that (at this segment) does NOT lie downhill; on a change
        # in the middle of a segment (rare) the side at the segment start wins - accepted simplification.
        mountain_is_left = side[i] > 0

        # Floor: carriageway top (road material) + bottom + both side faces (box).
        floor_builder.quad(
            [p3(left[i], floor_z[i]), p3(left[j], floor_z[j]), p3(right[j], floor_z[j]), p3(right[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
        roof_builder.quad(
            [p3(left[i], floor_bottom_z[i]), p3(right[i], floor_bottom_z[i]), p3(right[j], floor_bottom_z[j]), p3(left[j], floor_bottom_z[j])],
            [[u0, 0.0], [u0, across], [u1, across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        roof_builder.quad(
            [p3(left[i], floor_bottom_z[i]), p3(left[j], floor_bottom_z[j]), p3(left[j], floor_z[j]), p3(left[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, floor_h], [u0, floor_h]],
            [float(side_normal[0]), float(side_normal[1]), 0.0],
        )
        roof_builder.quad(
            [p3(right[i], floor_z[i]), p3(right[j], floor_z[j]), p3(right[j], floor_bottom_z[j]), p3(right[i], floor_bottom_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, floor_h], [u0, floor_h]],
            [-float(side_normal[0]), -float(side_normal[1]), 0.0],
        )

        # Roof: bottom/top (as before) + now additionally both side faces (box instead of slab).
        roof_builder.quad(
            [p3(left[i], roof_bottom_z[i]), p3(right[i], roof_bottom_z[i]), p3(right[j], roof_bottom_z[j]), p3(left[j], roof_bottom_z[j])],
            [[u0, 0.0], [u0, across], [u1, across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        roof_builder.quad(
            [p3(left[i], roof_top_z[i]), p3(left[j], roof_top_z[j]), p3(right[j], roof_top_z[j]), p3(right[i], roof_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
        # Roof side faces (roof_bottom_z to roof_top_z): only needed on the VALLEY SIDE - on the mountain side
        # the wall inner face (now reaching up to roof_top_z) already covers the same area, an additional
        # roof side face there would be coincident geometry (z-fighting).
        if not mountain_is_left:
            roof_builder.quad(
                [p3(left[i], roof_bottom_z[i]), p3(left[j], roof_bottom_z[j]), p3(left[j], roof_top_z[j]), p3(left[i], roof_top_z[i])],
                [[u0, 0.0], [u1, 0.0], [u1, roof_h], [u0, roof_h]],
                [float(side_normal[0]), float(side_normal[1]), 0.0],
            )
        if mountain_is_left:
            roof_builder.quad(
                [p3(right[i], roof_top_z[i]), p3(right[j], roof_top_z[j]), p3(right[j], roof_bottom_z[j]), p3(right[i], roof_bottom_z[i])],
                [[u0, 0.0], [u1, 0.0], [u1, roof_h], [u0, roof_h]],
                [-float(side_normal[0]), -float(side_normal[1]), 0.0],
            )

        # Mountain-side wall (box, wall_thickness further into the slope): ends flush with the
        # roof TOP EDGE (roof_top_z, not just roof_bottom_z) - hence 50 cm/roof_thickness higher than
        # the clear interior height.
        edge = left if mountain_is_left else right
        outer_edge = outer_left if mountain_is_left else outer_right
        wall_normal = [-side_normal[0], -side_normal[1], 0.0] if mountain_is_left else side_normal
        outward_normal = [-wall_normal[0], -wall_normal[1], 0.0]

        roof_builder.quad(  # inner face (visible from the interior, hidden from outside above the ceiling)
            [p3(edge[i], floor_z[i]), p3(edge[j], floor_z[j]), p3(edge[j], roof_top_z[j]), p3(edge[i], roof_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, wall_h], [u0, wall_h]],
            wall_normal,
        )
        roof_builder.quad(  # outer face, wall_thickness further into the slope
            [p3(outer_edge[i], floor_z[i]), p3(outer_edge[i], roof_top_z[i]), p3(outer_edge[j], roof_top_z[j]), p3(outer_edge[j], floor_z[j])],
            [[u0, 0.0], [u0, wall_h], [u1, wall_h], [u1, 0.0]],
            outward_normal,
        )
        roof_builder.quad(  # wall bottom (floor level, inner to outer edge)
            [p3(edge[i], floor_z[i]), p3(outer_edge[i], floor_z[i]), p3(outer_edge[j], floor_z[j]), p3(edge[j], floor_z[j])],
            [[u0, 0.0], [u0, wall_extra], [u1, wall_extra], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        roof_builder.quad(  # wall top, now flush with the roof top edge (inner to outer edge)
            [p3(edge[i], roof_top_z[i]), p3(edge[j], roof_top_z[j]), p3(outer_edge[j], roof_top_z[j]), p3(outer_edge[i], roof_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, wall_extra], [u0, wall_extra]],
            [0.0, 0.0, 1.0],
        )

        # Plinth (curb-like, as in bridges/bridge_mesh.py) on the valley-side open side - curb_width
        # offset inward from the carriageway edge, curb_height high.
        curb_edge = right if mountain_is_left else left
        curb_inner = inner_right if mountain_is_left else inner_left
        curb_outward = [-side_normal[0], -side_normal[1], 0.0] if mountain_is_left else side_normal

        roof_builder.quad(  # plinth top (concrete material like wall/roof, not carriageway material)
            [p3(curb_edge[i], curb_top_z[i]), p3(curb_edge[j], curb_top_z[j]), p3(curb_inner[j], curb_top_z[j]), p3(curb_inner[i], curb_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_w], [u0, curb_w]],
            [0.0, 0.0, 1.0],
        )
        roof_builder.quad(  # plinth outer face (facing the valley side)
            [p3(curb_edge[i], floor_z[i]), p3(curb_edge[j], floor_z[j]), p3(curb_edge[j], curb_top_z[j]), p3(curb_edge[i], curb_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_h], [u0, curb_h]],
            curb_outward,
        )
        roof_builder.quad(  # plinth inner face (facing the carriageway)
            [p3(curb_inner[i], curb_top_z[i]), p3(curb_inner[j], curb_top_z[j]), p3(curb_inner[j], floor_z[j]), p3(curb_inner[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_h], [u0, curb_h]],
            [-float(curb_outward[0]), -float(curb_outward[1]), 0.0],
        )

    end_cap_args = (
        left, right, outer_left, outer_right, inner_left, inner_right, floor_z, floor_bottom_z,
        roof_bottom_z, roof_top_z, curb_top_z, side, across, floor_h, roof_h, wall_extra, wall_h, curb_h, curb_w,
    )
    if cap_start:
        _add_end_caps(roof_builder, 0, xy[0] - xy[1], *end_cap_args)
    if cap_end:
        _add_end_caps(roof_builder, len(points) - 1, xy[-1] - xy[-2], *end_cap_args)

    cum = np.concatenate([[0.0], np.cumsum(steps)])
    total_len = float(cum[-1]) if len(cum) else 0.0
    column_positions = np.arange(column_spacing / 2.0, total_len, column_spacing) if total_len > 0 else np.array([])
    for s in column_positions:
        idx = max(1, min(int(np.searchsorted(cum, s)), len(points) - 1))
        t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
        # Flush on the plinth: centered on its centerline in plan (mid_left/mid_right, see
        # above) instead of on the carriageway edge - plinth and support have the same footprint.
        open_edge = mid_right if side[idx - 1] > 0 else mid_left
        cx = open_edge[idx - 1, 0] + t * (open_edge[idx, 0] - open_edge[idx - 1, 0])
        cy = open_edge[idx - 1, 1] + t * (open_edge[idx, 1] - open_edge[idx - 1, 1])
        floor_base = float(floor_z[idx - 1] + t * (floor_z[idx] - floor_z[idx - 1]))
        # Base on the plinth top edge (instead of floor level) - otherwise the support is half sunk into the plinth.
        # The top edge stays at floor_z + height (roof bottom edge, unchanged), so the support becomes
        # curb_height shorter than before.
        column_bottom = floor_base + curb_height
        column_top = floor_base + height
        # Profile aligned relative to the gallery direction (not axis-parallel to the world) - otherwise
        # the supports of diagonally running galleries visibly stand skewed to the wall/roof edge.
        column_direction = xy[idx] - xy[idx - 1]
        add_box_column(
            roof_builder, cx, cy, column_bottom, column_top, column_size, tile_m, direction=tuple(column_direction)
        )

    all_vertices = floor_builder.vertices + roof_builder.vertices
    all_uvs = floor_builder.uvs + roof_builder.uvs
    all_normals = floor_builder.normals + roof_builder.normals
    roof_offset = len(floor_builder.vertices)
    roof_faces = [[a + roof_offset, b + roof_offset, c + roof_offset] for a, b, c in roof_builder.faces]

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {floor_material: floor_builder.faces, roof_material: roof_faces},
    }


def _add_end_caps(
    builder: "MeshBuilder",
    idx: int,
    outward_xy: np.ndarray,
    left, right, outer_left, outer_right, inner_left, inner_right,
    floor_z, floor_bottom_z, roof_bottom_z, roof_top_z, curb_top_z,
    side, across, floor_h, roof_h, wall_extra, wall_h, curb_h, curb_w,
) -> None:
    """
    End face at one end (idx=0 or idx=len-1): full floor cross-section (box thickness) + full
    roof cross-section + wall cross-section (only its own footprint, inner to outer edge, up to
    roof_top_z - the wall ends flush with the roof top edge) + plinth cross-section on the
    support side - turns the open shell end into a clean, solid cut instead of a
    view into the hollow space.
    """
    norm = float(np.hypot(outward_xy[0], outward_xy[1]))
    normal = [float(outward_xy[0] / norm), float(outward_xy[1] / norm), 0.0] if norm > 1e-9 else [1.0, 0.0, 0.0]

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    builder.quad(  # floor end face
        [p3(left[idx], floor_bottom_z[idx]), p3(right[idx], floor_bottom_z[idx]), p3(right[idx], floor_z[idx]), p3(left[idx], floor_z[idx])],
        [[0.0, 0.0], [across, 0.0], [across, floor_h], [0.0, floor_h]],
        normal,
    )
    builder.quad(  # roof end face
        [p3(left[idx], roof_bottom_z[idx]), p3(right[idx], roof_bottom_z[idx]), p3(right[idx], roof_top_z[idx]), p3(left[idx], roof_top_z[idx])],
        [[0.0, 0.0], [across, 0.0], [across, roof_h], [0.0, roof_h]],
        normal,
    )

    mountain_is_left = side[idx] > 0
    edge_pt = left[idx] if mountain_is_left else right[idx]
    outer_pt = outer_left[idx] if mountain_is_left else outer_right[idx]
    builder.quad(  # wall end face (only the wall footprint: inner to outer edge, up to the roof top edge)
        [p3(edge_pt, floor_z[idx]), p3(outer_pt, floor_z[idx]), p3(outer_pt, roof_top_z[idx]), p3(edge_pt, roof_top_z[idx])],
        [[0.0, 0.0], [wall_extra, 0.0], [wall_extra, wall_h], [0.0, wall_h]],
        normal,
    )

    curb_edge_pt = right[idx] if mountain_is_left else left[idx]
    curb_inner_pt = inner_right[idx] if mountain_is_left else inner_left[idx]
    builder.quad(  # plinth end face (only the plinth footprint: carriageway edge to curb_width inward)
        [p3(curb_inner_pt, floor_z[idx]), p3(curb_edge_pt, floor_z[idx]), p3(curb_edge_pt, curb_top_z[idx]), p3(curb_inner_pt, curb_top_z[idx])],
        [[0.0, 0.0], [curb_w, 0.0], [curb_w, curb_h], [0.0, curb_h]],
        normal,
    )


def build_galleries(
    galleries: Sequence[Dict],
    ground_at: HeightAt,
    roof_material: str,
    height: float = 5.0,
    column_spacing: float = 6.0,
    roof_thickness: float = 0.5,
    floor_thickness: float = 5.0,
    wall_thickness: float = 5.0,
    column_size: float = 0.4,
    curb_height: float = 0.5,
    curb_width: float = 0.4,
) -> List[Dict]:
    """Mesh dicts for the DAE export, one per gallery (`galleries`: [{"id","coords","width","floor_material",
    "osm_tags"}, ...] - "osm_tags" optional, for resolve_open_side()). Both ends get an end face, also at the
    transition into a tunnel (there the round portal only closes the tube cross-section, see
    tunnel_portal.transition_regions())."""

    meshes = []
    for gallery in galleries:
        coords = gallery["coords"]
        if len(coords) < 2:
            continue
        mesh = build_gallery_mesh(
            coords, gallery["width"], height, ground_at, gallery["floor_material"], roof_material,
            column_spacing=column_spacing, roof_thickness=roof_thickness, floor_thickness=floor_thickness,
            wall_thickness=wall_thickness, column_size=column_size, curb_height=curb_height, curb_width=curb_width,
            # Default from the embankment logic (terrain_workflow._gallery_embedding), otherwise tag or terrain
            open_side=gallery.get("open_side") or resolve_open_side(gallery.get("osm_tags", {})),
        )
        meshes.append({"id": f"gallery_{gallery['id']}", **mesh})
    return meshes
