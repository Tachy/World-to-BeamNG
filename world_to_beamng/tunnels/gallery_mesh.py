"""
Galerien aus OSM-Linien (highway=* mit tunnel=avalanche_protector): wie ein Tunnel, aber talseitig offen (Dach +
Stützen statt einer zweiten Wand) - siehe Design-Spec Abschnitt 6. Keine Portal-Rahmen: Galerien sind keine in
den Fels geschnittenen Öffnungen, sondern offene Schutzbauten entlang der Straße - ihre Enden bleiben rechtwinklig.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def resolve_open_side(osm_tags: Dict) -> Optional[str]:
    """
    Liest die talseitig offene Wand direkt aus den OSM-Tags `avalanche_protector:left`/
    `avalanche_protector:right` (Wert "open"), falls vorhanden - "left"/"right" folgen dabei der
    Digitalisierungsrichtung der Way, exakt dieselbe Konvention wie offset_points()/valley_side().

    Deutlich zuverlässiger als der Höhenvergleich in valley_side(): das DGM erfasst am Bauwerk nicht
    das ursprüngliche Gelände, sondern die bereits fertige Galerie samt Erdüberwurf/Dach - "natürliche"
    Geländehöhe links/rechts der Centerline gibt es an dieser Stelle also gar nicht, die Galerie
    verschwindet dadurch im Zweifel komplett im (in Wirklichkeit gar nicht natürlichen) "Gelände".

    Returns:
        "left" | "right" | None (kein Tag vorhanden -> Aufrufer muss auf valley_side() zurückfallen)
    """
    if str(osm_tags.get("avalanche_protector:left", "")).lower() == "open":
        return "left"
    if str(osm_tags.get("avalanche_protector:right", "")).lower() == "open":
        return "right"
    return None


def valley_side(xy: np.ndarray, ground_at: HeightAt, half_width: float) -> np.ndarray:
    """
    Pro Punkt: +1.0, wenn die Seite RECHTS der Laufrichtung talwärts liegt (niedrigere natürliche Geländehöhe),
    sonst -1.0 (links talwärts). Gleiche Technik wie terrain.road_embedding.build_road_embankment_profiles()
    (natürliche Geländehöhe links/rechts der Centerline vergleichen).

    NUR ein Fallback für den (seltenen) Fall ohne `avalanche_protector:left`/`:right`-Tag (siehe
    resolve_open_side()) - das DGM an einer bestehenden Galerie zeigt bereits das Bauwerk selbst statt
    des ursprünglichen Hangs, ein Höhenvergleich links/rechts der Centerline ist dort bestenfalls eine
    grobe Näherung.
    """
    directions = np.diff(xy, axis=0)
    directions = np.vstack([directions, directions[-1:]])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    directions = directions / norms
    perp = np.column_stack([-directions[:, 1], directions[:, 0]])

    # WICHTIG: dieselbe Vorzeichen-Konvention wie offset_points() (left = point + normal, right = point - normal,
    # normal = (-dy,dx)) - sonst zeigt diese Funktion "links"/"rechts" spiegelverkehrt zu den left[]/right[]-Arrays,
    # die build_gallery_mesh() aus offset_points() für Wand/Stützen-Platzierung verwendet.
    left_xy = xy + perp * half_width
    right_xy = xy - perp * half_width
    left_z = np.asarray(ground_at(left_xy[:, 0], left_xy[:, 1]), dtype=float)
    right_z = np.asarray(ground_at(right_xy[:, 0], right_xy[:, 1]), dtype=float)
    return np.where(right_z < left_z, 1.0, -1.0)


def build_gallery_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    height: float,
    ground_at: HeightAt,
    floor_material: str,
    roof_material: str,
    column_spacing: float = 6.0,
    roof_thickness: float = 0.35,
    column_size: float = 0.4,
    tile_m: float = 5.0,
    open_side: Optional[str] = None,
) -> Dict:
    """
    Galerie-Mesh: Boden, Dach (Ober-/Unterseite), eine bergseitige Wand und Stützen auf der talseitig offenen Seite.

    Args:
        open_side: "left" | "right" | None - wenn gesetzt (aus resolve_open_side(), zuverlässiger OSM-Tag),
            gilt diese Seite für die GESAMTE Galerie als offen statt sie per valley_side() (Höhenvergleich,
            nur Fallback) punktweise zu bestimmen.

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], roof_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    roof_bottom_z = floor_z + height
    roof_top_z = roof_bottom_z + roof_thickness

    left, right = offset_points(xy, width / 2.0, closed=False)
    # +1 = rechts offen (Tal), -1 = links offen - siehe open_side/resolve_open_side()-Docstring.
    if open_side == "left":
        side = np.full(len(points), -1.0)
    elif open_side == "right":
        side = np.full(len(points), 1.0)
    else:
        side = valley_side(xy, ground_at, width / 2.0)

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / tile_m
    across = width / tile_m
    across_h = height / tile_m

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

        floor_builder.quad(
            [p3(left[i], floor_z[i]), p3(left[j], floor_z[j]), p3(right[j], floor_z[j]), p3(right[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
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

        # bergseitige Wand: die Seite, die (an diesem Segment) NICHT talwärts liegt; bei einem Wechsel mitten im
        # Segment (selten) gewinnt die Seite am Segment-Anfang - akzeptierte Vereinfachung.
        mountain_is_left = side[i] > 0
        edge = left if mountain_is_left else right
        wall_normal = [-side_normal[0], -side_normal[1], 0.0] if mountain_is_left else side_normal
        roof_builder.quad(
            [p3(edge[i], floor_z[i]), p3(edge[j], floor_z[j]), p3(edge[j], roof_bottom_z[j]), p3(edge[i], roof_bottom_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across_h], [u0, across_h]],
            wall_normal,
        )

    cum = np.concatenate([[0.0], np.cumsum(steps)])
    total_len = float(cum[-1]) if len(cum) else 0.0
    column_positions = np.arange(column_spacing / 2.0, total_len, column_spacing) if total_len > 0 else np.array([])
    for s in column_positions:
        idx = max(1, min(int(np.searchsorted(cum, s)), len(points) - 1))
        t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
        open_edge = right if side[idx - 1] > 0 else left
        cx = open_edge[idx - 1, 0] + t * (open_edge[idx, 0] - open_edge[idx - 1, 0])
        cy = open_edge[idx - 1, 1] + t * (open_edge[idx, 1] - open_edge[idx - 1, 1])
        cz = float(floor_z[idx - 1] + t * (floor_z[idx] - floor_z[idx - 1]))
        add_box_column(roof_builder, cx, cy, cz, cz + height, column_size, tile_m)

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


def build_galleries(
    galleries: Sequence[Dict],
    ground_at: HeightAt,
    roof_material: str,
    height: float = 5.0,
    column_spacing: float = 6.0,
    roof_thickness: float = 0.35,
    column_size: float = 0.4,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, eines je Galerie (`galleries`: [{"id","coords","width","floor_material",
    "osm_tags"}, ...] - "osm_tags" optional, für resolve_open_side())."""
    meshes = []
    for gallery in galleries:
        coords = gallery["coords"]
        if len(coords) < 2:
            continue
        mesh = build_gallery_mesh(
            coords, gallery["width"], height, ground_at, gallery["floor_material"], roof_material,
            column_spacing=column_spacing, roof_thickness=roof_thickness, column_size=column_size,
            open_side=resolve_open_side(gallery.get("osm_tags", {})),
        )
        meshes.append({"id": f"gallery_{gallery['id']}", **mesh})
    return meshes
