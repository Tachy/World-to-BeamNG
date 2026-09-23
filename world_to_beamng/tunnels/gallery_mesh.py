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
    roof_thickness: float = 0.5,
    floor_thickness: float = 5.0,
    wall_thickness: float = 5.0,
    column_size: float = 0.4,
    curb_height: float = 0.5,
    curb_width: float = 0.25,
    tile_m: float = 5.0,
    open_side: Optional[str] = None,
) -> Dict:
    """
    Galerie-Mesh: Boden, Dach und bergseitige Wand sind echte Quader (nicht nur dünne Flächen) - Boden
    floor_thickness nach unten, Dach roof_thickness nach oben, bergseitige Wand wall_thickness weiter in
    den Hang hinein (bündig mit der Dach-Oberkante). Dazu Stützen UND ein durchlaufender Sockel
    (curb_height/curb_width, wie der Bordstein bei Brücken) auf der talseitig offenen Seite (keine Wand
    dort). Beide Enden werden komplett verschlossen (Boden-/Dach-/Wand-Querschnitt) - wirkt wie ein
    sauberer Schnitt durchs Bauwerk, passend zu den (siehe extend_gallery_centerline_ends()) etwas über
    die OSM-Way-Grenze hinaus verlängerten Enden.

    Args:
        floor_thickness, wall_thickness: siehe config.GALLERY_FLOOR_THICKNESS/GALLERY_WALL_THICKNESS
        curb_height, curb_width: siehe config.GALLERY_CURB_HEIGHT/GALLERY_CURB_WIDTH - Sockel auf der
            Stützenseite, curb_width nach innen von der Fahrbahnkante versetzt
        open_side: "left" | "right" | None - wenn gesetzt (aus resolve_open_side(), zuverlässiger OSM-Tag),
            gilt diese Seite für die GESAMTE Galerie als offen statt sie per valley_side() (Höhenvergleich,
            nur Fallback) punktweise zu bestimmen.

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
    # Die Wand schließt bündig mit der Dach-OBERKANTE ab (nicht nur der Innenraum-Höhe) - reicht also
    # height + roof_thickness hoch, nicht nur height.
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
        # bergseitige Wand: die Seite, die (an diesem Segment) NICHT talwärts liegt; bei einem Wechsel
        # mitten im Segment (selten) gewinnt die Seite am Segment-Anfang - akzeptierte Vereinfachung.
        mountain_is_left = side[i] > 0

        # Boden: Fahrbahn-Oberseite (Straßenmaterial) + Unterseite + beide Randflächen (Quader).
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

        # Dach: Unter-/Oberseite (wie zuvor) + jetzt zusätzlich beide Randflächen (Quader statt Platte).
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
        # Dach-Randflächen (roof_bottom_z bis roof_top_z): nur auf der TALSEITE nötig - bergseitig deckt
        # die (jetzt bis roof_top_z reichende) Wand-Innenfläche dieselbe Fläche schon ab, eine zusätzliche
        # Dach-Randfläche dort wäre deckungsgleiche Geometrie (Z-Fighting).
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

        # Bergseitige Wand (Quader, wall_thickness weiter in den Hang hinein): schließt bündig mit der
        # Dach-OBERKANTE ab (roof_top_z, nicht nur roof_bottom_z) - deshalb 50 cm/roof_thickness höher als
        # die lichte Innenraum-Höhe.
        edge = left if mountain_is_left else right
        outer_edge = outer_left if mountain_is_left else outer_right
        wall_normal = [-side_normal[0], -side_normal[1], 0.0] if mountain_is_left else side_normal
        outward_normal = [-wall_normal[0], -wall_normal[1], 0.0]

        roof_builder.quad(  # Innenfläche (sichtbar aus dem Innenraum, oberhalb der Decke von außen verdeckt)
            [p3(edge[i], floor_z[i]), p3(edge[j], floor_z[j]), p3(edge[j], roof_top_z[j]), p3(edge[i], roof_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, wall_h], [u0, wall_h]],
            wall_normal,
        )
        roof_builder.quad(  # Außenfläche, wall_thickness weiter im Hang
            [p3(outer_edge[i], floor_z[i]), p3(outer_edge[i], roof_top_z[i]), p3(outer_edge[j], roof_top_z[j]), p3(outer_edge[j], floor_z[j])],
            [[u0, 0.0], [u0, wall_h], [u1, wall_h], [u1, 0.0]],
            outward_normal,
        )
        roof_builder.quad(  # Wand-Unterseite (Boden-Niveau, Innen- bis Außenkante)
            [p3(edge[i], floor_z[i]), p3(outer_edge[i], floor_z[i]), p3(outer_edge[j], floor_z[j]), p3(edge[j], floor_z[j])],
            [[u0, 0.0], [u0, wall_extra], [u1, wall_extra], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        roof_builder.quad(  # Wand-Oberseite, jetzt bündig mit der Dach-Oberkante (Innen- bis Außenkante)
            [p3(edge[i], roof_top_z[i]), p3(edge[j], roof_top_z[j]), p3(outer_edge[j], roof_top_z[j]), p3(outer_edge[i], roof_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, wall_extra], [u0, wall_extra]],
            [0.0, 0.0, 1.0],
        )

        # Sockel (Bordstein-artig, wie bridges/bridge_mesh.py) auf der talseitig offenen Seite - curb_width
        # nach innen von der Fahrbahnkante versetzt, curb_height hoch.
        curb_edge = right if mountain_is_left else left
        curb_inner = inner_right if mountain_is_left else inner_left
        curb_outward = [-side_normal[0], -side_normal[1], 0.0] if mountain_is_left else side_normal

        roof_builder.quad(  # Sockel-Oberseite (Beton-Material wie Wand/Dach, nicht Fahrbahn-Material)
            [p3(curb_edge[i], curb_top_z[i]), p3(curb_edge[j], curb_top_z[j]), p3(curb_inner[j], curb_top_z[j]), p3(curb_inner[i], curb_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_w], [u0, curb_w]],
            [0.0, 0.0, 1.0],
        )
        roof_builder.quad(  # Sockel-Außenfläche (zur Talseite gerichtet)
            [p3(curb_edge[i], floor_z[i]), p3(curb_edge[j], floor_z[j]), p3(curb_edge[j], curb_top_z[j]), p3(curb_edge[i], curb_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_h], [u0, curb_h]],
            curb_outward,
        )
        roof_builder.quad(  # Sockel-Innenfläche (zur Fahrbahn gerichtet)
            [p3(curb_inner[i], curb_top_z[i]), p3(curb_inner[j], curb_top_z[j]), p3(curb_inner[j], floor_z[j]), p3(curb_inner[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_h], [u0, curb_h]],
            [-float(curb_outward[0]), -float(curb_outward[1]), 0.0],
        )

    end_cap_args = (
        left, right, outer_left, outer_right, inner_left, inner_right, floor_z, floor_bottom_z,
        roof_bottom_z, roof_top_z, curb_top_z, side, across, floor_h, roof_h, wall_extra, wall_h, curb_h, curb_w,
    )
    _add_end_caps(roof_builder, 0, xy[0] - xy[1], *end_cap_args)
    _add_end_caps(roof_builder, len(points) - 1, xy[-1] - xy[-2], *end_cap_args)

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
        # Profil relativ zur Galerie-Richtung ausgerichtet (nicht achsenparallel zur Welt) - sonst stehen
        # die Stützen bei diagonal verlaufenden Galerien sichtbar schief zur Wand-/Dachkante.
        column_direction = xy[idx] - xy[idx - 1]
        add_box_column(roof_builder, cx, cy, cz, cz + height, column_size, tile_m, direction=tuple(column_direction))

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
    Stirnfläche an einem Ende (idx=0 oder idx=len-1): voller Boden-Querschnitt (Quader-Dicke) + voller
    Dach-Querschnitt + Wand-Querschnitt (nur deren eigener Fußabdruck, Innen- bis Außenkante, bis
    roof_top_z - die Wand schließt bündig mit der Dach-Oberkante ab) + Sockel-Querschnitt auf der
    Stützenseite - macht aus dem offenen Schalen-Ende einen sauberen, massiven Schnitt statt eines
    Blicks in den Hohlraum.
    """
    norm = float(np.hypot(outward_xy[0], outward_xy[1]))
    normal = [float(outward_xy[0] / norm), float(outward_xy[1] / norm), 0.0] if norm > 1e-9 else [1.0, 0.0, 0.0]

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    builder.quad(  # Boden-Stirnfläche
        [p3(left[idx], floor_bottom_z[idx]), p3(right[idx], floor_bottom_z[idx]), p3(right[idx], floor_z[idx]), p3(left[idx], floor_z[idx])],
        [[0.0, 0.0], [across, 0.0], [across, floor_h], [0.0, floor_h]],
        normal,
    )
    builder.quad(  # Dach-Stirnfläche
        [p3(left[idx], roof_bottom_z[idx]), p3(right[idx], roof_bottom_z[idx]), p3(right[idx], roof_top_z[idx]), p3(left[idx], roof_top_z[idx])],
        [[0.0, 0.0], [across, 0.0], [across, roof_h], [0.0, roof_h]],
        normal,
    )

    mountain_is_left = side[idx] > 0
    edge_pt = left[idx] if mountain_is_left else right[idx]
    outer_pt = outer_left[idx] if mountain_is_left else outer_right[idx]
    builder.quad(  # Wand-Stirnfläche (nur der Wand-Fußabdruck: Innen- bis Außenkante, bis zur Dach-Oberkante)
        [p3(edge_pt, floor_z[idx]), p3(outer_pt, floor_z[idx]), p3(outer_pt, roof_top_z[idx]), p3(edge_pt, roof_top_z[idx])],
        [[0.0, 0.0], [wall_extra, 0.0], [wall_extra, wall_h], [0.0, wall_h]],
        normal,
    )

    curb_edge_pt = right[idx] if mountain_is_left else left[idx]
    curb_inner_pt = inner_right[idx] if mountain_is_left else inner_left[idx]
    builder.quad(  # Sockel-Stirnfläche (nur der Sockel-Fußabdruck: Fahrbahnkante bis curb_width nach innen)
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
    curb_width: float = 0.25,
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
            column_spacing=column_spacing, roof_thickness=roof_thickness, floor_thickness=floor_thickness,
            wall_thickness=wall_thickness, column_size=column_size, curb_height=curb_height, curb_width=curb_width,
            open_side=resolve_open_side(gallery.get("osm_tags", {})),
        )
        meshes.append({"id": f"gallery_{gallery['id']}", **mesh})
    return meshes
