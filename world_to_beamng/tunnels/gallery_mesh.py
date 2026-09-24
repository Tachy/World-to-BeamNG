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


VALLEY_PROBE_OFFSETS = (10.0, 20.0, 40.0)  # Abstände über den Fahrbahnrand hinaus, in Metern


def valley_score(xy: np.ndarray, ground_at: HeightAt, half_width: float) -> np.ndarray:
    """
    Pro Punkt: Summe (Gelände links - Gelände rechts) in VALLEY_PROBE_OFFSETS Metern jenseits des Fahrbahnrands
    (positiv = rechts tiefer = Tal rechts). Bewusst AUSSERHALB der Einbettung gemessen: direkt neben der Galerie ist
    das Gelände nach der Einbettung flach (Fahrbahn, Bergwand-Saum, Böschung) - dort entschieden Zentimeter bzw.
    ein Gleichstand die Seite (Nuova strada 2026-09-24: beide Galerien zum Berg hin offen).
    """
    directions = np.diff(xy, axis=0)
    directions = np.vstack([directions, directions[-1:]])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    directions = directions / norms
    # dieselbe Vorzeichen-Konvention wie offset_points(): links = Punkt + (-dy, dx)
    perp = np.column_stack([-directions[:, 1], directions[:, 0]])
    score = np.zeros(len(xy))
    for offset in VALLEY_PROBE_OFFSETS:
        distance = half_width + offset
        left_xy, right_xy = xy + perp * distance, xy - perp * distance
        score += np.asarray(ground_at(left_xy[:, 0], left_xy[:, 1]), float) - np.asarray(ground_at(right_xy[:, 0], right_xy[:, 1]), float)
    return score


def valley_side(xy: np.ndarray, ground_at: HeightAt, half_width: float) -> np.ndarray:
    """
    Pro Punkt: +1.0, wenn die Seite RECHTS der Laufrichtung talwärts liegt, sonst -1.0 (siehe valley_score()).

    NUR ein Fallback für den Fall ohne `avalanche_protector:left`/`:right`-Tag (siehe resolve_open_side()).
    """
    return np.where(valley_score(xy, ground_at, half_width) > 0.0, 1.0, -1.0)


def gallery_open_side(osm_tags: Dict, coords, ground_at: HeightAt, width: float) -> str:
    """
    Offene (Tal-)Seite einer Galerie in Digitalisierungsrichtung: aus `avalanche_protector:left/right=open`, sonst aus
    dem Geländevergleich (Summe von valley_score() über die ganze Galerie, >= 0 -> "right"). EINE Stelle für Galerie-
    Mesh und Böschung (terrain_workflow), damit beide dieselbe Seite nehmen.
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
    Galerie-Mesh: Boden, Dach und bergseitige Wand sind echte Quader (nicht nur dünne Flächen) - Boden
    floor_thickness nach unten, Dach roof_thickness nach oben, bergseitige Wand wall_thickness weiter in
    den Hang hinein (bündig mit der Dach-Oberkante). Dazu ein durchlaufender Sockel (curb_height/curb_width,
    wie der Bordstein bei Brücken) auf der talseitig offenen Seite (keine Wand dort) UND Stützen, die BÜNDIG
    auf dem Sockel sitzen: im Grundriss auf dessen Mittellinie zentriert (curb_width == column_size ->
    Stützen-Außenkante == Fahrbahnkante == Dachkante, alles bündig) und in der Höhe auf der Sockel-Oberkante
    aufsetzend statt im Boden zu stecken (Stützenhöhe entsprechend um curb_height verkürzt, die Oberkante
    bleibt bei der Dach-Unterkante). Beide Enden werden komplett verschlossen (Boden-/Dach-/Wand-Querschnitt)
    - wirkt wie ein sauberer Schnitt durchs Bauwerk, exakt an den ursprünglichen OSM-Way-Grenzpunkten (keine
    künstliche Verlängerung der Centerline).

    Args:
        floor_thickness, wall_thickness: siehe config.GALLERY_FLOOR_THICKNESS/GALLERY_WALL_THICKNESS
        curb_height, curb_width: siehe config.GALLERY_CURB_HEIGHT/GALLERY_CURB_WIDTH - Sockel auf der
            Stützenseite, curb_width nach innen von der Fahrbahnkante versetzt; column_size sollte curb_width
            entsprechen, damit die Stütze bündig auf dem Sockel sitzt (siehe Docstring oben)
        open_side: "left" | "right" | None - wenn gesetzt (aus resolve_open_side(), zuverlässiger OSM-Tag),
            gilt diese Seite für die GESAMTE Galerie als offen. Ohne Tag gilt ebenfalls EINE Seite für die ganze
            Galerie: die Mehrheit der punktweisen valley_side() (Höhenvergleich, nur Fallback).
        cap_start, cap_end: Stirnfläche am Anfang/Ende bauen - False an einem Übergang zu einem Tunnel-Portal
            (dessen Stirnwand deckt den Galerie-Querschnitt ab; eine eigene Stirnfläche läge in derselben Ebene
            und flackerte).

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
    # Mittellinie des Sockel-Grundrisses (zwischen curb_inner und curb_edge = left/right) - die Stützen
    # sitzen dort zentriert (bündig mit dem Sockel-Fußabdruck, siehe Stützen-Schleife unten). Elementweiser
    # Mittelwert zweier offset_points()-Ergebnisse auf derselben (ggf. gehrungsgeschnittenen) Normalen ist
    # exakt gleichwertig zu einem eigenen offset_points()-Aufruf mit dem gemittelten Abstand.
    mid_left = (left + inner_left) / 2.0
    mid_right = (right + inner_right) / 2.0
    # +1 = rechts offen (Tal), -1 = links offen - siehe open_side/resolve_open_side()-Docstring.
    if open_side == "left":
        side = np.full(len(points), -1.0)
    elif open_side == "right":
        side = np.full(len(points), 1.0)
    else:
        # Ohne Tag: EINE Seite für die ganze Galerie (Mehrheit der punktweisen Talseite) - eine Galerie wechselt
        # nicht mittendrin die offene Seite, der punktweise Geländevergleich kippt am Bauwerk aber leicht.
        # Summe der Höhendifferenzen statt Punktzählung: kein stiller Gleichstand bei halb/halb
        total = float(valley_score(xy, ground_at, width / 2.0).sum())
        side = np.full(len(points), 1.0 if total >= 0.0 else -1.0)

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
        # Bündig auf dem Sockel: im Grundriss auf dessen Mittellinie zentriert (mid_left/mid_right, siehe
        # oben) statt auf der Fahrbahnkante - Sockel und Stütze haben denselben Fußabdruck.
        open_edge = mid_right if side[idx - 1] > 0 else mid_left
        cx = open_edge[idx - 1, 0] + t * (open_edge[idx, 0] - open_edge[idx - 1, 0])
        cy = open_edge[idx - 1, 1] + t * (open_edge[idx, 1] - open_edge[idx - 1, 1])
        floor_base = float(floor_z[idx - 1] + t * (floor_z[idx] - floor_z[idx - 1]))
        # Basis auf Sockel-Oberkante (statt Boden-Niveau) - sonst steckt die Stütze zur Hälfte im Sockel.
        # Oberkante bleibt bei floor_z + height (Dach-Unterkante, unverändert), die Stütze wird dadurch um
        # curb_height kürzer als zuvor.
        column_bottom = floor_base + curb_height
        column_top = floor_base + height
        # Profil relativ zur Galerie-Richtung ausgerichtet (nicht achsenparallel zur Welt) - sonst stehen
        # die Stützen bei diagonal verlaufenden Galerien sichtbar schief zur Wand-/Dachkante.
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
    curb_width: float = 0.4,
    transition_points: Sequence[Tuple[float, float]] = (),
    transition_tol: float = 0.5,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, eines je Galerie (`galleries`: [{"id","coords","width","floor_material",
    "osm_tags"}, ...] - "osm_tags" optional, für resolve_open_side()).

    transition_points: (x, y) der Übergangs-Portale (tunnel_portal.plan_tunnels(), portal["kind"] == "gallery") -
        ein Galerie-Ende, das höchstens transition_tol davon liegt, bekommt keine Stirnfläche."""

    def at_transition(point) -> bool:
        return any(np.hypot(point[0] - tx, point[1] - ty) <= transition_tol for tx, ty in transition_points)

    meshes = []
    for gallery in galleries:
        coords = gallery["coords"]
        if len(coords) < 2:
            continue
        mesh = build_gallery_mesh(
            coords, gallery["width"], height, ground_at, gallery["floor_material"], roof_material,
            column_spacing=column_spacing, roof_thickness=roof_thickness, floor_thickness=floor_thickness,
            wall_thickness=wall_thickness, column_size=column_size, curb_height=curb_height, curb_width=curb_width,
            # Vorgabe aus der Böschungslogik (terrain_workflow._gallery_embedding), sonst Tag bzw. Gelände
            open_side=gallery.get("open_side") or resolve_open_side(gallery.get("osm_tags", {})),
            cap_start=not at_transition(coords[0]), cap_end=not at_transition(coords[-1]),
        )
        meshes.append({"id": f"gallery_{gallery['id']}", **mesh})
    return meshes
