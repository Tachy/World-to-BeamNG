"""
Fahrbahnmarkierungen als eigene, schmale DecalRoads über der Fahrbahn - so wie BeamNGs eigene Levels es machen
(west_coast_usa: ~3100 `line_white`- und ~200 `line_dashed_short`-DecalRoads mit 0,15-0,2 m Breite). Weiße
Randlinien links und rechts, gestrichelte Leitlinien an den Fahrstreifengrenzen. Die Linien folgen der Knotenbreite
der Fahrbahn (also auch den weichen Breitenübergängen aus road_width_transitions.py). Hintergrund und Regeln siehe
docs/OSM_ROAD_ANALYSIS.md und docs/superpowers/plans/2026-09-24-road-markings-width-transitions.md.
"""

from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np

EDGE = "edge"
DIVIDER = "divider"
MAX_MITRE_FACTOR = 2.0  # spitze Knicke: Versatz höchstens doppelt so weit wie verlangt
BOUNDARY_EPS = 0.01  # Hindernisflächen um 1 cm schrumpfen, siehe junction_obstacles()


@dataclass(frozen=True)
class MarkingLayout:
    lanes: int


def parse_lanes(value) -> Optional[int]:
    """OSM-`lanes` als positive Ganzzahl, sonst None (fehlend, "2;3", "", "0", ...)."""
    try:
        lanes = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return lanes if lanes >= 1 else None


def marking_layout(
    tags: dict,
    width: float,
    internal_name: str,
    marked_highways: Collection[str],
    marked_surface: str,
    min_two_lane_width: float,
) -> Optional[MarkingLayout]:
    """
    Markierungs-Layout einer Straße oder None (keine Markierung): nur Straßentypen aus `marked_highways` mit der
    Oberfläche `marked_surface` (Asphalt) und ohne `lane_markings=no`. Fahrstreifen aus `lanes`; fehlt der Tag, ist
    eine Rampe (*_link) oder eine Straße schmaler als min_two_lane_width einspurig, alles andere zweispurig.
    """
    tags = tags or {}
    highway = str(tags.get("highway", ""))
    if tags.get("lane_markings") == "no" or highway not in marked_highways or internal_name != marked_surface:
        return None
    lanes = parse_lanes(tags.get("lanes"))
    if lanes is None:
        lanes = 1 if highway.endswith("_link") or width < min_two_lane_width else 2
    return MarkingLayout(lanes=lanes)


def line_offsets(widths: np.ndarray, lanes: int, edge_inset: float) -> List[Tuple[str, np.ndarray]]:
    """(Art, seitlicher Versatz je Knoten), positiv = links der Laufrichtung. Randlinien bei +-(Breite/2 -
    edge_inset), Leitlinien an den lanes-1 Fahrstreifengrenzen."""
    widths = np.asarray(widths, dtype=float)
    half = widths / 2.0
    lines = [(EDGE, half - edge_inset), (EDGE, -(half - edge_inset))]
    lines += [(DIVIDER, -half + k * widths / lanes) for k in range(1, lanes)]
    return lines


def offset_polyline(
    xy: np.ndarray, offsets: np.ndarray, start_normal: Optional[np.ndarray] = None, end_normal: Optional[np.ndarray] = None
) -> np.ndarray:
    """Polylinie mit Versatz je Knoten (positiv = links), an Knicken auf Gehrung. Nullsegmente (doppelte Knoten)
    übernehmen die Richtung des Nachbarsegments. `start_normal`/`end_normal` (Einheitsvektoren, links) ersetzen die
    Gehrungsrichtung am ersten/letzten Knoten - am Stoß zweier Straßen (siehe joint_normals())."""
    xy = np.asarray(xy, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-9
    if not valid.any():
        return xy.copy()
    normals = np.zeros_like(segments)
    normals[valid] = np.column_stack([-segments[valid, 1], segments[valid, 0]]) / lengths[valid, None]
    last = normals[int(np.argmax(valid))]
    for i in range(len(normals)):
        if valid[i]:
            last = normals[i]
        else:
            normals[i] = last

    result = np.empty_like(xy)
    count = len(xy)
    for i in range(count):
        before, after = normals[max(i - 1, 0)], normals[min(i, count - 2)]
        if i == 0 and start_normal is not None:
            miter = np.asarray(start_normal, dtype=float)
        elif i == count - 1 and end_normal is not None:
            miter = np.asarray(end_normal, dtype=float)
        else:
            miter = before + after
            norm = float(np.linalg.norm(miter))
            miter = after if norm < 1e-9 else miter / norm
        scale = 1.0 / max(float(np.dot(miter, after)), 1.0 / MAX_MITRE_FACTOR)
        result[i] = xy[i] + miter * offsets[i] * scale
    return result


def forward_indices(offset_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Indizes der Linienknoten, die in Fahrtrichtung vorankommen. In engen Kehren läuft die innere Linie sonst
    rückwärts (Versatz größer als der Kurvenradius) - diese Knoten fallen weg."""
    center_xy = np.asarray(center_xy, dtype=float)
    count = len(center_xy)
    kept = [0]
    for j in range(1, count):
        tangent = center_xy[min(j + 1, count - 1)] - center_xy[max(j - 1, 0)]
        if float(np.dot(offset_xy[j] - offset_xy[kept[-1]], tangent)) > 1e-9:
            kept.append(j)
    return np.array(kept, dtype=int)


def build_marking_lines(
    nodes: Sequence[Sequence[float]],
    layout: MarkingLayout,
    edge_inset: float,
    start_normal: Optional[np.ndarray] = None,
    end_normal: Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray]]:
    """(Art, (N, 3)-Linie) für alle Markierungslinien einer Straße aus ihren DecalRoad-Knoten [x, y, z, width];
    z je Linienknoten vom zugehörigen Fahrbahnknoten (BeamNG projiziert die Linie ohnehin aufs Terrain).
    `start_normal`/`end_normal`: gemeinsame Stoßnormale mit der Geradeaus-Fortsetzung (joint_normals()), damit die
    Linien beider Straßen an einem geknickten Stoß exakt aneinander anschließen."""
    arr = np.asarray(nodes, dtype=float)
    center_xy = arr[:, :2]
    lines = []
    for kind, offsets in line_offsets(arr[:, 3], layout.lanes, edge_inset):
        offset_xy = offset_polyline(center_xy, offsets, start_normal, end_normal)
        kept = forward_indices(offset_xy, center_xy)
        if len(kept) >= 2:
            lines.append((kind, np.column_stack([offset_xy[kept], arr[kept, 2]])))
    return lines


def joint_normals(roads: Sequence[Sequence[Sequence[float]]], pairs) -> Dict[Tuple[int, str], np.ndarray]:
    """
    Gemeinsame Linksnormale je Straßenende an einem Geradeaus-Stoß (`pairs` aus find_continuations()): die
    Winkelhalbierende beider Fahrtrichtungen, jeweils in der Laufrichtung der eigenen Straße. Ohne sie wird jede
    Straße senkrecht zu ihrem eigenen letzten Segment versetzt, und an einem Knick um den Winkel t klaffen die
    Linienenden um rund 2 * Versatz * sin(t/2) auseinander (außen Lücke, innen Überlappung).
    """
    from .road_width_transitions import outward_direction

    result: Dict[Tuple[int, str], np.ndarray] = {}
    for (ia, ea), (ib, eb) in pairs:
        da, db = outward_direction(roads[ia], ea), outward_direction(roads[ib], eb)
        if da is None or db is None:
            continue
        travel_a = da if ea == "start" else -da  # Fahrtrichtung der Straße am Stoß
        travel_b = db if eb == "start" else -db
        sign = 1.0 if ea != eb else -1.0  # gleiche Laufrichtung (Ende -> Anfang) oder gegenläufig
        joint = travel_a + sign * travel_b
        length = float(np.linalg.norm(joint))
        if length < 1e-9:
            continue
        joint = joint / length
        result[(ia, ea)] = np.array([-joint[1], joint[0]])
        result[(ib, eb)] = sign * np.array([-joint[1], joint[0]])
    return result


def clip_line(line: np.ndarray, obstacles, min_length: float) -> List[np.ndarray]:
    """Teile einer (N, 3)-Linie außerhalb von `obstacles` (shapely-Fläche, z.B. die Fahrbahnen einmündender Straßen)
    mit mindestens min_length Länge; z linear entlang der ursprünglichen Linie."""
    from shapely.geometry import LineString, Point

    shape = LineString(line[:, :2])
    if obstacles is None or obstacles.is_empty:
        pieces = [shape]
    else:
        rest = shape.difference(obstacles)
        pieces = list(getattr(rest, "geoms", [rest]))
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1))])
    result = []
    for piece in pieces:
        if piece.is_empty or piece.geom_type != "LineString" or piece.length < min_length:
            continue
        xy = np.asarray(piece.coords, dtype=float)
        s = np.array([shape.project(Point(p)) for p in xy])
        result.append(np.column_stack([xy, np.interp(s, cum, line[:, 2])]))
    return result


def road_surface_polygon(nodes: Sequence[Sequence[float]], clearance: float):
    """Fahrbahnfläche einer DecalRoad (Puffer um die Mittellinie mit der größten Knotenbreite, flache Enden),
    um `clearance` verbreitert."""
    from shapely.geometry import LineString

    arr = np.asarray(nodes, dtype=float)
    return LineString(arr[:, :2]).buffer(float(arr[:, 3].max()) / 2.0 + clearance, cap_style="flat")




def _side_junction(polygon, main_line, side_xy: np.ndarray, endpoint_tol: float):
    """
    (Mündungspunkt, Seite, eigene Halbebene) einer Straße, die mit genau einem Ende auf `main_line` (Mittellinie der
    markierten Straße) mündet - Seite +1 links, -1 rechts. None für Straßen, die die Mittellinie queren oder nur
    berühren.
    """
    from shapely.geometry import Point

    start, end = Point(side_xy[0]), Point(side_xy[-1])
    at_start = main_line.distance(start) <= endpoint_tol
    at_end = main_line.distance(end) <= endpoint_tol
    if at_start == at_end:
        return None
    reach = polygon.length
    left = main_line.buffer(reach, single_sided=True)
    right = main_line.buffer(-reach, single_sided=True)
    if polygon.intersection(left).area >= polygon.intersection(right).area:
        return (start if at_start else end), 1, left
    return (start if at_start else end), -1, right


def junction_obstacles(
    index: int,
    polygons: Sequence,
    tree,
    excluded: Collection[int],
    centerlines: Optional[Sequence[np.ndarray]] = None,
    endpoint_tol: float = 0.5,
):
    """
    Vereinigung der Fahrbahnflächen, die die Fläche `index` berühren - ohne sie selbst und ohne `excluded`
    (Geradeaus-Partner, Wege ohne Markierungslücke). None, wenn keine übrig bleibt. `tree`: shapely.STRtree über
    `polygons`.

    Mit `centerlines` ((N, 2) je Straße) wird die Fläche einer Einmündung auf ihre Seite der eigenen Mittellinie
    beschränkt: ihr flaches Ende steht senkrecht zu ihr selbst, nicht zur Hauptstraße, und reicht bei schräger
    Einmündung sonst über die Mittellinie - Leitlinie und gegenüberliegende Randlinie bekämen eine Lücke. Mündet am
    selben Punkt auch von der Gegenseite eine Straße (Kreuzung, in OSM an der Hauptstraße in zwei Ways geteilt),
    bleiben beide Flächen ganz, damit die Leitlinie in der Kreuzung unterbrochen wird.

    Um BOUNDARY_EPS geschrumpft: eine Nebenstraße beginnt am gemeinsamen Knoten auf der Mittellinie der Hauptstraße,
    ihre flache Kante liegt also genau auf deren Leitlinie. Ohne das Schrumpfen bekäme die Leitlinie an jeder
    T-Einmündung eine Lücke.
    """
    from shapely import unary_union
    from shapely.geometry import LineString

    candidates = [j for j in tree.query(polygons[index]) if j != index and j not in excluded]
    if not candidates:
        return None
    if centerlines is None:
        others = [polygons[j] for j in candidates]
    else:
        main_line = LineString(centerlines[index])
        junctions = {
            j: _side_junction(polygons[j], main_line, np.asarray(centerlines[j], dtype=float), endpoint_tol)
            for j in candidates
        }
        others = []
        for j in candidates:
            junction = junctions[j]
            if junction is None:
                others.append(polygons[j])
                continue
            point, side, own_half = junction
            crossing = any(
                other is not None and other[1] != side and other[0].distance(point) <= 2.0 * endpoint_tol
                for k, other in junctions.items()
                if k != j
            )
            others.append(polygons[j] if crossing else polygons[j].intersection(own_half))
    others = [polygon for polygon in others if not polygon.is_empty]
    if not others:
        return None
    return unary_union(others).buffer(-BOUNDARY_EPS)
