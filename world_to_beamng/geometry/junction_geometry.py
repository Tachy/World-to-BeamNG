"""
Konstruktion von sauberen Junctions mit zentralen Polygonen.

Fuer jede Junction (Kreuzung, T-Junction, etc.) wird ein zentrales Polygon gebaut:
- 4-Wege-Kreuzung -> Quad in der Mitte
- T-Kreuzung (3 Wege) -> Pentagon
- 2-Wege -> Quad

Die Strassen werden dann so gekuerzt, dass sie exakt an diesem Polygon andocken.
"""

import numpy as np
from scipy.spatial import cKDTree


def line_line_intersection(p1, d1, p2, d2):
    """
    Berechnet Schnittpunkt zweier 2D-Linien.

    Linie 1: p1 + t * d1
    Linie 2: p2 + s * d2

    Returns: Schnittpunkt (x, y) oder None wenn parallel
    """
    # Loese: p1 + t*d1 = p2 + s*d2
    # Als Matrix: [d1 | -d2] * [t, s]^T = p2 - p1
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = p2 - p1

    det = np.linalg.det(A)
    if abs(det) < 1e-10:  # Parallel
        return None

    t = np.linalg.solve(A, b)[0]
    intersection = p1 + t * d1
    return intersection


def build_junction_polygon(junction, road_polygons, road_width=7.0):
    """
    Baue Junction-Quad aus SCHNITTPUNKTEN der Strassenkanten.

    Die 4 Eckpunkte des Quads liegen auf den Schnittpunkten der
    left/right Kanten der angeschlossenen Strassen.

    Args:
        junction: Junction-Dict mit 'position', 'road_indices', 'direction_vectors'
        road_polygons: Wird nicht verwendet (nur fuer Signatur-Kompatibilität)
        road_width: Strassenbreite in Metern

    Returns:
        Junction-Polygon-Dict mit geometrisch korrektem Quad
    """
    junction_pos_2d = np.array(junction["position"][:2])
    junction_pos_3d = junction["position"]
    junction_z = junction_pos_3d[2]

    num_roads = len(junction["road_indices"])
    half_width = road_width / 2.0

    # Hole die Richtungsvektoren
    direction_vectors = junction.get("direction_vectors", {})

    if not direction_vectors or len(direction_vectors) < 2:
        # Fallback: Einfaches achsen-ausgerichtetes Quad
        vertices_2d = [
            (junction_pos_2d[0] - half_width, junction_pos_2d[1] - half_width),
            (junction_pos_2d[0] + half_width, junction_pos_2d[1] - half_width),
            (junction_pos_2d[0] + half_width, junction_pos_2d[1] + half_width),
            (junction_pos_2d[0] - half_width, junction_pos_2d[1] + half_width),
        ]
        vertices_3d = [(v[0], v[1], junction_z) for v in vertices_2d]

        return {
            "center": junction_pos_3d,
            "vertices_2d": vertices_2d,
            "vertices_3d": vertices_3d,
            "road_indices": junction["road_indices"],
            "type": _get_junction_type(num_roads),
            "num_roads": num_roads,
        }

    # Berechne left/right Kanten-Linien fuer jede Strasse
    road_edges = []
    for road_idx, direction in direction_vectors.items():
        direction = np.array(direction)
        norm = np.linalg.norm(direction)
        if norm > 0.01:
            direction = direction / norm

        # Perpendicular zur Strasse (left/right offset)
        perpendicular = np.array([-direction[1], direction[0]])

        # Left/Right edge Positionen am Junction-Punkt
        left_pos = junction_pos_2d + perpendicular * half_width
        right_pos = junction_pos_2d - perpendicular * half_width

        road_edges.append(
            {
                "road_idx": road_idx,
                "direction": direction,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "left_dir": direction,  # Kantenlinie verläuft parallel zur Strasse
                "right_dir": direction,
            }
        )

    # Sortiere Strassen nach Winkel (CCW)
    def angle_of_direction(edge):
        return np.arctan2(edge["direction"][1], edge["direction"][0])

    road_edges.sort(key=angle_of_direction)

    # Berechne Schnittpunkte benachbarter Kanten
    vertices_2d = []
    n = len(road_edges)

    for i in range(n):
        next_i = (i + 1) % n

        # Rechte Kante von Strasse i schneidet linke Kante von Strasse i+1
        edge1 = road_edges[i]
        edge2 = road_edges[next_i]

        intersection = line_line_intersection(
            edge1["right_pos"], edge1["right_dir"], edge2["left_pos"], edge2["left_dir"]
        )

        if intersection is not None:
            vertices_2d.append(tuple(intersection))
        else:
            # Fallback: Mittelwert
            mid = (edge1["right_pos"] + edge2["left_pos"]) / 2.0
            vertices_2d.append(tuple(mid))

    vertices_3d = [(v[0], v[1], junction_z) for v in vertices_2d]

    junction_poly = {
        "center": junction_pos_3d,
        "vertices_2d": vertices_2d,
        "vertices_3d": vertices_3d,
        "road_indices": junction["road_indices"],
        "type": _get_junction_type(num_roads),
        "num_roads": num_roads,
        "road_edges": road_edges,  # Speichere fuer Connector
    }

    return junction_poly


def _get_junction_type(num_roads):
    """Bestimme Typ basierend auf Anzahl Strassen."""
    if num_roads == 2:
        return "endpoint"
    elif num_roads == 3:
        return "t_junction"
    elif num_roads == 4:
        return "cross"
    else:
        return f"complex_{num_roads}"


def build_all_junction_polygons(junctions, road_polygons, road_width=7.0):
    """
    Baue Polygone fuer alle Junctions.

    Args:
        junctions: Liste aller Junctions
        road_polygons: Alle Strassen
        road_width: Strassenbreite

    Returns:
        Liste von Junction-Polygonen
    """
    junction_polys = []

    # Statistik nach Typ
    quad_count = 0
    t_junction_count = 0
    cross_count = 0

    for junction in junctions:
        poly = build_junction_polygon(junction, road_polygons, road_width)
        if poly:
            junction_polys.append(poly)
            # Zähle Typ
            poly_type = poly.get("type", "unknown")
            if poly_type == "quad":
                quad_count += 1
            elif poly_type == "t_junction":
                t_junction_count += 1
            elif poly_type == "cross":
                cross_count += 1

    print(f"  [i] {len(junction_polys)} Junction-Polygone gebaut:")
    if quad_count > 0:
        print(f"    - {quad_count} Quads (2-Wege)")
    if t_junction_count > 0:
        print(f"    - {t_junction_count} T-Junctions (3-Wege)")
    if cross_count > 0:
        print(f"    - {cross_count} Crosses (4-Wege)")

    return junction_polys


def truncate_roads_at_junctions(
    road_polygons, junctions, road_width=7.0, truncation_distance=3.5
):
    """
    Kuerze Strassen so, dass sie bei der Junction enden.

    Wichtig: Wenn eine Strasse an BEIDEN Enden eine Junction hat (z.B. start und end),
    wird sie zweimal gekuerzt (einmal vom Start, einmal vom Ende) - sie wird dadurch unterbrochen!

    Args:
        road_polygons: Alle Strassen (wird modifiziert)
        junctions: Alle Junctions
        road_width: Strassenbreite
        truncation_distance: Distanz von der Junction zur Strassen-Kante

    Returns:
        Modifizierte road_polygons
    """
    if not junctions:
        return road_polygons

    half_width = road_width / 2.0
    truncation_dist = truncation_distance if truncation_distance > 0 else half_width

    truncated_count = 0
    split_roads = 0  # Strassen die unterbrochen werden (start UND end Junction)

    for junction in junctions:
        junction_pos_2d = junction["position"][:2]

        for road_idx in junction["road_indices"]:
            if road_idx >= len(road_polygons):
                continue

            road = road_polygons[road_idx]
            coords = road["coords"]

            if len(coords) < 3:
                continue

            conn_types = junction["connection_types"].get(road_idx, [])

            # Pruefe ob diese Strasse an BEIDEN Enden dieser Junction angedockt ist
            # (Das bedeutet sie wird unterbrochen)
            if len(conn_types) > 1:
                split_roads += 1

            # Kuerze vom Start her
            if "start" in conn_types:
                new_coords = _truncate_from_start(
                    coords, junction_pos_2d, truncation_dist
                )
                if len(new_coords) >= 2:
                    road["coords"] = new_coords
                    truncated_count += 1

            # Kuerze vom Ende her
            if "end" in conn_types:
                new_coords = _truncate_from_end(
                    coords, junction_pos_2d, truncation_dist
                )
                if len(new_coords) >= 2:
                    road["coords"] = new_coords
                    truncated_count += 1

    if truncated_count > 0:
        print(f"  [i] {truncated_count} Strassen-Endpunkte gekuerzt")
    if split_roads > 0:
        print(f"  [i] {split_roads} Strassen unterbrochen (start UND end Junction)")

    return road_polygons


def _truncate_from_start(coords, junction_pos_2d, distance):
    """Entferne Punkte vom Anfang bis zur angegebenen Distanz zur Junction."""
    if len(coords) < 2:
        return coords

    junction_pos = np.array(junction_pos_2d)

    # Finde Index wo Distanz > distance
    cumulative_dist = 0
    for i in range(len(coords) - 1):
        p1 = np.array(coords[i][:2])
        p2 = np.array(coords[i + 1][:2])
        segment_dist = np.linalg.norm(p2 - p1)

        if cumulative_dist + segment_dist >= distance:
            # Schnitt an dieser Stelle
            # Interpoliere den genauen Punkt
            remaining = distance - cumulative_dist
            ratio = remaining / segment_dist if segment_dist > 0 else 0

            cut_point_xy = p1 + ratio * (p2 - p1)
            # Behalte Z-Koordinate vom ersten Punkt
            cut_point = (cut_point_xy[0], cut_point_xy[1], coords[i][2])

            return [cut_point] + list(coords[i + 1 :])

        cumulative_dist += segment_dist

    # Wenn nicht genug Distanz, gib Original zurueck
    return coords


def _truncate_from_end(coords, junction_pos_2d, distance):
    """Entferne Punkte vom Ende bis zur angegebenen Distanz zur Junction."""
    if len(coords) < 2:
        return coords

    # Reversiere, truncate, und reversiere zurueck
    coords_rev = list(reversed(coords))
    truncated_rev = _truncate_from_start(coords_rev, junction_pos_2d, distance)
    return list(reversed(truncated_rev))


def analyze_junction_polygons(junction_polys):
    """Gib Debug-Info ueber generierte Junction-Polygone."""
    if not junction_polys:
        print("  [i] Keine Junction-Polygone generiert")
        return

    types = {}
    total_vertices = 0

    for poly in junction_polys:
        t = poly["type"]
        types[t] = types.get(t, 0) + 1
        total_vertices += len(poly["vertices_3d"])

    print(
        f"  [i] {len(junction_polys)} Junction-Polygone generiert ({total_vertices} Vertices):"
    )

    # Detaillierte Aufschluesselung nach Typ
    for poly_type in sorted(types.keys()):
        count = types[poly_type]
        print(f"      - {count} {poly_type}")
