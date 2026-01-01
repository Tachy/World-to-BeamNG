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
