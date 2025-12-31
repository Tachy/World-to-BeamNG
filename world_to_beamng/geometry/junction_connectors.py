"""
Baue Verbindungs-Quads zwischen gekuerzte Strassen und zentrale Junction-Quads.

Diese Connectors fuellen die Luecke zwischen:
- Den gekuertzten Strassen-Enden (10m ab Junction)
- Den Eckpunkten des zentralen Junction-Quads
"""

import numpy as np


def build_junction_connectors(
    junction_polys, junctions, road_polygons, truncation_distance, road_width=7.0
):
    """
    Baue Verbindungs-Quads/Trapezoids zwischen Strassen-Enden und Junction-Quads.

    Das Junction-Quad ist ein einfaches Rechteck (7x7m).
    Fuer jede Strasse:
    - Finde die Fahrtrichtung zur Junction
    - Finde die entsprechende Kante des Junction-Quads
    - Verbinde die gekuerzte Strasse mit dieser Kante

    Args:
        junction_polys: List von Junction-Polygonen (einfache Rechtecke)
        junctions: List von Junctions mit Positionen und Richtungen
        road_polygons: Alle Strassen (mit gekuertzten coords)
        truncation_distance: Wie weit Strassen gekuerzt wurden (10m)
        road_width: Strassenbreite

    Returns:
        Liste von Connector-Polygonen (Trapezoid-Quads)
    """
    connectors = []
    half_width = road_width / 2.0

    for junction_idx, junction_poly in enumerate(junction_polys):
        junction = junctions[junction_idx]
        junction_pos_2d = np.array(junction["position"][:2])

        # WICHTIG: Nutze Z-Werte aus junction_poly["vertices_3d"], NICHT aus junction["position"]!
        # Das Junction-Quad verwendet die tatsächlichen Mesh-Z-Werte
        junction_vertices_3d = junction_poly.get("vertices_3d", [])
        if not junction_vertices_3d or len(junction_vertices_3d) < 3:
            continue

        # Junction-Quad Vertices (geometrisch korrekt aus Schnittpunkten)
        junction_vertices = junction_poly["vertices_2d"]

        if not junction_vertices or len(junction_vertices) < 3:
            continue

        # Hole die vorberechneten Edge-Daten (left/right Punkte pro Strasse)
        road_edge_data = junction_poly.get("road_edge_data", {})

        # Hole die Richtungsvektoren
        direction_vectors = junction.get("direction_vectors", {})

        # Gehe durch alle Strassen dieser Junction
        for road_idx in junction["road_indices"]:
            road = road_polygons[road_idx]
            coords = road["coords"]

            if len(coords) < 2:
                continue

            conn_types = junction["connection_types"].get(road_idx, [])
            direction = np.array(direction_vectors.get(road_idx, [1.0, 0.0]))

            # Bestimme: Strasse beginnt ("start") oder endet ("end") hier?
            if "start" in conn_types:
                # Strasse beginnt: gekuertzter Endpunkt ist coords[0]
                road_end_2d = np.array(coords[0][:2])
                road_end_z = coords[0][2]
                # Nächster Punkt fuer Richtungsberechnung
                road_next_2d = (
                    np.array(coords[1][:2]) if len(coords) > 1 else road_end_2d
                )
                # Richtung: WEG von Junction (von coords[0] zu coords[1])
                local_direction = road_next_2d - road_end_2d
            elif "end" in conn_types:
                # Strasse endet: gekuertzter Endpunkt ist coords[-1]
                road_end_2d = np.array(coords[-1][:2])
                road_end_z = coords[-1][2]
                # Vorheriger Punkt fuer Richtungsberechnung
                road_next_2d = (
                    np.array(coords[-2][:2]) if len(coords) > 1 else road_end_2d
                )
                # Richtung: WEG von Junction (von coords[-2] zu coords[-1])
                # WICHTIG: Hier muss die Richtung UMGEDREHT werden!
                local_direction = road_end_2d - road_next_2d
            else:
                continue

            # WICHTIG: Nutze die EXAKTE Richtung der Strasse an dieser Stelle
            # Das stellt sicher, dass die Rand-Punkte identisch mit dem Strassen-Mesh sind
            local_norm = np.linalg.norm(local_direction)
            if local_norm > 0.001:
                local_direction = local_direction / local_norm
            else:
                local_direction = direction  # Fallback

            # Berechne left/right Eckpunkte aus road_end_2d
            perp = np.array([-local_direction[1], local_direction[0]])
            road_left = road_end_2d + perp * half_width
            road_right = road_end_2d - perp * half_width

            # Überschreibe Z mit den Edge-Daten wenn vorhanden
            if road_idx in road_edge_data:
                road_end_z = road_edge_data[road_idx]["z"]

            # NEUE LOGIK: Finde die 2 Junction-Vertices, die zu dieser Strasse gehoeren
            # Berechne Richtung von junction_center zu road_end
            # Finde dann die 2 Vertices, die in diese Richtung zeigen

            road_center = (road_left + road_right) / 2.0
            direction_to_road = road_center - junction_pos_2d
            direction_to_road = direction_to_road / (
                np.linalg.norm(direction_to_road) + 1e-6
            )

            # Berechne den Winkel jedes Vertices zu dieser Richtung
            angles = []
            for v in junction_vertices:
                v_arr = np.array(v)
                vec_to_v = v_arr - junction_pos_2d
                if np.linalg.norm(vec_to_v) > 0.01:
                    vec_to_v = vec_to_v / np.linalg.norm(vec_to_v)
                    dot = np.dot(direction_to_road, vec_to_v)
                    angle = np.arccos(np.clip(dot, -1.0, 1.0))
                else:
                    angle = np.pi
                angles.append(angle)

            # Die 2 Vertices mit kleinstem Winkel zur Strassen-Richtung
            sorted_by_angle = sorted(range(len(angles)), key=lambda i: angles[i])
            idx_v1 = sorted_by_angle[0]
            idx_v2 = sorted_by_angle[1]

            # Hole die tatsächlichen 3D-Koordinaten aus dem Junction-Quad-Mesh
            junction_point1_3d = junction_vertices_3d[idx_v1]
            junction_point2_3d = junction_vertices_3d[idx_v2]

            # DEBUG fuer erstes Connector
            if junction_idx == 0 and road_idx == 0:
                print(
                    f"\n    DEBUG CONNECTOR Junction {junction_idx}, Road {road_idx}:"
                )
                print(
                    f"      Road-Ende (gekuerzt): ({road_end_2d[0]:.2f}, {road_end_2d[1]:.2f}), Z={road_end_z:.2f}"
                )
                print(
                    f"      Road-Richtung: ({local_direction[0]:.2f}, {local_direction[1]:.2f})"
                )
                print(f"      Road-Left: ({road_left[0]:.2f}, {road_left[1]:.2f})")
                print(f"      Road-Right: ({road_right[0]:.2f}, {road_right[1]:.2f})")
                print(
                    f"      Direction to Road: ({direction_to_road[0]:.2f}, {direction_to_road[1]:.2f})"
                )
                print(f"      Junction-Vertices Angles: {[f'{a:.2f}' for a in angles]}")
                print(
                    f"      Beste 2 Vertices: idx={idx_v1} (angle={angles[idx_v1]:.2f}), idx={idx_v2} (angle={angles[idx_v2]:.2f})"
                )
                print(
                    f"      Junction-Point 1: ({junction_point1_3d[0]:.2f}, {junction_point1_3d[1]:.2f}, {junction_point1_3d[2]:.2f})"
                )
                print(
                    f"      Junction-Point 2: ({junction_point2_3d[0]:.2f}, {junction_point2_3d[1]:.2f}, {junction_point2_3d[2]:.2f})"
                )
                print(f"      === CONNECTOR VERTICES ===")
                print(
                    f"      [0] Road-Left:  ({road_left[0]:.2f}, {road_left[1]:.2f}, {road_end_z:.2f})"
                )
                print(
                    f"      [1] Road-Right: ({road_right[0]:.2f}, {road_right[1]:.2f}, {road_end_z:.2f})"
                )
                print(
                    f"      [2] Junction-2: ({junction_point2_3d[0]:.2f}, {junction_point2_3d[1]:.2f}, {junction_point2_3d[2]:.2f})"
                )
                print(
                    f"      [3] Junction-1: ({junction_point1_3d[0]:.2f}, {junction_point1_3d[1]:.2f}, {junction_point1_3d[2]:.2f})"
                )

            # Baue ein Trapezoid-Quad:
            # road_left, road_right, junction_point2_3d, junction_point1_3d
            # (CCW Orientierung)
            connector = {
                "type": "junction_connector",
                "vertices_2d": [
                    tuple(road_left),
                    tuple(road_right),
                    (junction_point2_3d[0], junction_point2_3d[1]),
                    (junction_point1_3d[0], junction_point1_3d[1]),
                ],
                "vertices_3d": [
                    (road_left[0], road_left[1], road_end_z),
                    (road_right[0], road_right[1], road_end_z),
                    junction_point2_3d,  # Nutze exakte 3D-Koordinate vom Junction-Quad!
                    junction_point1_3d,  # Nutze exakte 3D-Koordinate vom Junction-Quad!
                ],
                "road_idx": road_idx,
                "junction_idx": junction_idx,
            }

            connectors.append(connector)

    return connectors


def connectors_to_faces(connector_polys, vertex_manager):
    """
    Konvertiere Connector-Polygone zu Mesh-Faces (Fan-Triangulation).

    Args:
        connector_polys: List von Connector-Polygonen
        vertex_manager: VertexManager zum Hinzufuegen der Vertices

    Returns:
        List von Faces [v0, v1, v2] (Triangles)
    """
    faces = []

    for connector in connector_polys:
        vertices_3d = connector["vertices_3d"]

        if len(vertices_3d) < 3:
            continue

        # Fuege alle Vertices hinzu
        v_indices = vertex_manager.add_vertices_batch_dedup_fast(vertices_3d)

        # Fan-Triangulation: v0 ist Zentrum
        # Fuer N Vertices: N-2 Triangles
        for i in range(1, len(v_indices) - 1):
            faces.append([v_indices[0], v_indices[i], v_indices[i + 1]])

    return faces
