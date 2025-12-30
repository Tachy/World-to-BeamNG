"""
Baue Verbindungs-Quads zwischen gekürzte Straßen und zentrale Junction-Quads.

Diese Connectors füllen die Lücke zwischen:
- Den gekürtzten Straßen-Enden (10m ab Junction)
- Den Eckpunkten des zentralen Junction-Quads
"""

import numpy as np


def build_junction_connectors(
    junction_polys, junctions, road_polygons, truncation_distance, road_width=7.0
):
    """
    Baue Verbindungs-Quads/Trapezoids zwischen Straßen-Enden und Junction-Quads.

    Das Junction-Quad ist ein einfaches Rechteck (7x7m).
    Für jede Straße:
    - Finde die Fahrtrichtung zur Junction
    - Finde die entsprechende Kante des Junction-Quads
    - Verbinde die gekürzte Straße mit dieser Kante

    Args:
        junction_polys: List von Junction-Polygonen (einfache Rechtecke)
        junctions: List von Junctions mit Positionen und Richtungen
        road_polygons: Alle Straßen (mit gekürtzten coords)
        truncation_distance: Wie weit Straßen gekürzt wurden (10m)
        road_width: Straßenbreite

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
        if not junction_vertices_3d:
            continue

        # Junction-Quad Vertices (geometrisch korrekt aus Schnittpunkten)
        junction_vertices = junction_poly["vertices_2d"]

        if not junction_vertices or len(junction_vertices) < 3:
            continue

        # Hole die vorberechneten Edge-Daten (left/right Punkte pro Straße)
        road_edge_data = junction_poly.get("road_edge_data", {})

        # Hole die Richtungsvektoren
        direction_vectors = junction.get("direction_vectors", {})

        # Gehe durch alle Straßen dieser Junction
        for road_idx in junction["road_indices"]:
            road = road_polygons[road_idx]
            coords = road["coords"]

            if len(coords) < 2:
                continue

            conn_types = junction["connection_types"].get(road_idx, [])
            direction = np.array(direction_vectors.get(road_idx, [1.0, 0.0]))

            # Bestimme: Straße beginnt ("start") oder endet ("end") hier?
            if "start" in conn_types:
                # Straße beginnt: gekürtzter Endpunkt ist coords[0]
                road_end_2d = np.array(coords[0][:2])
                road_end_z = coords[0][2]
                # Nächster Punkt für Richtungsberechnung
                road_next_2d = (
                    np.array(coords[1][:2]) if len(coords) > 1 else road_end_2d
                )
                # Richtung: WEG von Junction (von coords[0] zu coords[1])
                local_direction = road_next_2d - road_end_2d
            elif "end" in conn_types:
                # Straße endet: gekürtzter Endpunkt ist coords[-1]
                road_end_2d = np.array(coords[-1][:2])
                road_end_z = coords[-1][2]
                # Vorheriger Punkt für Richtungsberechnung
                road_next_2d = (
                    np.array(coords[-2][:2]) if len(coords) > 1 else road_end_2d
                )
                # Richtung: WEG von Junction (von coords[-2] zu coords[-1])
                # WICHTIG: Hier muss die Richtung UMGEDREHT werden!
                local_direction = road_end_2d - road_next_2d
            else:
                continue

            # WICHTIG: Nutze die EXAKTE Richtung der Straße an dieser Stelle
            # Das stellt sicher, dass die Rand-Punkte identisch mit dem Straßen-Mesh sind
            local_norm = np.linalg.norm(local_direction)
            if local_norm > 0.001:
                local_direction = local_direction / local_norm
            else:
                local_direction = direction  # Fallback

            # WICHTIG: Nutze die VORBERECHNETEN Edge-Vertices vom Road-Mesh!
            # Diese sind identisch mit den Mesh-Punkten und vermeiden Rundungsfehler
            if road_idx in road_edge_data:
                road_left = road_edge_data[road_idx]["left"]
                road_right = road_edge_data[road_idx]["right"]
                road_end_z = road_edge_data[road_idx]["z"]

                # DEBUG für erste Junction
                if junction_idx == 0 and road_idx == list(road_edge_data.keys())[0]:
                    print(f"    DEBUG Connector road {road_idx}:")
                    print(f"      road_left: {road_left}")
                    print(f"      road_right: {road_right}")
                    print(f"      road_end_z: {road_end_z}")
                    print(f"      road_end_2d (from coords): {road_end_2d}")
            else:
                # Fallback: Berechne Eckpunkte neu (sollte nicht passieren)
                perp = np.array([-local_direction[1], local_direction[0]])
                road_left = road_end_2d + perp * half_width
                road_right = road_end_2d - perp * half_width
                if junction_idx == 0:
                    print(f"    ⚠ FALLBACK für road {road_idx} - road_edge_data fehlt!")

            # NEUE LOGIK: Finde die 2 Junction-Vertices, die zu dieser Straße gehören
            # LOGIK: Das Junction-Quad hat 4 Vertices (sortiert CCW)
            # Für jede Straße müssen wir die 2 BENACHBARTEN Vertices wählen
            # Die zur Straße "gehören" = die beiden nächsten in der CCW-Reihenfolge
            #
            # Trick: Berechne Richtung von junction_center zu road_end
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

            # Die 2 Vertices mit kleinstem Winkel zur Straßen-Richtung
            sorted_by_angle = sorted(range(len(angles)), key=lambda i: angles[i])
            idx_left = sorted_by_angle[0]
            idx_right = sorted_by_angle[1]

            junction_vertex_idx_1 = idx_left
            junction_vertex_idx_2 = idx_right

            junction_point1 = np.array(junction_vertices[junction_vertex_idx_1])
            junction_point2 = np.array(junction_vertices[junction_vertex_idx_2])

            # Hole die tatsächlichen 3D-Koordinaten aus dem Junction-Quad-Mesh
            junction_point1_3d = junction_vertices_3d[junction_vertex_idx_1]
            junction_point2_3d = junction_vertices_3d[junction_vertex_idx_2]

            # Baue ein Trapezoid-Quad:
            # road_left, road_right, junction_point2, junction_point1
            # (CCW Orientierung)
            connector = {
                "type": "junction_connector",
                "vertices_2d": [
                    tuple(road_left),
                    tuple(road_right),
                    tuple(junction_point2),
                    tuple(junction_point1),
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
        vertex_manager: VertexManager zum Hinzufügen der Vertices

    Returns:
        List von Faces [v0, v1, v2] (Triangles)
    """
    faces = []

    for connector in connector_polys:
        vertices_3d = connector["vertices_3d"]

        if len(vertices_3d) < 3:
            continue

        # Füge alle Vertices hinzu
        v_indices = vertex_manager.add_vertices_batch_dedup_fast(vertices_3d)

        # Fan-Triangulation: v0 ist Zentrum
        # Für N Vertices: N-2 Triangles
        for i in range(1, len(v_indices) - 1):
            faces.append([v_indices[0], v_indices[i], v_indices[i + 1]])

    return faces
