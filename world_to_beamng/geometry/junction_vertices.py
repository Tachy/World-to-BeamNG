"""
Extraktion von Junction-Vertices - VEREINFACHTER Ansatz.

Die Junction-Quads werden direkt aus den edge-Punkten der 2 besten Straßen gebaut:
- Wähle 2 Straßen mit möglichst 90° Winkel
- Nimm deren left+right edge-Punkte als Quad-Eckpunkte
- Fertig!
"""

import numpy as np


def extract_junction_vertices_from_mesh(junctions, road_polygons, vertex_manager):
    """
    Berechne Junction-Quad Vertices aus edge-Punkten der Straßen.

    Für jede Junction:
    1. Sammle left/right edge-Punkte vom gekürtzten Ende jeder Straße
    2. Wähle die 2 Straßen mit bestem Winkel (möglichst 90°)
    3. Die 4 edge-Punkte dieser 2 Straßen = Junction-Quad-Eckpunkte
    4. Sortiere CCW um Junction-Punkt

    Args:
        junctions: Liste von Junction-Dicts
        road_polygons: Liste von Road-Polygons (gekürzt!)
        vertex_manager: VertexManager

    Returns:
        Dict: {junction_idx: {'vertices_2d': [...], 'vertices_3d': [...], ...}}
    """
    junction_vertices_map = {}
    debug_junction_idx = 0

    for junction_idx, junction in enumerate(junctions):
        junction_pos = np.array(junction["position"])
        road_indices = junction["road_indices"]
        connection_types = junction.get("connection_types", {})

        if junction_idx == debug_junction_idx:
            print(
                f"    Junction {junction_idx} @ {junction_pos[:2]}: {len(road_indices)} roads"
            )

        # Sammle edge-Punkte und Richtungen pro Straße
        road_data = {}  # {road_idx: {'left': point, 'right': point, 'direction': dir}}

        for road_idx in road_indices:
            if road_idx >= len(road_polygons):
                continue

            road = road_polygons[road_idx]
            coords = np.array(road.get("coords", []))
            conn_types = connection_types.get(road_idx, [])

            if coords is None or len(coords) < 2 or not conn_types:
                continue

            # Bestimme welches Ende der Straße an dieser Junction ist
            is_start = "start" in conn_types
            is_end = "end" in conn_types

            if is_start:
                p_truncated = coords[0]
                if len(coords) >= 2:
                    direction = coords[1][:2] - coords[0][:2]
                else:
                    direction = junction_pos[:2] - coords[0][:2]

            elif is_end:
                p_truncated = coords[-1]
                if len(coords) >= 2:
                    direction = coords[-1][:2] - coords[-2][:2]
                else:
                    direction = junction_pos[:2] - coords[-1][:2]
            else:
                continue

            # Normalisiere Richtung
            norm = np.linalg.norm(direction)
            if norm < 0.01:
                continue

            direction_normalized = direction / norm
            perpendicular = np.array(
                [-direction_normalized[1], direction_normalized[0]]
            )
            half_width = 3.5

            # Berechne left/right edge-Punkte
            left_point = p_truncated[:2] + perpendicular * half_width
            right_point = p_truncated[:2] - perpendicular * half_width

            road_data[road_idx] = {
                "left": left_point,
                "right": right_point,
                "direction": direction_normalized,
                "z": p_truncated[2],
            }

        if len(road_data) < 2:
            if junction_idx == debug_junction_idx:
                print(f"        → SKIP: zu wenig Straßen ({len(road_data)})")
            continue

        # Finde beste 2 Straßen (möglichst 90° Winkel)
        road_list = list(road_data.keys())
        best_pair = None
        best_angle_diff = float("inf")

        for i in range(len(road_list)):
            for j in range(i + 1, len(road_list)):
                road_a = road_list[i]
                road_b = road_list[j]

                dir_a = road_data[road_a]["direction"]
                dir_b = road_data[road_b]["direction"]

                dot = np.dot(dir_a, dir_b)
                dot = np.clip(dot, -1.0, 1.0)
                angle_rad = np.arccos(abs(dot))
                angle_deg = np.degrees(angle_rad)

                diff_from_90 = abs(angle_deg - 90.0)

                if diff_from_90 < best_angle_diff:
                    best_angle_diff = diff_from_90
                    best_pair = (road_a, road_b)

        if best_pair is None:
            if junction_idx == debug_junction_idx:
                print(f"        → SKIP: kein Straßenpaar")
            continue

        road_a, road_b = best_pair

        if junction_idx == debug_junction_idx:
            angle_between = 90.0 - best_angle_diff
            print(
                f"        → Beste Straßen: {road_a} & {road_b} (Winkel: {angle_between:.1f}°)"
            )

        # Die 4 Eckpunkte: left+right von beiden Straßen
        # WICHTIG: Behalte die individuellen Z-Werte, verwende NICHT z_avg!
        quad_vertices_3d = [
            (
                road_data[road_a]["left"][0],
                road_data[road_a]["left"][1],
                road_data[road_a]["z"],
            ),
            (
                road_data[road_a]["right"][0],
                road_data[road_a]["right"][1],
                road_data[road_a]["z"],
            ),
            (
                road_data[road_b]["left"][0],
                road_data[road_b]["left"][1],
                road_data[road_b]["z"],
            ),
            (
                road_data[road_b]["right"][0],
                road_data[road_b]["right"][1],
                road_data[road_b]["z"],
            ),
        ]

        # 2D Vertices für Sortierung
        quad_vertices_2d = [(v[0], v[1]) for v in quad_vertices_3d]

        # Sortiere CCW um Junction-Position
        def angle_from_junction(v):
            dx = v[0] - junction_pos[0]
            dy = v[1] - junction_pos[1]
            return np.arctan2(dy, dx)

        quad_vertices_3d.sort(key=angle_from_junction)

        # Aktualisiere 2D-Vertices nach Sortierung
        quad_vertices_2d = [(v[0], v[1]) for v in quad_vertices_3d]

        if junction_idx == debug_junction_idx:
            print(f"        → Final: 4 Eckpunkte")
            for i, v in enumerate(quad_vertices_3d):
                dist = np.sqrt(
                    (v[0] - junction_pos[0]) ** 2 + (v[1] - junction_pos[1]) ** 2
                )
                print(
                    f"           [{i}] ({v[0]:.1f}, {v[1]:.1f}, Z={v[2]:.1f}) dist={dist:.1f}m"
                )

        junction_vertices_map[junction_idx] = {
            "vertices": quad_vertices_3d,
            "vertices_2d": [(v[0], v[1]) for v in quad_vertices_3d],
            "vertices_3d": quad_vertices_3d,
            "road_indices": [road_a, road_b],  # Nur die 2 Straßen, die das Quad bilden
            "center": tuple(junction_pos),
            "road_edge_data": road_data,  # WICHTIG: Speichere Edge-Points für Connectoren!
        }

    return junction_vertices_map
