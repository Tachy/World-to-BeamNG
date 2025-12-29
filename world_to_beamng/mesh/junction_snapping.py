"""
T-Junction Snapping - Vereinfachter Ansatz.
Snap nur die einmündende Straße an die durchgehende.
"""

import numpy as np
from .. import config


def snap_junctions_with_edges(
    all_road_vertices, road_slope_polygons_2d, t_junctions, original_to_mesh_idx
):
    """
    Snappt T-Junctions durch einfaches Snappen an nächste Segment-Vertices.

    Algoritmus:
    1. Für jede T-Junction: Finde nächste Vertices der durchgehenden Straße
    2. Snap die einmündende Straße an diese Vertices (KEINE Änderung der durchgehenden!)
    """
    if not t_junctions:
        return all_road_vertices

    print(f"    → Snappe {len(t_junctions)} T-Junction Endpunkte...")

    all_road_vertices = list(all_road_vertices)

    from collections import defaultdict

    junctions_by_road = defaultdict(list)

    for junction in t_junctions:
        junctions_by_road[junction["through_road_idx"]].append(junction)

    vertex_offset = 0
    snapped_count = 0

    for mesh_idx, poly_data in enumerate(road_slope_polygons_2d):
        original_coords = poly_data["original_coords"]
        num_points = len(original_coords)

        if num_points < 2:
            vertex_offset += num_points * 2
            continue

        # Finde Original-Index
        original_road_idx = None
        for orig_idx, m_idx in original_to_mesh_idx.items():
            if m_idx == mesh_idx:
                original_road_idx = orig_idx
                break

        if original_road_idx is None:
            vertex_offset += num_points * 2
            continue

        if original_road_idx in junctions_by_road:
            junctions = junctions_by_road[original_road_idx]

            for junction in junctions:
                joining_road_original_idx = junction["joining_road_idx"]

                if joining_road_original_idx not in original_to_mesh_idx:
                    continue

                joining_mesh_idx = original_to_mesh_idx[joining_road_original_idx]
                joining_coords = road_slope_polygons_2d[joining_mesh_idx][
                    "original_coords"
                ]

                # Berechne Vertex-Offset für einmündende Straße
                joining_vertex_offset = sum(
                    len(road_slope_polygons_2d[i]["original_coords"]) * 2
                    for i in range(joining_mesh_idx)
                )

                # Bestimme Endpunkt-Indices der einmündenden Straße
                if junction["joining_is_start"]:
                    joining_left_idx = joining_vertex_offset + 0
                    joining_right_idx = joining_vertex_offset + len(joining_coords)
                    # Vorletzter Punkt für Richtungsbestimmung
                    prev_left_idx = (
                        joining_vertex_offset + 1
                        if len(joining_coords) > 1
                        else joining_left_idx
                    )
                    prev_right_idx = (
                        joining_vertex_offset + len(joining_coords) + 1
                        if len(joining_coords) > 1
                        else joining_right_idx
                    )
                else:
                    joining_left_idx = joining_vertex_offset + (len(joining_coords) - 1)
                    joining_right_idx = (
                        joining_vertex_offset
                        + len(joining_coords)
                        + (len(joining_coords) - 1)
                    )
                    # Vorletzter Punkt für Richtungsbestimmung
                    prev_left_idx = (
                        joining_vertex_offset + (len(joining_coords) - 2)
                        if len(joining_coords) > 1
                        else joining_left_idx
                    )
                    prev_right_idx = (
                        joining_vertex_offset
                        + len(joining_coords)
                        + (len(joining_coords) - 2)
                        if len(joining_coords) > 1
                        else joining_right_idx
                    )

                # Finde Segment der durchgehenden Straße
                param = junction["through_param"]
                segment_idx = int(param * (num_points - 1))
                segment_idx = min(segment_idx, num_points - 2)

                # Hole alle 4 Vertices des relevanten Segments
                left_seg1_idx = vertex_offset + segment_idx
                left_seg2_idx = vertex_offset + segment_idx + 1
                right_seg1_idx = vertex_offset + num_points + segment_idx
                right_seg2_idx = vertex_offset + num_points + segment_idx + 1

                left_seg1_pos = np.array(all_road_vertices[left_seg1_idx])
                left_seg2_pos = np.array(all_road_vertices[left_seg2_idx])
                right_seg1_pos = np.array(all_road_vertices[right_seg1_idx])
                right_seg2_pos = np.array(all_road_vertices[right_seg2_idx])

                # Hole einmündende Positionen
                joining_left_pos = np.array(all_road_vertices[joining_left_idx])
                joining_right_pos = np.array(all_road_vertices[joining_right_idx])

                # Hole vorletzte Positionen für Richtungsbestimmung
                prev_left_pos = np.array(all_road_vertices[prev_left_idx])
                prev_right_pos = np.array(all_road_vertices[prev_right_idx])

                # Berechne Richtungsvektor der einmündenden Straße (Mittellinie)
                joining_center = (joining_left_pos + joining_right_pos) / 2.0
                prev_center = (prev_left_pos + prev_right_pos) / 2.0
                direction_vector = joining_center[:2] - prev_center[:2]

                # Normalisiere Richtungsvektor
                direction_length = np.linalg.norm(direction_vector)
                if direction_length > 1e-6:
                    direction_vector = direction_vector / direction_length

                # Berechne Mittelpunkt des durchgehenden Segments
                through_center_seg = (
                    (left_seg1_pos + left_seg2_pos + right_seg1_pos + right_seg2_pos)
                    / 4.0
                )[:2]

                # Vektor von durchgehender Straße zur linken und rechten Seite
                left_center = ((left_seg1_pos + left_seg2_pos) / 2.0)[:2]
                right_center = ((right_seg1_pos + right_seg2_pos) / 2.0)[:2]

                to_left = left_center - through_center_seg
                to_right = right_center - through_center_seg

                # Skalarprodukt: Kommt die Straße eher von links oder rechts?
                dot_left = np.dot(direction_vector, to_left)
                dot_right = np.dot(direction_vector, to_right)

                # Die Seite mit positivem oder größerem Skalarprodukt ist die Seite,
                # von der die Straße kommt → dort anschließen
                if dot_right > dot_left:  # Vertauscht: War vorher dot_left > dot_right
                    # Einmündung gehört zur linken Seite
                    # Berechne Gesamtdistanz für beide mögliche Zuordnungen
                    dist_assignment1 = np.linalg.norm(
                        joining_left_pos[:2] - left_seg1_pos[:2]
                    ) + np.linalg.norm(joining_right_pos[:2] - left_seg2_pos[:2])
                    dist_assignment2 = np.linalg.norm(
                        joining_left_pos[:2] - left_seg2_pos[:2]
                    ) + np.linalg.norm(joining_right_pos[:2] - left_seg1_pos[:2])

                    if dist_assignment1 < dist_assignment2:
                        through_left_idx = left_seg1_idx
                        through_right_idx = left_seg2_idx
                        snap_left_pos = left_seg1_pos
                        snap_right_pos = left_seg2_pos
                    else:
                        through_left_idx = left_seg2_idx
                        through_right_idx = left_seg1_idx
                        snap_left_pos = left_seg2_pos
                        snap_right_pos = left_seg1_pos

                    # Aktualisiere BEIDE exakt auf die gleichen Positionen
                    all_road_vertices[through_left_idx] = tuple(snap_left_pos)
                    all_road_vertices[through_right_idx] = tuple(snap_right_pos)
                    all_road_vertices[joining_left_idx] = tuple(snap_left_pos)
                    all_road_vertices[joining_right_idx] = tuple(snap_right_pos)

                else:
                    # Einmündung gehört zur rechten Seite
                    # Berechne Gesamtdistanz für beide mögliche Zuordnungen
                    dist_assignment1 = np.linalg.norm(
                        joining_left_pos[:2] - right_seg1_pos[:2]
                    ) + np.linalg.norm(joining_right_pos[:2] - right_seg2_pos[:2])
                    dist_assignment2 = np.linalg.norm(
                        joining_left_pos[:2] - right_seg2_pos[:2]
                    ) + np.linalg.norm(joining_right_pos[:2] - right_seg1_pos[:2])

                    if dist_assignment1 < dist_assignment2:
                        through_left_idx = right_seg1_idx
                        through_right_idx = right_seg2_idx
                        snap_left_pos = right_seg1_pos
                        snap_right_pos = right_seg2_pos
                    else:
                        through_left_idx = right_seg2_idx
                        through_right_idx = right_seg1_idx
                        snap_left_pos = right_seg2_pos
                        snap_right_pos = right_seg1_pos

                    # Aktualisiere BEIDE exakt auf die gleichen Positionen
                    all_road_vertices[through_left_idx] = tuple(snap_left_pos)
                    all_road_vertices[through_right_idx] = tuple(snap_right_pos)
                    all_road_vertices[joining_left_idx] = tuple(snap_left_pos)
                    all_road_vertices[joining_right_idx] = tuple(snap_right_pos)

                snapped_count += 1

        vertex_offset += num_points * 2

    print(f"    → {snapped_count} Einmündungen gesnappt")

    return all_road_vertices
