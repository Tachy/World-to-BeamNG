"""
Stitching von Terrain-Lücken entlang Centerlines.
Iteriert über alle Straßen und füllt Löcher in Suchkreisen.
"""

import numpy as np
from shapely.geometry import LineString

from .. import config
from .stitch_local import find_boundary_polygons_in_circle, export_boundary_polygons_to_obj


def stitch_all_gaps(
    road_data_for_classification,
    vertex_manager,
    mesh,
    terrain_vertex_indices,
):
    """
    Iteriert über alle Centerlines und stitcht Löcher in Suchkreisen.

    DEBUG-Modus: Überspringt x Straßen, nimmt Mittelpunkt der (x+1)-ten Straße,
                 sucht dort Boundaries, bricht dann ab.

    Später (Production): Alle Suchkreise verarbeiten, Löcher triangulieren.
    """

    # ========== DEBUG-FLAGS (später entfernen) ==========
    DEBUG_SKIP_ROADS = 1  # Überspringe 4 Straßen
    DEBUG_SINGLE_CIRCLE = True  # Stoppe nach erstem Suchkreis

    print("Stitche Terrain-Gaps entlang Centerlines...")

    road_count = 0
    search_radius = config.CENTERLINE_SEARCH_RADIUS
    sample_spacing = config.CENTERLINE_SAMPLE_SPACING

    for road_info in road_data_for_classification:
        road_count += 1

        # DEBUG: Überspringe erste x Straßen
        if road_count <= DEBUG_SKIP_ROADS:
            continue

        # Ab hier: Straße #(DEBUG_SKIP_ROADS + 1)
        print(f"  Verarbeite Straße #{road_count}...")

        # Hole Centerline: Direkt aus trimmed_centerline (nicht "centerline_points"!)
        trimmed_centerline = road_info.get("trimmed_centerline", [])

        if not trimmed_centerline or len(trimmed_centerline) < 2:
            print(f"  [SKIP] Straße #{road_count}: Keine Centerline")
            continue

        centerline_3d = np.array(trimmed_centerline, dtype=float)
        centerline_points = centerline_3d[:, :2]  # Extrahiere XY
        centerline_linestring = LineString(centerline_points)

        if not centerline_linestring.is_valid or centerline_linestring.length == 0:
            continue

        # Sample entlang Centerline
        total_length = centerline_linestring.length
        num_samples = max(2, int(np.ceil(total_length / sample_spacing)) + 1)
        sample_distances = np.linspace(0, total_length, num_samples)

        # DEBUG: Nimm nur Mittelpunkt
        mid_distance = total_length / 2.0  # Mittelpunkt der Centerline
        mid_pt_2d = centerline_linestring.interpolate(mid_distance)

        # Z-Koordinate aus originaler 3D-Centerline interpolieren (oder Mittelpunkt nehmen)
        mid_idx_3d = len(centerline_3d) // 2
        mid_z = centerline_3d[mid_idx_3d, 2]

        centerline_midpoint = np.array([mid_pt_2d.x, mid_pt_2d.y, mid_z])

        print(
            f"  Suche Boundaries am Mittelpunkt: {centerline_midpoint[:2]}, Z={mid_z:.1f}, Radius={search_radius:.1f}m"
        )

        # Finde Boundaries in diesem Suchkreis
        boundaries = find_boundary_polygons_in_circle(
            centerline_point=centerline_midpoint,
            search_radius=search_radius,
            vertex_manager=vertex_manager,
            mesh=mesh,
            terrain_vertex_indices=terrain_vertex_indices,
            debug=True,
        )

        # Exportiere für Visualisierung
        export_boundary_polygons_to_obj(boundaries, centerline_midpoint, search_radius=search_radius)

        if boundaries:
            print(f"  [OK] {len(boundaries)} Boundary-Polygone gefunden und exportiert")
        else:
            print("  [i] Keine Boundary-Polygone gefunden (Kreis + Centerline exportiert)")

        # DEBUG: Sofort abbrechen nach erstem Suchkreis
        if DEBUG_SINGLE_CIRCLE:
            print("  [DEBUG] Single-Circle-Modus: Abbruch nach erstem Kreis")
            return []

    # Production-Rückgabe: Stitch-Faces
    return []
