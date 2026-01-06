"""
Mesh-Tile-Slicer: Sortiert Mesh in Tiles ohne Clipping

Terrain-Faces: werden nach Bounding Box sortiert (minimal schnitte)
Road-Faces: werden nach Centroid sortiert (KEIN Clipping um Vertex-Duplikation zu vermeiden)

Output: Pro Tile separate Vertices + Faces + Material-Zuordnung.
"""

import numpy as np
from collections import defaultdict


def slice_mesh_into_tiles(
    vertices, faces, materials_per_face, tile_size=400, grid_bounds=None
):
    """
    Clippt Mesh in 400×400m Tiles mit Sutherland-Hodgman.

    OPTIMIERUNG: Nur Road-Faces werden geclippt!
    Terrain-Faces werden direkt nach Bounding Box sortiert.

    Args:
        vertices: numpy array (n, 3) mit XYZ-Koordinaten
        faces: liste von [v0, v1, v2] (0-basiert)
        materials_per_face: liste von Material-Strings pro Face
                            z.B. ["terrain", "road", "terrain", ...]
        tile_size: Größe pro Tile in Metern (default: 400)
        grid_bounds: (x_min, x_max, y_min, y_max) oder None für Auto

    Returns:
        Dict: {(tile_x, tile_y): {
            "vertices": numpy array,
            "faces": liste,
            "materials": liste (pro geclipptem Face),
            "bounds": (x_min, x_max, y_min, y_max)
        }}
    """

    # Automatische Bounds aus Vertices wenn nicht gegeben
    if grid_bounds is None:
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        grid_bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())

    x_min, x_max, y_min, y_max = grid_bounds

    # Tile-Indizes berechnen
    tile_x_min = int(np.floor(x_min / tile_size))
    tile_x_max = int(np.ceil(x_max / tile_size))
    tile_y_min = int(np.floor(y_min / tile_size))
    tile_y_max = int(np.ceil(y_max / tile_size))

    print(f"  Slicing Mesh in {tile_size}m Tiles...")
    print(
        f"    Tile-Range: X [{tile_x_min}...{tile_x_max}], Y [{tile_y_min}...{tile_y_max}]"
    )

    # Sammle Faces pro Tile
    tiles_data = {}  # Verwende normales Dict statt defaultdict (schneller)

    # ===== VEKTORISIERTE TERRAIN-FACE ZUORDNUNG =====
    # Konvertiere zu NumPy Arrays einmalig
    faces_array = np.array(faces, dtype=np.int32)
    materials_array = np.array(
        materials_per_face if materials_per_face else ["unknown"] * len(faces)
    )

    terrain_mask = materials_array == "terrain"
    terrain_face_indices = np.where(terrain_mask)[0]

    if len(terrain_face_indices) > 0:
        # Extrahiere alle Terrain-Face Vertices auf einmal
        terrain_faces = faces_array[terrain_face_indices]

        # Hole XY-Koordinaten für alle Vertices der Terrain-Faces
        v0_coords = vertices[terrain_faces[:, 0], :2]
        v1_coords = vertices[terrain_faces[:, 1], :2]
        v2_coords = vertices[terrain_faces[:, 2], :2]

        # Berechne Bounding Boxes vektorisiert
        bbox_x_min = np.minimum(
            np.minimum(v0_coords[:, 0], v1_coords[:, 0]), v2_coords[:, 0]
        )
        bbox_y_min = np.minimum(
            np.minimum(v0_coords[:, 1], v1_coords[:, 1]), v2_coords[:, 1]
        )

        # Tile-Indizes berechnen
        tile_x_indices = np.floor(bbox_x_min / tile_size).astype(np.int32)
        tile_y_indices = np.floor(bbox_y_min / tile_size).astype(np.int32)

        # Gruppiere nach Tiles - direkt ohne unique
        for i in range(len(terrain_face_indices)):
            tile_key = (tile_x_indices[i], tile_y_indices[i])
            if tile_key not in tiles_data:
                tiles_data[tile_key] = {"face_indices": [], "materials": []}
            tiles_data[tile_key]["face_indices"].append((terrain_face_indices[i], None))
            tiles_data[tile_key]["materials"].append(f"tile_{tile_key[0]}_{tile_key[1]}")

    # ===== ROAD-FACES: KEIN CLIPPING - nur zu Tile-Center zuordnen =====
    # Road-Faces werden NICHT geclippt! Dies verursacht Vertex-Duplikation
    # und sporadische falsche Vernetzung an Tile-Grenzen.
    # Stattdessen: Face als ganzes zum Tile mit Centroid zuordnen.
    road_face_indices = np.where(~terrain_mask)[0]

    for i in range(len(road_face_indices)):
        face_idx = road_face_indices[i]
        v0, v1, v2 = faces[face_idx]
        p0 = vertices[v0, :2]
        p1 = vertices[v1, :2]
        p2 = vertices[v2, :2]

        material = (
            materials_per_face[face_idx]
            if face_idx < len(materials_per_face)
            else "unknown"
        )

        # ===== Road-Faces: Centroid-basierte Zuordnung OHNE Clipping =====
        # Berechne Centroid des Faces
        centroid_x = (p0[0] + p1[0] + p2[0]) / 3.0
        centroid_y = (p0[1] + p1[1] + p2[1]) / 3.0

        # Bestimme Tile basierend auf Centroid
        tile_x = int(np.floor(centroid_x / tile_size))
        tile_y = int(np.floor(centroid_y / tile_size))

        # Initialisiere Tile-Daten falls noch nicht vorhanden
        if (tile_x, tile_y) not in tiles_data:
            tiles_data[(tile_x, tile_y)] = {"face_indices": [], "materials": []}

        # Füge Original-Face UNGECLIPPt hinzu (poly_or_none = None bedeutet: Original-Face)
        tiles_data[(tile_x, tile_y)]["face_indices"].append((face_idx, None))
        tiles_data[(tile_x, tile_y)]["materials"].append(material)

    # Konvertiere zu Output-Format
    result = {}

    for (tile_x, tile_y), tile_info in tiles_data.items():
        # Verwende Dict für schnelleres Lookup
        tile_vertex_mapping = {}  # {original_vertex_idx: local_vertex_idx}
        tile_vertices_set = {}  # {(x, y, z): local_vertex_idx}

        tile_vertices_list = []
        tile_faces_list = []
        tile_materials_list = []

        for idx, (face_idx, poly_or_none) in enumerate(tile_info["face_indices"]):
            material = tile_info["materials"][idx]

            # ===== FALL 1: Original Face (poly_or_none ist None) =====
            if poly_or_none is None:
                # Direkt Vertices vom Original-Face verwenden
                v0, v1, v2 = faces[face_idx]

                # Für jedes Vertex: zu tile_vertices hinzufügen (OPTIMIERT mit Index-Mapping)
                local_indices = []
                for v_idx in [v0, v1, v2]:
                    if v_idx not in tile_vertex_mapping:
                        # Neues Vertex - verwende Index statt Tuple!
                        tile_vertex_mapping[v_idx] = len(tile_vertices_list)
                        tile_vertices_list.append(list(vertices[v_idx]))

                    local_indices.append(tile_vertex_mapping[v_idx])

                tile_faces_list.append(local_indices)
                tile_materials_list.append(material)
                continue

            # ===== FALL 2: Geclipptes Polygon (Road) =====
            clipped_poly = poly_or_none

            # Koordinaten vom geclippten Polygon (XY nur)
            coords_xy = np.array(
                clipped_poly.exterior.coords[:-1]
            )  # Letzter Punkt = erster

            # Z-Werte aus Original-Face interpolieren (barycentric, vektorisiert)
            v0_idx, v1_idx, v2_idx = faces[face_idx]
            p0 = vertices[v0_idx]
            p1 = vertices[v1_idx]
            p2 = vertices[v2_idx]

            # Extrahiere XYZ
            x0, y0, z0 = p0
            x1, y1, z1 = p1
            x2, y2, z2 = p2

            # Vektorisierte Koordinaten
            x = coords_xy[:, 0]
            y = coords_xy[:, 1]

            # Determinante für barycentrische Koordinaten
            det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

            if abs(det) < 1e-12:
                # Degenerated triangle - alle Punkte bekommen average Z
                z = np.full(len(coords_xy), (z0 + z1 + z2) / 3.0)
            else:
                # Barycentrische Koordinaten (vektorisiert)
                lambda1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
                lambda2 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
                lambda3 = 1.0 - lambda1 - lambda2

                # Z interpolieren (vektorisiert)
                z = lambda1 * z0 + lambda2 * z1 + lambda3 * z2

            # Kombiniere zu 3D Punkten
            face_vertices_3d = np.column_stack([coords_xy, z])

            # Dedupliziere Vertices im Tile
            face_indices_for_tile = []
            for point_3d in face_vertices_3d:
                point_3d_tuple = tuple(point_3d)
                if point_3d_tuple not in tile_vertices_set:
                    tile_vertices_set[point_3d_tuple] = len(tile_vertices_list)
                    tile_vertices_list.append(list(point_3d))
                face_indices_for_tile.append(tile_vertices_set[point_3d_tuple])

            # Trianguliere geclipptes Polygon (einfache Fan-Triangulation)
            if len(face_indices_for_tile) >= 3:
                for i in range(1, len(face_indices_for_tile) - 1):
                    tile_faces_list.append(
                        [
                            face_indices_for_tile[0],
                            face_indices_for_tile[i],
                            face_indices_for_tile[i + 1],
                        ]
                    )
                    tile_materials_list.append(material)

        # Speichere Tile-Daten
        if tile_vertices_list:
            result[(tile_x, tile_y)] = {
                "vertices": np.array(tile_vertices_list, dtype=np.float32),
                "faces": tile_faces_list,
                "materials": tile_materials_list,
                "bounds": (
                    tile_x * tile_size,
                    (tile_x + 1) * tile_size,
                    tile_y * tile_size,
                    (tile_y + 1) * tile_size,
                ),
            }

    print(f"  [OK] {len(result)} Tiles erstellt (mit Clipping)")
    return result

    return result


def get_tile_grid_info(vertices, tile_size=400):
    """
    Gibt Info über das Tile-Grid basierend auf Vertices.

    Returns:
        {
            "tile_size": int,
            "bounds": (x_min, x_max, y_min, y_max),
            "tile_range_x": (min_x, max_x),
            "tile_range_y": (min_y, max_y),
            "num_tiles_x": int,
            "num_tiles_y": int,
        }
    """
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())

    x_min, x_max, y_min, y_max = bounds

    tile_x_min = int(np.floor(x_min / tile_size))
    tile_x_max = int(np.ceil(x_max / tile_size))
    tile_y_min = int(np.floor(y_min / tile_size))
    tile_y_max = int(np.ceil(y_max / tile_size))

    return {
        "tile_size": tile_size,
        "bounds": bounds,
        "tile_range_x": (tile_x_min, tile_x_max),
        "tile_range_y": (tile_y_min, tile_y_max),
        "num_tiles_x": tile_x_max - tile_x_min,
        "num_tiles_y": tile_y_max - tile_y_min,
    }
