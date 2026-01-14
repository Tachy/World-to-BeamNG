"""
Mesh-Tile-Slicer: Sortiert Mesh in Tiles ohne Clipping

Terrain-Faces: werden nach Bounding Box sortiert (minimal schnitte)
Road-Faces: werden nach Centroid sortiert (KEIN Clipping um Vertex-Duplikation zu vermeiden)

Output: Pro Tile separate Vertices + Faces + Material-Zuordnung.
"""

import numpy as np
from collections import defaultdict


def slice_mesh_into_tiles(
    vertices, faces, materials_per_face, tile_size=400, grid_bounds=None, vertex_normals=None, mesh_obj=None
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
        mesh_obj: Optional Mesh-Objekt mit uv_indices + uvs (indexed UV-System)

    Returns:
        Dict: {(tile_x, tile_y): {
            "vertices": numpy array,
            "faces": liste,
            "materials": liste (pro geclipptem Face),
            "bounds": (x_min, x_max, y_min, y_max),
            "uv_indices": Dict {face_idx: [uv_idx0, uv_idx1, uv_idx2]},
            "global_uvs": Liste [(u0,v0), (u1,v1), ...]
        }}

    DESIGN: UV-Koordinaten nutzen Indexed-System (wie Vertices)!
    Input: mesh_obj.uv_indices + mesh_obj.uvs
    Output: tile_data["uv_indices"] + tile_data["global_uvs"] (pro Tile)

    """

    # Automatische Bounds aus Vertices wenn nicht gegeben
    if grid_bounds is None:
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        grid_bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())

    x_min, x_max, y_min, y_max = grid_bounds

    # Tile-Indizes berechnen (korrigiert)
    # floor() für Min, floor() für Max (da wir bereits im Tile-Index-System sind)
    tile_x_min = int(np.floor(x_min / tile_size))
    tile_x_max = int(np.floor(x_max / tile_size))  # Nicht ceil()!
    tile_y_min = int(np.floor(y_min / tile_size))
    tile_y_max = int(np.floor(y_max / tile_size))  # Nicht ceil()!

    print(f"  Slicing Mesh in {tile_size}m Tiles...")
    print(f"    Tile-Range: X [{tile_x_min}...{tile_x_max}], Y [{tile_y_min}...{tile_y_max}]")

    # Sammle Faces pro Tile
    tiles_data = {}  # Verwende normales Dict statt defaultdict (schneller)

    # ===== VEKTORISIERTE TERRAIN-FACE ZUORDNUNG =====
    # Konvertiere zu NumPy Arrays einmalig
    faces_array = np.array(faces, dtype=np.int32)
    materials_array = np.array(materials_per_face if materials_per_face else ["unknown"] * len(faces))

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
        bbox_x_min = np.minimum(np.minimum(v0_coords[:, 0], v1_coords[:, 0]), v2_coords[:, 0])
        bbox_y_min = np.minimum(np.minimum(v0_coords[:, 1], v1_coords[:, 1]), v2_coords[:, 1])

        # Tile-Indizes berechnen
        tile_x_indices = np.floor(bbox_x_min / tile_size).astype(np.int32)
        tile_y_indices = np.floor(bbox_y_min / tile_size).astype(np.int32)

        # Gruppiere nach Tiles - direkt ohne unique
        for i in range(len(terrain_face_indices)):
            tile_key = (tile_x_indices[i], tile_y_indices[i])
            if tile_key not in tiles_data:
                tiles_data[tile_key] = {"face_indices": [], "materials": []}
            tiles_data[tile_key]["face_indices"].append((terrain_face_indices[i], None))
            # Berechne Welt-Koordinaten für Material-Namen
            corner_x = tile_key[0] * tile_size
            corner_y = tile_key[1] * tile_size
            tiles_data[tile_key]["materials"].append(f"tile_{corner_x}_{corner_y}")

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

        material = materials_per_face[face_idx] if face_idx < len(materials_per_face) else "unknown"

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
    has_normals = vertex_normals is not None

    for (tile_x, tile_y), tile_info in tiles_data.items():
        # Verwende Dict für schnelleres Lookup
        tile_vertex_mapping = {}  # {original_vertex_idx: local_vertex_idx}
        tile_vertices_set = {}  # {(x, y, z): local_vertex_idx}

        tile_vertices_list = []
        tile_faces_list = []
        tile_materials_list = []
        tile_normals_list = [] if has_normals else None
        tile_original_face_indices = []
        tile_uv_indices = {}  # {new_face_idx: [uv_idx0, uv_idx1, uv_idx2]} - UV-Indizes für dieses Tile

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
                        tile_vertex_mapping[v_idx] = len(tile_vertices_list)
                        tile_vertices_list.append(list(vertices[v_idx]))
                        if has_normals:
                            tile_normals_list.append(list(vertex_normals[v_idx]))

                    local_indices.append(tile_vertex_mapping[v_idx])

                new_face_idx = len(tile_faces_list)
                tile_faces_list.append(local_indices)
                tile_materials_list.append(material)
                tile_original_face_indices.append(face_idx)

                # Kopiere UV-Indizes für dieses Face (falls vorhanden)
                if mesh_obj and face_idx in mesh_obj.uv_indices:
                    tile_uv_indices[new_face_idx] = mesh_obj.uv_indices[face_idx]
                continue

            # ===== FALL 2: Geclipptes Polygon (Road) =====
            clipped_poly = poly_or_none

            # Koordinaten vom geclippten Polygon (XY nur)
            coords_xy = np.array(clipped_poly.exterior.coords[:-1])  # Letzter Punkt = erster

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

            # Approximative Normal für neue Vertices (Mittel der Original-Vertex-Normalen)
            avg_normal = None
            if has_normals:
                n0 = vertex_normals[v0_idx]
                n1 = vertex_normals[v1_idx]
                n2 = vertex_normals[v2_idx]
                avg = n0 + n1 + n2
                norm = np.linalg.norm(avg)
                if norm > 1e-12:
                    avg_normal = (avg / norm).tolist()

            # Dedupliziere Vertices im Tile
            face_indices_for_tile = []
            for point_3d in face_vertices_3d:
                point_3d_tuple = tuple(point_3d)
                if point_3d_tuple not in tile_vertices_set:
                    tile_vertices_set[point_3d_tuple] = len(tile_vertices_list)
                    tile_vertices_list.append(list(point_3d))
                    if has_normals:
                        tile_normals_list.append(avg_normal if avg_normal is not None else [0.0, 0.0, 1.0])
                face_indices_for_tile.append(tile_vertices_set[point_3d_tuple])

            # Trianguliere geclipptes Polygon (einfache Fan-Triangulation)
            if len(face_indices_for_tile) >= 3:
                for i in range(1, len(face_indices_for_tile) - 1):
                    new_face_idx = len(tile_faces_list)
                    tile_faces_list.append(
                        [
                            face_indices_for_tile[0],
                            face_indices_for_tile[i],
                            face_indices_for_tile[i + 1],
                        ]
                    )
                    tile_materials_list.append(material)
                    tile_original_face_indices.append(face_idx)

                    # Kopiere UV-Indizes vom Original-Face (falls vorhanden)
                    if mesh_obj and face_idx in mesh_obj.uv_indices:
                        tile_uv_indices[new_face_idx] = mesh_obj.uv_indices[face_idx]

        # Speichere Tile-Daten
        if tile_vertices_list:
            # === BERECHNE UVs PRO TILE MIT TILE-GRENZEN (CRITICAL FIX) ===
            # WICHTIG: UVs müssen auf die Tile-Grenzen normalisiert werden, nicht auf Vertex-Bounds!
            # Grund: Nach Clipping sind Vertex-Bounds != Tile-Grenzen
            # Dadurch entsteht Versatz bei Texture-Alignment zwischen Tiles.
            # Lösung: Nutze exakte Tile-Grenzen als Bounds für UV-Normalisierung.

            tile_vertices_array = np.array(tile_vertices_list, dtype=np.float32)

            # Berechne EXAKTE Tile-Grenzen (nicht aus Vertices!)
            # Tile-Indizes sind: tile_x, tile_y
            # Tile-Grenzen sind: [tile_x * tile_size, (tile_x+1) * tile_size] × [tile_y * tile_size, (tile_y+1) * tile_size]
            x_min_tile = tile_x * tile_size
            x_max_tile = (tile_x + 1) * tile_size
            y_min_tile = tile_y * tile_size
            y_max_tile = (tile_y + 1) * tile_size

            x_range = x_max_tile - x_min_tile  # immer tile_size (z.B. 500m)
            y_range = y_max_tile - y_min_tile  # immer tile_size (z.B. 500m)

            # Berechne UVs für alle Vertices im Tile
            # Nutze Tile-Grenzen, nicht Vertex-Bounds!
            all_uvs_x = (tile_vertices_array[:, 0] - x_min_tile) / x_range
            all_uvs_y = (tile_vertices_array[:, 1] - y_min_tile) / y_range

            # Dedupliziere UVs (float16 für Precision)
            all_uvs_x_f16 = np.float16(all_uvs_x)
            all_uvs_y_f16 = np.float16(all_uvs_y)

            tile_uvs = []  # Liste von (u, v) Tupeln
            uv_lookup = {}  # (u_f16, v_f16) → uv_idx
            vertex_to_uv_idx = {}  # tile_vertex_idx → uv_idx

            for tile_v_idx in range(len(tile_vertices_array)):
                u = all_uvs_x_f16[tile_v_idx]
                v = all_uvs_y_f16[tile_v_idx]
                uv_key = (u, v)

                if uv_key not in uv_lookup:
                    uv_lookup[uv_key] = len(tile_uvs)
                    tile_uvs.append((float(u), float(v)))

                vertex_to_uv_idx[tile_v_idx] = uv_lookup[uv_key]

            # Setze UV-Indizes für alle Faces
            # WICHTIG: Road-UVs werden zentral in mesh.preserve_road_uvs() gehandhabt!
            tile_uv_indices_computed = {}

            # Hole preservierte Road-UV-Daten aus mesh_obj
            road_uv_data = getattr(mesh_obj, "road_uv_data", {}) if mesh_obj else {}

            for face_idx, face in enumerate(tile_faces_list):
                v0, v1, v2 = face

                # Prüfe ob dieses Face eine Road mit speziellen UVs ist
                original_face_idx = (
                    tile_original_face_indices[face_idx] if face_idx < len(tile_original_face_indices) else None
                )

                if original_face_idx in road_uv_data:
                    # Road-Face mit speziellen UVs - füge sie ins Tile-UV-System ein
                    uv_coords = road_uv_data[original_face_idx]  # [(u0,v0), (u1,v1), (u2,v2)]
                    tile_uv_indices_new = []

                    for u, v in uv_coords:
                        # Füge UV ins Tile-Pool ein (dedupliziert mit float16)
                        uv_key = (np.float16(u), np.float16(v))
                        if uv_key not in uv_lookup:
                            uv_lookup[uv_key] = len(tile_uvs)
                            tile_uvs.append((float(u), float(v)))
                        tile_uv_indices_new.append(uv_lookup[uv_key])

                    tile_uv_indices_computed[face_idx] = tile_uv_indices_new
                else:
                    # Terrain-Face - nutze berechnete UVs aus Tile-Grenzen
                    tile_uv_indices_computed[face_idx] = [
                        vertex_to_uv_idx[v0],
                        vertex_to_uv_idx[v1],
                        vertex_to_uv_idx[v2],
                    ]
            result[(tile_x, tile_y)] = {
                "vertices": tile_vertices_array,
                "faces": tile_faces_list,
                "materials": tile_materials_list,
                "normals": np.array(tile_normals_list, dtype=np.float32) if has_normals else None,
                "original_face_indices": tile_original_face_indices,
                "vertex_mapping": tile_vertex_mapping,  # Mapping: orig_vertex_idx → tile_local_idx
                "uv_indices": tile_uv_indices_computed,  # PRO-TILE berechnete UV-Indizes
                "global_uvs": tile_uvs,  # PRO-TILE berechnete UVs im Bereich [0,1]
                "bounds": (
                    x_min_tile,  # Exakte Tile-Grenzen, nicht Vertex-Bounds!
                    x_max_tile,
                    y_min_tile,
                    y_max_tile,
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
