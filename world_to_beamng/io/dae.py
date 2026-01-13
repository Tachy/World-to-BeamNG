"""
DAE (Collada) Exporter für Mesh (alle Tiles in einer Datei).

Exportiert eine .dae Datei mit:
- Mehrere Geometrien (ein Chunk = eine Geometrie)
- Material-Gruppen (terrain, road, etc.)
- Pro Face eindeutige Material-Zuordnung
"""

import numpy as np
import os


def export_merged_dae(
    tiles_dict,
    output_path,
    tile_size=400,
    mesh_obj=None,
):
    """
    Exportiert alle Tiles als EINE .dae (Collada 1.4.1) mit mehreren Geometrien.

    REFACTORED: Verwendet jetzt DAEExporter und liest UVs direkt aus Mesh.face_uvs.

    Args:
        tiles_dict: Dictionary von tile_slicer.slice_mesh_into_tiles()
                    Format: {(tile_x, tile_y): {"vertices": [...], "faces": [...], "materials": [...]}}
        output_path: Ziel-Dateipfad (wird als Vorlage verwendet, echte Datei nutzt Koordinaten)
        tile_size: Tile-Größe in Metern (zur Koordinaten-Umrechnung)
        mesh_obj: Optional: Mesh-Objekt mit face_uvs für UV-Koordinaten

    Returns:
        Tatsächlicher DAE-Dateiname mit Koordinaten-Index
    """
    from ..managers import DAEExporter

    # Konvertiere tiles_dict zu DAEExporter Format
    meshes = []
    for (tile_x, tile_y), tile_data in sorted(tiles_dict.items()):
        vertices = np.array(tile_data["vertices"])
        faces = tile_data["faces"]
        materials_per_face = tile_data.get("materials", [])
        tile_normals = tile_data.get("normals")
        original_face_indices = tile_data.get("original_face_indices", [])
        vertex_mapping = tile_data.get("vertex_mapping", {})  # orig_vertex_idx → tile_local_idx

        if len(faces) == 0:
            continue

        # Extrahiere UVs aus mesh_obj.face_uvs falls vorhanden
        # Erstelle per-Vertex UV-Array basierend auf den Faces
        explicit_uvs = None

        # Hole vertex_mapping
        vertex_mapping = tile_data.get("vertex_mapping", {})  # orig_vertex_idx → tile_local_idx

        if mesh_obj is not None and hasattr(mesh_obj, "face_uvs") and len(mesh_obj.face_uvs) > 0:
            num_vertices = len(vertices)
            uv_array = np.zeros((num_vertices, 2), dtype=np.float32)
            uv_assigned = np.zeros(num_vertices, dtype=bool)

            # Strategie: mesh_obj.face_uvs hat Road-Face-UVs mit Keys 0, 1, 2, ...
            # (der 0-ten Road-Face Indices, nicht Mesh-Indices!)
            # Aber wie mappen wir zu original_face_indices?
            #
            # LÖSUNG: Die original_face_indices enthalten SOWOHL Terrain-Faces ALS AUCH Road-Faces
            # Die ERSTEN N Terrain-Faces, dann die Road-Faces!
            #
            # Die road_face_indices speichert die FINALEN Mesh-Indices der ungeclippten Roads
            # Das ist NICHT hilfreich.
            #
            # Neu-Strategie: Nutze tile_data["road_uvs"] falls vorhanden!

            road_uvs_list = tile_data.get("road_uvs", [])

            if road_uvs_list:
                # road_uvs_list ist parallel zu road_face_indices
                for face_local_idx, orig_face_idx in enumerate(tile_data.get("road_face_indices", [])):
                    if face_local_idx < len(road_uvs_list):
                        face_uv_dict = road_uvs_list[face_local_idx]

                        # Mappe Original-Vertex-Indizes zu tile-local und speichere UVs
                        for orig_vertex_idx, uv in face_uv_dict.items():
                            if orig_vertex_idx in vertex_mapping:
                                tile_local_idx = vertex_mapping[orig_vertex_idx]
                                uv_array[tile_local_idx] = uv
                                uv_assigned[tile_local_idx] = True

            assigned_count = np.sum(uv_assigned)
            if assigned_count > 0:
                # Road-UVs wurden zugewiesen, ABER: Terrain-Vertices brauchen noch UVs!
                # Fülle fehlende UVs mit Terrain-Fallback
                tile_bounds = tile_data.get("bounds", None)
                if tile_bounds:
                    x_min, x_max, y_min, y_max = tile_bounds
                    for i, vertex in enumerate(vertices):
                        if not uv_assigned[i]:  # Nur für Vertices OHNE Road-UVs
                            x, y, z = vertex
                            u = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
                            v = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
                            uv_array[i] = [u, v]

                explicit_uvs = uv_array

        # Falls KEINE Road-UVs vorhanden: Nutze tile_bounds für ALLE Vertices (reiner Terrain-Fall)
        if explicit_uvs is None:
            tile_bounds = tile_data.get("bounds", None)
            if tile_bounds:
                x_min, x_max, y_min, y_max = tile_bounds
                # Für Terrain: einfache Projektion basierend auf tile_bounds
                num_vertices = len(vertices)
                uv_array = np.zeros((num_vertices, 2), dtype=np.float32)
                for i, vertex in enumerate(vertices):
                    x, y, z = vertex
                    u = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
                    v = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
                    uv_array[i] = [u, v]
                explicit_uvs = uv_array

        # Gruppiere Faces pro Material
        # WICHTIG: "terrain" und "unknown" Materialien werden dem Tile-Material zugewiesen!
        tile_material_name = f"tile_{tile_x}_{tile_y}"
        faces_by_material = {}
        for idx, face in enumerate(faces):
            mat_name = materials_per_face[idx] if idx < len(materials_per_face) else "unknown"
            
            # Stitch-Faces und ungeclippte Terrain-Faces haben material="terrain" oder "unknown"
            # → Weise sie dem Tile-Material zu (z.B. "tile_-1000_-1000")
            if mat_name in ("terrain", "unknown"):
                mat_name = tile_material_name
            
            faces_by_material.setdefault(mat_name, []).append(face)

        # WICHTIG: UV-Koordinaten müssen die Tile-Bounds reflektieren, nicht Vertex-Bounds!
        # Berechne absolute Tile-Bounds in lokalen Koordinaten
        tile_bounds = tile_data.get("bounds", None)
        if tile_bounds:
            # bounds = (x_min, x_max, y_min, y_max)
            x_min, x_max, y_min, y_max = tile_bounds
        else:
            # Fallback: Berechne aus tile_x/tile_y
            x_min = tile_x * tile_size
            x_max = (tile_x + 1) * tile_size
            y_min = tile_y * tile_size
            y_max = (tile_y + 1) * tile_size

        meshes.append(
            {
                "id": f"tile_{tile_x}_{tile_y}",
                "vertices": vertices,
                "faces": faces_by_material,
                "normals": tile_normals,
                "uvs": explicit_uvs,  # Explizite UVs aus mesh_obj.face_uvs (falls vorhanden)
                "uv_offset": (0.0, 0.0),
                "uv_scale": (1.0, 1.0),
                "tile_bounds": (x_min, x_max, y_min, y_max),  # Für korrekte UV-Berechnung mit echten Bounds
            }
        )

    # Berechne minimale Koordinaten für Dateinamen
    if tiles_dict:
        min_tile_x = min(tile_x for tile_x, tile_y in tiles_dict.keys())
        min_tile_y = min(tile_y for tile_x, tile_y in tiles_dict.keys())
        corner_x = min_tile_x * tile_size
        corner_y = min_tile_y * tile_size

        output_dir = os.path.dirname(output_path)
        actual_output_path = os.path.join(output_dir, f"terrain_{corner_x}_{corner_y}.dae")
        actual_dae_filename = f"terrain_{corner_x}_{corner_y}.dae"
    else:
        actual_output_path = output_path
        actual_dae_filename = os.path.basename(output_path)

    # Erstelle Material-Textur-Mapping
    from ..managers import MaterialManager
    from .. import config

    material_manager = MaterialManager(beamng_dir="")

    # Sammle alle verwendeten Material-Namen
    all_material_names = set()
    for mesh_data in meshes:
        faces_dict = mesh_data.get("faces", {})
        if isinstance(faces_dict, dict):
            all_material_names.update(faces_dict.keys())

    # Erstelle Materials für alle Namen
    for mat_name in all_material_names:
        if mat_name.startswith("tile_"):
            # Terrain-Material: Extrahiere Koordinaten aus Name
            try:
                parts = mat_name.split("_")
                corner_x = int(parts[1])
                corner_y = int(parts[2])
                texture_path = config.RELATIVE_DIR_TEXTURES + f"tile_{corner_x}_{corner_y}.dds"
                material_manager.add_terrain_material(corner_x, corner_y, texture_path)
            except (IndexError, ValueError):
                print(f"  ⚠ Konnte Koordinaten nicht aus Material-Name extrahieren: {mat_name}")
        else:
            # Road-Material: Hole Properties aus OSM_MAPPER
            # Versuche verschiedene OSM-Tag-Kombinationen
            road_props = config.OSM_MAPPER.get_road_properties({"surface": mat_name})
            if not road_props or road_props.get("internal_name") != mat_name:
                # Fallback 1: Suche in allen highway_defaults
                found = False
                for highway_type, props in config.OSM_MAPPER.config.get("highway_defaults", {}).items():
                    if props.get("internal_name") == mat_name:
                        road_props = props
                        found = True
                        break

                # Fallback 2: Suche in surface_overrides
                if not found:
                    for surface_type, props in config.OSM_MAPPER.config.get("surface_overrides", {}).items():
                        if props.get("internal_name") == mat_name:
                            road_props = props
                            found = True
                            break

                if not found:
                    print(f"  ⚠ Material {mat_name} nicht in OSM_MAPPER gefunden")
                    continue

            material_manager.add_road_material(mat_name, road_props)

    # Extrahiere Textur-Pfade aus Materials
    material_textures = {}
    for mat_name, mat_data in material_manager.materials.items():
        stages = mat_data.get("Stages", [])
        if stages and "baseColorMap" in stages[0]:
            material_textures[mat_name] = stages[0]["baseColorMap"]

    # Export mit DAEExporter
    exporter = DAEExporter()
    exporter.export_multi_mesh(
        output_path=actual_output_path, meshes=meshes, with_uv=True, material_textures=material_textures
    )

    total_vertices = sum(len(m["vertices"]) for m in meshes)
    total_faces = sum(sum(len(f) for f in m["faces"].values()) for m in meshes)
    print(f"  [OK] DAE exportiert: {actual_dae_filename}")
    print(f"    -> {len(meshes)} Tiles, {total_vertices} Vertices, {total_faces} Faces")
    print(f"    -> {len(material_textures)} Materialien mit Texturen")

    return actual_dae_filename


def create_terrain_materials_json(tiles_dict, level_name="World_to_BeamNG", tile_size=400):
    """
    Erstellt materials.json Einträge für Terrain-Tiles.

    Args:
        tiles_dict: Dictionary von tile_slicer.slice_mesh_into_tiles()
        level_name: Name des BeamNG Levels (für Texturpfade)
        tile_size: Tile-Größe in Metern (zur Koordinaten-Umrechnung)

    Returns:
        Dict mit Material-Definitionen
    """
    from ..managers import MaterialManager
    from .. import config

    manager = MaterialManager(beamng_dir="")  # Nur für dict-Erstellung

    for tile_x, tile_y in sorted(tiles_dict.keys()):
        if len(tiles_dict[(tile_x, tile_y)]["faces"]) == 0:
            continue

        # Berechne Welt-Koordinaten der Tile-Ecke
        corner_x = tile_x * tile_size
        corner_y = tile_y * tile_size

        # Texturpfad
        texture_path = config.RELATIVE_DIR_TEXTURES + f"tile_{corner_x}_{corner_y}.dds"

        # Füge Material über Manager hinzu
        manager.add_terrain_material(corner_x, corner_y, texture_path)

    return manager.materials


def create_terrain_items_json(dae_filename):
    """
    Erstellt items.json Eintrag für terrain.dae.

    Args:
        dae_filename: Dateiname der Terrain-DAE (mit Koordinaten-Index, z.B. "terrain_0_0.dae")

    Returns:
        Dict mit TSStatic-Eintrag (Key und __name basierend auf dae_filename)
    """
    from ..managers import ItemManager
    import os

    manager = ItemManager(beamng_dir="")  # Nur für dict-Erstellung
    item_name = os.path.splitext(dae_filename)[0]

    manager.add_terrain(item_name, dae_filename)

    return manager.items[item_name]


def export_terrain_materials_json(tiles_dict, output_dir, level_name="World_to_BeamNG", tile_size=400):
    """
    Exportiert/merged main.materials.json für Terrain.

    Args:
        tiles_dict: Tiles Dictionary
        output_dir: BeamNG Level-Root
        level_name: Level-Name für Pfade
        tile_size: Tile-Größe in Metern (zur Koordinaten-Umrechnung)

    Returns:
        Pfad zur materials.json
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    materials_file = output_path / "main.materials.json"

    materials = create_terrain_materials_json(tiles_dict, level_name, tile_size)

    # Wenn existiert: merge
    if materials_file.exists():
        with open(materials_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(materials)
        materials = existing

    with open(materials_file, "w", encoding="utf-8") as f:
        json.dump(materials, f, indent=2)

    print(f"  [✓] Materials JSON: {materials_file.name} ({len(materials)} Materialien)")
    return str(materials_file)
