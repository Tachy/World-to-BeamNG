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
        faces = list(tile_data["faces"])
        materials_per_face = tile_data.get("materials", [])
        tile_normals = tile_data.get("normals")

        if len(faces) == 0:
            continue

        # Nutze UV-Indizes direkt aus tile_data (kein Konversion mehr nötig!)
        face_uv_indices = tile_data.get("uv_indices", {})
        global_uvs = tile_data.get("global_uvs", [])

        # Konvertiere zu numpy
        explicit_uvs = np.array(global_uvs, dtype=np.float32) if global_uvs else None

        # Für Faces ohne UV-Indizes: 1:1 Mapping (Terrain-Fallback)
        for face_idx, face in enumerate(faces):
            if face_idx not in face_uv_indices:
                face_uv_indices[face_idx] = list(face)  # uv_idx == v_idx

        # Füge fallback UVs für Vertices hinzu, die noch keine UVs haben (1:1 Projektion)
        # Das ist wichtig für Terrain-Faces, die keine expliziten UVs bekommen haben
        tile_bounds = tile_data.get("bounds", None)
        if tile_bounds and explicit_uvs is not None:
            x_min, x_max, y_min, y_max = tile_bounds
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0

            # Füge UVs für alle Vertices hinzu, die in face_uv_indices als 1:1 Mapping referenziert werden
            for v_idx in range(len(vertices)):
                if v_idx >= len(explicit_uvs):
                    # Berechne UV aus Vertex-Position
                    x, y, z = vertices[v_idx]
                    u = (x - x_min) / x_range
                    v = (y - y_min) / y_range
                    # Erweitere explicit_uvs
                    explicit_uvs = np.vstack([explicit_uvs, np.array([[u, v]], dtype=np.float32)])

        materials_per_face = tile_data.get("materials", [])

        # Gruppiere Faces pro Material mit separaten UV-Indizes
        # WICHTIG: "terrain" und "unknown" Materialien werden dem Tile-Material zugewiesen!
        # Verwende Weltkoordinaten (tile_x * tile_size), nicht Tile-Indizes!
        tile_coord_x = tile_x * tile_size
        tile_coord_y = tile_y * tile_size
        tile_material_name = f"tile_{tile_coord_x}_{tile_coord_y}"
        faces_by_material = {}
        uv_indices_by_material = {}  # Separate UV-Indizes pro Material

        for idx, face in enumerate(faces):
            mat_name = materials_per_face[idx] if idx < len(materials_per_face) else "unknown"

            # Stitch-Faces und ungeclippte Terrain-Faces haben material="terrain" oder "unknown"
            # → Weise sie dem Tile-Material zu (z.B. "tile_-1000_-1000")
            if mat_name in ("terrain", "unknown"):
                mat_name = tile_material_name

            faces_by_material.setdefault(mat_name, []).append(face)

            # Hole UV-Indizes für dieses Face
            uv_ids = face_uv_indices.get(idx, list(face))  # Fallback: uv_idx == v_idx
            uv_indices_by_material.setdefault(mat_name, []).append(uv_ids)

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
                "id": f"tile_{tile_coord_x}_{tile_coord_y}",
                "vertices": vertices,
                "faces": faces_by_material,
                "uv_indices": uv_indices_by_material,  # NEU: Separate UV-Indizes pro Material
                "normals": tile_normals,
                "uvs": explicit_uvs,  # Global UV-Array
                "uv_offset": (0.0, 0.0),
                "uv_scale": (1.0, 1.0),
                "tile_bounds": (x_min, x_max, y_min, y_max),
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
