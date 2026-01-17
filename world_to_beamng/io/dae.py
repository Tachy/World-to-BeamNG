"""
DAE (Collada) Exporter für Mesh.

NEUE ARCHITEKTUR:
- Exportiert SEPARATE DAE-Dateien pro Tile (tile_X_Y.dae)
- Jede DAE hat NUR EINE Geometrie
- Verhindert Z-Fighting durch überlappende Geometrien
- Besseres Culling durch separate TSStatics
"""

import numpy as np
import os


def export_separate_tile_daes(
    tiles_dict,
    output_dir,
    tile_size=400,
    mesh_obj=None,
):
    """
    Exportiert jedes Tile als SEPARATE .dae Datei.

    NEUE LÖSUNG: Statt einer DAE mit 16 Geometrien → 16 DAEs mit je 1 Geometrie!
    Verhindert überlappende Rendering an Tile-Grenzen.

    Args:
        tiles_dict: Dictionary von tile_slicer.slice_mesh_into_tiles()
                    Format: {(tile_x, tile_y): {"vertices": [...], "faces": [...], "materials": [...]}}
        output_dir: Ziel-Verzeichnis für DAE-Dateien
        tile_size: Tile-Größe in Metern
        mesh_obj: Optional: Mesh-Objekt mit face_uvs für UV-Koordinaten

    Returns:
        Liste von exportierten DAE-Dateinamen
    """
    from ..managers import DAEExporter, MaterialManager
    from .. import config

    os.makedirs(output_dir, exist_ok=True)

    exported_files = []
    material_manager = MaterialManager(beamng_dir="")

    # Exportiere jedes Tile als separate DAE
    for (tile_x, tile_y), tile_data in sorted(tiles_dict.items()):
        vertices = np.array(tile_data["vertices"])
        faces = list(tile_data["faces"])
        materials_per_face = tile_data.get("materials", [])
        tile_normals = tile_data.get("normals")

        if len(faces) == 0:
            continue

        # === FILTER: Ignoriere Mini-Tiles (Stitch-Artefakte) ===
        # Tiles mit weniger als 10 Faces sind wahrscheinlich Fehler beim Gap-Filling
        min_faces_threshold = 10
        if len(faces) < min_faces_threshold:
            print(f"      [Filter] Ignoriere Mini-Tile tile_{tile_x * tile_size}_{tile_y * tile_size}: nur {len(faces)} Faces")
            continue

        # Berechne Welt-Koordinaten
        corner_x = tile_x * tile_size
        corner_y = tile_y * tile_size
        tile_name = f"tile_{corner_x}_{corner_y}"
        dae_filename = f"{tile_name}.dae"
        dae_path = os.path.join(output_dir, dae_filename)

        # Nutze UV-Indizes direkt aus tile_data
        face_uv_indices = tile_data.get("uv_indices", {})
        global_uvs = tile_data.get("global_uvs", [])

        # Validierung
        if len(face_uv_indices) != len(faces):
            raise ValueError(
                f"Tile ({tile_x}, {tile_y}): {len(faces)} Faces aber nur {len(face_uv_indices)} UV-Index-Sets!"
            )

        explicit_uvs = np.array(global_uvs, dtype=np.float32) if global_uvs else None

        # Gruppiere Faces pro Material
        tile_material_name = tile_name  # "tile_X_Y"
        faces_by_material = {}
        uv_indices_by_material = {}

        for idx, face in enumerate(faces):
            mat_name = materials_per_face[idx] if idx < len(materials_per_face) else "unknown"

            # Terrain/Unknown → Tile-Material
            if mat_name in ("terrain", "unknown"):
                mat_name = tile_material_name

            faces_by_material.setdefault(mat_name, []).append(face)
            uv_ids = face_uv_indices.get(idx, list(face))
            uv_indices_by_material.setdefault(mat_name, []).append(uv_ids)

        # Tile-Bounds
        tile_bounds = tile_data.get("bounds", None)
        if tile_bounds:
            x_min, x_max, y_min, y_max = tile_bounds
        else:
            x_min = corner_x
            x_max = corner_x + tile_size
            y_min = corner_y
            y_max = corner_y + tile_size

        # Erstelle Material-Definitionen
        all_material_names = set(faces_by_material.keys())

        for mat_name in all_material_names:
            if mat_name.startswith("tile_"):
                # Terrain-Material
                texture_path = config.RELATIVE_DIR_TEXTURES + f"{mat_name}.dds"
                material_manager.add_terrain_material(corner_x, corner_y, texture_path)
            else:
                # Road-Material
                road_props = config.OSM_MAPPER.get_road_properties({"surface": mat_name})
                if not road_props or road_props.get("internal_name") != mat_name:
                    # Fallback-Suche
                    found = False
                    for highway_type, props in config.OSM_MAPPER.config.get("highway_defaults", {}).items():
                        if props.get("internal_name") == mat_name:
                            road_props = props
                            found = True
                            break

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

        # Extrahiere Textur-Pfade
        material_textures = {}
        for mat_name, mat_data in material_manager.materials.items():
            stages = mat_data.get("Stages", [])
            if stages and "baseColorMap" in stages[0]:
                material_textures[mat_name] = stages[0]["baseColorMap"]

        # Erstelle Mesh-Daten für DAEExporter
        mesh_data = {
            "id": tile_name,
            "vertices": vertices,
            "faces": faces_by_material,
            "uv_indices": uv_indices_by_material,
            "normals": tile_normals,
            "uvs": explicit_uvs,
            "uv_offset": (0.0, 0.0),
            "uv_scale": (1.0, 1.0),
            "tile_bounds": (x_min, x_max, y_min, y_max),
        }

        # Export mit DAEExporter (SINGLE Mesh!)
        exporter = DAEExporter()
        exporter.export_multi_mesh(
            output_path=dae_path,
            meshes=[mesh_data],  # NUR DIESES EINE Tile!
            with_uv=True,
            material_textures=material_textures,
        )

        total_faces = sum(len(f) for f in faces_by_material.values())
        print(
            f"    [OK] {dae_filename}: {len(vertices)} Vertices, {total_faces} Faces, {len(faces_by_material)} Materialien"
        )

        exported_files.append(dae_filename)

    return exported_files


def export_merged_dae(
    tiles_dict,
    output_path,
    tile_size=400,
    mesh_obj=None,
):
    """
    DEPRECATED: Exportiert alle Tiles als EINE .dae (Collada 1.4.1) mit mehreren Geometrien.

    PROBLEM: Führt zu überlappenden Geometrien → Z-Fighting!
    LÖSUNG: Nutze export_separate_tile_daes() stattdessen!

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

        # Nutze UV-Indizes direkt aus tile_data (KEINE FALLBACKS!)
        face_uv_indices = tile_data.get("uv_indices", {})
        global_uvs = tile_data.get("global_uvs", [])

        # Konvertiere zu numpy
        explicit_uvs = np.array(global_uvs, dtype=np.float32) if global_uvs else None

        # Validierung: Alle Faces MÜSSEN UV-Indizes haben
        if len(face_uv_indices) != len(faces):
            raise ValueError(
                f"Tile ({tile_x}, {tile_y}): {len(faces)} Faces aber nur {len(face_uv_indices)} UV-Index-Sets! "
                f"UV-Berechnung im tile_slicer hat nicht alle Faces abgedeckt."
            )

        materials_per_face = tile_data.get("materials", [])

        # Gruppiere Faces pro Material mit separaten UV-Indizes
        # WICHTIG: "terrain" und "unknown" Materialien werden dem Tile-Material zugewiesen!
        # Verwende Weltkoordinaten (tile_x * tile_size), nicht Tile-Indizes!
        tile_coord_x = tile_x * tile_size
        tile_coord_y = tile_y * tile_size
        tile_material_name = f"tile_{tile_coord_x}_{tile_coord_y}"
        faces_by_material = {}
        uv_indices_by_material = {}  # Separate UV-Indizes pro Material

        # DEBUG: Zähle original Materialien
        original_materials = {}
        for idx in range(len(faces)):
            mat = materials_per_face[idx] if idx < len(materials_per_face) else "unknown"
            original_materials[mat] = original_materials.get(mat, 0) + 1

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

        # DEBUG: Zeige Material-Verteilung für dieses Tile
        if tile_coord_x == -1000 and tile_coord_y == -1000:  # Nur erstes Tile
            print(f"  [DEBUG] Tile {tile_material_name}:")
            print(f"    Original-Materialien: {original_materials}")
            print(f"    Nach Remapping: {dict((k, len(v)) for k, v in faces_by_material.items())}")

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
            # NUR Materialien mit echten Faces hinzufügen (nicht leere Platzhalter)
            for mat_name, face_list in faces_dict.items():
                if len(face_list) > 0:  # ← Nur wenn Faces existieren
                    all_material_names.add(mat_name)

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
