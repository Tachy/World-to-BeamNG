"""
DAE (Collada) Exporter für Mesh.

NEUE ARCHITEKTUR:
- Exportiert SEPARATE DAE-Dateien pro Tile (tile_X_Y.dae)
- Jede DAE hat NUR EINE Geometrie
- Verhindert Z-Fighting durch überlappende Geometrien
- Besseres Culling durch separate TSStatics
"""

import numpy as np
from pathlib import Path
import logging
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def export_separate_tile_daes(
    tiles_dict,
    output_dir,
    material_manager,
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
        material_manager: MaterialManager-Instanz (Materials werden hier registriert)
        tile_size: Tile-Größe in Metern
        mesh_obj: Optional: Mesh-Objekt mit face_uvs für UV-Koordinaten

    Returns:
        Liste von exportierten DAE-Dateinamen
    """
    from ..managers import DAEExporter
    from .. import config

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    exported_files = []

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
            logger.info(
    f"      [Filter] Ignoriere Mini-Tile tile_{tile_x * tile_size}_{tile_y * tile_size}: nur {len(faces
)} Faces"
            )
            continue

        # Berechne Welt-Koordinaten
        corner_x = tile_x * tile_size
        corner_y = tile_y * tile_size
        tile_name = f"tile_{corner_x}_{corner_y}"
        dae_filename = f"{tile_name}.dae"
        dae_path = Path(output_dir) / dae_filename

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
                texture_path = str(config.RELATIVE_DIR_TEXTURES / f"{mat_name}.dds")
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
                        logger.info(f"  ⚠ Material {mat_name} nicht in OSM_MAPPER gefunden")
                        continue

                material_manager.add_road_material(mat_name, road_props)

        # Extrahiere Textur-Pfade aus MaterialManager
        material_textures = material_manager.extract_textures()

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
        exporter = DAEExporter(material_manager=material_manager)
        exporter.export_multi_mesh(
            output_path=dae_path,
            meshes=[mesh_data],  # NUR DIESES EINE Tile!
            with_uv=True,
            material_textures=material_textures,
        )

        total_faces = sum(len(f) for f in faces_by_material.values())
        logger.info(
    f"    [OK] {dae_filename}: {len(vertices
)} Vertices, {total_faces} Faces, {len(faces_by_material)} Materialien"
        )

        exported_files.append(dae_filename)

    return exported_files


def create_terrain_materials_json(tiles_dict, material_manager, level_name="World_to_BeamNG", tile_size=400):
    """
    Erstellt materials.json Einträge für Terrain-Tiles.

    REFACTORED: Nutzt jetzt übergebenen MaterialManager statt lokale Instanz.

    Args:
        tiles_dict: Dictionary von tile_slicer.slice_mesh_into_tiles()
        material_manager: MaterialManager-Instanz
        level_name: Name des BeamNG Levels (für Texturpfade)
        tile_size: Tile-Größe in Metern (zur Koordinaten-Umrechnung)

    Returns:
        Dict mit Material-Definitionen
    """
    from .. import config

    # Registriere Materials direkt im übergebenen Manager (KEIN lokaler Manager mehr)

    min_faces_threshold = 10  # Gleicher Filter wie in export_separate_tile_daes

    for tile_x, tile_y in sorted(tiles_dict.keys()):
        tile_data = tiles_dict[(tile_x, tile_y)]
        faces = tile_data.get("faces", [])

        if len(faces) == 0:
            continue

        # === FILTER: Ignoriere Mini-Tiles (Stitch-Artefakte) ===
        # Tiles mit weniger als 10 Faces sind wahrscheinlich Fehler beim Gap-Filling
        if len(faces) < min_faces_threshold:
            continue

        # Berechne Welt-Koordinaten der Tile-Ecke
        corner_x = tile_x * tile_size
        corner_y = tile_y * tile_size

        # Texturpfad
        texture_path = str(config.RELATIVE_DIR_TEXTURES / f"tile_{corner_x}_{corner_y}.dds")

        # Füge Material über Manager hinzu
        material_manager.add_terrain_material(corner_x, corner_y, texture_path)

    return material_manager.materials


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

    logger.info(f"  [✓] Materials JSON: {materials_file.name} ({len(materials)} Materialien)")
    return str(materials_file)
