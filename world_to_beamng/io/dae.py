"""
DAE (Collada) Exporter für Mesh (alle Tiles in einer Datei).

Exportiert eine .dae Datei mit:
- Mehrere Geometrien (ein Chunk = eine Geometrie)
- Material-Gruppen (terrain, road, etc.)
- Pro Face eindeutige Material-Zuordnung
"""

import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement, ElementTree
from pathlib import Path


def export_merged_dae(
    tiles_dict,
    output_path,
    tile_size=400,
):
    """
    Exportiert alle Tiles als EINE .dae (Collada 1.4.1) mit mehreren Geometrien.

    Args:
        tiles_dict: Dictionary von tile_slicer.slice_mesh_into_tiles()
                    Format: {(tile_x, tile_y): {"vertices": [...], "faces": [...], "materials": [...]}}
        output_path: Ziel-Dateipfad (wird als Vorlage verwendet, echte Datei nutzt Koordinaten)
        tile_size: Tile-Größe in Metern (zur Koordinaten-Umrechnung)

    Returns:
        Tatsächlicher DAE-Dateiname mit Koordinaten-Index
    """

    # ============ DAE XML Structure ============

    # Root
    collada = Element("COLLADA")
    collada.set("xmlns", "http://www.collada.org/2005/11/COLLADASchema")
    collada.set("version", "1.4.1")

    # Asset (Metadaten)
    asset = SubElement(collada, "asset")
    created = SubElement(asset, "created")
    created.text = "2026-01-02T00:00:00"
    modified = SubElement(asset, "modified")
    modified.text = "2026-01-02T00:00:00"
    unit = SubElement(asset, "unit")
    unit.set("name", "meter")
    unit.set("meter", "1")
    up_axis = SubElement(asset, "up_axis")
    up_axis.text = "Z_UP"

    # Sammle alle verwendeten Materialnamen über alle Tiles
    material_names = set()
    for tile_data in tiles_dict.values():
        mat_list = tile_data.get("materials", []) or []
        material_names.update(mat_list)
    # Fallback Material nur anlegen, wenn wirklich referenziert wird

    # Library: Materials (ein Eintrag pro Materialname)
    lib_materials = SubElement(collada, "library_materials")
    for mat_name in sorted(material_names):
        mat = SubElement(lib_materials, "material")
        mat.set("id", mat_name)
        mat.set("name", mat_name)
        effect = SubElement(mat, "instance_effect")
        effect.set("url", f"#effect_{mat_name}")

    # Library: Effects (Dummy-Effects ohne Texturen - werden über materials.json verlinkt)
    lib_effects = SubElement(collada, "library_effects")
    for mat_name in sorted(material_names):
        effect = SubElement(lib_effects, "effect")
        effect.set("id", f"effect_{mat_name}")

        profile_common = SubElement(effect, "profile_COMMON")
        technique = SubElement(profile_common, "technique")
        technique.set("sid", "common")

        phong = SubElement(technique, "phong")
        diffuse = SubElement(phong, "diffuse")
        color = SubElement(diffuse, "color")
        color.set("sid", "diffuse")
        color.text = "0.8 0.8 0.8 1"  # Grau als Fallback

    # Library: Geometries (eine pro Tile)
    lib_geometries = SubElement(collada, "library_geometries")

    tile_count = 0
    for (tile_x, tile_y), tile_data in sorted(tiles_dict.items()):
        vertices = np.array(tile_data["vertices"])
        faces = tile_data["faces"]
        materials_per_face = tile_data["materials"]

        if len(faces) == 0:
            continue  # Leere Tiles überspringen

        tile_count += 1
        mesh_name = f"tile_{tile_x}_{tile_y}"

        geometry = SubElement(lib_geometries, "geometry")
        geometry.set("id", f"{mesh_name}_geometry")
        geometry.set("name", mesh_name)

        mesh = SubElement(geometry, "mesh")

        # ============ Vertices für dieses Tile ============
        vertices_source_id = f"{mesh_name}_vertices"
        vertices_source = SubElement(mesh, "source")
        vertices_source.set("id", vertices_source_id)

        # Float Array mit Vertices
        vertices_array = SubElement(vertices_source, "float_array")
        vertices_array.set("id", f"{vertices_source_id}_array")
        vertices_array.set("count", str(len(vertices) * 3))

        # OPTIMIERT: NumPy reshape + tobytes statt genexpr
        if isinstance(vertices, np.ndarray):
            vertex_data = " ".join(map(str, vertices.ravel()))
        else:
            vertex_data = " ".join(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices)
        vertices_array.text = vertex_data

        # Accessor
        vertices_accessor = SubElement(vertices_source, "technique_common")
        accessor = SubElement(vertices_accessor, "accessor")
        accessor.set("source", f"#{vertices_source_id}_array")
        accessor.set("count", str(len(vertices)))
        accessor.set("stride", "3")

        param_x = SubElement(accessor, "param")
        param_x.set("name", "X")
        param_x.set("type", "float")

        param_y = SubElement(accessor, "param")
        param_y.set("name", "Y")
        param_y.set("type", "float")

        param_z = SubElement(accessor, "param")
        param_z.set("name", "Z")
        param_z.set("type", "float")

        # ============ UV-Koordinaten für dieses Tile ============
        # Berechne UV-Koordinaten basierend auf Vertex-Position im Tile
        # Jedes Tile ist 500x500m groß, UV sollte 0-1 sein
        tile_size_m = 500.0
        tile_origin_x = tile_x * tile_size_m
        tile_origin_y = tile_y * tile_size_m

        # OPTIMIERT: Vektorisiert mit NumPy
        if isinstance(vertices, np.ndarray):
            uv_coords = np.column_stack(
                [
                    (vertices[:, 0] - tile_origin_x) / tile_size_m,
                    (vertices[:, 1] - tile_origin_y) / tile_size_m,
                ]
            )
        else:
            uv_coords = []
            for v in vertices:
                u = (v[0] - tile_origin_x) / tile_size_m
                v_coord = (v[1] - tile_origin_y) / tile_size_m
                uv_coords.append((u, v_coord))

        uv_source_id = f"{mesh_name}_uv"
        uv_source = SubElement(mesh, "source")
        uv_source.set("id", uv_source_id)

        uv_array = SubElement(uv_source, "float_array")
        uv_array.set("id", f"{uv_source_id}_array")
        uv_array.set("count", str(len(uv_coords) * 2))

        # OPTIMIERT: NumPy reshape statt genexpr
        if isinstance(uv_coords, np.ndarray):
            uv_data = " ".join(map(str, uv_coords.ravel()))
        else:
            uv_data = " ".join(f"{u:.6f} {v:.6f}" for u, v in uv_coords)
        uv_array.text = uv_data

        uv_accessor = SubElement(uv_source, "technique_common")
        uv_acc = SubElement(uv_accessor, "accessor")
        uv_acc.set("source", f"#{uv_source_id}_array")
        uv_acc.set("count", str(len(uv_coords)))
        uv_acc.set("stride", "2")

        param_s = SubElement(uv_acc, "param")
        param_s.set("name", "S")
        param_s.set("type", "float")

        param_t = SubElement(uv_acc, "param")
        param_t.set("name", "T")
        param_t.set("type", "float")

        # Vertices Element
        vertices_elem = SubElement(mesh, "vertices")
        vertices_elem.set("id", f"{mesh_name}_vertices_vertex")

        vert_input = SubElement(vertices_elem, "input")
        vert_input.set("semantic", "POSITION")
        vert_input.set("source", f"#{vertices_source_id}")

        # ============ Faces (pro Material gruppiert) ============
        materials_per_face = tile_data.get("materials", []) or []
        if len(materials_per_face) != len(faces):
            materials_per_face = ["unknown"] * len(faces)

        # Gruppiere Faces pro Material
        faces_by_material = {}
        for idx, face in enumerate(faces):
            mat_name = materials_per_face[idx] if idx < len(materials_per_face) else "unknown"
            faces_by_material.setdefault(mat_name, []).append(face)

        for mat_name, face_list in faces_by_material.items():
            if not face_list:
                continue

            triangles = SubElement(mesh, "triangles")
            triangles.set("count", str(len(face_list)))
            triangles.set("material", mat_name)

            # Input: Vertex Indices
            tri_input_vertex = SubElement(triangles, "input")
            tri_input_vertex.set("semantic", "VERTEX")
            tri_input_vertex.set("source", f"#{mesh_name}_vertices_vertex")
            tri_input_vertex.set("offset", "0")

            # Input: UV Coordinates
            tri_input_uv = SubElement(triangles, "input")
            tri_input_uv.set("semantic", "TEXCOORD")
            tri_input_uv.set("source", f"#{uv_source_id}")
            tri_input_uv.set("offset", "1")
            tri_input_uv.set("set", "0")

            p_elem = SubElement(triangles, "p")
            face_data_parts = []
            for face in face_list:
                for vertex_idx in face:
                    face_data_parts.append(str(vertex_idx))
                    face_data_parts.append(str(vertex_idx))  # UV Index = Vertex Index
            p_elem.text = " ".join(face_data_parts)

    # ============ Library: Visual Scenes ============
    lib_visual_scenes = SubElement(collada, "library_visual_scenes")
    visual_scene = SubElement(lib_visual_scenes, "visual_scene")
    visual_scene.set("id", "default_scene")
    visual_scene.set("name", "default_scene")

    # Erstelle Nodes für alle Tiles
    for (tile_x, tile_y), tile_data in sorted(tiles_dict.items()):
        if len(tile_data["faces"]) == 0:
            continue

        mesh_name = f"tile_{tile_x}_{tile_y}"

        node = SubElement(visual_scene, "node")
        node.set("id", f"{mesh_name}_node")
        node.set("name", mesh_name)

        # Instanz der Geometrie
        instance_geometry = SubElement(node, "instance_geometry")
        instance_geometry.set("url", f"#{mesh_name}_geometry")

        # Bind Material (nur ein Material pro Tile)
        bind_material = SubElement(instance_geometry, "bind_material")
        technique_common = SubElement(bind_material, "technique_common")

        mat_name = f"tile_{tile_x}_{tile_y}"
        instance_material = SubElement(technique_common, "instance_material")
        instance_material.set("symbol", mat_name)
        instance_material.set("target", f"#{mat_name}")

        # Bind Vertex Input für UV
        bind_vertex_input = SubElement(instance_material, "bind_vertex_input")
        bind_vertex_input.set("semantic", "UVSET0")
        bind_vertex_input.set("input_semantic", "TEXCOORD")
        bind_vertex_input.set("input_set", "0")

    # ============ Scene (Root) ============
    scene = SubElement(collada, "scene")
    SubElement(scene, "instance_visual_scene").set("url", "#default_scene")

    # ============ Schreibe DAE ============
    # Berechne minimale Koordinaten für Index
    if tiles_dict:
        min_tile_x = min(tile_x for tile_x, tile_y in tiles_dict.keys())
        min_tile_y = min(tile_y for tile_x, tile_y in tiles_dict.keys())
        corner_x = min_tile_x * tile_size
        corner_y = min_tile_y * tile_size

        # Benenne DAE-Datei um (ersetze "terrain.dae" durch "terrain_<x>_<y>.dae")
        output_dir = os.path.dirname(output_path)
        actual_output_path = os.path.join(output_dir, f"terrain_{corner_x}_{corner_y}.dae")
        actual_dae_filename = f"terrain_{corner_x}_{corner_y}.dae"
    else:
        actual_output_path = output_path
        actual_dae_filename = os.path.basename(output_path)

    os.makedirs(os.path.dirname(actual_output_path) or ".", exist_ok=True)

    tree = ElementTree(collada)
    tree.write(actual_output_path, encoding="utf-8", xml_declaration=True)

    return actual_dae_filename

    total_vertices = sum(len(t["vertices"]) for t in tiles_dict.values())
    total_faces = sum(len(t["faces"]) for t in tiles_dict.values())

    print(f"  [OK] DAE exportiert: {output_path}")
    print(f"    -> {tile_count} Tiles, {total_vertices} Vertices, {total_faces} Faces")

    return output_path


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
    materials = {}

    for tile_x, tile_y in sorted(tiles_dict.keys()):
        if len(tiles_dict[(tile_x, tile_y)]["faces"]) == 0:
            continue

        # Berechne Welt-Koordinaten der Tile-Ecke (wie in aerial.py)
        corner_x = tile_x * tile_size
        corner_y = tile_y * tile_size

        # Material-Name, Key und Texturpfad alle mit Welt-Koordinaten
        mat_name = f"tile_{corner_x}_{corner_y}"
        texture_path = f"/levels/{level_name}/art/shapes/textures/tile_{corner_x}_{corner_y}.jpg"

        materials[mat_name] = {
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "Stages": [{"colorMap": texture_path, "specularPower": 1, "pixelSpecular": True}],
            "groundModelName": "grass",  # Physik für Terrain
        }

    return materials


def create_terrain_items_json(dae_filename):
    """
    Erstellt items.json Eintrag für terrain.dae.

    Args:
        dae_filename: Dateiname der Terrain-DAE (mit Koordinaten-Index, z.B. "terrain_0_0.dae")

    Returns:
        Dict mit TSStatic-Eintrag (Key und __name basierend auf dae_filename)
    """
    # Extrahiere Basename ohne Extension (z.B. "terrain_0_0.dae" -> "terrain_0_0")
    import os

    item_name = os.path.splitext(dae_filename)[0]

    return {
        "__name": item_name,
        "class": "TSStatic",
        "shapeName": dae_filename,
        "position": [0, 0, 0],
        "rotation": [0, 0, 1, 0],
        "scale": [1, 1, 1],
        "collisionType": "Visible Mesh",  # Befahrbar
    }


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
