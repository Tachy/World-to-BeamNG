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
):
    """
    Exportiert alle Tiles als EINE .dae (Collada 1.4.1) mit mehreren Geometrien.

    Args:
        tiles_dict: Dictionary von tile_slicer.slice_mesh_into_tiles()
                    Format: {(tile_x, tile_y): {"vertices": [...], "faces": [...], "materials": [...]}}
        output_path: Ziel-Dateipfad (z.B. "terrain.dae")
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

    # Sammle alle verwendeten Materialien
    all_materials = set()
    for tile_data in tiles_dict.values():
        all_materials.update(tile_data["materials"])
    unique_materials = sorted(all_materials)
    material_map = {mat: f"material_{i}" for i, mat in enumerate(unique_materials)}

    # Library: Images (Textur-Referenzen für Tiles)
    lib_images = SubElement(collada, "library_images")
    for tile_x, tile_y in sorted(tiles_dict.keys()):
        image = SubElement(lib_images, "image")
        image.set("id", f"tile_{tile_x}_{tile_y}_image")
        image.set("name", f"tile_{tile_x}_{tile_y}")
        init_from = SubElement(image, "init_from")
        init_from.text = f"textures/tile_{tile_x}_{tile_y}.jpg"

    # Library: Materials
    lib_materials = SubElement(collada, "library_materials")
    for mat_name in unique_materials:
        mat = SubElement(lib_materials, "material")
        mat.set("id", material_map[mat_name])
        mat.set("name", mat_name)
        effect = SubElement(mat, "instance_effect")
        effect.set("url", f"#effect_{material_map[mat_name]}")

    # Library: Effects (mit Texture Support)
    lib_effects = SubElement(collada, "library_effects")

    # Erstelle Effects für jedes Tile (da jedes Tile seine eigene Textur hat)
    for tile_x, tile_y in sorted(tiles_dict.keys()):
        if len(tiles_dict[(tile_x, tile_y)]["faces"]) == 0:
            continue

        effect_id = f"tile_{tile_x}_{tile_y}_effect"
        effect = SubElement(lib_effects, "effect")
        effect.set("id", effect_id)

        profile_common = SubElement(effect, "profile_COMMON")

        # NewParam für Texture Surface
        newparam_surface = SubElement(profile_common, "newparam")
        newparam_surface.set("sid", f"tile_{tile_x}_{tile_y}_surface")
        surface = SubElement(newparam_surface, "surface")
        surface.set("type", "2D")
        init_from = SubElement(surface, "init_from")
        init_from.text = f"tile_{tile_x}_{tile_y}_image"

        # NewParam für Texture Sampler
        newparam_sampler = SubElement(profile_common, "newparam")
        newparam_sampler.set("sid", f"tile_{tile_x}_{tile_y}_sampler")
        sampler2d = SubElement(newparam_sampler, "sampler2D")
        source = SubElement(sampler2d, "source")
        source.text = f"tile_{tile_x}_{tile_y}_surface"

        technique = SubElement(profile_common, "technique")
        technique.set("sid", "common")

        phong = SubElement(technique, "phong")
        diffuse = SubElement(phong, "diffuse")
        texture = SubElement(diffuse, "texture")
        texture.set("texture", f"tile_{tile_x}_{tile_y}_sampler")
        texture.set("texcoord", "UVSET0")

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
            vertex_data = " ".join(
                f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices
            )
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

        # ============ Faces (getrennt nach Material) ============
        # Gruppiere Faces nach ihrem Material
        faces_by_material = {}
        for face_idx, face in enumerate(faces):
            mat = materials_per_face[face_idx]
            if mat not in faces_by_material:
                faces_by_material[mat] = []
            faces_by_material[mat].append(face)

        # Erstelle separate triangles Elemente für jedes Material
        for mat_name in sorted(faces_by_material.keys()):
            mat_faces = faces_by_material[mat_name]
            mat_id = material_map[mat_name]

            triangles = SubElement(mesh, "triangles")
            triangles.set("count", str(len(mat_faces)))
            triangles.set("material", mat_name)  # ← Echten Material-Namen, nicht ID!

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

            # Face-Daten: vertex_idx uv_idx vertex_idx uv_idx ...
            p_elem = SubElement(triangles, "p")
            face_data_parts = []
            for face in mat_faces:
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

        # Bind Materials (alle verwendeten Materialien in diesem Tile)
        bind_material = SubElement(instance_geometry, "bind_material")
        technique_common = SubElement(bind_material, "technique_common")

        # Binde alle Materialien die in diesem Tile verwendet werden
        for mat_name in sorted(unique_materials):
            mat_id = material_map[mat_name]
            instance_material = SubElement(technique_common, "instance_material")
            instance_material.set(
                "symbol", mat_name
            )  # ← Echten Material-Namen als Symbol
            instance_material.set("target", f"#{mat_id}")

            # Bind Vertex Input für UV
            bind_vertex_input = SubElement(instance_material, "bind_vertex_input")
            bind_vertex_input.set("semantic", "UVSET0")
            bind_vertex_input.set("input_semantic", "TEXCOORD")
            bind_vertex_input.set("input_set", "0")

    # ============ Scene (Root) ============
    scene = SubElement(collada, "scene")
    SubElement(scene, "instance_visual_scene").set("url", "#default_scene")

    # ============ Schreibe DAE ============
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tree = ElementTree(collada)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    total_vertices = sum(len(t["vertices"]) for t in tiles_dict.values())
    total_faces = sum(len(t["faces"]) for t in tiles_dict.values())

    print(f"  [OK] DAE exportiert: {output_path}")
    print(f"    -> {tile_count} Tiles, {total_vertices} Vertices, {total_faces} Faces")
    print(f"    -> Materialien: {', '.join(unique_materials)}")

    return output_path
