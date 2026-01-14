"""
DAE (Collada) Loader für Mesh-Viewer.

Parsed .dae Dateien und extrahiert:
- Vertices (aus float_array)
- Faces (aus triangles)
- Material-Zuordnung pro Face
"""

# Nutze lxml für schnelleres XML-Parsing (2-3x schneller als xml.etree)
from lxml import etree as ET
import numpy as np
from pathlib import Path


def load_dae_tile(filepath):
    """
    Lade eine DAE-Datei und extrahiere Vertices, Faces, Materialien und UV-Koordinaten.

    Returns:
        {
            "vertices": np.array (n, 3),
            "faces": list of [v0, v1, v2],
            "materials": list of material names,
            "materials_per_face": dict {material_name: [face_indices]},
            "tiles": dict {tile_name: {"faces": [...], "uvs": [...]}},
        }
    """

    # Lade und parse DAE mit lxml (huge_tree erlaubt große Textknoten, z.B. Horizon)
    parser = ET.XMLParser(huge_tree=True, resolve_entities=False)
    tree = ET.parse(str(filepath), parser)
    root = tree.getroot()

    # Namespace handling (Collada verwendet Namespaces)
    ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}
    ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

    # ===== Extrahiere alle Geometrien (Tiles) =====
    tiles_info = {}  # {tile_name: {"faces": [...], "uvs": [...]}}
    all_vertices = []
    all_faces = []
    all_materials = []
    vertex_offset = 0

    for geometry in root.findall(".//collada:geometry", ns):
        tile_name = geometry.get("name", "unknown")
        mesh = geometry.find("collada:mesh", ns)

        if mesh is None:
            continue

        # Extrahiere Vertices für dieses Tile
        tile_vertices = []
        tile_uvs = []

        for source in mesh.findall("collada:source", ns):
            source_id = source.get("id", "")
            float_array = source.find("collada:float_array", ns)

            if float_array is None:
                continue

            # Schnelleres Parsing mit np.fromstring()
            data_str = float_array.text.strip()
            data = np.fromstring(data_str, sep=" ", dtype=np.float32)

            technique = source.find("collada:technique_common", ns)
            if technique is None:
                print(f"[!] technique_common nicht gefunden in source {source_id}")
                continue

            accessor = technique.find("collada:accessor", ns)
            if accessor is None:
                print(f"[!] accessor nicht gefunden in source {source_id}")
                continue

            stride = int(accessor.get("stride", "3"))

            # Vertices (stride 3, params: X, Y, Z)
            if stride == 3 and "vertices" in source_id and len(data) % 3 == 0:
                tile_vertices = data.reshape(-1, 3)

            # UV-Koordinaten (stride 2, params: S, T)
            elif stride == 2 and "uv" in source_id and len(data) % 2 == 0:
                tile_uvs = data.reshape(-1, 2)

        # Extrahiere Faces für dieses Tile
        tile_faces = []
        tile_faces_local = []  # Mit lokalen Indizes für UV-Mapping

        for triangles in mesh.findall("collada:triangles", ns):
            material_name = triangles.get("material", "unknown")

            # Finde p (indices)
            p = triangles.find("collada:p", ns)
            if p is None:
                print(f"[!] p (indices) nicht gefunden in triangles für tile {tile_name}")
                continue

            if p.text is None:
                print(f"[!] p.text ist leer in triangles für tile {tile_name}")
                continue

            # Schnelleres Parsing mit np.fromstring()
            indices = np.fromstring(p.text.strip(), sep=" ", dtype=np.int32)

            # Prüfe ob UV-Koordinaten UND Normals vorhanden sind
            inputs = triangles.findall("collada:input", ns)
            has_normals = any(inp.get("semantic") == "NORMAL" for inp in inputs)
            has_uvs = any(inp.get("semantic") == "TEXCOORD" for inp in inputs)

            # Stride-Berechnung: VERTEX=1, NORMAL=1, UV=1
            stride = 1  # Minimum: nur VERTEX
            if has_normals:
                stride += 1
            if has_uvs:
                stride += 1

            if stride > 1:
                # Indices format: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2 ... (bei stride=3)
                # oder: v0 n0 v1 n1 v2 n2 ... (bei stride=2, nur Normals)
                # oder: v0 uv0 v1 uv1 v2 uv2 ... (bei stride=2, nur UVs)
                for i in range(0, len(indices), 3 * stride):  # stride indices pro vertex * 3 vertices
                    if i + (3 * stride - 1) < len(indices):
                        # Globale Indizes für merged mesh
                        face_global = [
                            indices[i] + vertex_offset,
                            indices[i + stride] + vertex_offset,
                            indices[i + 2*stride] + vertex_offset,
                        ]
                        # Lokale Indizes für Tile (nur Vertex-Indices, ohne Normals/UVs)
                        face_local = [indices[i], indices[i + stride], indices[i + 2*stride]]

                        tile_faces.append(face_global)
                        tile_faces_local.append(face_local)
                        all_materials.append(material_name)
            else:
                # Standard format: v0 v1 v2
                for i in range(0, len(indices), 3):
                    if i + 2 < len(indices):
                        face_global = [
                            indices[i] + vertex_offset,
                            indices[i + 1] + vertex_offset,
                            indices[i + 2] + vertex_offset,
                        ]
                        face_local = [indices[i], indices[i + 1], indices[i + 2]]

                        tile_faces.append(face_global)
                        tile_faces_local.append(face_local)
                        all_materials.append(material_name)

        # Speichere Tile-Info
        if len(tile_vertices) > 0:
            tiles_info[tile_name] = {
                "faces": tile_faces,  # Globale Indizes
                "faces_local": tile_faces_local,  # Lokale Indizes für UV
                "vertices": tile_vertices,  # NumPy Array (nicht tolist()!)
                "uvs": tile_uvs if len(tile_uvs) > 0 else np.array([]),  # NumPy Array
            }

            all_vertices.append(tile_vertices)
            all_faces.extend(tile_faces)
            vertex_offset += len(tile_vertices)

    # ===== Merge alle Vertices =====
    merged_vertices = np.vstack(all_vertices) if all_vertices else np.array([])

    # WORKAROUND: Filtere degenerierte Faces (Duplikate) heraus
    # Bug im DAE-Export sorgt dafür, dass alle Faces aktuell degeneriert sind!
    valid_faces = []
    valid_materials = []
    degenerate_count = 0
    
    for face, mat in zip(all_faces, all_materials):
        # Prüfe ob alle 3 Vertex-Indizes unterschiedlich sind
        if len(set(face)) == 3:
            valid_faces.append(face)
            valid_materials.append(mat)
        else:
            degenerate_count += 1
    
    if degenerate_count > 0:
        print(f"  [!] WARNUNG: {degenerate_count} degenerierte Faces gefiltert ({100*degenerate_count/len(all_faces):.1f}%)")
        print(f"      Verbleibende valide Faces: {len(valid_faces)}")
    
    all_faces = valid_faces
    all_materials = valid_materials

    # Erstelle materials_per_face dict
    materials_per_face = {}
    for idx, mat in enumerate(all_materials):
        if mat not in materials_per_face:
            materials_per_face[mat] = []
        materials_per_face[mat].append(idx)

    return {
        "vertices": merged_vertices,
        "faces": all_faces,
        "materials": all_materials,  # Pro Face
        "materials_per_face": materials_per_face,  # Indizes pro Material
        "tiles": tiles_info,  # Per-Tile Info mit UVs
        "filepath": str(filepath),
    }


def load_all_dae_tiles(tiles_dir="tiles_dae"):
    """
    Lade alle DAE-Tiles aus einem Verzeichnis.

    Returns:
        {
            "tile_X_Y": {...dae_data...},
            ...
        }
    """
    tiles_data = {}
    tiles_path = Path(tiles_dir)

    if not tiles_path.exists():
        print(f"Verzeichnis nicht gefunden: {tiles_dir}")
        return tiles_data

    for dae_file in sorted(tiles_path.glob("tile_*.dae")):
        try:
            tile_id = dae_file.stem  # z.B. "tile_0_0"
            data = load_dae_tile(str(dae_file))

            if len(data["vertices"]) > 0:
                tiles_data[tile_id] = data
                print(f"  Tile {tile_id}: {len(data['vertices'])} vertices, {len(data['faces'])} faces")

        except Exception as e:
            print(f"  Fehler beim Laden von {dae_file}: {e}")

    return tiles_data


def merge_dae_tiles(tiles_data, tile_ids=None):
    """
    Merge mehrere DAE-Tiles in ein großes Mesh.

    Args:
        tiles_data: Output von load_all_dae_tiles()
        tile_ids: Liste von Tile-IDs zum Mergen, oder None für alle

    Returns:
        {
            "vertices": np.array (n, 3),
            "faces": list of [v0, v1, v2],
            "materials": list per Face,
            "tile_origins": dict {tile_id: offset_index},
        }
    """
    if tile_ids is None:
        tile_ids = sorted(tiles_data.keys())

    all_vertices = []
    all_faces = []
    all_materials = []
    tile_origins = {}
    vertex_offset = 0

    for tile_id in tile_ids:
        if tile_id not in tiles_data:
            continue

        tile = tiles_data[tile_id]
        vertices = tile["vertices"]
        faces = tile["faces"]
        materials = tile["materials"]

        tile_origins[tile_id] = vertex_offset

        # Vertices hinzufügen
        all_vertices.append(vertices)

        # Faces mit Offset hinzufügen
        for face in faces:
            all_faces.append([v + vertex_offset for v in face])

        # Materials hinzufügen
        all_materials.extend(materials)

        vertex_offset += len(vertices)

    return {
        "vertices": np.vstack(all_vertices) if all_vertices else np.array([]),
        "faces": all_faces,
        "materials": all_materials,
        "tile_origins": tile_origins,
    }
