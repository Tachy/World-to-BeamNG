"""
DAE (Collada) loader for the mesh viewer.

Parses .dae files and extracts:
- Vertices (from float_array)
- Faces (from triangles)
- Material assignment per face
"""

# Use lxml for faster XML parsing (2-3x faster than xml.etree)
from lxml import etree as ET
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()
import numpy as np
from pathlib import Path, PurePosixPath


def load_dae_tile(filepath):
    """
    Load a DAE file and extract vertices, faces, materials and UV coordinates.

    Returns:
        {
            "vertices": np.array (n, 3),
            "faces": list of [v0, v1, v2],
            "materials": list of material names,
            "materials_per_face": dict {material_name: [face_indices]},
            "tiles": dict {tile_name: {"faces": [...], "uvs": [...]}},
        }
    """

    # Load and parse the DAE with lxml (huge_tree allows large text nodes, e.g. horizon)
    parser = ET.XMLParser(huge_tree=True, resolve_entities=False)
    tree = ET.parse(str(filepath), parser)
    root = tree.getroot()

    # Namespace handling (Collada uses namespaces)
    ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}
    ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

    # ===== Extract all geometries (tiles) =====
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

        # Extract vertices for this tile
        tile_vertices = []
        tile_uvs = []

        for source in mesh.findall("collada:source", ns):
            source_id = source.get("id", "")
            float_array = source.find("collada:float_array", ns)

            if float_array is None or float_array.text is None:
                continue

            # Faster parsing with np.fromstring()
            data_str = float_array.text.strip()
            data = np.fromstring(data_str, sep=" ", dtype=np.float32)

            technique = source.find("collada:technique_common", ns)
            if technique is None:
                logger.error(f"[!] technique_common not found in source {source_id}")
                continue

            accessor = technique.find("collada:accessor", ns)
            if accessor is None:
                logger.error(f"[!] accessor not found in source {source_id}")
                continue

            stride = int(accessor.get("stride", "3"))

            # Vertices (stride 3, params: X, Y, Z)
            if stride == 3 and "vertices" in source_id and len(data) % 3 == 0:
                tile_vertices = data.reshape(-1, 3)

            # UV coordinates (stride 2, params: S, T)
            elif stride == 2 and "uv" in source_id and len(data) % 2 == 0:
                tile_uvs = data.reshape(-1, 2)

        # Extract faces for this tile
        # IMPORTANT: DAE has indexed UVs - one vertex can have several different UVs!
        # Solution: duplicate vertices for each unique (vertex_idx, uv_idx) pair

        tile_faces = []
        tile_vertices_expanded = []  # Expanded vertices (duplicated)
        tile_uvs_expanded = []  # Corresponding UVs
        vertex_uv_to_new_idx = {}  # (vertex_idx, uv_idx) → new_vertex_idx

        for triangles in mesh.findall("collada:triangles", ns):
            material_name = triangles.get("material", "unknown")

            # Find p (indices)
            p = triangles.find("collada:p", ns)
            if p is None:
                logger.error(f"[!] p (indices) not found in triangles for tile {tile_name}")
                continue

            if p.text is None:
                logger.error(f"[!] p.text is empty in triangles for tile {tile_name}")
                continue

            # Faster parsing with np.fromstring()
            indices = np.fromstring(p.text.strip(), sep=" ", dtype=np.int32)

            # Find input offsets dynamically
            inputs = triangles.findall("collada:input", ns)
            input_offsets = {}
            for inp in inputs:
                semantic = inp.get("semantic")
                offset = int(inp.get("offset", "0"))
                input_offsets[semantic] = offset

            # Stride = number of inputs
            stride = len(inputs)

            # IMPORTANT: UVs must be read with the correct UV indices from <p>!
            has_uvs = "TEXCOORD" in input_offsets

            if stride > 1:
                # Indices format: [offset0_data, offset1_data, offset2_data, ...] * vertices
                # e.g. with stride=3: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2 ...

                vertex_offset_in_stride = input_offsets.get("VERTEX", 0)
                uv_offset_in_stride = input_offsets.get("TEXCOORD", 2)

                for i in range(0, len(indices), 3 * stride):  # stride indices per vertex * 3 vertices
                    if i + (3 * stride - 1) < len(indices):
                        face = []

                        # For each of the 3 vertices in the face
                        for v_local in range(3):
                            v_idx = indices[i + v_local * stride + vertex_offset_in_stride]

                            # Determine UV index (if present)
                            if has_uvs and len(tile_uvs) > 0:
                                uv_idx = indices[i + v_local * stride + uv_offset_in_stride]
                                pair_key = (v_idx, uv_idx)

                                # Check whether this combination already exists
                                if pair_key not in vertex_uv_to_new_idx:
                                    # New expanded vertex
                                    new_idx = len(tile_vertices_expanded)
                                    vertex_uv_to_new_idx[pair_key] = new_idx

                                    # Add vertex and UV
                                    if v_idx < len(tile_vertices) and uv_idx < len(tile_uvs):
                                        tile_vertices_expanded.append(tile_vertices[v_idx])
                                        tile_uvs_expanded.append(tile_uvs[uv_idx])
                                    else:
                                        logger.error(
                                            f"[!] Index out of range: v_idx={v_idx}/{len(tile_vertices
)}, uv_idx={uv_idx}/{len(tile_uvs)}"
                                        )
                                        tile_vertices_expanded.append([0, 0, 0])
                                        tile_uvs_expanded.append([0, 0])

                                face.append(vertex_uv_to_new_idx[pair_key])
                            else:
                                # No UV mapping - use the original vertex index
                                if v_idx not in vertex_uv_to_new_idx:
                                    new_idx = len(tile_vertices_expanded)
                                    vertex_uv_to_new_idx[v_idx] = new_idx
                                    if v_idx < len(tile_vertices):
                                        tile_vertices_expanded.append(tile_vertices[v_idx])
                                    else:
                                        tile_vertices_expanded.append([0, 0, 0])

                                face.append(vertex_uv_to_new_idx[v_idx])

                        # Global indices (with new offset)
                        face_global = [idx + vertex_offset for idx in face]
                        tile_faces.append(face_global)
                        all_materials.append(material_name)
            else:
                # stride==1: standard format: v0 v1 v2 (vertices only, no UVs/normals)
                for i in range(0, len(indices), 3):
                    if i + 2 < len(indices):
                        face = []
                        for v_local in range(3):
                            v_idx = indices[i + v_local]

                            if v_idx not in vertex_uv_to_new_idx:
                                new_idx = len(tile_vertices_expanded)
                                vertex_uv_to_new_idx[v_idx] = new_idx
                                if v_idx < len(tile_vertices):
                                    tile_vertices_expanded.append(tile_vertices[v_idx])
                                else:
                                    tile_vertices_expanded.append([0, 0, 0])

                            face.append(vertex_uv_to_new_idx[v_idx])

                        face_global = [idx + vertex_offset for idx in face]
                        tile_faces.append(face_global)
                        all_materials.append(material_name)

        # Store tile info with expanded vertices
        if len(tile_vertices_expanded) > 0:
            # Convert to NumPy arrays
            final_vertices = np.array(tile_vertices_expanded, dtype=np.float32)
            final_uvs = np.array(tile_uvs_expanded, dtype=np.float32) if len(tile_uvs_expanded) > 0 else np.array([])

            # Create local faces (indices relative to this tile, 0-based)
            tile_faces_local = []
            for face_global in tile_faces:
                face_local = [idx - vertex_offset for idx in face_global]
                tile_faces_local.append(face_local)

            tiles_info[tile_name] = {
                "faces": tile_faces,  # Global indices
                "faces_local": tile_faces_local,  # Local indices (for the viewer)
                "vertices": final_vertices,  # Expanded vertices
                "uvs": final_uvs,  # Corresponding UVs (1:1 to vertices)
            }

            all_vertices.append(final_vertices)
            all_faces.extend(tile_faces)
            vertex_offset += len(final_vertices)

    # ===== Merge all vertices =====
    merged_vertices = np.vstack(all_vertices) if all_vertices else np.array([])

    # WORKAROUND: filter out degenerate faces (duplicates)
    # Bug in the DAE export currently makes all faces degenerate!
    valid_faces = []
    valid_materials = []
    degenerate_count = 0

    for face, mat in zip(all_faces, all_materials):
        # Check whether all 3 vertex indices are different
        if len(set(face)) == 3:
            valid_faces.append(face)
            valid_materials.append(mat)
        else:
            degenerate_count += 1

    if degenerate_count > 0:
        logger.error(
            f"  [!] WARNING: filtered {degenerate_count} degenerate faces ({100*degenerate_count/len(all_faces
):.1f}%)"
        )
        logger.info(f"      Remaining valid faces: {len(valid_faces)}")

    all_faces = valid_faces
    all_materials = valid_materials

    # Create materials_per_face dict
    materials_per_face = {}
    for idx, mat in enumerate(all_materials):
        if mat not in materials_per_face:
            materials_per_face[mat] = []
        materials_per_face[mat].append(idx)

    return {
        "vertices": merged_vertices,
        "faces": all_faces,
        "materials": all_materials,  # Per face
        "materials_per_face": materials_per_face,  # Indices per material
        "tiles": tiles_info,  # Per-tile info with UVs
        "filepath": str(filepath),
    }


def load_all_dae_tiles(tiles_dir="tiles_dae"):
    """
    Load all DAE tiles from a directory.

    Returns:
        {
            "tile_X_Y": {...dae_data...},
            ...
        }
    """
    tiles_data = {}
    tiles_path = Path(tiles_dir)

    if not tiles_path.exists():
        logger.error(f"Directory not found: {tiles_dir}")
        return tiles_data

    for dae_file in sorted(tiles_path.glob("tile_*.dae")):
        try:
            tile_id = dae_file.stem  # e.g. "tile_0_0"
            data = load_dae_tile(str(dae_file))

            if len(data["vertices"]) > 0:
                tiles_data[tile_id] = data
                logger.info(f"  Tile {tile_id}: {len(data['vertices'])} vertices, {len(data['faces'])} faces")

        except Exception as e:
            logger.error(f"  Error loading {dae_file}: {e}")

    return tiles_data


def merge_dae_tiles(tiles_data, tile_ids=None):
    """
    Merge several DAE tiles into one large mesh.

    Args:
        tiles_data: Output of load_all_dae_tiles()
        tile_ids: List of tile IDs to merge, or None for all

    Returns:
        {
            "vertices": np.array (n, 3),
            "faces": list of [v0, v1, v2],
            "materials": list per face,
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

        # Add vertices
        all_vertices.append(vertices)

        # Add faces with offset
        for face in faces:
            all_faces.append([v + vertex_offset for v in face])

        # Add materials
        all_materials.extend(materials)

        vertex_offset += len(vertices)

    return {
        "vertices": np.vstack(all_vertices) if all_vertices else np.array([]),
        "faces": all_faces,
        "materials": all_materials,
        "tile_origins": tile_origins,
    }


def load_all_dae_files(beamng_dir, items_json_path, resolve_path_func=None):
    """
    Load all DAE files from items.level.json.

    Searches for shapeName entries in items.level.json and loads the referenced DAE files.

    Args:
        beamng_dir: BeamNG directory (config.BEAMNG_DIR)
        items_json_path: Full path to items.level.json
        resolve_path_func: Function for resolving BeamNG paths
                         (e.g. _resolve_beamng_path from dae_viewer.py)

    Returns:
        Tuple (dae_files, tile_data):
        - dae_files: List of (item_name, dae_path) tuples
        - tile_data: List of (item_name, data) tuples (loaded with load_dae_tile)
    """
    import json

    dae_files = []
    tile_data = []

    # Load items.level.json
    if not items_json_path.exists():
        logger.error(f"  [!] items.level.json not found: {items_json_path}")
        return dae_files, tile_data

    try:
        # items.level.json is JSONL (JSON Lines) - each line is a separate JSON object
        items = {}
        with open(items_json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item_data = json.loads(line)
                    # Extract item name (usually "name" or "id")
                    item_name = item_data.get("name") or item_data.get("id") or str(len(items))
                    items[item_name] = item_data
                except json.JSONDecodeError as e:
                    logger.error(f"  [!] Error parsing line: {e}")
                    continue

        logger.info(f"  [DAE Loader] {len(items)} items found")

    except Exception as e:
        logger.error(f"  [!] Error loading items.level.json: {e}")
        return dae_files, tile_data

    # Extract DAE paths from items
    for item_name, item_data in items.items():
        if not isinstance(item_data, dict):
            continue

        shape_name = item_data.get("shapeName")
        if not shape_name or not shape_name.endswith(".dae"):
            continue

        # Resolve path
        if resolve_path_func:
            dae_path = resolve_path_func(shape_name)
        else:
            # Fallback: simple path resolution
            dae_path = beamng_dir / PurePosixPath(shape_name)

        # Check whether the file exists
        if dae_path and Path(dae_path).exists():  # dae_path can be str or Path. Path(dae_path) makes it always Path
            dae_files.append((item_name, dae_path))
            logger.info(f"  [DAE Loader] ✓ {item_name}")
        else:
            logger.error(f"  [!] DAE not found: {shape_name}")

    logger.info(f"  [DAE Loader] {len(dae_files)} DAE files found")

    # Load all DAE files
    for item_name, dae_path in dae_files:
        try:
            data = load_dae_tile(dae_path)
            if data:
                tile_data.append((item_name, data))
                logger.info(f"  [DAE Loader] ✓ Loaded: {item_name}")
            else:
                logger.error(f"  [!] No data in {item_name}")
        except Exception as e:
            logger.error(f"  [!] Error loading {item_name}: {e}")

    logger.info(f"  [DAE Loader] {len(tile_data)} DAE files loaded successfully")
    return dae_files, tile_data


def load_forest_data(beamng_dir):
    """
    Load forest.forest4.json centrally.

    Args:
        beamng_dir: BeamNG directory (config.BEAMNG_DIR)

    Returns:
        Dict with forest data, or None on error
    """
    import json

    forest_json_path = Path(beamng_dir) / "forest" / "forest.forest4.json"

    if not forest_json_path.exists():
        logger.error(f"  [!] forest.forest4.json not found: {forest_json_path}")
        return None

    try:
        trees = []
        with open(forest_json_path, "r", encoding="utf-8") as f:
            # Read JSONL format: each line is a separate JSON object
            for line in f:
                line = line.strip()
                if line:  # Ignore empty lines
                    trees.append(json.loads(line))

        forest_data = {"trees": trees}
        logger.info(f"  [DAE Loader] ✓ forest.forest4.json (JSONL) loaded: {len(trees)} instances")
        return forest_data
    except Exception as e:
        logger.error(f"  [!] Error loading forest.forest4.json: {e}")
        return None


def load_all_viewer_data(beamng_dir, items_json_path, resolve_path_func=None):
    """
    Central function: loads ALL viewer data once.

    This function is called TWICE:
    1. Initial load in __init__
    2. Reload (L key) in reload_dae_file()

    Args:
        beamng_dir: BeamNG directory (config.BEAMNG_DIR)
        items_json_path: Full path to items.level.json
        resolve_path_func: Function for resolving BeamNG paths

    Returns:
        Dict with all loaded data:
        {
            "dae_files": List of (item_name, dae_path),
            "tile_data": List of (item_name, data),
            "forest_data": Dict with the forest.forest4.json content, or None,
            "status": "success" | "partial" | "error"
        }
    """
    result = {"dae_files": [], "tile_data": [], "forest_data": None, "status": "success"}

    # Load DAE files
    logger.info("[Loader] Loading all viewer data...")
    dae_files, tile_data = load_all_dae_files(beamng_dir, items_json_path, resolve_path_func)
    result["dae_files"] = dae_files
    result["tile_data"] = tile_data

    if not tile_data:
        result["status"] = "partial"

    # Load forest data
    forest_data = load_forest_data(beamng_dir)
    result["forest_data"] = forest_data

    if not forest_data:
        result["status"] = "partial"

    logger.info(f"[Loader] ✓ All data loaded")
    return result
