"""
LoD2 building data processing for BeamNG.

Loads 3D building models from CityGML files (LGL Baden-Württemberg),
transforms them into the local coordinate system and exports them
as TSStatic objects for BeamNG.

Format: CityGML 2km x 2km tiles in ZIP archives
Output: .dae files per tile + main.items.json entries
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from lxml import etree
import zipfile
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def load_citygml_from_zip(zip_path: Path) -> List[etree.Element]:
    """
    Extracts CityGML files from a ZIP archive.

    Args:
        zip_path: Path to the ZIP archive

    Returns:
        List of XML ElementTree roots
    """
    buildings = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith((".gml", ".xml")):
                    with zip_ref.open(file_info) as gml_file:
                        tree = etree.parse(gml_file)
                        buildings.append(tree.getroot())
    except Exception as e:
        logger.error(f"[!] Error loading {zip_path}: {e}")

    return buildings


def parse_citygml_buildings(gml_root: etree.Element, local_offset=None) -> List[Dict]:
    """
    Parses CityGML data and extracts building geometry.

    IMPORTANT: This function returns ONLY the RAW UTM coordinates, WITHOUT normalization!
    The normalization happens later in normalize_buildings_full() with the full 3D offset.

    Args:
        gml_root: XML root of the CityGML document
        local_offset: NOT USED - only for backward compatibility

    Returns:
        List of building dicts with:
        - 'id': Building ID
        - 'walls': List of (vertices, faces) for walls (UTM coordinates!)
        - 'roofs': List of (vertices, faces) for roofs (UTM coordinates!)
        - 'bounds': (min_x, min_y, min_z, max_x, max_y, max_z) in UTM
    """
    # Namespaces for CityGML 1.0 (LGL Baden-Württemberg)
    namespaces = {
        "gml": "http://www.opengis.net/gml",
        "bldg": "http://www.opengis.net/citygml/building/1.0",
        "core": "http://www.opengis.net/citygml/1.0",
    }

    buildings = []

    # Find all Building objects
    for city_object in gml_root.findall(".//core:cityObjectMember", namespaces):
        building_elem = city_object.find("bldg:Building", namespaces)
        if building_elem is None:
            continue

        building_id = building_elem.get("{http://www.opengis.net/gml}id", "unknown")

        walls = []
        roofs = []
        all_vertices = []

        # Find all boundedBy elements
        for bounded in building_elem.findall(".//bldg:boundedBy", namespaces):
            # WallSurface
            wall_surface = bounded.find("bldg:WallSurface", namespaces)
            if wall_surface is not None:
                wall_geom = _extract_surface_geometry(wall_surface, namespaces, local_offset)
                if wall_geom:
                    walls.extend(wall_geom)
                    for verts, _ in wall_geom:
                        all_vertices.append(verts)

            # RoofSurface
            roof_surface = bounded.find("bldg:RoofSurface", namespaces)
            if roof_surface is not None:
                roof_geom = _extract_surface_geometry(roof_surface, namespaces, local_offset)
                if roof_geom:
                    roofs.extend(roof_geom)
                    for verts, _ in roof_geom:
                        all_vertices.append(verts)

        # Compute the bounding box
        if all_vertices:
            all_verts_combined = np.vstack(all_vertices)
            bounds = (
                float(np.min(all_verts_combined[:, 0])),
                float(np.min(all_verts_combined[:, 1])),
                float(np.min(all_verts_combined[:, 2])),
                float(np.max(all_verts_combined[:, 0])),
                float(np.max(all_verts_combined[:, 1])),
                float(np.max(all_verts_combined[:, 2])),
            )

            buildings.append({"id": building_id, "walls": walls, "roofs": roofs, "bounds": bounds})

    return buildings


def _extract_surface_geometry(
    surface_elem: etree.Element, namespaces: Dict, local_offset: Tuple[float, float]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts geometry from a WallSurface or RoofSurface.

    IMPORTANT: Returns RAW UTM coordinates, WITHOUT normalization!
    The normalization happens later, centrally, in normalize_buildings_full().

    Args:
        surface_elem: XML element for the surface
        namespaces: XML namespaces
        local_offset: NOT USED - only for compatibility

    Returns:
        List of (vertices, faces) tuples with UTM coordinates
    """
    geometries = []

    # Find all Polygon elements
    for polygon in surface_elem.findall(".//gml:Polygon", namespaces):
        # Exterior ring (main polygon)
        exterior = polygon.find(".//gml:exterior//gml:posList", namespaces)
        if exterior is None or not exterior.text:
            continue

        # Parse coordinates (x y z x y z ...)
        coords_text = exterior.text.strip().split()
        coords = np.array([float(c) for c in coords_text]).reshape(-1, 3)

        if len(coords) < 3:
            continue

        # IMPORTANT: NO normalization here! Return RAW UTM coordinates
        coords_utm = coords.astype(np.float64)

        # Create faces (triangulation for polygons)
        n_verts = len(coords_utm)
        if n_verts == 3:
            # Triangle
            faces = np.array([[0, 1, 2]])
        elif n_verts == 4:
            # Quad -> 2 triangles
            faces = np.array([[0, 1, 2], [0, 2, 3]])
        else:
            # Polygon -> fan triangulation
            faces = []
            for i in range(1, n_verts - 1):
                faces.append([0, i, i + 1])
            faces = np.array(faces)

        geometries.append((coords_utm, faces))

    return geometries


def normalize_buildings_full(
    buildings: List[Dict],
    local_offset: Tuple[float, float, float],
) -> List[Dict]:
    """
    Normalizes building vertices into the local coordinate system (X, Y, Z).

    CENTRAL NORMALIZATION after the import:
    - All vertices are normalized immediately with global_offset (incl. Z)
    - X, Y, Z coordinates are transformed consistently
    - Buildings stand directly at the correct elevation

    Args:
        buildings: List of building dicts
        local_offset: (origin_x, origin_y, z_min) - 3D offset

    Returns:
        List of normalized building dicts
    """
    if not buildings or len(local_offset) < 3:
        return buildings

    ox, oy, oz = local_offset[0], local_offset[1], local_offset[2]
    normalized = []

    for building in buildings:
        # Copy the building structure
        building_norm = {
            "id": building.get("id"),
            "walls": [],
            "roofs": [],
            "bounds": building.get("bounds"),
        }

        # Normalize all wall vertices
        for verts, faces in building.get("walls", []):
            # Verts: (N, 3) array - convert to float for subtraction
            verts_norm = np.asarray(verts, dtype=np.float64).copy()
            verts_norm[:, 0] -= ox
            verts_norm[:, 1] -= oy
            verts_norm[:, 2] -= oz
            building_norm["walls"].append((verts_norm, faces))

        # Normalize all roof vertices
        for verts, faces in building.get("roofs", []):
            # Verts: (N, 3) array - convert to float for subtraction
            verts_norm = np.asarray(verts, dtype=np.float64).copy()
            verts_norm[:, 0] -= ox
            verts_norm[:, 1] -= oy
            verts_norm[:, 2] -= oz
            building_norm["roofs"].append((verts_norm, faces))

        # IMPORTANT: RECOMPUTE bounds from the normalized vertices!
        # (Do not simply subtract the offset, since parse_citygml_buildings has already
        # partially normalized X/Y)
        all_normalized_verts = []
        for verts, _ in building_norm["walls"]:
            all_normalized_verts.append(verts)
        for verts, _ in building_norm["roofs"]:
            all_normalized_verts.append(verts)

        if all_normalized_verts:
            all_verts_combined = np.vstack(all_normalized_verts)
            bounds = (
                float(np.min(all_verts_combined[:, 0])),
                float(np.min(all_verts_combined[:, 1])),
                float(np.min(all_verts_combined[:, 2])),
                float(np.max(all_verts_combined[:, 0])),
                float(np.max(all_verts_combined[:, 1])),
                float(np.max(all_verts_combined[:, 2])),
            )
            building_norm["bounds"] = bounds

        normalized.append(building_norm)

    return normalized


def cache_lod2_buildings(
    lod2_dir: str,
    bbox: Tuple[float, float, float, float],
    local_offset: Tuple[float, float, float],
    cache_dir: str,
    height_hash: str,
) -> str:
    """
    Loads and caches LoD2 building data - with immediate 3D normalization.

    Args:
        lod2_dir: Directory with ZIP files
        bbox: (min_lat, min_lon, max_lat, max_lon) in WGS84
        local_offset: (x_offset, y_offset, z_offset) in local coordinates - 3D!
        cache_dir: Cache directory
        height_hash: Hash for cache validation

    Returns:
        Path to the cache file
    """
    from ..geometry.coordinates import transformer_to_utm

    # Convert the bbox from WGS84 (lat/lon) to the source CRS (the same CRS as everywhere else in the
    # pipeline - previously a different, hard-coded UTM zone variant than elsewhere in the code was used here)
    min_x_utm, min_y_utm = transformer_to_utm.transform(bbox[1], bbox[0])  # lon, lat
    max_x_utm, max_y_utm = transformer_to_utm.transform(bbox[3], bbox[2])
    bbox_utm = (min_x_utm, min_y_utm, max_x_utm, max_y_utm)

    bbox_utm = (min_x_utm, min_y_utm, max_x_utm, max_y_utm)

    # Cache key: use height_hash (tile_hash) directly for uniform consistency
    # All cache files for this tile (OSM, LoD2, elevations, grid) use the same hash
    cache_file = Path(cache_dir) / f"lod2_{height_hash}.pkl"

    if cache_file.exists():
        logger.debug(f"  [i] LoD2 cache found: {cache_file.name}")
        return str(cache_file)

    logger.info(f"[9] Loading LoD2 building data from {lod2_dir}...")

    lod2_path = Path(lod2_dir)
    if not lod2_path.exists():
        logger.error(f"  [!] LoD2 directory not found: {lod2_dir}")
        return None

    # Collect all ZIP files
    zip_files = list(lod2_path.glob("*.zip"))
    if not zip_files:
        logger.error(f"  [!] No ZIP files found in {lod2_dir}")
        return None

    logger.debug(f"  [i] {len(zip_files)} ZIP archives found")

    # CENTRAL PIPELINE: parse → bbox filter (UTM) → normalization (ONCE!)
    all_buildings_raw_utm = []  # RAW UTM buildings before filtering
    total_parsed = 0

    # PHASE 1: Parse all buildings from the ZIPs (RAW UTM coordinates)
    for zip_path in zip_files:
        gml_roots = load_citygml_from_zip(zip_path)
        for gml_root in gml_roots:
            buildings_raw = parse_citygml_buildings(gml_root, None)
            all_buildings_raw_utm.extend(buildings_raw)
            total_parsed += len(buildings_raw)

    logger.debug(f"  [i] {total_parsed} buildings parsed from ZIPs")

    # PHASE 2: bbox filtering in UTM coordinates (BEFORE normalization!)
    buildings_in_bbox_utm = []
    for building in all_buildings_raw_utm:
        bounds = building.get("bounds")
        if not bounds:
            continue
        # bounds = (min_x, min_y, min_z, max_x, max_y, max_z) in UTM
        center_x = (bounds[0] + bounds[3]) / 2
        center_y = (bounds[1] + bounds[4]) / 2
        # Check whether the center lies within the bbox
        if bbox_utm[0] <= center_x <= bbox_utm[2] and bbox_utm[1] <= center_y <= bbox_utm[3]:
            buildings_in_bbox_utm.append(building)

    logger.debug(f"  [i] {len(buildings_in_bbox_utm)} buildings found in the UTM bbox")

    # PHASE 3: CENTRAL normalization ONCE (NEVER again afterwards!)
    all_buildings = normalize_buildings_full(buildings_in_bbox_utm, local_offset)
    logger.info(f"  [✓] {len(all_buildings)} buildings normalized")

    # Write the pickle cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(all_buildings, f)

    logger.info(f"  [✓] {len(all_buildings)} buildings cached")

    return str(cache_file)


def load_buildings_from_cache(cache_file: str) -> List[Dict]:
    """Loads buildings from the cache."""
    if not cache_file or not Path(cache_file).exists():
        return []

    with open(cache_file, "rb") as f:
        return pickle.load(f)


def create_items_json_entry(dae_path: str, tile_x: int, tile_y: int, item_manager, item_name: str = None) -> Dict:
    """
    Creates an items.json entry for a building tile.

    REFACTORED: Now uses the passed ItemManager instead of a local instance.

    Args:
        dae_path: Relative path to the .dae file
        tile_x, tile_y: Tile coordinates (world coordinates of the upper left corner)
        item_manager: ItemManager instance
        item_name: Optional - item name (default: buildings_tile_<x>_<y>)

    Returns:
        Dict for items.json
    """
    # Register the item directly in the passed manager (NO local manager anymore)
    dae_filename = Path(dae_path).name
    item_name = item_name or f"buildings_tile_{tile_x}_{tile_y}"

    item_manager.add_building(item_name, dae_filename, position=(0, 0, 0))

    return item_manager.items[item_name]
