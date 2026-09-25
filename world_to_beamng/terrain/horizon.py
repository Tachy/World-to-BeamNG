"""
Horizon layer - generates a low-resolution horizon mesh from DGM30 and Sentinel-2.

Pipeline:
1. Load DGM30 data (30 m resolution) from cache/dgm30/*.tif (Copernicus DEM GLO-30, automatically
   downloaded by terrain/dgm30_fetch.py) and clip it to the horizon area
   (±config.HORIZON_HALF_SIZE_M around the core area). Two independent cache layers for this: per
   tile (_cached_geotiff_as_xyz(), area-independent - survives a change of the core area)
   and the finished combined/clipped point cloud (_dgm30_cache_file(), bound to
   tile_hash).
2. Load Sentinel-2 RGB satellite images
3. Generate horizon grid (config.HORIZON_GRID_SPACING)
4. Texture with Sentinel-2 RGB
5. Export as DAE with materials
"""

import hashlib
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import json
from world_to_beamng.logging_config import LoggerConfig

from .. import config

logger = LoggerConfig.get_logger()


def _load_geotiff_as_xyz(geotiff_path):
    """
    Converts a GeoTIFF to XYZ format (coordinates + height values), in UTM (absolute, NOT shifted
    to local) - see _cached_geotiff_as_xyz() for the per-tile cached, area-independent
    variant that the calling _load_local_dgm30() actually uses.

    Samples the 30 m resolution down to a 200 m grid for faster processing.

    Args:
        geotiff_path: path to the GeoTIFF

    Returns:
        Tuple (height_points, height_elevations) in UTM
    """
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError:
        logger.error("  [!] rasterio not installed. Install: pip install rasterio")
        return None, None

    try:
        with rasterio.open(geotiff_path) as src:
            # Check the CRS and reproject if necessary
            src_crs = src.crs

            # Target: the pipeline's resolved source CRS (default EPSG:25832, ETRS89/UTM32N;
            # automatically detected for GeoTIFF elevation data, see geometry.coordinates)
            from ..geometry.coordinates import get_source_crs_epsg

            dst_crs = f"EPSG:{get_source_crs_epsg()}"

            # If the source CRS is not UTM, reproject
            if src_crs and src_crs.to_string() != dst_crs:
                logger.debug(f"  [i] Reprojecting from {src_crs.to_string()} to {dst_crs}")

                from rasterio.warp import calculate_default_transform, reproject, Resampling

                # Compute the new transform and dimensions
                transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds
                )

                # Create a temporary array for the reprojected data
                dem_data = np.empty((height, width), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=dem_data,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

                # Compute the new bounds in UTM
                from rasterio.transform import array_bounds

                bounds = array_bounds(height, width, transform)
                x_min, y_min, x_max, y_max = bounds[0], bounds[1], bounds[2], bounds[3]

            else:
                # Already UTM
                dem_data = src.read(1).astype(np.float32)
                transform = src.transform
                bounds = src.bounds
                x_min, y_min, x_max, y_max = bounds.left, bounds.bottom, bounds.right, bounds.top

            rows, cols = dem_data.shape

            # DEBUG: show bounds
            logger.debug(f"  [DEBUG] UTM Bounds: X=[{x_min:.2f}..{x_max:.2f}], Y=[{y_min:.2f}..{y_max:.2f}]")
            logger.debug(f"  [DEBUG] Width: {x_max - x_min:.2f}m, height: {y_max - y_min:.2f}m")

            # Create 200 m grid
            grid_spacing = 200.0
            x_coords = np.arange(x_min, x_max + grid_spacing * 0.5, grid_spacing)
            y_coords = np.arange(y_min, y_max + grid_spacing * 0.5, grid_spacing)

            logger.debug(f"  [i] Sampling {rows}×{cols} GeoTIFF (30m) onto {len(x_coords)}×{len(y_coords)} grid (200m)")

            height_points = []
            height_elevations = []

            # Sample onto 200 m grid
            for y in y_coords:
                for x in x_coords:
                    # Convert UTM back to pixel coordinates
                    col, row = ~transform * (x, y)
                    col_int, row_int = int(col), int(row)

                    # Check whether within bounds
                    if 0 <= row_int < rows and 0 <= col_int < cols:
                        z = dem_data[row_int, col_int]

                        # Ignore NoData values
                        if not np.isnan(z) and not np.isinf(z):
                            height_points.append([x, y])
                            height_elevations.append(z)

            if not height_points:
                logger.error("  [!] No valid height points in the GeoTIFF")
                return None, None

            height_points = np.array(height_points)
            height_elevations = np.array(height_elevations)

            logger.debug(f"  [OK] {len(height_elevations)} height points (200m grid) loaded from GeoTIFF")

            return height_points, height_elevations

    except Exception as e:
        logger.error(f"  [!] Error loading the GeoTIFF: {e}")
        return None, None


def _dgm30_tile_cache_file(tif_file):
    """
    Cache file of the 200 m grid conversion of ONE single DGM30 tile, in UTM (absolute).

    Independent of the core area (tile_hash) - Copernicus DEM tiles are reusable worldwide,
    their expensive conversion (read GeoTIFF, reproject if needed, downsample to 200 m grid) does not
    have to be recomputed when the core area changes (e.g. between two test regions) -
    only the subsequent combination/clipping/shift into local coordinates depends
    on the core area (see _dgm30_cache_file()).

    The name contains the file signature (size + modification time): replacing the tile file
    (e.g. a different version/source) immediately forces a recomputation instead of returning a wrong old
    tile.
    """
    st = tif_file.stat()
    signature = f"{tif_file.name}:{st.st_size}:{int(st.st_mtime)}"
    return config.CACHE_DIR / f"dgm30_tile_{hashlib.sha1(signature.encode('utf-8')).hexdigest()[:16]}.npz"


def _cached_geotiff_as_xyz(tif_file):
    """Like _load_geotiff_as_xyz(), but with a cache per tile (see _dgm30_tile_cache_file())."""
    cache_file = _dgm30_tile_cache_file(tif_file)
    if cache_file.exists():
        logger.debug(f"  [OK] DGM30 tile cache found: {tif_file.name} (already available as 200m grid)")
        data = np.load(cache_file)
        return data["points"], data["elevations"]

    points, elevations = _load_geotiff_as_xyz(str(tif_file))
    if points is not None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, points=points, elevations=elevations)

    return points, elevations


def _dgm30_cache_file(dgm30_path, tile_hash):
    """
    Cache file of the clipped DGM30 points (None without tile_hash).

    The name contains the DGM30 files (name, size, modification time): anyone who adds missing tiles thus does not
    get the old, incomplete cache back.
    """
    if not tile_hash:
        return None
    files = sorted(list(dgm30_path.glob("*.tif")) + list(dgm30_path.glob("*.tiff"))) if dgm30_path.exists() else []
    signature = "|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files)
    return config.CACHE_DIR / f"dgm30_horizon_{tile_hash}_{hashlib.sha1(signature.encode('utf-8')).hexdigest()[:10]}.npz"


def load_dgm30_tiles(dgm30_dir, bbox_utm, local_offset=None, tile_hash=None):
    """
    Loads DGM30 elevation data from GeoTIFF files (data/DGM30/*.tif) and clips it to the horizon area.

    Several files may be in the folder (e.g. several 1° tiles of the Copernicus DEM GLO-30); their points
    are combined. The files themselves have to be downloaded (see README).

    Args:
        dgm30_dir: directory with DGM30 files (e.g. data/DGM30/)
        bbox_utm: (min_x, max_x, min_y, max_y) in UTM meters - the horizon area
        local_offset: (ox, oy, oz) Optional - points are converted directly into local coordinates
        tile_hash: Optional - hash for cache consistency

    Returns:
        Tuple (height_points, height_elevations) or (None, None)
    """
    dgm30_path = Path(dgm30_dir)

    # Check the cache first (we assume it already contains local coordinates)
    cache_file = _dgm30_cache_file(dgm30_path, tile_hash)
    if cache_file is not None and cache_file.exists():
        logger.debug(f"  [OK] DGM30 cache found: {cache_file.name} (already local)")
        data = np.load(cache_file)
        return data["points"], data["elevations"]

    if dgm30_path.exists():
        height_points, height_elevations = _load_local_dgm30(
            dgm30_path, tile_hash, local_offset=local_offset, area_utm=bbox_utm
        )
        if height_points is not None:
            return height_points, height_elevations

    logger.error(f"  [!] No DGM30 files (*.tif) in {dgm30_path} - download the tiles, see README")
    return None, None


# From this gap at the edge of the horizon area on, a compass direction counts as not covered (the grid is 200 m fine,
# a tile rarely ends exactly at the edge)
_DGM30_EDGE_TOLERANCE_M = 1000.0


def clip_dgm30_to_area(points, elevations, area_utm, local_offset=None):
    """
    Discards DGM30 points outside the horizon area and reports where the data does not cover it.

    Without clipping, the extent of the loaded tiles determines the size of the horizon: whole 1° tiles would give
    a much larger horizon than the (100 km wide) texture.

    Args:
        points: (N, 2) points, local (with local_offset) or UTM (without)
        elevations: (N,) heights
        area_utm: (min_x, max_x, min_y, max_y) in UTM meters
        local_offset: (ox, oy[, oz]) - origin of the local coordinates, otherwise UTM

    Returns:
        (points, elevations, missing) with missing = compass directions ("west", "east", "south", "north") in which the
        points end more than _DGM30_EDGE_TOLERANCE_M short of the edge of the area
    """
    ox, oy = (local_offset[0], local_offset[1]) if local_offset is not None else (0.0, 0.0)
    x_min, x_max, y_min, y_max = area_utm[0] - ox, area_utm[1] - ox, area_utm[2] - oy, area_utm[3] - oy

    inside = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    points, elevations = points[inside], elevations[inside]
    if not len(elevations):
        return points, elevations, []

    tolerance = _DGM30_EDGE_TOLERANCE_M
    missing = []
    if points[:, 0].min() > x_min + tolerance:
        missing.append("west")
    if points[:, 0].max() < x_max - tolerance:
        missing.append("east")
    if points[:, 1].min() > y_min + tolerance:
        missing.append("south")
    if points[:, 1].max() < y_max - tolerance:
        missing.append("north")
    return points, elevations, missing


def _load_local_dgm30(dgm30_path, tile_hash=None, local_offset=None, area_utm=None):
    """
    Loads DGM30 from locally stored GeoTIFF files.

    Args:
        dgm30_path: Path object of the directory
        tile_hash: Optional - hash for cache
        local_offset: Optional – store directly in local coordinates
        area_utm: Optional – (min_x, max_x, min_y, max_y) in UTM: points outside are discarded

    Returns:
        Tuple (height_points, height_elevations) or (None, None)
    """
    # Search for GeoTIFF files
    tif_files = list(dgm30_path.glob("*.tif")) + list(dgm30_path.glob("*.tiff"))

    if not tif_files:
        logger.debug(f"  [i] No GeoTIFF files found in {dgm30_path}")
        return None, None

    logger.debug(f"  [i] Loading {len(tif_files)} GeoTIFF file(s)...")

    # Load first GeoTIFF (several are combined)
    all_points = []
    all_elevations = []

    for tif_file in tif_files:
        logger.debug(f"    - {tif_file.name}")
        points, elevations = _cached_geotiff_as_xyz(tif_file)

        if points is not None:
            all_points.append(points)
            all_elevations.append(elevations)

    if not all_points:
        logger.error(f"  [!] No DGM30 data loaded from GeoTIFF")
        return None, None

    # Combine all data (still in UTM, absolute - see _load_geotiff_as_xyz())
    height_points = np.vstack(all_points) if len(all_points) > 1 else all_points[0]
    height_elevations = np.concatenate(all_elevations) if len(all_elevations) > 1 else all_elevations[0]

    logger.debug(f"  [OK] {len(height_elevations)} points (200m grid) loaded from {len(tif_files)} GeoTIFF(s)")

    if local_offset is not None:
        # Shift only AFTER combining (no longer per tile, see _cached_geotiff_as_xyz()) -
        # makes the per-tile cache reusable independent of the area.
        ox, oy, oz = local_offset
        height_points = height_points - np.array([ox, oy])
        height_elevations = height_elevations - oz

    if area_utm is not None:
        height_points, height_elevations, missing = clip_dgm30_to_area(height_points, height_elevations, area_utm, local_offset)
        if not len(height_elevations):
            logger.error("  [!] The DGM30 files lie completely outside the horizon area - wrong tiles?")
            return None, None
        if missing:
            logger.warning(
                f"  [!] The DGM30 files do not cover the horizon area in the {', '.join(missing)} - "
                "download the missing tiles (see README); the horizon ends earlier there"
            )
        logger.debug(f"  [OK] Cropped to the horizon area: {len(height_elevations)} points")

    # Save cache
    cache_file = _dgm30_cache_file(dgm30_path, tile_hash)
    if cache_file is not None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, points=height_points, elevations=height_elevations)
        logger.debug(f"  [OK] DGM30 cache created: {cache_file.name}")

    return height_points, height_elevations


def enhance_sentinel2_image(image, contrast_factor=1.25, brightness_factor=0.88, color_factor=1.18):
    """
    Enhances Sentinel-2 satellite images to match DOP20 ground imagery.

    Sentinel-2 is often too pale and too bright compared to DOP20:
    - Increases contrast more (1.25 vs 1.18 for DOP20)
    - Reduces brightness more (0.88 vs 0.92 for DOP20)
    - Increases color saturation more (1.18 vs 1.12 for DOP20)

    Args:
        image: PIL Image (RGB)
        contrast_factor: contrast multiplier (1.25 = +25%, default)
        brightness_factor: brightness multiplier (0.88 = -12%, default)
        color_factor: color saturation multiplier (1.18 = +18%, default)

    Returns:
        Enhanced PIL Image
    """
    from PIL import ImageEnhance

    # Make sure the image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Increase contrast (more than DOP20)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Reduce brightness more (darker, to match DOP20)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Increase color saturation more (more vivid)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    return image


def load_sentinel2_geotiff(sentinel2_file, bbox_utm, tile_hash=None):
    """
    Loads the Sentinel-2 RGB GeoTIFF with georeferencing.

    The GeoTIFF must be georeferenced (with metadata for the coordinate transformation).

    Args:
        sentinel2_file: path of the GeoTIFF file (provided by sentinel2_fetch.ensure_horizon_texture())
        bbox_utm: (min_x, max_x, min_y, max_y) in UTM meters
        tile_hash: Optional - hash for cache

    Returns:
        Tuple (image_array, bounds_utm, transform) or None
        image_array: (H, W, 3) numpy array RGB [0-255]
        bounds_utm: (x_min, y_min, x_max, y_max) in UTM
        transform: rasterio affine transform
    """
    try:
        import rasterio
    except ImportError:
        logger.error("  [!] rasterio not installed. Install: pip install rasterio")
        return None

    tif_file = Path(sentinel2_file)

    if not tif_file.is_file():
        logger.error(f"  [!] Sentinel-2 GeoTIFF not found: {tif_file}")
        return None

    try:
        with rasterio.open(tif_file) as src:
            # Read RGB bands (band 1, 2, 3)
            if src.count >= 3:
                rgb_data = np.dstack([src.read(i) for i in range(1, 4)])
            elif src.count == 1:
                # Grayscale to RGB
                band = src.read(1)
                rgb_data = np.dstack([band, band, band])
            else:
                logger.error(f"    [!] Unexpected band count: {src.count}")
                return None

            # Extract metadata
            transform = src.transform
            bounds = src.bounds
            bounds_utm = (bounds.left, bounds.bottom, bounds.right, bounds.top)

            logger.debug(f"    - {tif_file.name}: {src.width}×{src.height} ({src.crs})")
            logger.debug(
                f"      UTM Bounds: X=[{bounds.left:.0f}..{bounds.right:.0f}], Y=[{bounds.bottom:.0f}..{bounds.top:.0f}]"
            )
            logger.debug(f"      Width: {bounds.right - bounds.left:.0f}m, height: {bounds.top - bounds.bottom:.0f}m")

            # Normalize to 0-255 if necessary
            if rgb_data.max() > 255:
                rgb_data = (rgb_data / rgb_data.max() * 255).astype(np.uint8)
            else:
                rgb_data = rgb_data.astype(np.uint8)

            # Enhance Sentinel-2 colors to match DOP20
            from PIL import Image, ImageEnhance

            pil_image = Image.fromarray(rgb_data, "RGB")
            pil_image = enhance_sentinel2_image(pil_image)
            rgb_data = np.array(pil_image)

            logger.debug(f"  [OK] Sentinel-2 loaded: {rgb_data.shape}")

            return rgb_data, bounds_utm, transform

    except Exception as e:
        logger.error(f"    [!] Error loading: {e}")
        return None


def generate_horizon_mesh(
    height_points,
    height_elevations,
    local_offset,
    tile_bounds=None,
    terrain_height_at=None,
):
    """
    Generates the horizon mesh with its own VertexManager.

    ARCHITECTURE:
    - UVs are generated AFTER the mesh (in horizon_workflow.py)

    OPTIMIZATIONS:
    - Vectorized batch VertexManager insertion
    - Efficient grid indexing
    - NO UV computation (comes later!)
    - No redundant lookups
    - Filter quads over terrain tiles (optional)

    Args:
        height_points: (N, 2) grid points already in local coordinates
        height_elevations: (N,) height values in local coordinates
        local_offset: (ox, oy, oz) transformation – only used here for consistency/logging
        tile_bounds: Optional - list of (x_min, y_min, x_max, y_max) tuples in local coordinates
                     for skipping quads that lie above terrain
        terrain_height_at: Optional - height query of the terrain heightmap. With tile_bounds
                        the horizon is then built with an exactly fitting hole, edge ring and
                        height transition (terrain/horizon_seam.py) - without stitching to a
                        terrain mesh. Without: old behavior (coarse hole, DGM30 heights).

    Returns:
        Tuple (mesh, nx, ny)
        mesh: mesh object with VertexManager
        nx, ny: grid dimensions (for texturing)
    """
    from ..mesh.vertex_manager import VertexManager
    from ..mesh.mesh import Mesh

    _ = local_offset  # kept for the caller signature; data is already local

    if terrain_height_at is not None and tile_bounds:
        from ..mesh.vertex_manager import VertexManager
        from ..mesh.mesh import Mesh
        from .horizon_seam import build_horizon_geometry

        hole = (
            min(b[0] for b in tile_bounds),
            min(b[1] for b in tile_bounds),
            max(b[2] for b in tile_bounds),
            max(b[3] for b in tile_bounds),
        )
        vertices, faces, nx, ny = build_horizon_geometry(
            height_points,
            height_elevations,
            hole,
            terrain_height_at,
            spacing=config.HORIZON_GRID_SPACING,
            seam_step=config.HORIZON_SEAM_STEP,
            blend_distance=config.HORIZON_BLEND_DISTANCE,
            flange_inset=config.HORIZON_FLANGE_INSET,
            flange_sink=config.HORIZON_FLANGE_SINK,
        )
        vm = VertexManager(tolerance=0.001)
        indices = vm.add_vertices_direct_nohash(vertices)
        mesh = Mesh(vm)
        mesh.faces = list(map(tuple, faces))
        mesh.uvs = []
        mesh.uv_indices = {}
        logger.debug(
            f"  [OK] Horizon with matching terrain hole {hole}: {len(vertices)} vertices, {len(faces)} triangles"
        )
        return mesh, nx, ny

    # Points and heights are already available locally
    local_points = height_points
    local_elevations = height_elevations

    # Create a regular grid from irregular points
    # Determine grid dimensions (200 m spacing)
    x_min, x_max = local_points[:, 0].min(), local_points[:, 0].max()
    y_min, y_max = local_points[:, 1].min(), local_points[:, 1].max()

    grid_spacing = 200.0
    x_coords = np.arange(x_min, x_max + grid_spacing * 0.5, grid_spacing)
    y_coords = np.arange(y_min, y_max + grid_spacing * 0.5, grid_spacing)

    nx = len(x_coords)
    ny = len(y_coords)

    logger.debug(f"  [i] Creating horizon mesh: {nx}×{ny} grid")

    # Create grid with nearest-neighbor interpolation
    from scipy.spatial import cKDTree

    tree = cKDTree(local_points)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Find nearest points
    distances, indices = tree.query(grid_points_flat)
    grid_elevations = local_elevations[indices]

    # Create 3D vertices - horizon 50 m below Z level for core mesh separation
    vertices = np.column_stack([grid_points_flat, grid_elevations])

    vm = VertexManager(tolerance=0.001)

    mesh = Mesh(vm)

    # === OPTIMIZATION: batch vertex insertion (WITHOUT hash lookup!) ===
    # For regular grids: all vertices are distinct, no dedup needed!
    # Use add_vertices_direct_nohash() instead of individual add_vertex() calls (258k calls!)
    vertex_indices = np.array(vm.add_vertices_direct_nohash(vertices), dtype=int)

    # Reshape vertex_indices into an (ny, nx) grid for easy access
    vertex_grid = vertex_indices.reshape(ny, nx)

    # === OPTIMIZATION 3: vectorized quad filtering over tile bounds ===
    quads_mask = np.ones((ny - 1, nx - 1), dtype=bool)

    if tile_bounds:
        logger.debug(f"  [i] Filtering {len(tile_bounds)} terrain tiles (2x2 km) with vectorized lookup...")

        import time

        t0 = time.time()

        # Pre-compute all 4 vertex positions for each quad (vectorized)
        # Quad (y, x) has vertices at:
        #   v0 = (x_coords[x], y_coords[y])       - bottom left
        #   v1 = (x_coords[x+1], y_coords[y])     - bottom right
        #   v2 = (x_coords[x], y_coords[y+1])     - top left
        #   v3 = (x_coords[x+1], y_coords[y+1])   - top right

        # Create meshgrids for all 4 corners
        x_left = x_coords[:-1]
        x_right = x_coords[1:]
        y_bottom = y_coords[:-1]
        y_top = y_coords[1:]

        # For each tile, check whether at least one vertex lies inside it
        for tile_x_min, tile_y_min, tile_x_max, tile_y_max in tile_bounds:
            # For each quad: check whether ANY vertex lies in the tile
            # A vertex lies in the tile if: tile_x_min <= x < tile_x_max AND tile_y_min <= y < tile_y_max

            # Check all 4 vertices (vectorized over all quads)
            # v0 (bottom left): (x_left, y_bottom)
            v0_inside = ((x_left >= tile_x_min) & (x_left < tile_x_max))[:, None] & (
                (y_bottom >= tile_y_min) & (y_bottom < tile_y_max)
            )[None, :]

            # v1 (bottom right): (x_right, y_bottom)
            v1_inside = ((x_right >= tile_x_min) & (x_right < tile_x_max))[:, None] & (
                (y_bottom >= tile_y_min) & (y_bottom < tile_y_max)
            )[None, :]

            # v2 (top left): (x_left, y_top)
            v2_inside = ((x_left >= tile_x_min) & (x_left < tile_x_max))[:, None] & (
                (y_top >= tile_y_min) & (y_top < tile_y_max)
            )[None, :]

            # v3 (top right): (x_right, y_top)
            v3_inside = ((x_right >= tile_x_min) & (x_right < tile_x_max))[:, None] & (
                (y_top >= tile_y_min) & (y_top < tile_y_max)
            )[None, :]

            # Remove the quad if AT LEAST ONE vertex lies in the tile
            tile_mask = v0_inside | v1_inside | v2_inside | v3_inside
            quads_mask &= ~tile_mask.T  # Transpose because the meshgrid is (x, y) instead of (y, x)

        skipped_count = np.sum(~quads_mask)
        if skipped_count > 0:
            logger.debug(f"  [OK] {skipped_count} quads above the terrain filtered ({time.time() - t0:.2f}s)")

    # === OPTIMIZATION 4: batch insert of direct arrays (NO deduplication needed) ===
    # Store faces & UVs directly without add_face() overhead

    valid_quads = np.argwhere(quads_mask)  # (N, 2) array with (y, x) indices

    if len(valid_quads) == 0:
        logger.error("  [!] No quads to generate (all filtered)")
        return mesh, nx, ny

    # Create face arrays vectorized
    y_indices = valid_quads[:, 0]
    x_indices = valid_quads[:, 1]

    # Vertex indices for all quads at once
    v0 = vertex_grid[y_indices, x_indices]
    v1 = vertex_grid[y_indices, x_indices + 1]
    v2 = vertex_grid[y_indices + 1, x_indices]
    v3 = vertex_grid[y_indices + 1, x_indices + 1]

    # Create two triangles per quad directly
    faces_tri1 = np.column_stack([v0, v1, v2])
    faces_tri2 = np.column_stack([v1, v3, v2])

    # Combine and store directly (NO add_face loops!)
    faces_array = np.vstack([faces_tri1, faces_tri2]).astype(int)
    mesh.faces = list(map(tuple, faces_array))

    # === UVs are NOT generated here! ===
    # They are generated in horizon_workflow.py AFTER stitching (for all vertices together)
    mesh.uvs = []
    mesh.uv_indices = {}

    face_count = len(mesh.faces)

    logger.debug(f"  [OK] {face_count} triangles generated")
    logger.debug(f"  [OK] {len(mesh.uvs)} UVs (1 per vertex, without deduplication)")

    return mesh, nx, ny


def texture_horizon_mesh(vertices, horizon_image, nx, ny, bounds_utm, transform, global_offset):
    """
    Maps the Sentinel-2 RGB texture onto the horizon mesh with correct georeferencing.

    Args:
        vertices: (M, 3) mesh vertices in local coordinates
        horizon_image: (H, W, 3) RGB array
        nx, ny: grid dimensions
        bounds_utm: (x_min, y_min, x_max, y_max) texture bounds in UTM
        transform: rasterio affine transform
        global_offset: (ox, oy, oz) for converting back local → UTM

    Returns:
        Dict with texture information
    """
    if horizon_image is None:
        return {"texture_path": None, "uv_map": None}

    config.BEAMNG_DIR_TEXTURES.mkdir(parents=True, exist_ok=True)

    # Convert mesh vertices back to UTM for coordinate mapping
    ox, oy, oz = global_offset
    vertices_utm = vertices.copy()
    vertices_utm[:, 0] += ox
    vertices_utm[:, 1] += oy

    # Check the match
    mesh_x_min, mesh_x_max = vertices_utm[:, 0].min(), vertices_utm[:, 0].max()
    mesh_y_min, mesh_y_max = vertices_utm[:, 1].min(), vertices_utm[:, 1].max()

    tex_x_min, tex_y_min, tex_x_max, tex_y_max = bounds_utm

    logger.debug(f"  [i] Coordinate check:")
    logger.debug(f"      Mesh (UTM):    X=[{mesh_x_min:.0f}..{mesh_x_max:.0f}], Y=[{mesh_y_min:.0f}..{mesh_y_max:.0f}]")
    logger.debug(f"      Texture (UTM): X=[{tex_x_min:.0f}..{tex_x_max:.0f}], Y=[{tex_y_min:.0f}..{tex_y_max:.0f}]")

    # Compute overlap
    overlap_x = (min(mesh_x_max, tex_x_max) - max(mesh_x_min, tex_x_min)) / (mesh_x_max - mesh_x_min) * 100
    overlap_y = (min(mesh_y_max, tex_y_max) - max(mesh_y_min, tex_y_min)) / (mesh_y_max - mesh_y_min) * 100

    logger.debug(f"      Overlap: X={overlap_x:.1f}%, Y={overlap_y:.1f}%")

    # Save temporarily as TIF for texconv
    import tempfile
    import subprocess

    temp_tif = Path(tempfile.gettempdir()) / "horizon_temp.tif"

    img_pil = Image.fromarray(horizon_image.astype("uint8"), "RGB")
    img_pil.save(temp_tif, "TIFF")

    # Convert to DDS with texconv.exe (BC1, 8192x8192, mipmaps)
    from ..io.texconv import ensure_texconv

    texconv_exe = ensure_texconv()
    dds_output = config.BEAMNG_DIR_TEXTURES / "horizon_sentinel2.dds"

    # texconv parameters:
    # -f BC1_UNORM: BC1 compression
    # -w 8192 -h 8192: target resolution
    # -m 0: full mipmap chain
    # -o: output directory
    # -y: overwrite without asking
    cmd = [
        str(texconv_exe),
        "-f",
        "BC1_UNORM",
        "-w",
        str(config.HORIZON_IMAGE_SIZE_PX),
        "-h",
        str(config.HORIZON_IMAGE_SIZE_PX),
        "-m",
        "0",
        "-y",
        "-o",
        str(config.BEAMNG_DIR_TEXTURES),
        str(temp_tif),
    ]

    logger.debug(f"  [i] Converting to DDS (BC1, 8192x8192, mipmaps)...")
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    # texconv names the output after the input: horizon_temp.dds -> rename
    texconv_output = config.BEAMNG_DIR_TEXTURES / "horizon_temp.dds"
    if texconv_output.exists():
        if dds_output.exists():
            dds_output.unlink()
        texconv_output.rename(dds_output)

    # Clean up
    if temp_tif.exists():
        temp_tif.unlink()

    logger.debug(f"  [OK] Horizon texture (DDS) saved: {dds_output}")

    # Relative paths for materials.json
    relative_texture_path = str(config.RELATIVE_DIR_TEXTURES / "horizon_sentinel2.dds")

    return {
        "texture_path": relative_texture_path,
        "image_size": horizon_image.shape,
        "bounds_utm": bounds_utm,
        "mesh_coverage": (overlap_x, overlap_y),
    }


def export_horizon_dae(mesh, texture_info, output_dir, level_name="default", global_offset=None, tile_bounds=None):
    """
    Exports the horizon mesh as a DAE (Collada) file with deduplicated UVs.

    Tile bounds are already filtered during mesh generation (see generate_horizon_mesh).
    This function is solely responsible for the DAE export.

    Args:
        mesh: mesh object with central VertexManager + deduplicated UVs (already filtered)
        texture_info: dict with texture information (bounds_utm, mesh_coverage)
        output_dir: target directory (BeamNG level directory)
        level_name: name of the level
        global_offset: (ox, oy, oz) for UTM conversion
        tile_bounds: (UNUSED - only for API compatibility, filtering happens in generate_horizon_mesh)

    Returns:
        Path to the generated DAE file
    """
    # tile_bounds is no longer needed here - filtering already happens during mesh generation!
    _ = tile_bounds  # Unused - filtering happens in generate_horizon_mesh()

    dae_path = Path(output_dir) / "art" / "shapes" / "terrain_horizon.dae"
    dae_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute UV offsets based on coordinate mismatch
    bounds_utm = texture_info.get("bounds_utm", None)
    vertices = mesh.vertex_manager.vertices
    faces = mesh.faces  # Extract faces from the mesh

    logger.debug(f"  [i] DAE-Export: {len(vertices)} Vertices, {len(faces)} Faces")

    if bounds_utm and global_offset:
        ox, oy, oz = global_offset
        tex_x_min, tex_y_min, tex_x_max, tex_y_max = bounds_utm

        # Mesh bounds in UTM
        mesh_x_min, mesh_x_max = vertices[:, 0].min() + ox, vertices[:, 0].max() + ox
        mesh_y_min, mesh_y_max = vertices[:, 1].min() + oy, vertices[:, 1].max() + oy

        # Offsets in UTM meters
        offset_x_m = mesh_x_min - tex_x_min
        offset_y_m = mesh_y_min - tex_y_min

        # Texture size: 100 km × 100 km for ±50 km
        tex_width_m = tex_x_max - tex_x_min
        tex_height_m = tex_y_max - tex_y_min

        # UV offset (normalized to 0..1)
        uv_offset_x = offset_x_m / tex_width_m
        uv_offset_y = offset_y_m / tex_height_m

        # Mesh size in local coordinates
        mesh_width_m = vertices[:, 0].max() - vertices[:, 0].min()
        mesh_height_m = vertices[:, 1].max() - vertices[:, 1].min()

        # UV scale (mesh size to texture size)
        uv_scale_x = mesh_width_m / tex_width_m
        uv_scale_y = mesh_height_m / tex_height_m

        logger.debug(f"  [i] UV mapping with offset:")
        logger.debug(f"      UV-Offset: ({uv_offset_x:.4f}, {uv_offset_y:.4f})")
        logger.debug(f"      UV scale: ({uv_scale_x:.4f}, {uv_scale_y:.4f})")
    else:
        uv_offset_x, uv_offset_y = 0.0, 0.0
        uv_scale_x, uv_scale_y = 1.0, 1.0

    # Scale deduplicated UVs VECTORIZED with offset and scale
    uvs_array = np.array(mesh.uvs, dtype=np.float32)
    scaled_uvs_array = uvs_array.copy()
    scaled_uvs_array[:, 0] = uv_offset_x + uvs_array[:, 0] * uv_scale_x
    scaled_uvs_array[:, 1] = uv_offset_y + uvs_array[:, 1] * uv_scale_y

    # Convert to list for compatibility
    scaled_uvs = [(u, v) for u, v in scaled_uvs_array]

    # Write DAE with a StringIO buffer (faster than direct file I/O)
    from io import StringIO

    buffer = StringIO()

    # Write everything into the buffer (much faster than directly into a file)
    f = buffer
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema">\n')

    # Asset
    f.write("  <asset>\n")
    f.write("    <created>2025-01-07T00:00:00</created>\n")
    f.write("    <modified>2025-01-07T00:00:00</modified>\n")
    f.write("  </asset>\n")

    # === Library Materials ===
    f.write("  <library_materials>\n")
    f.write('    <material id="horizon_terrain" name="horizon_terrain">\n')
    f.write('      <instance_effect url="#horizon_terrain_effect"/>\n')
    f.write("    </material>\n")
    f.write("  </library_materials>\n")

    # === Library Effects ===
    f.write("  <library_effects>\n")
    f.write('    <effect id="horizon_terrain_effect">\n')
    f.write("      <profile_COMMON>\n")
    f.write('        <technique sid="common">\n')
    if texture_info and texture_info.get("texture_path"):
        # With texture
        f.write("          <phong>\n")
        f.write("            <diffuse>\n")
        f.write("              <color>1.0 1.0 1.0 1.0</color>\n")
        f.write("            </diffuse>\n")
        f.write("            <shininess>\n")
        f.write("              <float>1.0</float>\n")
        f.write("            </shininess>\n")
        f.write("          </phong>\n")
    else:
        # Without texture - fallback color
        f.write("          <phong>\n")
        f.write("            <diffuse>\n")
        f.write("              <color>0.8 0.8 0.8 1.0</color>\n")
        f.write("            </diffuse>\n")
        f.write("            <shininess>\n")
        f.write("              <float>1.0</float>\n")
        f.write("            </shininess>\n")
        f.write("          </phong>\n")
    f.write("        </technique>\n")
    f.write("      </profile_COMMON>\n")
    f.write("    </effect>\n")
    f.write("  </library_effects>\n")

    # Library Geometries
    f.write("  <library_geometries>\n")
    f.write('    <geometry id="horizon_mesh" name="horizon">\n')
    f.write("      <mesh>\n")

    # === Vertices Source ===
    f.write('        <source id="horizon_vertices">\n')
    f.write(f'          <float_array id="horizon_vertices_array" count="{len(vertices) * 3}">')

    # Write vertices with minimal overhead
    for vertex in vertices:
        f.write(f"\n{vertex[0]:.2f} {vertex[1]:.2f} {vertex[2]:.2f}")

    f.write("\n          </float_array>\n")
    f.write("          <technique_common>\n")
    f.write(f'            <accessor source="#horizon_vertices_array" count="{len(vertices)}" stride="3">\n')
    f.write('              <param name="X" type="float"/>\n')
    f.write('              <param name="Y" type="float"/>\n')
    f.write('              <param name="Z" type="float"/>\n')
    f.write("            </accessor>\n")
    f.write("          </technique_common>\n")
    f.write("        </source>\n")

    # === Normals Source (for BeamNG compatibility) ===
    # Compute smooth normals VECTORIZED from the faces
    normals = np.zeros((len(vertices), 3), dtype=np.float32)

    # Convert faces to a NumPy array for vectorized processing
    faces_array = np.array(faces, dtype=np.int32)

    # Extract all vertices for all faces at once
    v0_all = vertices[faces_array[:, 0]]
    v1_all = vertices[faces_array[:, 1]]
    v2_all = vertices[faces_array[:, 2]]

    # Compute edges vectorized
    edge1_all = v1_all - v0_all
    edge2_all = v2_all - v0_all

    # Compute face normals vectorized
    face_normals = np.cross(edge1_all, edge2_all)
    face_normals_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        face_normals, face_normals_len, out=np.zeros_like(face_normals), where=face_normals_len > 0
    )

    # Accumulate face normals into vertex normals
    for i, face in enumerate(faces_array):
        normals[face[0]] += face_normals[i]
        normals[face[1]] += face_normals[i]
        normals[face[2]] += face_normals[i]

    # Normalize vertex normals vectorized
    normals_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, normals_len, out=np.tile([0.0, 0.0, 1.0], (len(normals), 1)), where=normals_len > 0)

    f.write('        <source id="horizon_normals">\n')
    f.write(f'          <float_array id="horizon_normals_array" count="{len(normals) * 3}">')
    # Write normals
    for normal in normals:
        f.write(f"\n{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
    f.write("\n          </float_array>\n")
    f.write("          <technique_common>\n")
    f.write(f'            <accessor source="#horizon_normals_array" count="{len(vertices)}" stride="3">\n')
    f.write('              <param name="X" type="float"/>\n')
    f.write('              <param name="Y" type="float"/>\n')
    f.write('              <param name="Z" type="float"/>\n')
    f.write("            </accessor>\n")
    f.write("          </technique_common>\n")
    f.write("        </source>\n")

    # === UV Coordinates Source (deduplicated from mesh.uvs) ===
    f.write('        <source id="horizon_uvs">\n')
    f.write(f'          <float_array id="horizon_uvs_array" count="{len(scaled_uvs) * 2}">')

    # Write deduplicated UV coordinates
    for u, v in scaled_uvs:
        f.write(f"\n{u:.6f} {v:.6f}")

    f.write("\n          </float_array>\n")
    f.write("          <technique_common>\n")
    f.write(f'            <accessor source="#horizon_uvs_array" count="{len(scaled_uvs)}" stride="2">\n')
    f.write('              <param name="S" type="float"/>\n')
    f.write('              <param name="T" type="float"/>\n')
    f.write("            </accessor>\n")
    f.write("          </technique_common>\n")
    f.write("        </source>\n")
    f.write('        <vertices id="horizon_vertices_input">\n')
    f.write('          <input semantic="POSITION" source="#horizon_vertices"/>\n')
    f.write("        </vertices>\n")

    # === Triangles (with deduplicated UV indices and normals) ===
    f.write(f'        <triangles material="horizon_terrain" count="{len(faces)}">\n')
    f.write('          <input semantic="VERTEX" source="#horizon_vertices_input" offset="0"/>\n')
    f.write('          <input semantic="NORMAL" source="#horizon_normals" offset="1"/>\n')
    f.write('          <input semantic="TEXCOORD" source="#horizon_uvs" offset="2" set="0"/>\n')
    f.write("          <p>")

    # Write face indices with normals + deduplicated UV indices from mesh.uv_indices
    # Format: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
    for face_idx, face in enumerate(faces):
        if face_idx in mesh.uv_indices:
            uv_indices = mesh.uv_indices[face_idx]
            f.write(
                f"\n{face[0]} {face[0]} {uv_indices[0]} {face[1]} {face[1]} {uv_indices[1]} {face[2]} {face[2]} {uv_indices[2]}"
            )
        else:
            # Fallback: use vertex indices as UV indices (should not occur)
            f.write(f"\n{face[0]} {face[0]} {face[0]} {face[1]} {face[1]} {face[1]} {face[2]} {face[2]} {face[2]}")

    f.write("\n          </p>\n")
    f.write("        </triangles>\n")

    # Close mesh, geometry
    f.write("      </mesh>\n")
    f.write("    </geometry>\n")
    f.write("  </library_geometries>\n")

    # === Library Visual Scenes ===
    f.write("  <library_visual_scenes>\n")
    f.write('    <visual_scene id="Scene" name="Scene">\n')
    f.write('      <node id="Horizon" name="Horizon" type="NODE">\n')
    f.write('        <instance_geometry url="#horizon_mesh">\n')
    f.write("          <bind_material>\n")
    f.write("            <technique_common>\n")
    f.write('              <instance_material symbol="horizon_terrain" target="#horizon_terrain"/>\n')
    f.write("            </technique_common>\n")
    f.write("          </bind_material>\n")
    f.write("        </instance_geometry>\n")
    f.write("      </node>\n")
    f.write("    </visual_scene>\n")
    f.write("  </library_visual_scenes>\n")

    # === Scene ===
    f.write("  <scene>\n")
    f.write('    <instance_visual_scene url="#Scene"/>\n')
    f.write("  </scene>\n")

    f.write("</COLLADA>\n")

    # Write the buffer contents to the file at once (much faster)
    with open(dae_path, "w", encoding="utf-8") as file:
        file.write(buffer.getvalue())
    buffer.close()

    logger.debug(f"  [OK] DAE exported with deduplicated UVs: {dae_path.name}")
    logger.debug(f"  [OK] UV statistics: {len(mesh.uvs)} deduplicated UVs, {len(mesh.faces)} faces")

    # Check whether the file exists
    if dae_path.exists():
        file_size = dae_path.stat().st_size
        logger.debug(f"      File size: {file_size:,} bytes")
    else:
        logger.warning(f"      Horizon DAE does not exist: {dae_path}")

    return dae_path.name
