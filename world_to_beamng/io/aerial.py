"""
Aerial image processing - extracts and tiles aerial photos.
"""

import json
import os
import zipfile
import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional
from PIL import Image, ImageEnhance
from io import BytesIO
from world_to_beamng.logging_config import LoggerConfig
from .. import config

# This module itself builds large canvases from its own, trusted geodata (it does not open a
# foreign file) - PIL's decompression bomb protection (default limit ~89.5 million pixels) triggers
# here for no reason: even a 2 km tile at fine resolution (e.g. 0.1 m/px Swiss orthophotos)
# is far above it.
Image.MAX_IMAGE_PIXELS = None

logger = LoggerConfig.get_logger()


def parse_world_file(tfw_data):
    """
    Parses World File (.tfw) data.

    Args:
        tfw_data: Bytes or string of the .tfw file

    Returns:
        Dict with pixel_size_x, pixel_size_y, x_origin, y_origin
    """
    if isinstance(tfw_data, bytes):
        tfw_data = tfw_data.decode("utf-8")

    lines = tfw_data.strip().split("\n")
    if len(lines) < 6:
        return None

    try:
        pixel_size_x = float(lines[0])
        pixel_size_y = float(lines[3])
        x_origin = float(lines[4])
        y_origin = float(lines[5])

        return {
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "x_origin": x_origin,
            "y_origin": y_origin,
        }
    except (ValueError, IndexError):
        return None


def extract_images_from_zips(aerial_dir=config.AERIAL_DATA_DIR):
    """
    Extracts all images with georeferencing from ZIP files.

    Georeferencing comes either from an accompanying .tfw file (world_info["crs_epsg"] then stays
    None - the source CRS is assumed as before) OR, if there is no .tfw, from embedded GeoTIFF
    tags in the image itself (world_info["crs_epsg"] set). A plain JPG/PNG without a .tfw and
    without geo tags remains an error case (world_info=None, discarded later).

    Args:
        aerial_dir: Path to the aerial photo directory (config.AERIAL_DATA_DIR)

    Returns:
        List of (image_name, image_data_bytes, world_file_info) tuples
    """
    aerial_path = Path(aerial_dir)
    images = []

    if not aerial_path.exists():
        logger.error(f"[!] Directory {aerial_dir} does not exist")
        return images

    zip_files = list(aerial_path.glob("*.zip"))

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                file_list = zip_ref.namelist()

                # Find image files (TIF, TIFF, JPG, JPEG, PNG)
                image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
                image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in image_extensions)]

                for img_file in image_files:
                    img_data = zip_ref.read(img_file)

                    # Look for the matching .tfw file
                    # Replace the image extension with .tfw (e.g. .tif → .tfw)
                    img_path = Path(img_file)
                    tfw_file = str(img_path.with_suffix(".tfw"))

                    world_info = None
                    if tfw_file in file_list:
                        tfw_data = zip_ref.read(tfw_file)
                        world_info = parse_world_file(tfw_data)
                    else:
                        # Debugging: look for a .tfw with the same stem name (case-insensitive)
                        base_name = img_path.stem.lower()
                        for f in file_list:
                            if f.lower().endswith(".tfw") and Path(f).stem.lower() == base_name:
                                tfw_data = zip_ref.read(f)
                                world_info = parse_world_file(tfw_data)
                                break

                    if world_info is None:
                        # No .tfw found - the image itself may have embedded GeoTIFF
                        # georeferencing (no .tfw needed, e.g. many generic GeoTIFF portals)
                        world_info = _read_geotiff_world_info(f"/vsizip/{zip_path}/{img_file}")

                    images.append((img_file, img_data, world_info))

        except Exception as e:
            logger.error(f"[!] Error reading {zip_path.name}: {e}")

    return images


def _read_geotiff_world_info(path_or_vsi):
    """
    Reads CRS + world_info (pixel_size_x/y, x_origin, y_origin = upper left pixel corner) from a
    georeferenced raster via rasterio - embedded GeoTIFF tags, no .tfw needed.

    Returns:
        dict like parse_world_file(), plus "crs_epsg" (can be None if the CRS has no EPSG code -
        known limitation), or None if there is no CRS/no real geotransform.
    """
    import rasterio

    try:
        with rasterio.open(path_or_vsi) as src:
            if src.crs is None or src.transform.is_identity:
                return None
            t = src.transform
            return {
                "pixel_size_x": t.a,
                "pixel_size_y": t.e,
                "x_origin": t.c,
                "y_origin": t.f,
                "crs_epsg": src.crs.to_epsg(),
            }
    except Exception:
        return None


def extract_loose_images(aerial_dir=config.AERIAL_DATA_DIR):
    """
    Loose raster files directly in the directory (*.tif, *.tiff) - not in a ZIP. Georeferencing
    as in extract_images_from_zips(): embedded GeoTIFF tags preferred, otherwise an accompanying
    .tfw file of the same name.

    Returns:
        List of (image_name, image_path: Path, world_info) - image_path (not bytes!), since loose
        GeoTIFFs can be arbitrarily large; see _open_image().
    """
    aerial_path = Path(aerial_dir)
    images = []
    if not aerial_path.exists():
        return images

    paths = sorted(aerial_path.glob("*.tif")) + sorted(aerial_path.glob("*.tiff"))
    for path in paths:
        world_info = _read_geotiff_world_info(str(path))
        if world_info is None:
            tfw = path.with_suffix(".tfw")
            if tfw.exists():
                world_info = parse_world_file(tfw.read_bytes())
                if world_info is not None:
                    world_info["crs_epsg"] = None
        images.append((path.name, path, world_info))
    return images


def extract_georeferenced_images(aerial_dir=config.AERIAL_DATA_DIR):
    """
    Combines extract_images_from_zips() (ZIP, bytes-based) and extract_loose_images() (loose
    file, Path-based) into one uniform list - the source format no longer matters afterwards, both
    continue through the same _open_image()/_prepare_image_for_compositing() path.

    Returns:
        List of (image_name, source: bytes|Path, world_info|None)
    """
    return extract_images_from_zips(aerial_dir) + extract_loose_images(aerial_dir)


def _open_image(source):
    """Opens a source image - source is either bytes (from a ZIP) or a Path (loose file)."""
    return Image.open(BytesIO(source)) if isinstance(source, (bytes, bytearray)) else Image.open(source)


def _reproject_image_to_source_crs(source, dst_epsg):
    """
    Reprojects a single georeferenced image to dst_epsg via rasterio (model: the reprojection
    logic already present in terrain/horizon_image.py::build_horizon_image()).

    Args:
        source: bytes (from a ZIP) or Path/str (loose file)
        dst_epsg: Target EPSG code

    Returns:
        (PIL.Image RGB, world_info) in the target CRS
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    def _reproject(src):
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{dst_epsg}", src.width, src.height, *src.bounds
        )
        dst = np.zeros((3, height, width), dtype=src.dtypes[0])
        for band in range(1, min(src.count, 3) + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=dst[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=f"EPSG:{dst_epsg}",
                resampling=Resampling.bilinear,
            )
        image = Image.fromarray(np.moveaxis(dst, 0, -1)).convert("RGB")
        world_info = {
            "pixel_size_x": transform.a,
            "pixel_size_y": transform.e,
            "x_origin": transform.c,
            "y_origin": transform.f,
            "crs_epsg": dst_epsg,
        }
        return image, world_info

    if isinstance(source, (bytes, bytearray)):
        with rasterio.io.MemoryFile(source) as memfile, memfile.open() as src:
            return _reproject(src)
    with rasterio.open(source) as src:
        return _reproject(src)


def _prepare_image_for_compositing(source, world_info, dst_epsg):
    """
    Opens a source image for compositing and reprojects it to the target CRS if needed. For
    .tfw pairs (world_info["crs_epsg"] is None, assumed source CRS as before) it NEVER
    reprojects - no behavior difference for the existing LGL-BW path.

    Returns:
        (PIL.Image, world_info) - world_info unchanged, except on reprojection (then the
        georeferencing recomputed in the target CRS)
    """
    src_epsg = world_info.get("crs_epsg")
    if src_epsg is not None and src_epsg != dst_epsg:
        return _reproject_image_to_source_crs(source, dst_epsg)
    return _open_image(source), world_info


def enhance_dop20_image(image, contrast_factor=1.18, brightness_factor=0.92, color_factor=1.12):
    """
    Enhances DOP20 images for a more natural look in BeamNG.

    DOP20 images are often too pale and too bright - this function increases:
    - Contrast (more dynamic range)
    - Color saturation (more vivid)
    - Reduces brightness (more natural)

    Args:
        image: PIL Image
        contrast_factor: Contrast multiplier (1.18 = +18%, default)
        brightness_factor: Brightness multiplier (0.92 = -8%, default)
        color_factor: Color saturation multiplier (1.12 = +12%, default)

    Returns:
        Enhanced PIL Image
    """
    # Make sure the image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Reduce brightness (darker)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Increase color saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    return image


AERIAL_PHOTO_FILENAME = "aerial_photo.png"


def process_aerial_images(aerial_dir, output_dir, grid_bounds, global_offset, target_pixel_size=None):
    """
    Composes all aerial photos into ONE contiguous photo for the entire
    grid_bounds area (instead of many small 500m tiles).

    Background (research 2026-09-18): according to the official docs, BeamNG's v1.5 terrain
    material system is designed for a SMALL number of repeating materials
    ("keep terrain material counts much lower than the technical
    limit"), not for many (16-25) unique 4096px textures. With many
    large, unique materials, BeamNG's texture atlas packer displayed individual tiles
    rotated when packing, even though the source files were
    demonstrably correct (each on its own AND as a composed mosaic
    without gaps). With only one material for the entire area, this
    packing problem disappears completely.

    Uses .tfw World Files to position each source image exactly on
    a common canvas at native resolution, then scales the result
    down to target_pixel_size.

    Args:
        aerial_dir: Directory with ZIP archives
        output_dir: Target directory for the composed photo
        grid_bounds: (min_x, max_x, min_y, max_y) in local coordinates
        global_offset: (utm_x, utm_y, utm_z) tuple - UTM offset for the coordinate transformation
        target_pixel_size: Edge length (pixels) of the output photo (default: config.TERRAIN_BASE_TEX_PIXEL_SIZE)

    Returns:
        1 if a photo was saved, otherwise 0
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE

    from ..geometry.coordinates import get_source_crs_epsg

    dst_epsg = get_source_crs_epsg()

    images = extract_georeferenced_images(aerial_dir)
    if not images:
        logger.debug("  [i] No aerial photos found")
        return 0

    images_with_geo = [(name, data, info) for name, data, info in images if info is not None]
    if not images_with_geo:
        logger.error(f"  [!] No georeferencing found (missing .tfw files or embedded GeoTIFF tags?)")
        return 0

    logger.debug(f"  [i] {len(images_with_geo)} aerial photos with georeferencing found")

    grid_min_x, grid_max_x, grid_min_y, grid_max_y = grid_bounds
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y
    offset_x, offset_y = global_offset[:2]

    # Native resolution as the reference for the canvas (all DOP20 tiles of
    # a region have the same pixel size, e.g. 0.2m/px).
    native_pixel_size = abs(images_with_geo[0][2]["pixel_size_x"])
    canvas_w = max(1, round(grid_width / native_pixel_size))
    canvas_h = max(1, round(grid_height / native_pixel_size))
    logger.info(
        f"  [i] Building combined aerial photo: {grid_width:.0f}m x {grid_height:.0f}m "
        f"@ {native_pixel_size}m/px = {canvas_w}x{canvas_h}px native -> {target_pixel_size}x{target_pixel_size}px"
    )

    # Fill color for possible gaps (no aerial photo coverage) - muted green instead of
    # black/magenta, so that missing border areas do not stand out garishly.
    canvas = Image.new("RGB", (canvas_w, canvas_h), (70, 95, 55))

    pasted = 0
    for img_name, img_data, world_info in images_with_geo:
        try:
            image, world_info = _prepare_image_for_compositing(img_data, world_info, dst_epsg)
            image = enhance_dop20_image(image)
            pixel_size = abs(world_info["pixel_size_x"])

            # The .tfw origin is the upper left (northwest) pixel corner.
            img_local_x = world_info["x_origin"] - offset_x
            img_local_y = world_info["y_origin"] - offset_y

            if not math.isclose(pixel_size, native_pixel_size, rel_tol=1e-6):
                scale = pixel_size / native_pixel_size
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )

            # Position on the canvas: the canvas origin is the
            # northwest corner (grid_min_x, grid_max_y), row 0 = north -
            # standard image convention, no tile bucket arithmetic needed anymore.
            px = round((img_local_x - grid_min_x) / native_pixel_size)
            py = round((grid_max_y - img_local_y) / native_pixel_size)

            canvas.paste(image, (px, py))
            pasted += 1
        except Exception as e:
            logger.error(f"  [!] Error processing {img_name}: {e}")
            continue

    if pasted == 0:
        logger.error("  [!] No aerial photos could be placed")
        return 0

    canvas = canvas.resize((target_pixel_size, target_pixel_size), Image.Resampling.LANCZOS)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / AERIAL_PHOTO_FILENAME
    canvas.save(filepath, "PNG")

    logger.info(f"  [OK] Combined aerial photo from {pasted} source images saved: {filepath}")
    return 1


AERIAL_SIGNATURE_FILENAME = "aerial_photo.json"
AERIAL_SIGNATURE_VERSION = 2
SINGLE_PHOTO_NAME = AERIAL_PHOTO_FILENAME[: -len(".png")]  # "aerial_photo"


def process_aerial_tiles(aerial_dir, output_dir, photos, global_offset, target_pixel_size=None):
    """
    Four-image mode: builds a separate aerial photo per entry in `photos` (one photo per DGM1 tile).

    Each source image is read and enhanced only ONCE and then placed into all photos it touches
    (a source image can span a tile boundary). Positioning works as for the combined photo via the
    .tfw georeferencing; each photo is scaled from the native resolution (0.2 m/px) to target_pixel_size.

    Args:
        photos: [{"name": "aerial_photo_0", "bounds": (x_min, x_max, y_min, y_max)}] in local coordinates
        global_offset: (utm_x, utm_y, ...) for converting the source image origins to local

    Returns:
        Number of saved photos
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE

    from ..geometry.coordinates import get_source_crs_epsg

    dst_epsg = get_source_crs_epsg()

    images = [(n, d, i) for n, d, i in extract_georeferenced_images(aerial_dir) if i is not None]
    if not images:
        logger.error("  [!] No georeferenced aerial photos found")
        return 0

    offset_x, offset_y = global_offset[:2]
    native = abs(images[0][2]["pixel_size_x"])

    canvases = []
    for photo in photos:
        x_min, x_max, y_min, y_max = photo["bounds"]
        size = (max(1, round((x_max - x_min) / native)), max(1, round((y_max - y_min) / native)))
        canvases.append(Image.new("RGB", size, (70, 95, 55)))  # muted green for gaps
        logger.info(f"  [i] Building {photo['name']}: {x_max - x_min:.0f}m x {y_max - y_min:.0f}m @ {native}m/px = {size[0]}x{size[1]}px -> {target_pixel_size}px")

    for img_name, img_data, world_info in images:
        try:
            image, world_info = _prepare_image_for_compositing(img_data, world_info, dst_epsg)
            image = enhance_dop20_image(image)
            pixel_size = abs(world_info["pixel_size_x"])
            if not math.isclose(pixel_size, native, rel_tol=1e-6):
                scale = pixel_size / native
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS
                )
            img_x = world_info["x_origin"] - offset_x  # .tfw origin = upper left (northwest) pixel corner
            img_y = world_info["y_origin"] - offset_y
            for photo, canvas in zip(photos, canvases):
                x_min, _, _, y_max = photo["bounds"]
                px = round((img_x - x_min) / native)
                py = round((y_max - img_y) / native)
                if px >= canvas.width or py >= canvas.height or px + image.width <= 0 or py + image.height <= 0:
                    continue  # source image lies outside this tile
                canvas.paste(image, (px, py))  # PIL clips at the edges
        except Exception as e:
            logger.error(f"  [!] Error processing {img_name}: {e}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for photo, canvas in zip(photos, canvases):
        canvas.resize((target_pixel_size, target_pixel_size), Image.Resampling.LANCZOS).save(
            Path(output_dir) / f"{photo['name']}.png", "PNG"
        )
        saved += 1
    logger.info(f"  [OK] {saved} tile aerial photos saved")
    return saved


def _aerial_source_files(aerial_dir):
    """ZIPs AND loose raster files (*.tif/*.tiff) - both count as source images (see extract_georeferenced_images())."""
    p = Path(aerial_dir)
    return sorted(p.glob("*.zip")) + sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))


def aerial_photos_signature(aerial_dir, photos, global_offset, target_pixel_size=None):
    """
    Describes WHAT the aerial photos were built for: which photos (name + area), origin, resolution, source images.

    Without this information the exporter does not detect outdated photos - e.g. the 2 km photo of a single
    DGM1 tile, which would simply be stretched to double the area after switching to four tiles (4 km).
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE
    sources = [[path.name, path.stat().st_size] for path in _aerial_source_files(aerial_dir)]
    return {
        "version": AERIAL_SIGNATURE_VERSION,
        "photos": [{"name": p["name"], "bounds": [round(float(v), 3) for v in p["bounds"]]} for p in photos],
        "global_offset": [round(float(v), 3) for v in global_offset[:2]],
        "target_pixel_size": int(target_pixel_size),
        "sources": sources,
    }


def write_aerial_photo_signature(output_dir, signature):
    (Path(output_dir) / AERIAL_SIGNATURE_FILENAME).write_text(json.dumps(signature, indent=2), encoding="utf-8")


def aerial_photo_is_current(output_dir, signature):
    """True if ALL photos exist and were built with exactly this signature (photos without a signature count as outdated)."""
    signature_file = Path(output_dir) / AERIAL_SIGNATURE_FILENAME
    if not signature_file.exists():
        return False
    photos = signature.get("photos") or [{"name": SINGLE_PHOTO_NAME}]
    if not all((Path(output_dir) / f"{p['name']}.png").exists() for p in photos):
        return False
    try:
        return json.loads(signature_file.read_text(encoding="utf-8")) == signature
    except (OSError, ValueError):
        return False


def _remove_stale_photos(output_dir, keep_names):
    """Removes photos of the respective other mode (aerial_photo.png or aerial_photo_<k>.png) - about 130 MB each."""
    import re

    for path in Path(output_dir).glob("aerial_photo*.png"):
        if re.fullmatch(r"aerial_photo(_\d+)?\.png", path.name) and path.stem not in keep_names:
            path.unlink()
            logger.info(f"  [i] Removed outdated aerial photo: {path.name}")


def ensure_aerial_photos(aerial_dir, output_dir, photos, global_offset, target_pixel_size=None):
    """
    Builds the aerial photos only if they are missing or do not match the current area/tile layout.

    Args:
        photos: [{"name", "bounds"}]; a single entry "aerial_photo" = combined photo, otherwise one photo per tile

    Returns:
        "current" (matches, nothing to do), "built" (newly built), "failed" (build failed)
        or "none" (no source images - existing photos remain unchanged)
    """
    if not Path(aerial_dir).exists() or not _aerial_source_files(aerial_dir):
        return "none"

    signature = aerial_photos_signature(aerial_dir, photos, global_offset, target_pixel_size)
    names = {p["name"] for p in photos}
    if aerial_photo_is_current(output_dir, signature):
        _remove_stale_photos(output_dir, names)
        return "current"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if len(photos) == 1 and photos[0]["name"] == SINGLE_PHOTO_NAME:
        built = process_aerial_images(aerial_dir, output_dir, photos[0]["bounds"], global_offset, target_pixel_size)
    else:
        built = process_aerial_tiles(aerial_dir, output_dir, photos, global_offset, target_pixel_size)
    if built <= 0:
        return "failed"
    write_aerial_photo_signature(output_dir, signature)
    _remove_stale_photos(output_dir, names)
    return "built"


MINIMAP_SUBDIR = "minimap"
MINIMAP_FILENAME = "terrain.png"


def build_minimap_image(textures_dir, output_path, photos, terrain_bounds_local, target_pixel_size=None):
    """
    Builds the BigMap preview image (info.json field "minimap") from the already built aerial photo PNGs
    (aerial_photo*.png in textures_dir, see ensure_aerial_photos()) - does not read the source images again,
    but only places the finished photos, scaled down, on a common canvas.

    Same convention as process_aerial_images()/process_aerial_tiles(): row 0 = north, canvas origin
    = (terrain_bounds_local[0], terrain_bounds_local[3]) = (x_min, y_max), muted green for gaps.

    Args:
        textures_dir: Directory with the finished aerial_photo*.png (config.BEAMNG_DIR_TEXTURES)
        output_path: Target PNG path
        photos: [{"name", "bounds": (x_min, x_max, y_min, y_max)}] - same list as passed to ensure_aerial_photos()
        terrain_bounds_local: (x_min, x_max, y_min, y_max) of the ENTIRE terrain area in local coordinates
        target_pixel_size: Edge length (pixels) of the minimap (default: config.MINIMAP_PIXEL_SIZE)

    Returns:
        True on success, False if a source photo is missing (no exception - the minimap is optional)
    """
    if target_pixel_size is None:
        target_pixel_size = config.MINIMAP_PIXEL_SIZE

    x_min, x_max, y_min, y_max = terrain_bounds_local
    width_m, height_m = x_max - x_min, y_max - y_min
    if width_m <= 0 or height_m <= 0:
        return False

    px_per_m_x = target_pixel_size / width_m
    px_per_m_y = target_pixel_size / height_m
    canvas = Image.new("RGB", (target_pixel_size, target_pixel_size), (70, 95, 55))  # muted green for gaps

    source_paths = [Path(textures_dir) / f"{photo['name']}.png" for photo in photos]
    if not all(path.exists() for path in source_paths):
        return False

    def load_tile(photo, source_path):
        bx_min, bx_max, by_min, by_max = photo["bounds"]
        tile_w = max(1, round((bx_max - bx_min) * px_per_m_x))
        tile_h = max(1, round((by_max - by_min) * px_per_m_y))
        # reducing_gap: first shrink by an integer factor with a box filter, then LANCZOS to the target size -
        # about 8x faster than LANCZOS over the full 8192 photo, equally sharp at minimap resolution
        with Image.open(source_path) as source:
            tile = source.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS, reducing_gap=3.0)
        px = round((bx_min - x_min) * px_per_m_x)
        py = round((y_max - by_max) * px_per_m_y)
        return tile, (px, py)

    # PNG decoding (about 1.4 s per 8192 photo) releases the GIL - threads really run in parallel.
    # At most 4 at a time: each decoded photo occupies about 200 MB.
    workers = max(1, min(4, len(photos), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for tile, position in executor.map(load_tile, photos, source_paths):
            canvas.paste(tile, position)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")
    return True


MINIMAP_SIGNATURE_FILENAME = "aerial_minimap.json"
MINIMAP_SIGNATURE_VERSION = 1


def minimap_signature(textures_dir, photos, terrain_bounds_local, target_pixel_size=None):
    """
    Describes WHAT the minimap was built from: photos (name + area), terrain area, resolution and
    size + modification time of each source photo - a newly built aerial photo thus automatically makes the minimap outdated.
    """
    if target_pixel_size is None:
        target_pixel_size = config.MINIMAP_PIXEL_SIZE
    sources = []
    for photo in photos:
        path = Path(textures_dir) / f"{photo['name']}.png"
        stat = path.stat() if path.exists() else None
        sources.append([path.name, stat.st_size if stat else None, stat.st_mtime_ns if stat else None])
    return {
        "version": MINIMAP_SIGNATURE_VERSION,
        "photos": [{"name": p["name"], "bounds": [round(float(v), 3) for v in p["bounds"]]} for p in photos],
        "terrain_bounds": [round(float(v), 3) for v in terrain_bounds_local],
        "target_pixel_size": int(target_pixel_size),
        "sources": sources,
    }


def ensure_minimap_image(textures_dir, output_path, photos, terrain_bounds_local, target_pixel_size=None):
    """
    Builds the minimap only if it is missing or no longer matches the aerial photos/area (see minimap_signature()).
    The signature is stored next to aerial_photo.json in textures_dir, not in the level's minimap folder.

    Returns:
        "current" (matches, nothing to do), "built" (newly built) or "missing" (source photo missing)
    """
    signature = minimap_signature(textures_dir, photos, terrain_bounds_local, target_pixel_size)
    signature_file = Path(textures_dir) / MINIMAP_SIGNATURE_FILENAME
    if Path(output_path).exists() and signature_file.exists():
        try:
            if json.loads(signature_file.read_text(encoding="utf-8")) == signature:
                return "current"
        except (OSError, ValueError):
            pass

    if not build_minimap_image(textures_dir, output_path, photos, terrain_bounds_local, target_pixel_size):
        return "missing"
    signature_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return "built"


def minimap_info_json_fields(x_min, y_max, size_m, relative_file=None):
    """
    info.json fields for the minimap: "size" (terrain extent) and "minimap" (image + position).

    Args:
        x_min, y_max: Northwest corner of the terrain area in local coordinates (= image origin, row 0 = north)
        size_m: Edge length of the (square) terrain area in meters
        relative_file: Path relative to the level root (default: "{MINIMAP_SUBDIR}/{MINIMAP_FILENAME}")

    Returns:
        {"size": [...], "minimap": [...]} for merging in ItemManager.set_info_json_fields()
    """
    file = relative_file or f"{MINIMAP_SUBDIR}/{MINIMAP_FILENAME}"
    return {
        "size": [size_m, size_m],
        "minimap": [{"file": file, "size": [size_m, size_m], "offset": [x_min, y_max]}],
    }


POI_PREVIEW_SUBDIR = "spawn_previews"


def _photo_containing(photos, xy):
    """Photo tile whose bounds contain xy - otherwise the one with the nearest center (fallback for
    a POI right at the tile edge/just outside due to rounding)."""
    x, y = xy
    for photo in photos:
        bx_min, bx_max, by_min, by_max = photo["bounds"]
        if bx_min <= x <= bx_max and by_min <= y <= by_max:
            return photo

    def _center_dist(photo):
        bx_min, bx_max, by_min, by_max = photo["bounds"]
        return math.hypot(x - (bx_min + bx_max) / 2.0, y - (by_min + by_max) / 2.0)

    return min(photos, key=_center_dist) if photos else None


def _load_rgb_photo(source_path: Path, image_cache: Optional[Dict[Path, "Image.Image"]]) -> "Image.Image":
    """Loads an aerial photo as RGB (fully decoded), optionally reused via `image_cache`.

    For a larger export, the composed aerial photo PNGs are often >100 MB; `.convert("RGB")`
    ALWAYS decodes the entire image (PNG does not support partial decoding), regardless of the
    later crop. Without a cache, every call (e.g. per POI preview image, up to
    config.MAX_POI_SPAWN_POINTS times for the same photo tile) pays this decoding cost again.
    """
    if image_cache is not None and source_path in image_cache:
        cached = image_cache[source_path]
        # Future: decoded in advance in the background by PoiPreviewBuilder - errors arrive here as OSError
        return cached.result() if isinstance(cached, Future) else cached
    rgb = _decode_rgb_photo(source_path)
    if image_cache is not None:
        image_cache[source_path] = rgb
    return rgb


def _decode_rgb_photo(source_path: Path) -> "Image.Image":
    with Image.open(source_path) as img:
        return img.convert("RGB")  # standalone copy, independent of the file handle


def build_poi_preview_image(
    textures_dir, output_path, photos, position_xy, crop_size_m=None, target_pixel_size=None,
    image_cache: Optional[Dict[Path, "Image.Image"]] = None,
):
    """
    Preview image for a POI spawn point (info.json spawnPoints[].preview, see
    lua/ge/extensions/core/levels.lua): square crop from the already built aerial photo,
    top-down view, POI centered - does not read any source image again, only the finished
    aerial_photo*.png (see ensure_aerial_photos()), same convention as build_minimap_image() (row 0 = north).

    Args:
        textures_dir: Directory with the finished aerial_photo*.png (config.BEAMNG_DIR_TEXTURES)
        output_path: Target image path (.jpg)
        photos: [{"name", "bounds": (x_min, x_max, y_min, y_max)}] - same list as passed to ensure_aerial_photos()
        position_xy: (x, y) of the POI in local coordinates
        crop_size_m: Edge length (meters) of the crop (default: config.POI_PREVIEW_CROP_SIZE_M)
        target_pixel_size: Edge length (pixels) of the saved image (default: config.POI_PREVIEW_PIXEL_SIZE)
        image_cache: optional dict {path: already decoded RGB image}, kept open by the caller across
            multiple calls (see PoiPreviewBuilder) - saves repeated decoding of the same image
            when several POIs lie on the same photo tile.

    Returns:
        True on success, False if no matching photo tile could be found/read (the
        preview image is optional - BeamNG otherwise falls back to the level preview image)
    """
    if crop_size_m is None:
        crop_size_m = config.POI_PREVIEW_CROP_SIZE_M
    if target_pixel_size is None:
        target_pixel_size = config.POI_PREVIEW_PIXEL_SIZE

    photo = _photo_containing(photos, position_xy)
    if photo is None:
        return False
    source_path = Path(textures_dir) / f"{photo['name']}.png"
    if not source_path.exists():
        return False

    bx_min, bx_max, by_min, by_max = photo["bounds"]
    width_m, height_m = bx_max - bx_min, by_max - by_min
    if width_m <= 0 or height_m <= 0:
        return False

    x, y = position_xy
    half = crop_size_m / 2.0
    try:
        source = _load_rgb_photo(source_path, image_cache)
        px_per_m_x = source.width / width_m
        px_per_m_y = source.height / height_m

        left = (x - half - bx_min) * px_per_m_x
        right = (x + half - bx_min) * px_per_m_x
        top = (by_max - (y + half)) * px_per_m_y  # row 0 = north
        bottom = (by_max - (y - half)) * px_per_m_y

        # Clamp to the image border (POI near the tile edge): the crop stays within the image, is then
        # just no longer exactly centered - better than an empty/truncated preview image.
        left, right = max(0.0, left), min(float(source.width), right)
        top, bottom = max(0.0, top), min(float(source.height), bottom)
        if right - left < 2 or bottom - top < 2:
            return False

        crop = source.crop((round(left), round(top), round(right), round(bottom)))
        crop = crop.resize((target_pixel_size, target_pixel_size), Image.Resampling.LANCZOS)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, "JPEG", quality=85)
    except OSError as exc:
        logger.debug(f"  [POI-Preview] {source_path} skipped: {exc}")
        return False
    return True


POI_PREVIEW_SIGNATURE_FILENAME = "aerial_poi_previews.json"


class PoiPreviewBuilder:
    """
    preview_builder for ItemManager.save(): (object_name, (x, y)) -> preview image path relative to the level root
    or None - see build_poi_preview_image().

    A preview image is only re-cropped if it is missing or the position, crop or the
    source photo (size + modification time) have changed; the signatures are stored next to aerial_photo.json in
    textures_dir. If a crop is needed after all, it decodes ALL photo tiles on the first miss
    in parallel in the background (about 1.4 s per 8192 PNG, otherwise the main time sink of the step when serial).

    Call close() after the last call: it frees the decoded photos (about 200 MB each) and
    writes the signatures.
    """

    def __init__(self, textures_dir, level_dir, photos):
        self.textures_dir = Path(textures_dir)
        self.level_dir = Path(level_dir)
        self.photos = photos
        self.reused = 0
        self.built = 0
        self._image_cache: Dict[Path, object] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._signature_file = self.textures_dir / POI_PREVIEW_SIGNATURE_FILENAME
        try:
            self._previous = json.loads(self._signature_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._previous = {}
        self._current: Dict[str, dict] = {}

    def _signature(self, position_xy):
        photo = _photo_containing(self.photos, position_xy)
        if photo is None:
            return None
        path = self.textures_dir / f"{photo['name']}.png"
        if not path.exists():
            return None
        stat = path.stat()
        return {
            "photo": photo["name"],
            "bounds": [round(float(v), 3) for v in photo["bounds"]],
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "position": [round(float(v), 2) for v in position_xy[:2]],
            "crop_size_m": float(config.POI_PREVIEW_CROP_SIZE_M),
            "pixel_size": int(config.POI_PREVIEW_PIXEL_SIZE),
        }

    def _prefetch_photos(self):
        """Decode all photo tiles in parallel (at most 4 at a time, decoding releases the GIL)."""
        if self._executor is not None:
            return
        paths = [self.textures_dir / f"{p['name']}.png" for p in self.photos]
        paths = [p for p in paths if p.exists() and p not in self._image_cache]
        self._executor = ThreadPoolExecutor(max_workers=max(1, min(4, len(paths), os.cpu_count() or 1)))
        for path in paths:
            self._image_cache[path] = self._executor.submit(_decode_rgb_photo, path)

    def __call__(self, object_name: str, position_xy) -> Optional[str]:
        relative_path = f"{POI_PREVIEW_SUBDIR}/{object_name}.jpg"
        output_path = self.level_dir / relative_path
        signature = self._signature(position_xy)
        if signature is None:
            return None
        if output_path.exists() and self._previous.get(object_name) == signature:
            self._current[object_name] = signature
            self.reused += 1
            return relative_path

        self._prefetch_photos()
        if not build_poi_preview_image(
            self.textures_dir, output_path, self.photos, position_xy, image_cache=self._image_cache
        ):
            return None
        self._current[object_name] = signature
        self.built += 1
        return relative_path

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._image_cache.clear()
        if self._current != self._previous:
            self._signature_file.write_text(json.dumps(self._current, indent=2), encoding="utf-8")
