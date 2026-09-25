"""
Builds the .ter layer map (material index per raster cell) from two sources:

1. Aerial photo fallback: ONE photo material (AERIAL_PHOTO_MATERIAL_NAME) for the
   whole area - io/aerial.py assembles the overall photo itself from all
   DOP20 source images (see research 2026-09-18: many unique
   500m-tile materials overwhelm BeamNG's terrain atlas packer).
2. OSM land use: polygons from data/osm_to_beamng.json["landuse_mappings"]
   are burned into the layer map by priority (spec section 6).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from affine import Affine
from PIL import Image
from rasterio.features import rasterize

from ..osm.landuse_polygons import AREA_TAG_KEYS

# Flat placeholders for the mandatory texture slots that BeamNG's v1.5 terrain
# material system requires for EVERY layer (see ensure_flat_pbr_placeholders()).
# Values follow the PBR "neutral" convention: normal points straight up,
# roughness/height are mid-gray (neutral), AO is white (no additional
# shading), baseColor is arbitrary (always used with Strength=0).
_FLAT_PLACEHOLDER_COLORS = {
    "baseColor": (128, 128, 128),
    "normal": (128, 128, 255),
    "roughness": (128, 128, 128),
    "ao": (255, 255, 255),
    "height": (128, 128, 128),
}

# Pixel size of the detail textures: must exactly match the detailTexSize of the
# TerrainMaterialTextureSet (see build_terrain_material_texture_set()).
DETAIL_TEX_SIZE = 1024

# Strength with which the gray detail texture is laid over the aerial photo
# (BeamNG's own grass materials use 0.15-0.2, see east_coast_usa).
DEFAULT_DETAIL_STRENGTH = 0.25

EMPTY_RASTER_VALUE = 255

# Fallback for areas that do have a landuse/natural/leisure tag but match no (active) category:
# generic grass instead of an unpainted photo remainder (see get_landuse_category()). Categories with
# "active": false (e.g. regions like natural=mountain_range, which are not a real ground surface) are
# excluded - they are deliberately known, but never a terrain layer, not even the generic default.
DEFAULT_LANDUSE_CATEGORY = "meadow"


def get_landuse_category(osm_tags: Dict, landuse_mappings: Dict) -> Optional[str]:
    """
    Determines the landuse_mappings category for an OSM element.

    Each category lists its associated tag values under "osm_tags", e.g.
    {"landuse": ["meadow", "grass"], "natural": ["grassland"]}. If several
    categories match (e.g. landuse=meadow + natural=wood), the one with the
    highest "priority" wins. Categories without "osm_tags" (e.g. the "base"
    fallback entry) are never assigned. "osm_exclude_tags" (same format)
    excludes elements again, e.g. water without the dry retention
    basins (basin=detention).

    If no active category matches, but the tag value is also not listed explicitly
    anywhere (not even as "active": false), DEFAULT_LANDUSE_CATEGORY applies
    (see there) - an area with an unknown landuse-like tag thus becomes
    generic grass instead of simply staying unpainted (photo remainder).

    Returns:
        Category name or None if there is no match (or explicitly excluded)
    """
    best_category = None
    best_priority = None
    excluded = False
    for category, data in landuse_mappings.items():
        category_tags = data.get("osm_tags")
        if not category_tags:
            continue
        if not any(osm_tags.get(key) in values for key, values in category_tags.items()):
            continue
        if data.get("active", True) is False:
            excluded = True
            continue
        if any(osm_tags.get(key) in values for key, values in data.get("osm_exclude_tags", {}).items()):
            continue
        priority = data.get("priority", 0)
        if best_priority is None or priority > best_priority:
            best_category, best_priority = category, priority
    has_area_tag = any(key in osm_tags for key in AREA_TAG_KEYS)
    if best_category is None and not excluded and has_area_tag and DEFAULT_LANDUSE_CATEGORY in landuse_mappings:
        return DEFAULT_LANDUSE_CATEGORY
    return best_category


AERIAL_PHOTO_MATERIAL_NAME = "aerial_photo"
PHOTO_GROUND_MODEL = "ASPHALT"  # groundmodels.json only knows UPPERCASE names; matches the previous engine fallback


def build_photo_fallback_layer(size: int) -> Tuple[np.ndarray, List[str]]:
    """
    Builds the base layer map: the ENTIRE area gets ONE single aerial photo
    material (AERIAL_PHOTO_MATERIAL_NAME), no longer a separate material
    per 500m tile.

    Background (research 2026-09-18): BeamNG's v1.5 terrain material system
    is designed for a small number of repeating materials, not for
    many (16-25) unique 4096px textures - the atlas packer displayed
    individual tiles visibly rotated, even though the source files
    were demonstrably correct. io/aerial.py now assembles the aerial photo itself
    into ONE overall photo (see process_aerial_images()); the
    layer map therefore only has to point everywhere to the same material index (0).

    Returns:
        (layer_map, material_names) - layer_map is (size, size) uint8 filled
        with zeros, material_names = [AERIAL_PHOTO_MATERIAL_NAME]
    """
    layer_map = np.zeros((size, size), dtype=np.uint8)
    return layer_map, [AERIAL_PHOTO_MATERIAL_NAME]


def paint_landuse_materials(
    layer_map: np.ndarray,
    material_names: List[str],
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    landuse_polygons: List[Dict],
    landuse_mappings: Dict,
    background_category: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Burns OSM land use polygons into the layer map, prioritized by
    landuse_mappings[category]["priority"] (higher priority wins on
    overlap, see spec section 6/8).

    Args:
        layer_map: (size, size) uint8, is NOT modified (a copy is returned)
        material_names: existing material list (photo fallback names)
        landuse_polygons: List of {"osm_tags": Dict, "geometry": shapely.Polygon}
                          in local (grid) coordinates
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        background_category: optional - category name from landuse_mappings with which the ENTIRE area
            is filled first, BEFORE the polygons are burned (independent of their priority - the
            background does not take part in the priority sorting, every real polygon covers it).
            Closes the gap that get_landuse_category()'s DEFAULT_LANDUSE_CATEGORY fallback leaves open:
            that only applies to an EXISTING but unknown landuse tag value - areas with NO
            land use polygon at all (no OSM element covers them) would otherwise stay on the photo
            fallback forever, even though DEFAULT_LANDUSE_CATEGORY="meadow" is meant to prevent exactly
            that. None (default) = previous behavior, no change. A "keep_photo" category or a name
            unknown in landuse_mappings is a no-op (there is no material that could be painted).

    Returns:
        (new layer_map, extended material_names)
    """
    result = layer_map.copy()
    names = list(material_names)
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(square_size, square_size)

    if background_category is not None:
        bg_data = landuse_mappings.get(background_category)
        if bg_data is not None and not bg_data.get("keep_photo"):
            bg_source = landuse_mappings[bg_data.get("use_material_of", background_category)]
            bg_name = bg_source["internal_name"]
            if bg_name not in names:
                names.append(bg_name)
            result[:] = names.index(bg_name)

    scored = []
    for poly in landuse_polygons:
        category = get_landuse_category(poly["osm_tags"], landuse_mappings)
        if category is None:
            continue
        category_data = landuse_mappings[category]
        # keep_photo categories (commercial areas) have no material of their own: they restore
        # the aerial photo (index 0) over the layers beneath.
        # use_material_of (water): own area with its own priority, but the material
        # of another category (meadow ground beneath the water).
        if category_data.get("keep_photo"):
            internal_name = None
        else:
            material_source = landuse_mappings[category_data.get("use_material_of", category)]
            internal_name = material_source["internal_name"]
        scored.append((category_data.get("priority", 0), poly["geometry"], internal_name))

    # Sort ascending by priority -> high priority is burned last (on top)
    scored.sort(key=lambda item: item[0])

    for _priority, geometry, internal_name in scored:
        if internal_name is None:
            material_index = 0
        else:
            if internal_name not in names:
                if len(names) >= 254:
                    raise ValueError(
                        f"More than 254 materials ({len(names)} already present, "
                        f"another land use category '{internal_name}' would exceed the "
                        f"limit) - reduce the land use categories"
                    )
                names.append(internal_name)
            material_index = names.index(internal_name)

        _burn_geometry(result, geometry, material_index, transform, size)

    return result, names


def _burn_geometry(target: np.ndarray, geometry, value: int, transform: Affine, size: int) -> None:
    """
    Burns a geometry into `target` (size x size), only within the window of its bounding box.

    Same result as rasterizing over the whole map, but per polygon only as large as the polygon itself
    (with 4096² cells and over a thousand polygons otherwise several seconds of idle work).
    """
    if geometry is None or geometry.is_empty:
        return
    inverse = ~transform
    x0, y0, x1, y1 = geometry.bounds
    cols = [inverse * (x0, y0), inverse * (x1, y1)]
    col_lo = max(0, int(np.floor(min(c[0] for c in cols))) - 1)
    col_hi = min(size, int(np.ceil(max(c[0] for c in cols))) + 2)
    row_lo = max(0, int(np.floor(min(c[1] for c in cols))) - 1)
    row_hi = min(size, int(np.ceil(max(c[1] for c in cols))) + 2)
    if col_lo >= col_hi or row_lo >= row_hi:
        return
    window = rasterize(
        [(geometry, value)],
        out_shape=(row_hi - row_lo, col_hi - col_lo),
        transform=transform * Affine.translation(col_lo, row_lo),
        fill=EMPTY_RASTER_VALUE,
        dtype="uint8",
    )
    hit = window != EMPTY_RASTER_VALUE
    view = target[row_lo:row_hi, col_lo:col_hi]
    view[hit] = window[hit]


def mask_layer_map_with_photo(
    layer_map: np.ndarray,
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    geometries: List,
    buffer: float = 0.0,
) -> np.ndarray:
    """
    Resets the layer map under the geometries to the aerial photo (index 0).

    Reason: ground vegetation (GroundCover) grows on the terrain LAYER. Roads are
    decals above the terrain - without this masking, grass would grow through roads and
    buildings as soon as a land use layer lies beneath. Nothing grows on the
    photo layer.

    Args:
        geometries: shapely geometries in local coordinates (e.g. road surfaces,
            building footprints)
        buffer: Buffer in meters around each geometry (e.g. road shoulder)

    Returns:
        New layer_map (input remains unchanged)
    """
    result = layer_map.copy()
    shapes = []
    for geometry in geometries:
        if buffer:
            geometry = geometry.buffer(buffer)
        if geometry is not None and not geometry.is_empty:
            shapes.append((geometry, 1))
    if not shapes:
        return result

    transform = Affine.translation(origin_x, origin_y) * Affine.scale(square_size, square_size)
    mask = rasterize(shapes, out_shape=(size, size), transform=transform, fill=0, dtype="uint8")
    result[mask == 1] = 0
    return result


def mark_padding_as_holes(layer_map: np.ndarray, data_cols: int, data_rows: int) -> np.ndarray:
    """
    Marks the padded excess border of the power-of-two heightmap as terrain hole.

    The .ter size is a power of two (e.g. 2048), the real data is smaller (2001).
    The border beyond the data is mere extrapolation - as a hole (layer 255) neither
    terrain is rendered nor collides there; the horizon covers this strip
    (terrain/horizon_seam.py). The visible terrain thus ends exactly at the data border.

    Args:
        layer_map: (size, size) layer indices, layer_map[row, col]
        data_cols, data_rows: Number of real data columns (x) and rows (y), respectively

    Returns:
        New layer_map (input remains unchanged)
    """
    from .ter_writer import EMPTY_LAYER_VALUE

    result = layer_map.copy()
    result[:, data_cols:] = EMPTY_LAYER_VALUE
    result[data_rows:, :] = EMPTY_LAYER_VALUE
    return result


DETAIL_TEXTURE_KEYS = ("detailColorMap", "detailNormalMap")


def ensure_landuse_detail_textures_sized(
    landuse_mappings: Dict,
    detail_tex_size: int,
    beamng_dir: Path,
    textures_dir: Path,
    level_name: str,
) -> Dict:
    """
    Scales detailColorMap/detailNormalMap of all active land use categories
    to detail_tex_size and returns a copy of landuse_mappings with the
    (possibly new) paths.

    Reason: baseColorDetailTex/normalDetailTex of ALL TerrainMaterial entries
    must have exactly the detailTexSize of the TerrainMaterialTextureSet -
    otherwise BeamNG reports "dont have required size" and renders the
    "warning texture" (uniformly gray ground) for the ENTIRE material, see
    research 2026-09-18. BeamNG's terrain textures are already 1024 px in size;
    the scaling only takes effect for deviating sizes.

    Only level-local textures (path starts with "levels/{level_name}/",
    i.e. copied here by io/beamng_assets.py::ensure_shared_textures()) are
    touched; other paths remain unchanged.
    """
    target_size = (detail_tex_size, detail_tex_size)
    level_prefix = f"levels/{level_name}/"
    textures_dir.mkdir(parents=True, exist_ok=True)
    result: Dict = {}

    for category, data in landuse_mappings.items():
        data = dict(data)
        if data.get("active", True) is not False:
            for key in DETAIL_TEXTURE_KEYS:
                rel_path = data.get(key)
                if not rel_path or not rel_path.lstrip("/").startswith(level_prefix):
                    continue
                fs_path = beamng_dir / Path(rel_path.lstrip("/")).relative_to(level_prefix)
                if not fs_path.is_file():
                    continue
                with Image.open(fs_path) as img:
                    if img.size == target_size:
                        continue
                    resized_name = f"_terrain_detail_{fs_path.stem}_{detail_tex_size}.png"
                    resized_path = textures_dir / resized_name
                    if not resized_path.exists():
                        img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS).save(resized_path, "PNG")
                data[key] = f"/levels/{level_name}/art/shapes/textures/{resized_name}"
        result[category] = data

    return result


def ensure_flat_pbr_placeholders(
    textures_dir: Path,
    level_name: str,
    base_tex_size: int,
    detail_tex_size: int = 1024,
    macro_tex_size: int = 1024,
) -> Dict[str, Dict[str, str]]:
    """
    Creates (once) flat placeholder PNGs for the mandatory texture slots and
    returns their level paths, nested by tier (base/detail/macro).

    Reason: according to the official docs, BeamNG's v1.5 terrain material editor
    (https://documentation.beamng.com/modding/levels/level_formats/terrain/)
    saves NO TerrainMaterial with an empty texture slot - all 5 channels (baseColor,
    normal, roughness, ao, height) need base, detail AND macro textures,
    otherwise BeamNG renders the material as "warning texture" (uniformly
    dark gray ground). We only have real baseColor base data (aerial photo/
    land use texture) - for the remaining slots neutral placeholders suffice,
    whose detail/macro share is additionally muted via *DetailStrength/*MacroStrength=0
    (see build_terrain_material_entries()).

    IMPORTANT: The placeholders must have exactly the pixel size per tier declared in
    the TerrainMaterialTextureSet (baseTexSize/detailTexSize/
    macroTexSize) - otherwise BeamNG logs "dont have required size of W-H" and
    likewise renders the "warning texture" (see research 2026-09-18: a
    generic 8x8 image was NOT enough, even though the texture slot itself was filled).

    Returns:
        {"base": {"normal": "/levels/.../_flat_normal_4096.png", ...},
         "detail": {"baseColor": ..., "normal": ..., ...},
         "macro": {...}}
    """
    textures_dir.mkdir(parents=True, exist_ok=True)
    tier_sizes = {"base": base_tex_size, "detail": detail_tex_size, "macro": macro_tex_size}
    paths: Dict[str, Dict[str, str]] = {}

    for tier, size in tier_sizes.items():
        paths[tier] = {}
        for channel, color in _FLAT_PLACEHOLDER_COLORS.items():
            filename = f"_flat_{channel}_{size}.png"
            filepath = textures_dir / filename
            if not filepath.exists():
                Image.new("RGB", (size, size), color).save(filepath, "PNG")
            paths[tier][channel] = f"/levels/{level_name}/art/shapes/textures/{filename}"

    return paths


def _add_required_pbr_slots(entry: Dict, placeholders: Dict[str, Dict[str, str]]) -> None:
    """
    Adds the 13 mandatory fields to a TerrainMaterial dict (baseColor detail/
    macro as well as normal/roughness/ao/height for each of base+detail+macro) that
    build_terrain_material_entries() has no real data for - see
    ensure_flat_pbr_placeholders().

    The detail/macro share is damped to zero via *Strength=[0, 0], so that
    the neutral placeholder content plays no visible role either way.
    baseColorBaseTex/-Size is already set by the caller.
    """
    zero = [0.0, 0.0]
    # setdefault: real textures that are already set (e.g. the detail texture of a
    # land use category) must not be overwritten by placeholders.
    entry.setdefault("baseColorDetailTex", placeholders["detail"]["baseColor"])
    entry.setdefault("baseColorDetailStrength", zero)
    entry.setdefault("baseColorMacroTex", placeholders["macro"]["baseColor"])
    entry.setdefault("baseColorMacroStrength", zero)

    for channel in ("normal", "roughness", "ao", "height"):
        entry.setdefault(f"{channel}BaseTex", placeholders["base"][channel])
        entry.setdefault(f"{channel}DetailTex", placeholders["detail"][channel])
        entry.setdefault(f"{channel}DetailStrength", zero)
        entry.setdefault(f"{channel}MacroTex", placeholders["macro"][channel])
        entry.setdefault(f"{channel}MacroStrength", zero)


def build_terrain_material_entries(
    material_names: List[str],
    photo_tile_names: List[str],
    landuse_mappings: Dict,
    level_name: str,
    photo_extent_size: float,
    placeholders: Dict[str, str],
    variant_parents: Optional[Dict[str, Tuple[str, str]]] = None,
    photo_extents: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict]:
    """
    Builds TerrainMaterial JSON entries for materials.json (schema verified
    against BeamNG's official docs, see _add_required_pbr_slots()).

    Args:
        material_names: all layer map material names in index order
        photo_tile_names: subset of material_names that refer to the
                          assembled aerial photo (currently only
                          [AERIAL_PHOTO_MATERIAL_NAME], see
                          build_photo_fallback_layer())
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        level_name: for the photo texture path
        photo_extent_size: Edge length (meters) of the entire exported
                          area covered by the ONE aerial photo (no longer
                          a 500m tile size - the photo does not repeat,
                          but covers the complete area once)
        placeholders: from ensure_flat_pbr_placeholders() - mandatory texture slots
                      for which we have no real data
        variant_parents: four-image mode (see terrain/photo_tiles.py): variant -> (layer, photo
                      material of the tile), e.g. {"mat_grass_t1": ("mat_grass", "aerial_photo_1")}. The
                      variant uses the photo of its tile as base color and the detail texture of the layer.
        photo_extents: photo material -> edge length in cells (tile size); without an entry
                      photo_extent_size applies

    Returns:
        {material_name: {...TerrainMaterial JSON...}}
    """
    variant_parents = variant_parents or {}
    photo_extents = photo_extents or {}
    entries: Dict[str, Dict] = {}
    landuse_by_internal_name = {v["internal_name"]: v for v in landuse_mappings.values() if v.get("internal_name")}
    photo_tile_set = set(photo_tile_names)

    for name in material_names:
        if name in photo_tile_set:
            entry = {
                "internalName": name,
                "class": "TerrainMaterial",
                "persistentId": str(uuid4()),
                # .png, not .dds: BeamNG's terrain atlas packer expects a PNG
                # source texture and caches it to DDS itself (see io/aerial.py).
                "baseColorBaseTex": f"/levels/{level_name}/art/shapes/textures/{name}.png",
                "baseColorBaseTexSize": photo_extents.get(name, photo_extent_size),
                # Built-up/unmapped areas; without it BeamNG logs "ground model not found ... using asphalt"
                "groundmodelName": PHOTO_GROUND_MODEL,
            }
            _add_required_pbr_slots(entry, placeholders)
            entries[name] = entry
            continue

        # Four-image mode: variant of a layer with the photo of its tile as base color
        layer_name, photo_name = variant_parents.get(name, (name, photo_tile_names[0]))
        category_data = landuse_by_internal_name.get(layer_name)
        if category_data is None:
            continue

        # Color from the aerial photo (same base texture as the photo material),
        # the land use lies on top as a gray detail texture. BeamNG's
        # terrain textures are detail textures (near-greyscale) - as a base
        # they would yield uniformly gray areas.
        strength = float(category_data.get("detailStrength", DEFAULT_DETAIL_STRENGTH))
        entry = {
            "internalName": name,
            "class": "TerrainMaterial",
            "persistentId": str(uuid4()),
            "baseColorBaseTex": f"/levels/{level_name}/art/shapes/textures/{photo_name}.png",
            "baseColorBaseTexSize": photo_extents.get(photo_name, photo_extent_size),
            "baseColorDetailTex": category_data["detailColorMap"],
            "baseColorDetailStrength": [strength, 0.0],
        }
        if category_data.get("detailNormalMap"):
            entry["normalDetailTex"] = category_data["detailNormalMap"]
            entry["normalDetailStrength"] = [1.0, 0.0]
        if category_data.get("groundModelName"):
            # BeamNG's groundmodels.json only knows UPPERCASE names; without
            # groundmodelName BeamNG logs "ground model not found ... using asphalt".
            entry["groundmodelName"] = str(category_data["groundModelName"]).upper()
        _add_required_pbr_slots(entry, placeholders)
        entries[name] = entry

    return entries


def build_terrain_material_texture_set(name: str, base_tex_size: int = 512) -> Dict[str, Dict]:
    """
    Builds the TerrainMaterialTextureSet entry that TerrainBlock.materialTextureSet
    references. TerrainBlock resolves this name via the SimObject "name" field
    (NOT "internalName" - that is only relevant for TerrainMaterial layer references from the
    .ter file), see the official schema example at
    https://documentation.beamng.com/modding/levels/level_formats/terrain/:
    {"class": "TerrainMaterialTextureSet", "name": "...", "baseTexSize": [w,h], ...}.
    Without the "name" field BeamNG does not find the set ("Failed to find
    TerrainMaterialTextureSet with name: ...") and crashes on the first terrain draw
    with "D3D12: root cbv with 0 gpu va", because the terrain material constants were never
    bound.

    Args:
        name: Name of the set (must match TerrainBlock.materialTextureSet exactly)
        base_tex_size: Atlas resolution in pixels (square) for baseColor textures

    Returns:
        {name: {...TerrainMaterialTextureSet JSON...}}
    """
    return {
        name: {
            "class": "TerrainMaterialTextureSet",
            "name": name,
            "baseTexSize": [base_tex_size, base_tex_size],
            "detailTexSize": [DETAIL_TEX_SIZE, DETAIL_TEX_SIZE],
            "macroTexSize": [1024, 1024],
        }
    }
