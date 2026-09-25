"""
Four-image mode: one aerial photo per photo tile.

Instead of ONE overall photo (for 4x4 km only 0.5 m/px at 8192 px), each photo tile gets its
own 8192 px photo (2 km -> 0.244 m/px). Each photo is its own terrain material
(`aerial_photo_<k>`), which is painted only in its tile.

The photo tiling is a FIXED grid over the total area (build_processing_tile_grid(),
tile size config.PHOTO_TILE_SIZE_M) - independent of the size/number of the raw elevation data tiles
(which, depending on the source, can be e.g. 1 km instead of 2 km in size, see utils/tile_scanner.py). For LGL
Baden-Württemberg this coincides by chance with the 2x2 km DGM1 ZIPs, but there is no longer any connection.

The land use layers carry the aerial photo as base color. They are therefore kept per tile as a variant
`<layer>_t<k>` (base color = photo of the tile). So that the rest of the pipeline (painting the land use,
masks under roads/buildings, holes) can keep working unchanged on "logical" layers, the layer map
is split into these physical materials only at the VERY END (expand_layers_per_tile).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

HOLE_VALUE = 255  # ter_writer.EMPTY_LAYER_VALUE
MAX_MATERIALS = 254


def build_processing_tile_grid(bbox_utm: Tuple[float, float, float, float], tile_size_m: float) -> List[Dict]:
    """
    Fixed, regular tile grid over a total area - independent of the size/number of the
    raw elevation data tiles that actually supply this area (see module docstring).

    Args:
        bbox_utm: (x_min, x_max, y_min, y_max) of the total area (e.g. utils.tile_scanner.compute_global_bbox())
        tile_size_m: edge length of a tile in meters (config.PHOTO_TILE_SIZE_M)

    Returns:
        List of {"bbox_utm": (x0, x1, y0, y1)} - the same schema as the raw scan tiles, which
        photo_tile_specs() expects. The last row/column is clamped to the actual bbox edge
        instead of being exactly tile_size_m in size - build_tile_index_map() (below) tolerates this,
        since it only works with sorted start coordinates via searchsorted and does not require a
        uniform tile size.
    """
    if tile_size_m <= 0:
        raise ValueError(f"tile_size_m must be positive, is {tile_size_m}")

    x_min, x_max, y_min, y_max = bbox_utm
    x_starts = np.arange(x_min, x_max, tile_size_m) if x_max > x_min else np.array([x_min])
    y_starts = np.arange(y_min, y_max, tile_size_m) if y_max > y_min else np.array([y_min])

    return [
        {"bbox_utm": (float(x0), float(min(x0 + tile_size_m, x_max)), float(y0), float(min(y0 + tile_size_m, y_max)))}
        for y0 in y_starts
        for x0 in x_starts
    ]


def photo_tile_specs(tiles: Sequence[Dict], global_offset: Sequence[float]) -> List[Dict]:
    """
    One photo per tile in stable order (south first, then west first).

    Returns:
        [{"index", "name": "aerial_photo_<k>", "bounds": (x_min, x_max, y_min, y_max) in local coordinates}]
    """
    ox, oy = float(global_offset[0]), float(global_offset[1])
    ordered = sorted(tiles, key=lambda t: (t["bbox_utm"][2], t["bbox_utm"][0]))
    specs = []
    for index, tile in enumerate(ordered):
        x0, x1, y0, y1 = tile["bbox_utm"]
        specs.append({"index": index, "name": f"aerial_photo_{index}", "bounds": (x0 - ox, x1 - ox, y0 - oy, y1 - oy)})
    return specs


def build_tile_index_map(size: int, origin_x: float, origin_y: float, square_size: float, specs: Sequence[Dict]) -> np.ndarray:
    """
    Tile index per heightmap cell (row = y, column = x), shape (size, size).

    The outermost data cell (x/y = tile maximum) and the border padded to a power of two belong to the
    nearest edge tile - no cell is left without a tile.
    """
    xs = origin_x + np.arange(size) * square_size
    ys = origin_y + np.arange(size) * square_size
    x_starts = sorted({s["bounds"][0] for s in specs})
    y_starts = sorted({s["bounds"][2] for s in specs})
    col_tile = np.clip(np.searchsorted(x_starts, xs, side="right") - 1, 0, len(x_starts) - 1)
    row_tile = np.clip(np.searchsorted(y_starts, ys, side="right") - 1, 0, len(y_starts) - 1)

    lookup = np.zeros((len(y_starts), len(x_starts)), dtype=np.int16)
    for spec in specs:
        lookup[y_starts.index(spec["bounds"][2]), x_starts.index(spec["bounds"][0])] = spec["index"]
    return lookup[np.ix_(row_tile, col_tile)]


def expand_layers_per_tile(
    layer_map: np.ndarray,
    material_names: Sequence[str],
    tile_index_map: np.ndarray,
    photo_names: Sequence[str],
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]], Dict[str, Tuple[str, str]]]:
    """
    Splits the logical layer map (index 0 = photo) per tile into physical materials.

    Returns:
        (new layer_map, new material names, variants, parents)
        variants: layer -> its tile variants (only those actually occurring), e.g. {"mat_grass": ["mat_grass_t0", ...]}
        parents: variant -> (layer, photo material of the tile), for the base color of the variant
        Holes (255) stay holes.
    """
    names: List[str] = list(photo_names)  # photo materials first: index k = photo of tile k
    index_of = {name: i for i, name in enumerate(names)}
    variants: Dict[str, List[str]] = {}
    parents: Dict[str, Tuple[str, str]] = {}

    def register(name: str) -> int:
        if name not in index_of:
            if len(names) >= MAX_MATERIALS:
                raise ValueError(
                    f"More than {MAX_MATERIALS} materials after the split per tile ({len(names)} already) - "
                    f"use fewer land use categories or tiles"
                )
            index_of[name] = len(names)
            names.append(name)
        return index_of[name]

    result = np.full(layer_map.shape, HOLE_VALUE, dtype=np.uint8)
    for logical, logical_name in enumerate(material_names):
        in_layer = layer_map == logical
        if not in_layer.any():
            continue
        for tile in np.unique(tile_index_map[in_layer]):
            cells = in_layer & (tile_index_map == tile)
            if logical == 0:
                physical = photo_names[int(tile)]
            else:
                physical = f"{logical_name}_t{int(tile)}"
                variants.setdefault(logical_name, []).append(physical)
                parents[physical] = (logical_name, photo_names[int(tile)])
            result[cells] = register(physical)
    return result, names, variants, parents


def split_layers_by_tile(
    layer_map: np.ndarray,
    material_names: Sequence[str],
    tiles: Sequence[Dict],
    global_offset: Sequence[float],
    origin_x: float,
    origin_y: float,
    square_size: float,
) -> Dict:
    """
    Everything process_tile needs for four-image mode, in one call.

    Returns:
        {"layer_map", "material_names", "photo_tile_names", "photo_extents", "layer_variants",
         "variant_parents", "specs"}; photo_extents = edge length per photo in cells (tile size)
    """
    specs = photo_tile_specs(tiles, global_offset)
    index_map = build_tile_index_map(layer_map.shape[0], origin_x, origin_y, square_size, specs)
    photo_names = [spec["name"] for spec in specs]
    new_map, names, variants, parents = expand_layers_per_tile(layer_map, material_names, index_map, photo_names)
    return {
        "layer_map": new_map,
        "material_names": names,
        "photo_tile_names": photo_names,
        "photo_extents": {s["name"]: (s["bounds"][1] - s["bounds"][0]) / square_size for s in specs},
        "layer_variants": variants,
        "variant_parents": parents,
        "specs": specs,
    }

