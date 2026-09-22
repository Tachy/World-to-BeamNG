"""
Vier-Bilder-Modus: ein Luftbild pro Foto-Kachel.

Statt EINES Gesamtfotos (bei 4x4 km nur 0,5 m/px bei 8192 px) bekommt jede Foto-Kachel ihr
eigenes 8192-px-Foto (2 km -> 0,244 m/px). Jedes Foto ist ein eigenes Terrain-Material
(`aerial_photo_<k>`), das nur in seiner Kachel gemalt wird.

Die Foto-Kachelung ist ein FESTER Raster über die Gesamtfläche (build_processing_tile_grid(),
Kachelgröße config.PHOTO_TILE_SIZE_M) - unabhängig von der Größe/Anzahl der rohen Höhendaten-Kacheln
(die je nach Quelle z.B. 1 km statt 2 km groß sein können, siehe utils/tile_scanner.py). Bei LGL
Baden-Württemberg deckt sich das zufällig mit den 2x2-km-DGM1-ZIPs, ist aber kein Zusammenhang mehr.

Die Landnutzungs-Schichten tragen das Luftbild als Basisfarbe. Sie werden deshalb je Kachel als Variante
`<schicht>_t<k>` geführt (Basisfarbe = Foto der Kachel). Damit die restliche Pipeline (Malen der Landnutzung,
Masken unter Straßen/Gebäuden, Löcher) unverändert auf "logischen" Schichten arbeiten kann, wird die Layer-Map
erst GANZ AM ENDE in diese physischen Materialien aufgeteilt (expand_layers_per_tile).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

HOLE_VALUE = 255  # ter_writer.EMPTY_LAYER_VALUE
MAX_MATERIALS = 254


def build_processing_tile_grid(bbox_utm: Tuple[float, float, float, float], tile_size_m: float) -> List[Dict]:
    """
    Fester, regelmäßiger Kachelraster über eine Gesamtfläche - unabhängig von der Größe/Anzahl der
    rohen Höhendaten-Kacheln, die diese Fläche tatsächlich liefern (siehe Moduldocstring).

    Args:
        bbox_utm: (x_min, x_max, y_min, y_max) der Gesamtfläche (z.B. utils.tile_scanner.compute_global_bbox())
        tile_size_m: Kantenlänge einer Kachel in Metern (config.PHOTO_TILE_SIZE_M)

    Returns:
        Liste von {"bbox_utm": (x0, x1, y0, y1)} - dasselbe Schema wie die rohen Scan-Kacheln, das
        photo_tile_specs() erwartet. Die letzte Zeile/Spalte wird auf die tatsächliche BBox-Kante
        geklemmt statt exakt tile_size_m groß zu sein - build_tile_index_map() (unten) verträgt das,
        da es nur mit sortierten Start-Koordinaten per searchsorted arbeitet, keine einheitliche
        Kachelgröße voraussetzt.
    """
    if tile_size_m <= 0:
        raise ValueError(f"tile_size_m muss positiv sein, ist {tile_size_m}")

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
    Ein Foto je Kachel in stabiler Reihenfolge (Süden zuerst, dann Westen zuerst).

    Returns:
        [{"index", "name": "aerial_photo_<k>", "bounds": (x_min, x_max, y_min, y_max) in lokalen Koordinaten}]
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
    Kachel-Index je Heightmap-Zelle (Zeile = y, Spalte = x), shape (size, size).

    Die äußerste Datenzelle (x/y = Kachel-Maximum) und der auf die Zweierpotenz aufgefüllte Rand gehören zur
    nächstgelegenen Randkachel - keine Zelle bleibt ohne Kachel.
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
    Teilt die logische Layer-Map (Index 0 = Foto) pro Kachel in physische Materialien auf.

    Returns:
        (neue layer_map, neue Materialnamen, variants, parents)
        variants: Schicht -> ihre Kachel-Varianten (nur tatsächlich vorkommende), z.B. {"mat_grass": ["mat_grass_t0", ...]}
        parents: Variante -> (Schicht, Foto-Material der Kachel), für die Basisfarbe der Variante
        Löcher (255) bleiben Löcher.
    """
    names: List[str] = list(photo_names)  # Foto-Materialien zuerst: Index k = Foto der Kachel k
    index_of = {name: i for i, name in enumerate(names)}
    variants: Dict[str, List[str]] = {}
    parents: Dict[str, Tuple[str, str]] = {}

    def register(name: str) -> int:
        if name not in index_of:
            if len(names) >= MAX_MATERIALS:
                raise ValueError(
                    f"Mehr als {MAX_MATERIALS} Materialien nach der Aufteilung pro Kachel ({len(names)} bereits) - "
                    f"weniger Landnutzungs-Kategorien oder Kacheln verwenden"
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
    Alles, was process_tile für den Vier-Bilder-Modus braucht, in einem Aufruf.

    Returns:
        {"layer_map", "material_names", "photo_tile_names", "photo_extents", "layer_variants",
         "variant_parents", "specs"}; photo_extents = Kantenlänge je Foto in Zellen (Kachelgröße)
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

