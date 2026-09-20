"""
Echtes Wasser für Bäche und Wasserflächen.

In BeamNG sind Wasseroberfläche, Wellen, Unterwasser-Nebel und Auftrieb eigene Objekte - das Luftbild
allein reicht nicht:

- Bäche werden `River`-Splines: Knoten [x, y, z, Breite, Tiefe, nx, ny, nz] entlang der OSM-Linie.
- Teiche/Seen werden `WaterBlock`-Quader (Oberfläche = Position.z, Tiefe nach unten). Ein Block ist
  immer ein Rechteck, das Polygon deshalb mit kleinen Blöcken gekachelt, die es samt einem Rand überdecken.
  Der Spiegel ist der Mittelwert der drei tiefsten Randpunkte; wo das Gelände höher liegt, ist das Wasser
  verdeckt, wo es tiefer liegt (das ganze Loch), sichtbar. Bäche enden am Ufer (cut_line_by_area).

Das DGM1 bleibt bis auf die Teichmulden unverändert (carve_pond_basins: innerhalb des OSM-Polygons 50 cm
tiefer, Böschung 45 Grad nach innen); die Wasserhöhen werden aus dem Gelände abgeleitet. Bäche liegen
knapp über dem Rinnenboden (Wasser nur in der Rinne sichtbar, wo das Gelände dahinter höher ist);
das Wasser fällt flussabwärts nur und verschwindet unter Dämmen/Durchlässen im Gelände.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]
ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]
Bounds = Tuple[float, float, float, float]

UNDERGROUND_TAGS = ("tunnel", "culvert")
NORMAL_UP = [0.0, 0.0, 1.0]


def _parse_width(value: Optional[str]) -> Optional[float]:
    """OSM-`width` wie "4", "3.5 m" oder "2,5" -> Meter oder None."""
    if not value:
        return None
    match = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def select_waterways(osm_data: Sequence[Dict], to_local: ToLocal, widths: Dict[str, float]) -> List[Dict]:
    """
    Sichtbare Bäche/Flüsse aus OSM-Ways: nur Arten aus `widths` (Standardbreite je Art), keine
    unterirdischen Abschnitte (Tunnel/Durchlass). Die Breite kommt aus dem `width`-Tag, sonst dem Standard.

    Returns:
        [{"waterway", "width", "coords": [(x, y), ...]}] in lokalen Koordinaten, Richtung wie in OSM
    """
    result = []
    for element in osm_data:
        tags = element.get("tags") or {}
        kind = tags.get("waterway")
        if element.get("type") != "way" or kind not in widths:
            continue
        if any(tags.get(key) not in (None, "no") for key in UNDERGROUND_TAGS):
            continue
        coords = to_local(element.get("geometry") or [])
        if len(coords) < 2:
            continue
        result.append({"waterway": kind, "width": _parse_width(tags.get("width")) or widths[kind], "coords": coords})
    return result


def clip_line_to_bounds(coords: Sequence[Tuple[float, float]], bounds: Bounds) -> List[List[Tuple[float, float]]]:
    """Schneidet eine Linie auf das Terrain zu; verlässt sie es und kehrt zurück, entstehen mehrere Teile."""
    clipped = LineString(coords).intersection(box(*bounds))
    if clipped.is_empty:
        return []
    parts = list(clipped.geoms) if isinstance(clipped, MultiLineString) else [clipped]
    return [list(part.coords) for part in parts if isinstance(part, LineString) and part.length > 0]


def cut_line_by_area(
    coords: Sequence[Tuple[float, float]], area: Optional[BaseGeometry], min_length: float = 1.0
) -> List[List[Tuple[float, float]]]:
    """
    Schneidet die Teile einer Linie weg, die in `area` (z.B. Teichfläche) liegen: ein Bach endet am Ufer.
    Läuft er durch den Teich, entstehen zwei Stücke; Reststücke unter `min_length` Meter entfallen.
    """
    line = LineString(coords)
    if area is None or area.is_empty:
        return [list(coords)]
    rest = line.difference(area)
    if rest.is_empty:
        return []
    parts = list(rest.geoms) if hasattr(rest, "geoms") else [rest]
    return [list(part.coords) for part in parts if isinstance(part, LineString) and part.length >= min_length]


def _resample(coords: Sequence[Tuple[float, float]], spacing: float) -> np.ndarray:
    line = LineString(coords)
    count = max(2, int(round(line.length / spacing)) + 1)
    distances = np.linspace(0.0, line.length, count)
    return np.array([line.interpolate(d).coords[0] for d in distances])


def _perpendicular(points: np.ndarray) -> np.ndarray:
    tangents = np.gradient(points, axis=0)
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms == 0] = 1.0
    return np.column_stack([-tangents[:, 1], tangents[:, 0]]) / norms[:, None]


def _snap_to_channel(points: np.ndarray, height_at: HeightAt, search: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zieht die Linie seitlich (±search) auf den tiefsten Punkt quer zum Lauf - die Rinne im DGM1. Die OSM-Linie
    liegt oft 1-2 m daneben; dort läge die Wasserfläche verdeckt unter dem Ufer. Die Versätze werden
    geglättet (gleitender Median, dann Mittel), damit der Bach nicht von Rand zu Rand springt.

    Returns:
        (verschobene Punkte, Rinnenboden-Höhe an diesen Punkten)
    """
    perp = _perpendicular(points)
    offsets = np.linspace(-search, search, 9)
    samples = np.stack([np.asarray(height_at(*(points + perp * o).T), dtype=float) for o in offsets])
    best = offsets[np.argmin(samples, axis=0)]

    def smooth(values: np.ndarray, window: int, reducer) -> np.ndarray:
        half = window // 2
        padded = np.pad(values, half, mode="edge")
        return np.array([reducer(padded[i : i + window]) for i in range(len(values))])

    best = smooth(smooth(best, 5, np.median), 3, np.mean)
    snapped = points + perp * best[:, None]
    # Boden an der verschobenen Stelle: Minimum über eine kleine Umgebung quer (fängt Rest-Versatz ab)
    fine = np.linspace(-0.75, 0.75, 4)
    bottom = np.stack([np.asarray(height_at(*(snapped + perp * o).T), dtype=float) for o in fine]).min(axis=0)
    return snapped, bottom


def build_river_nodes(
    coords: Sequence[Tuple[float, float]],
    height_at: HeightAt,
    width: float,
    depth: float,
    spacing: float = 10.0,
    lift: float = 0.2,
    search: float = 2.0,
) -> List[List[float]]:
    """
    River-Knoten [x, y, z, Breite, Tiefe, 0, 0, 1] flussabwärts.

    Die Knoten werden seitlich auf die Rinne im DGM1 gezogen (siehe _snap_to_channel). Wasserhöhe =
    Rinnenboden + `lift`, danach als laufendes Minimum flussabwärts: das Wasser fällt nur.
    Steigt das Gelände (Damm, Straße, Durchlass), bleibt der Wasserspiegel darunter und ist dort unsichtbar.
    Liegt das Ende höher als der Anfang, wird die Linie umgedreht (OSM-Richtung war dann nicht das Gefälle).
    """
    points = _resample(coords, spacing)
    points, bottom = _snap_to_channel(points, height_at, search)
    if bottom[-1] > bottom[0] + 1.0:
        points, bottom = points[::-1], bottom[::-1]
    level = np.minimum.accumulate(bottom + lift)
    return [[float(x), float(y), float(z), float(width), float(depth), *NORMAL_UP] for (x, y), z in zip(points, level)]


def split_nodes(nodes: List[List[float]], max_nodes: int) -> List[List[List[float]]]:
    """Teilt lange Knotenlisten in Stücke <= max_nodes; aufeinanderfolgende Stücke teilen einen Knoten (lückenlos)."""
    if len(nodes) <= max_nodes:
        return [nodes]
    chunks, start = [], 0
    while start < len(nodes) - 1:
        end = min(start + max_nodes, len(nodes))
        chunks.append(nodes[start:end])
        start = end - 1
    return chunks


IDENTITY_ROTATION = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def is_pond_area(tags: Dict[str, str]) -> bool:
    """
    Wasserfläche nach OSM-Tags: natürliches Wasser, Becken oder Stausee. Trockene Hochwasser-Rückhaltebecken
    (`basin=detention`) sind nie Wasser, sondern Wiese (siehe landuse_mappings["meadow"]).
    """
    if tags.get("basin") == "detention":
        return False
    return tags.get("natural") == "water" or tags.get("landuse") in ("basin", "reservoir")


def select_pond_areas(landuse_polygons: Sequence[Dict], bounds: Bounds) -> List[BaseGeometry]:
    """Wasserflächen (siehe is_pond_area) als Geometrien, auf das Terrain (`bounds` = xmin, ymin, xmax, ymax) zugeschnitten."""
    terrain_box = box(*bounds)
    areas = []
    for polygon in landuse_polygons:
        if not is_pond_area(polygon["osm_tags"]):
            continue
        geometry = polygon["geometry"].intersection(terrain_box)
        if not geometry.is_empty and geometry.area > 0:
            areas.append(geometry)
    return areas


def carve_pond_basins(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    areas: Sequence[BaseGeometry],
    depth: float = 0.5,
    slope_deg: float = 45.0,
) -> np.ndarray:
    """
    Legt das Terrain innerhalb der Wasserflächen tiefer: alle Rasterpunkte im Polygon um `depth` Meter, mit einer
    Böschung von `slope_deg` Grad nach innen (Absenkung = Abstand zum Rand * tan(Winkel), höchstens `depth`; bei
    45 Grad also 1 m Absenkung je Meter nach innen, volle Tiefe nach 0,5 m). Außerhalb und auf dem Rand ändert sich
    nichts. Überlappen sich Flächen, gilt je Punkt die größere Absenkung.

    heights[i, j] gehört zur Weltposition (origin_x + j * square_size, origin_y + i * square_size).
    Gibt eine neue Heightmap zurück, `heights` bleibt unverändert.
    """
    from shapely import contains_xy, distance, points

    rows, cols = heights.shape
    lowering = np.zeros_like(heights, dtype=float)
    slope = float(np.tan(np.radians(slope_deg)))
    for area in areas:
        if area is None or area.is_empty:
            continue
        min_x, min_y, max_x, max_y = area.bounds
        j0 = max(0, int(np.floor((min_x - origin_x) / square_size)))
        j1 = min(cols, int(np.ceil((max_x - origin_x) / square_size)) + 1)
        i0 = max(0, int(np.floor((min_y - origin_y) / square_size)))
        i1 = min(rows, int(np.ceil((max_y - origin_y) / square_size)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        xs, ys = np.meshgrid(origin_x + np.arange(j0, j1) * square_size, origin_y + np.arange(i0, i1) * square_size)
        inside = contains_xy(area, xs, ys)
        if not inside.any():
            continue
        edge_distance = distance(points(xs[inside], ys[inside]), area.boundary)
        window = lowering[i0:i1, j0:j1]
        window[inside] = np.maximum(window[inside], np.minimum(depth, edge_distance * slope))
    return heights - lowering


def pond_level(polygon: BaseGeometry, height_at: HeightAt, count: int = 3, rim_step: float = 1.0) -> float:
    """Wasserspiegel: Mittelwert der `count` tiefsten Höhenpunkte auf dem Polygonrand (Punkte alle `rim_step` m)."""
    ring = polygon.exterior
    samples = max(count, int(np.ceil(ring.length / rim_step)))
    points = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0.0, ring.length, samples, endpoint=False)])
    heights = np.sort(np.asarray(height_at(points[:, 0], points[:, 1]), dtype=float))
    return float(heights[:count].mean())


def _cells_covering(shape: BaseGeometry, cell: float) -> List[Tuple[float, float, float, float]]:
    """Rasterzellen der Kantenlänge `cell`, die `shape` berühren, zu Zeilen-Rechtecken verschmolzen."""
    prepared = prep(shape)
    min_x, min_y, max_x, max_y = shape.bounds
    rectangles = []
    y = min_y
    while y < max_y - 1e-9:
        run_start = None
        x = min_x
        while x < max_x - 1e-9:
            hit = prepared.intersects(box(x, y, x + cell, y + cell))
            if hit and run_start is None:
                run_start = x
            if not hit and run_start is not None:
                rectangles.append((run_start, y, x, y + cell))
                run_start = None
            x += cell
        if run_start is not None:
            rectangles.append((run_start, y, x, y + cell))
        y += cell
    return rectangles


def build_pond_blocks(
    polygon: BaseGeometry,
    height_at: HeightAt,
    depth: float = 3.0,
    cell: float = 6.0,
    margin: float = 2.0,
) -> List[Dict]:
    """
    `WaterBlock`-Quader (position = Mitte, Oberfläche auf position.z; scale = Breite, Länge, Tiefe), die das
    Polygon samt `margin` Meter Rand lückenlos überdecken.

    Der Spiegel gilt für das ganze Gewässer: Mittelwert der drei tiefsten Randpunkte (siehe pond_level). Die
    Blöcke reichen über den Rand hinaus, damit das Wasser das ganze Loch füllt, wo das Gelände unter dem
    Spiegel liegt; wo es höher liegt, ist es verdeckt. Die Zellgröße passt sich der Teichgröße an (mindestens
    1 m), damit auch kleine Teiche sauber gefüllt werden.
    """
    if polygon is None or polygon.is_empty:
        return []
    polygon = polygon.buffer(0)
    pieces = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    blocks = []
    for piece in pieces:
        if piece.area < 1.0:
            continue
        level = pond_level(piece, height_at)
        step = min(cell, max(1.0, np.sqrt(piece.area) / 3.0))
        for x0, y0, x1, y1 in _cells_covering(piece.buffer(margin), step):
            blocks.append(
                {
                    "position": [(x0 + x1) / 2.0, (y0 + y1) / 2.0, level],
                    "scale": [x1 - x0, y1 - y0, float(depth)],
                    "rotationMatrix": list(IDENTITY_ROTATION),
                }
            )
    return blocks
