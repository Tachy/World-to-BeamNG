"""
Vineyard Generator: Rebzeilen für Weinberg-Flächen (landuse=vineyard).

Die Reben sind Forest-Items (grape_vine: ein Zeilensegment, dessen X-Achse die
Zeilenrichtung ist; siehe io/vineyard_assets.py). Pro Weinberg-Polygon
werden gerade, parallele Zeilen erzeugt. Standardmäßig laufen sie entlang der
Falllinie (Steigungsgradient); jedes Segment folgt außerdem der Hangneigung in
Zeilenrichtung und steht aufrecht.

Format pro Instanz (BeamNG .forest4.json Schema, wie ForestInstanceGenerator):
    {"type": "grape_vine", "pos": [x, y, z], "rotationMatrix": [9 Werte, zeilenweise], "scale": 1.0}
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from shapely import contains_xy
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..terrain.terrain_materials import get_landuse_category

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]

GRADIENT_SAMPLE_STEP = 4.0  # Abstand der Stichproben für die Falllinie in Metern
GRADIENT_DIFF_STEP = 1.0  # Schrittweite der zentralen Differenz in Metern
MIN_ROW_FILL = 0.95  # kürzere Zeilenstücke (in Segmentlängen) bleiben leer


def make_height_sampler(heights: np.ndarray, origin_x: float, origin_y: float, square_size: float) -> HeightAt:
    """
    Bilineare Höhenabfrage auf der exportierten Terrain-Heightmap.

    heights[i, j] gehört zur Weltposition (origin_x + j * square_size, origin_y + i * square_size);
    außerhalb des Rasters wird an den Rand geklemmt.
    """
    rows, cols = heights.shape

    def height_at(x, y):
        fx = np.clip((np.asarray(x, dtype=float) - origin_x) / square_size, 0.0, cols - 1.001)
        fy = np.clip((np.asarray(y, dtype=float) - origin_y) / square_size, 0.0, rows - 1.001)
        j0 = fx.astype(int)
        i0 = fy.astype(int)
        tj = fx - j0
        ti = fy - i0
        return (
            heights[i0, j0] * (1 - ti) * (1 - tj)
            + heights[i0 + 1, j0] * ti * (1 - tj)
            + heights[i0, j0 + 1] * (1 - ti) * tj
            + heights[i0 + 1, j0 + 1] * ti * tj
        )

    return height_at


def _canonical(vec: np.ndarray) -> np.ndarray:
    """Achsen haben kein Vorzeichen: eindeutige Richtung mit ux > 0 (bzw. uy > 0 bei ux ≈ 0)."""
    vec = vec / np.linalg.norm(vec)
    if vec[0] < -1e-9 or (abs(vec[0]) <= 1e-9 and vec[1] < 0):
        vec = -vec
    return vec


def _sample_points(polygon: BaseGeometry, step: float) -> np.ndarray:
    min_x, min_y, max_x, max_y = polygon.bounds
    xs = np.arange(min_x + step / 2, max_x, step)
    ys = np.arange(min_y + step / 2, max_y, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    inside = contains_xy(polygon, grid_x, grid_y)
    return np.column_stack([grid_x[inside], grid_y[inside]])


def _gradient_samples(polygon: BaseGeometry, height_at: HeightAt):
    """(gx, gy) des Geländegefälles (m/m) an Stichproben im Polygon oder None."""
    points = _sample_points(polygon, GRADIENT_SAMPLE_STEP)
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    h = GRADIENT_DIFF_STEP
    gx = (height_at(x + h, y) - height_at(x - h, y)) / (2 * h)
    gy = (height_at(x, y + h) - height_at(x, y - h)) / (2 * h)
    return gx, gy


def _fall_line(polygon: BaseGeometry, height_at: HeightAt):
    """(Achse der Falllinie als Einheitsvektor, mittleres Gefälle in m/m) oder None."""
    samples = _gradient_samples(polygon, height_at)
    if samples is None:
        return None
    gx, gy = samples
    # Strukturtensor statt Mittel der Gradienten: Gefälle nach links und rechts (Kuppe)
    # hebt sich im Mittel auf, gehört aber zur selben Achse.
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    return _canonical(axis), float(np.mean(np.hypot(gx, gy)))


def _direction_agreement(polygon: BaseGeometry, height_at: HeightAt, min_slope_percent: float) -> float:
    """
    Wie einheitlich ist die Falllinie im Polygon? 1.0 = überall dieselbe Achse, 0 = wild
    gemischt (mittlere Resultierende der verdoppelten, nach Gefälle gewichteten Winkel).
    Bei fast ebenem Gelände ist die Falllinie bedeutungslos -> 1.0 (kein Teilen nötig).
    """
    samples = _gradient_samples(polygon, height_at)
    if samples is None:
        return 1.0
    gx, gy = samples
    weight = np.hypot(gx, gy)
    if weight.mean() * 100.0 < min_slope_percent:
        return 1.0
    doubled = 2.0 * np.arctan2(gy, gx)
    resultant = np.hypot(np.sum(weight * np.cos(doubled)), np.sum(weight * np.sin(doubled)))
    return float(resultant / np.sum(weight))


def _long_axis(polygon: BaseGeometry) -> np.ndarray:
    """Längsachse des kleinsten umschließenden Rechtecks."""
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        min_x, min_y, max_x, max_y = polygon.bounds
        return _canonical(np.array([1.0, 0.0]) if max_x - min_x >= max_y - min_y else np.array([0.0, 1.0]))
    coords = np.array(rectangle.exterior.coords)
    edge_a, edge_b = coords[1] - coords[0], coords[2] - coords[1]
    return _canonical(edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b)


def compute_row_direction(
    polygon: BaseGeometry, height_at: HeightAt, orientation: str = "gradient", min_slope_percent: float = 2.0
) -> np.ndarray:
    """
    Zeilenrichtung (2D-Einheitsvektor) für ein Weinberg-Polygon.

    orientation="gradient": Zeilen entlang der Falllinie (Steigungsgradient);
    orientation="contour": Zeilen entlang der Höhenlinien. Bei fast ebenem Gelände
    (Gefälle < min_slope_percent) ist die Falllinie unzuverlässig - dann läuft die
    Zeile entlang der Längsachse der Fläche.
    """
    fall = _fall_line(polygon, height_at)
    if fall is None or fall[1] * 100.0 < min_slope_percent:
        return _long_axis(polygon)
    axis = fall[0]
    if orientation == "contour":
        return _canonical(np.array([-axis[1], axis[0]]))
    return axis


def _polygon_parts(geometry: BaseGeometry) -> List[BaseGeometry]:
    return [g for g in getattr(geometry, "geoms", [geometry]) if g.geom_type == "Polygon" and g.area > 1e-6]


def _halve(polygon: BaseGeometry) -> List[BaseGeometry]:
    """Teilt ein Polygon mit einem Schnitt quer zur Längsachse in der Mitte in zwei Teile."""
    a = _long_axis(polygon)
    b = np.array([-a[1], a[0]])
    coords = np.array(polygon.exterior.coords) if polygon.geom_type == "Polygon" else np.array(polygon.envelope.exterior.coords)
    t = coords @ a
    t_lo, t_hi, t_mid = float(t.min()) - 1.0, float(t.max()) + 1.0, float((t.min() + t.max()) / 2.0)
    big = 1e6
    first = Polygon([t_lo * a - big * b, t_mid * a - big * b, t_mid * a + big * b, t_lo * a + big * b])
    second = Polygon([t_mid * a - big * b, t_hi * a - big * b, t_hi * a + big * b, t_mid * a + big * b])
    return _polygon_parts(polygon.intersection(first)) + _polygon_parts(polygon.intersection(second))


def split_by_direction(
    polygon: BaseGeometry,
    height_at: HeightAt,
    max_spread_deg: float,
    min_area: float,
    min_slope_percent: float = 2.0,
    max_depth: int = 6,
) -> List[BaseGeometry]:
    """
    Teilt ein großes Polygon in Blöcke, solange die Falllinie darin zu stark schwankt.

    Gerade Rebzeilen können nur EINER Richtung folgen; bei gekrümmtem Hang weicht
    diese vom lokalen Gefälle ab. Ein Block wird halbiert, wenn die Falllinie im Mittel
    mehr als max_spread_deg von der Hauptachse abweicht und beide Hälften mindestens
    min_area groß bleiben. Jeder Block bekommt danach seine eigene Zeilenrichtung.

    Returns:
        Liste von Polygonen, die das Eingabe-Polygon ohne Überlappung überdecken.
    """
    threshold = float(np.cos(np.radians(2.0 * max_spread_deg)))
    result: List[BaseGeometry] = []
    stack = [(polygon, 0)]
    while stack:
        block, depth = stack.pop()
        if (
            depth >= max_depth
            or block.area < 2.0 * min_area
            or _direction_agreement(block, height_at, min_slope_percent) >= threshold
        ):
            result.append(block)
            continue
        halves = _halve(block)
        if len(halves) < 2:
            result.append(block)
            continue
        stack.extend((half, depth + 1) for half in halves)
    return result


def build_exclusion_geometry(shapes: Sequence[BaseGeometry], margin: float) -> Optional[BaseGeometry]:
    """Vereinigung der um `margin` Meter gepufferten Flächen (Wege, Gebäude) oder None."""
    # Ungültige Polygone (Selbstüberschneidung) reparieren: die Vereinigung würde daran scheitern, die Pufferung
    # je Polygon hat sie bisher stillschweigend bereinigt.
    parts = [shape if shape.is_valid else shape.buffer(0) for shape in shapes if shape is not None and not shape.is_empty]
    # Erst vereinigen, dann einmal puffern (Minkowski-Summe: gleiches Ergebnis, aber deutlich schneller)
    return unary_union(parts).buffer(margin) if parts else None


def _line_parts(geometry: BaseGeometry) -> List[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        parts = []
        for part in geometry.geoms:
            parts.extend(_line_parts(part))
        return parts
    return []


def _rotation_matrices(forward: np.ndarray) -> np.ndarray:
    """
    Zeilenweise 3x3-Rotationsmatrizen als (N, 9)-Array für N normierte Zeilenrichtungen
    (N, 3): ZEILEN = Modell-X (Zeilenrichtung, folgt der Hangneigung), Y, Z (aufrecht,
    senkrecht zu X). BeamNG liest die Achsen als Zeilen; belegt an BeamNGs eigenen
    Weinbergen (italy): dort folgt Zeile 0 zu 98,5 % dem Geländegefälle, Spalte 0 ist
    negativ korreliert. Spalten würden die Neigung invertieren (Reben tauchen in den
    Hang) und die Richtung spiegeln (quer zum Hang).
    """
    up = np.array([0.0, 0.0, 1.0]) - forward[:, 2:3] * forward
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    side = np.cross(up, forward)
    return np.stack([forward, side, up], axis=1).reshape(-1, 9)


def _polygon_instances(polygon: BaseGeometry, height_at: HeightAt, rows: Dict, exclusion, rng) -> List[Dict]:
    orientation = rows.get("orientation", "gradient")
    spacing = float(rows["row_spacing"])
    segment = float(rows["segment_length"])
    scale_min, scale_max = rows.get("scale_range", [1.0, 1.0])

    area = polygon.buffer(-float(rows.get("edge_margin", 0.0)))
    if exclusion is not None:
        area = area.difference(exclusion)
    if area.is_empty:
        return []

    u = compute_row_direction(polygon, height_at, orientation, float(rows.get("min_slope_percent", 2.0)))
    v = np.array([-u[1], u[0]])

    corners = np.array(list(area.envelope.exterior.coords))
    t_all, v_all = corners @ u, corners @ v
    t_min, t_max = float(t_all.min()) - 1.0, float(t_all.max()) + 1.0
    v_min, v_max = float(v_all.min()), float(v_all.max())

    centers = []
    row_count = int(np.floor((v_max - v_min) / spacing)) + 1
    v_start = v_min + ((v_max - v_min) - (row_count - 1) * spacing) / 2.0
    for row in range(row_count):
        v_coord = v_start + row * spacing
        line = LineString([t_min * u + v_coord * v, t_max * u + v_coord * v])
        for part in _line_parts(line.intersection(area)):
            start, end = np.array(part.coords[0]), np.array(part.coords[-1])
            t0, t1 = sorted((float(start @ u), float(end @ u)))
            # Die Zeile reicht bis exakt an Polygonrand bzw. Ausschlusszone (kein Rest an den Enden): die Segmente
            # werden auf die ganze Länge verteilt, Anzahl = nächste ganze Zahl. Nach unten gerundet (Reste > 0,5
            # Segment) rücken sie gleichmäßig auseinander, nach oben gerundet stehen die Endsegmente bündig am Rand
            # und die inneren überlappen leicht - so ragt nie ein Segment über den Rand hinaus. Reststücke unter
            # MIN_ROW_FILL Segmentlängen bleiben leer.
            length = t1 - t0
            if length < segment * MIN_ROW_FILL:
                continue
            fill = length / segment
            count = max(1, int(np.floor(fill + 0.5)))
            if count == 1:
                t = np.array([(t0 + t1) / 2.0])
            elif fill >= count:
                t = t0 + (np.arange(count) + 0.5) * (length / count)
            else:
                t = np.linspace(t0 + segment / 2.0, t1 - segment / 2.0, count)
            centers.append(t[:, None] * u + v_coord * v)
    if not centers:
        return []

    # Alle Segmente eines Blocks auf einmal: Höhen der Segmentenden, Neigung, Matrix.
    center = np.vstack(centers)
    end_a, end_b = center - u * segment / 2.0, center + u * segment / 2.0
    z_a, z_b = height_at(end_a[:, 0], end_a[:, 1]), height_at(end_b[:, 0], end_b[:, 1])
    forward = np.column_stack([np.full(len(center), segment * u[0]), np.full(len(center), segment * u[1]), z_b - z_a])
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    z_center = height_at(center[:, 0], center[:, 1])
    matrices = _rotation_matrices(forward).tolist()
    scales = rng.uniform(scale_min, scale_max, size=len(center))  # gleiche Zufallsfolge wie einzeln gezogen

    return [
        {"type": rows["item"], "pos": [x, y, z], "rotationMatrix": matrix, "scale": scale}
        for (x, y), z, matrix, scale in zip(center.tolist(), z_center.tolist(), matrices, scales.tolist())
    ]


def generate_vineyard_instances(
    polygon: BaseGeometry, height_at: HeightAt, rows: Dict, exclusion: Optional[BaseGeometry] = None
) -> List[Dict]:
    """
    Erzeugt die Rebzeilen-Instanzen für ein Weinberg-(Multi-)Polygon.

    Args:
        polygon: Weinberg-Fläche in lokalen Koordinaten (MultiPolygon: jeder Teil
            bekommt seine eigene Zeilenrichtung)
        height_at: Höhenabfrage (siehe make_height_sampler())
        rows: Zeilen-Einstellungen aus landuse_mappings["vineyard"]["rows"]
            (item, orientation, row_spacing, segment_length, edge_margin,
            min_slope_percent, scale_range)
        exclusion: Bereiche ohne Reben (z.B. Wege, Gebäude)

    Returns:
        Liste von Forest-Instanzen; deterministisch für gleiche Eingaben.
    """
    # MultiPolygon/GeometryCollection (z.B. nach dem Verschnitt mit dem Terrain-Rechteck)
    # in einzelne Polygone zerlegen; Linien-/Punktreste sind keine Flächen.
    parts = _polygon_parts(polygon)
    spread = rows.get("max_direction_spread_deg")
    instances = []
    for part in parts:
        blocks = (
            split_by_direction(
                part,
                height_at,
                float(spread),
                float(rows.get("min_block_area", 2500.0)),
                float(rows.get("min_slope_percent", 2.0)),
            )
            if spread is not None
            else [part]
        )
        for block in blocks:
            centroid = block.centroid
            rng = np.random.RandomState(int(abs(centroid.x * 1000.0 + centroid.y)) % (2**31))
            instances.extend(_polygon_instances(block, height_at, rows, exclusion, rng))
    return instances


def generate_vineyards(
    landuse_polygons: Sequence[Dict],
    landuse_mappings: Dict,
    height_at: HeightAt,
    exclusion: Optional[BaseGeometry] = None,
    bounds: Optional[BaseGeometry] = None,
) -> List[Dict]:
    """
    Erzeugt Rebzeilen für alle Weinberg-Polygone (Kategorie mit "rows"-Einstellungen).

    Args:
        landuse_polygons: Ergebnis von osm.landuse_polygons.build_landuse_polygons()
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        height_at: Höhenabfrage
        exclusion: Bereiche ohne Reben
        bounds: Ausdehnung des Terrains. Die OSM-Abfrage reicht darüber hinaus, und
            außerhalb gibt es keine Höhendaten (die Heightmap klemmt am Rand) - dort
            würden Reben in der Luft schweben.
    """
    instances = []
    for polygon in landuse_polygons:
        category = get_landuse_category(polygon["osm_tags"], landuse_mappings)
        rows = landuse_mappings.get(category, {}).get("rows") if category else None
        if not rows:
            continue
        geometry = polygon["geometry"]
        if bounds is not None:
            geometry = geometry.intersection(bounds)
            if geometry.is_empty:
                continue
        instances.extend(generate_vineyard_instances(geometry, height_at, rows, exclusion))
    return instances
