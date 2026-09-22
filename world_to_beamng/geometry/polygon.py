"""
Polygon-Operationen und Strassen-Extraktion.
"""

import numpy as np
from shapely.geometry import Polygon

from ..terrain.elevation import get_elevations_for_points
from ..geometry.coordinates import transformer_to_utm
from ..config import OSM_MAPPER
from .. import config
from .road_structures import classify_structure
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()


def clip_road_polygons(road_polygons, grid_bounds_local, margin=3.0):
    """
    Clippt Strassen-Polygone am Grid-Rand mit Margin.

    Args:
        road_polygons: Liste von Strassen-Dictionaries mit 'coords'
        grid_bounds_local: (min_x, max_x, min_y, max_y) in lokalen Koordinaten
        margin: Abstand vom Grid-Rand in Metern (default 3.0)
                Positiv = Straßen werden VOR dem Rand geschnitten
                Negativ = Straßen werden ÜBER den Rand hinaus erweitert

    Returns:
        Geclippte road_polygons (Strassen die komplett ausserhalb liegen werden entfernt)
    """
    if not config.ENABLE_ROAD_CLIPPING:
        return road_polygons

    if not grid_bounds_local:
        return road_polygons

    min_x, max_x, min_y, max_y = grid_bounds_local

    # WICHTIG: Margin-Semantik:
    # - Positiv (z.B. +10): Clip-Box wird KLEINER → Straßen enden 10m VOR dem Grid-Rand
    # - Negativ (z.B. -20): Clip-Box wird GRÖSSER → Straßen enden 20m HINTER dem Grid-Rand
    # Formel: clip_min = min + margin (bei margin=-20 → -1000 + (-20) = -1020 ✓)
    clip_min_x = min_x + margin
    clip_max_x = max_x - margin
    clip_min_y = min_y + margin
    clip_max_y = max_y - margin

    clipped_roads = []
    removed_count = 0
    segment_count = 0
    split_count = 0

    for road in road_polygons:
        coords = road["coords"]

        # WICHTIG: Punkte, die durch das Clipping entfernt wurden, dürfen NICHT
        # einfach übersprungen werden - sonst werden die verbleibenden,
        # tatsächlich weit auseinanderliegenden Punkte (z.B. von einer Straße,
        # die weit ausserhalb des Tiles einen Bogen macht und an zwei ganz
        # unterschiedlichen Stellen wieder ins Tile hineinragt) zu einer
        # einzigen, künstlichen "Teleport"-Gerade zusammengefasst. Das erzeugt
        # eine falsche Centerline mit falschen Z-Werten, die dann als riesige,
        # unnatürliche Klippe im Böschungs-Blend landet. Stattdessen: an jeder
        # Lücke einen neuen, eigenständigen Strassen-Abschnitt beginnen (wie im
        # alten Mesh-Workflow, wo Strassen am Rand tatsächlich endeten).
        runs = []
        current_run = []
        for x, y, z in coords:
            if clip_min_x <= x <= clip_max_x and clip_min_y <= y <= clip_max_y:
                current_run.append((x, y, z))
            elif current_run:
                runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)

        osm_tags = road.get("osm_tags", {})
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        max_seg = config.GRID_SPACING

        for run_idx, new_coords in enumerate(runs):
            if len(new_coords) < 2:
                removed_count += 1
                continue

            # Unterteile lange Segmente nach Clipping (um grosse Luecken innerhalb
            # eines zusammenhängenden Abschnitts zu füllen, z.B. bei grob
            # abgetasteten OSM-Ways)
            final_coords = []
            for i, coord in enumerate(new_coords):
                final_coords.append(coord)

                # Wenn nicht das letzte Segment
                if i < len(new_coords) - 1:
                    next_coord = new_coords[i + 1]
                    # Berechne Distanz zum nächsten Punkt
                    dist = np.sqrt(
                        (next_coord[0] - coord[0]) ** 2
                        + (next_coord[1] - coord[1]) ** 2
                        + (next_coord[2] - coord[2]) ** 2
                    )

                    # Wenn Segment länger als max_seg, interpoliere Zwischenpunkte
                    if dist > max_seg:
                        num_intermediate = int(np.ceil(dist / max_seg)) - 1
                        for j in range(1, num_intermediate + 1):
                            t = j / (num_intermediate + 1)
                            inter_point = (
                                coord[0] + t * (next_coord[0] - coord[0]),
                                coord[1] + t * (next_coord[1] - coord[1]),
                                coord[2] + t * (next_coord[2] - coord[2]),
                            )
                            final_coords.append(inter_point)

            road_id = road["id"]
            if len(runs) > 1:
                # Mehrere getrennte Abschnitte aus derselben Strasse -> eindeutige IDs
                road_id = f"{road_id}_c{run_idx}" if isinstance(road_id, str) else road_id * 1000 + run_idx
                split_count += 1

            clipped_roads.append(
                {
                    "id": road_id,
                    "coords": final_coords,
                    "name": road["name"],
                    "osm_tags": osm_tags,  # OSM-Tags durchreichen
                }
            )
            segment_count += len(coords) - len(final_coords)

    if removed_count > 0 or segment_count > 0 or split_count > 0:
        logger.info(
            f"  Clipping: {removed_count} Strassen(-Abschnitte) entfernt, "
            f"{segment_count} Punkte ausserhalb des Grids entfernt, "
            f"{split_count} Strassen am Rand in getrennte Abschnitte gesplittet"
        )

    return clipped_roads


def drop_close_nodes(nodes, min_dist):
    """Entfernt Knoten, die (in XY) näher als min_dist am vorherigen behaltenen Knoten liegen.

    Start- und Endknoten bleiben exakt erhalten (Junction-Anschluss an
    Nachbarstraßen). Ist das letzte Segment zu kurz, wird stattdessen der
    vorletzte Knoten entfernt.

    Args:
        nodes: Liste von Knoten [x, y, z, ...] (weitere Einträge wie die Breite
            bleiben unverändert)
        min_dist: Mindestabstand in Metern

    Returns:
        Gefilterte Knotenliste, oder [] wenn die Straße nach dem Filtern
        unbrauchbar kurz ist (weniger als 2 Knoten oder Start-Ende-Abstand < min_dist).
    """
    if len(nodes) < 2:
        return []

    def dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    kept = [nodes[0]]
    for node in nodes[1:-1]:
        if dist(node, kept[-1]) >= min_dist:
            kept.append(node)

    last = nodes[-1]
    if len(kept) > 1 and dist(last, kept[-1]) < min_dist:
        kept.pop()
    kept.append(last)

    if dist(kept[0], kept[-1]) < min_dist:
        return []
    return kept


def resample_road_xy_only(xy_coords, target_spacing):
    """Resampled Centerline auf XY-Ebene mit fixer Schrittweite.

    Args:
        xy_coords: Liste von (x, y) Koordinaten
        target_spacing: Ziel-Abstand zwischen Punkten in Metern

    Returns:
        Liste von resampleten (x, y) Koordinaten
    """
    if len(xy_coords) < 2:
        return xy_coords

    coords_arr = np.array(xy_coords)

    # Berechne kumulative Distanz
    diffs = np.diff(coords_arr, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum[-1]

    if total_len < 1e-6:
        return xy_coords

    # Berechne Sample-Positionen
    num_samples = max(2, int(np.ceil(total_len / target_spacing)) + 1)
    t = np.linspace(0.0, total_len, num_samples)

    # Interpoliere x, y
    x = np.interp(t, cum, coords_arr[:, 0])
    y = np.interp(t, cum, coords_arr[:, 1])

    # Stelle sicher, dass Start/End exakt bleiben (wichtig für Junctions!)
    x[0], y[0] = coords_arr[0, 0], coords_arr[0, 1]
    x[-1], y[-1] = coords_arr[-1, 0], coords_arr[-1, 1]

    return list(zip(x, y))


def _linear_elevation_profile(coords):
    """Ersetzt die Z-Werte durch lineare Interpolation zwischen Anfangs- und Endpunkt (Bogenlänge-gewichtet);
    Start/Ende bleiben exakt erhalten (dort schließt die normale Straße an, siehe Design-Spec Abschnitt 2)."""
    arr = np.array(coords, dtype=float)
    xy = arr[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return coords
    t = cum / total
    z = arr[0, 2] + t * (arr[-1, 2] - arr[0, 2])
    return [(float(x), float(y), float(zz)) for (x, y), zz in zip(xy, z)]


def apply_structure_elevation_profiles(road_polygons):
    """Brücken/Tunnel/Galerien (siehe geometry.road_structures.classify_structure) bekommen ein lineares
    Höhenprofil zwischen ihren Endpunkten statt der rohen DGM-Höhe an jedem Punkt - siehe Design-Spec Abschnitt 2
    (z.B. der 16,9 km lange Gotthard-Straßentunnel bekommt sonst die Bergrücken-Höhe darüber zugewiesen)."""
    for road in road_polygons:
        if classify_structure(road.get("osm_tags", {})) != "surface" and len(road["coords"]) >= 2:
            road["coords"] = _linear_elevation_profile(road["coords"])
    return road_polygons


def get_road_polygons(roads, bbox, height_points, height_elevations, global_offset, tile_hash=None):
    """Extrahiert Strassen-Polygone mit ihren Koordinaten und Hoehen (NEUE PIPELINE).

    Pipeline:
    1. OSM → lokale XY (ohne Z)
    2. Resampling auf XY-Ebene (fixer Schritt)
    3. Höhen-Sampling auf verdichteten Punkten
    4. Optional: mildes XY-Smoothing

    Args:
        roads: OSM-Strassen-Daten
        bbox: (lat_min, lon_min, lat_max, lon_max) BBox
        height_points: Höhendaten-Punkte (LOKAL, bereits normalisiert!)
        height_elevations: Z-Werte (LOKAL, bereits normalisiert!)
        global_offset: (origin_x, origin_y) für Koordinaten-Transformation
        tile_hash: Optional - tile_hash für Cache-Konsistenz
    """
    road_polygons = []

    # Sammle alle Koordinaten fuer Batch-Verarbeitung
    all_coords = []
    road_indices = []

    for way in roads:
        if "geometry" not in way:
            continue

        pts = [[p["lat"], p["lon"]] for p in way["geometry"]]
        if len(pts) < 2:
            continue

        road_indices.append((len(all_coords), len(all_coords) + len(pts), way))
        all_coords.extend(pts)

    if not all_coords:
        return road_polygons

    # Batch-UTM-Transformation (vektorisiert) - NUR XY
    lats = np.array([c[0] for c in all_coords])
    lons = np.array([c[1] for c in all_coords])
    xs_utm, ys_utm = transformer_to_utm.transform(lons, lats)

    # Transformiere zu lokalen Koordinaten mit global_offset
    ox, oy = global_offset
    xs = xs_utm - ox
    ys = ys_utm - oy

    # Erstelle temporäre road_polygons mit XY (ohne Z)
    temp_roads_xy = []
    for start_idx, end_idx, way in road_indices:
        xy_coords = [(xs[i], ys[i]) for i in range(start_idx, end_idx)]
        osm_tags = way.get("tags", {})
        temp_roads_xy.append(
            {
                "id": way["id"],
                "xy_coords": xy_coords,
                "name": osm_tags.get("name", f"road_{way['id']}"),
                "osm_tags": osm_tags,
            }
        )

    # SCHRITT 2: XY-Resampling (verdichte Centerlines VOR Höhen-Sampling)
    logger.info(f"  Resample Centerlines auf XY-Ebene...")
    points_before_resampling = sum(len(r["xy_coords"]) for r in temp_roads_xy)

    for road in temp_roads_xy:
        osm_tags = road.get("osm_tags", {})
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        target_spacing = road_width * config.SAMPLE_SPACING_FACTOR

        resampled_xy = resample_road_xy_only(road["xy_coords"], target_spacing)
        road["xy_coords"] = resampled_xy

    points_after_resampling = sum(len(r["xy_coords"]) for r in temp_roads_xy)
    logger.info(
        f"    -> {points_before_resampling} Punkte → {points_after_resampling} Punkte ({points_after_resampling - points_before_resampling:+d})"
    )

    # SCHRITT 3: Batch-Elevation-Lookup auf den resampleten XY-Punkten
    logger.info(f"  Lade Elevations für {points_after_resampling} resampelte Punkte...")

    # Sammle alle XY-Punkte für Batch-Lookup
    all_xy_flat = []
    road_xy_indices = []
    for road in temp_roads_xy:
        start = len(all_xy_flat)
        all_xy_flat.extend(road["xy_coords"])
        end = len(all_xy_flat)
        road_xy_indices.append((start, end, road))

    # Konvertiere XY zurück zu Lat/Lon für Elevation-Lookup
    xs_flat = np.array([xy[0] for xy in all_xy_flat])
    ys_flat = np.array([xy[1] for xy in all_xy_flat])

    # Transformiere zu UTM und dann zu Lat/Lon
    xs_utm_flat = xs_flat + ox
    ys_utm_flat = ys_flat + oy
    lons_flat, lats_flat = transformer_to_utm.transform(xs_utm_flat, ys_utm_flat, direction="INVERSE")

    latlon_coords = [[lats_flat[i], lons_flat[i]] for i in range(len(lats_flat))]
    all_elevations = get_elevations_for_points(
        latlon_coords, bbox, height_points, height_elevations, global_offset, height_hash=tile_hash
    )

    # Erstelle finale road_polygons mit XYZ
    for start_idx, end_idx, road in road_xy_indices:
        xyz_coords = [(all_xy_flat[i][0], all_xy_flat[i][1], all_elevations[i]) for i in range(start_idx, end_idx)]
        road_polygons.append(
            {
                "id": road["id"],
                "coords": xyz_coords,
                "name": road["name"],
                "osm_tags": road["osm_tags"],
            }
        )

    # SCHRITT 3b: Brücken/Tunnel/Galerien bekommen ein lineares Höhenprofil statt der rohen DGM-Abtastung
    # (siehe Design-Spec Abschnitt 2) - VOR dem Smoothing, damit dieses auf dem bereits korrekten Profil arbeitet.
    road_polygons = apply_structure_elevation_profiles(road_polygons)

    # SCHRITT 4: Optional - mildes XY-Smoothing (Z bleibt erhalten oder nur leicht geglättet)
    if config.ENABLE_ROAD_SMOOTHING:
        logger.info(f"  Mildes XY-Smoothing...")
        road_polygons = smooth_roads_xy_only(road_polygons)
    else:
        logger.info(f"  Smoothing SKIP (config.ENABLE_ROAD_SMOOTHING=False)")

    return road_polygons


def smooth_roads_xy_only(road_polygons):
    """Mildes XY+Z-Smoothing mit konfigurierbarer Stärke.

    Glättet XY und Z mit Chaikin-Filter.
    Iterationen und Gewichtung sind über Config steuerbar:
    - ROAD_SMOOTH_ITERATIONS: 1-3 (höher = glatter)
    - ROAD_SMOOTH_WEIGHT: 0.5-0.9 (höher = weniger Glättung)

    Returns:
        Modifizierte road_polygons mit geglätteten Koordinaten
    """
    total_points = sum(len(road["coords"]) for road in road_polygons)

    # Config-Parameter
    iterations = max(1, config.ROAD_SMOOTH_ITERATIONS)
    weight_center = config.ROAD_SMOOTH_WEIGHT  # z.B. 0.75
    weight_neighbor = (1.0 - weight_center) / 2.0  # z.B. 0.125

    for road in road_polygons:
        coords = road["coords"]
        if len(coords) < 3:
            continue

        coords_arr = np.array(coords)
        smoothed_arr = coords_arr.copy()

        # Chaikin-Glättung für XYZ
        for iteration in range(iterations):
            temp = smoothed_arr.copy()
            for i in range(1, len(smoothed_arr) - 1):
                # Glätte XYZ mit konfigurierbarem Gewicht
                temp[i, 0] = (
                    weight_center * smoothed_arr[i, 0]
                    + weight_neighbor * smoothed_arr[i - 1, 0]
                    + weight_neighbor * smoothed_arr[i + 1, 0]
                )
                temp[i, 1] = (
                    weight_center * smoothed_arr[i, 1]
                    + weight_neighbor * smoothed_arr[i - 1, 1]
                    + weight_neighbor * smoothed_arr[i + 1, 1]
                )
                temp[i, 2] = (
                    weight_center * smoothed_arr[i, 2]
                    + weight_neighbor * smoothed_arr[i - 1, 2]
                    + weight_neighbor * smoothed_arr[i + 1, 2]
                )
            smoothed_arr = temp

        # Start/End exakt beibehalten (wichtig für Junctions!)
        smoothed_arr[0] = coords_arr[0]
        smoothed_arr[-1] = coords_arr[-1]

        road["coords"] = [(p[0], p[1], p[2]) for p in smoothed_arr]

    logger.info(f"    -> {total_points} Punkte geglättet (XY+Z, {iterations} Iter., Weight={weight_center:.2f})")
    return road_polygons
