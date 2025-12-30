"""
Polygon-Operationen und Straßen-Extraktion.
"""

import numpy as np
from shapely.geometry import Polygon

from ..terrain.elevation import get_elevations_for_points
from ..geometry.coordinates import transformer_to_utm
from .. import config


def clip_road_polygons(road_polygons, grid_bounds_local, margin=3.0):
    """
    Clippt Straßen-Polygone am Grid-Rand mit Margin.

    Args:
        road_polygons: Liste von Straßen-Dictionaries mit 'coords'
        grid_bounds_local: (min_x, max_x, min_y, max_y) in lokalen Koordinaten
        margin: Abstand vom Grid-Rand in Metern (default 3.0)

    Returns:
        Geclippte road_polygons (Straßen die komplett außerhalb liegen werden entfernt)
    """
    if not grid_bounds_local:
        return road_polygons

    min_x, max_x, min_y, max_y = grid_bounds_local
    clip_min_x = min_x + margin
    clip_max_x = max_x - margin
    clip_min_y = min_y + margin
    clip_max_y = max_y - margin

    clipped_roads = []
    removed_count = 0
    segment_count = 0

    for road in road_polygons:
        coords = road["coords"]
        new_coords = []

        for x, y, z in coords:
            # Hard Clip: Punkt außerhalb → verwerfen
            if clip_min_x <= x <= clip_max_x and clip_min_y <= y <= clip_max_y:
                new_coords.append((x, y, z))

        # Straße behalten wenn mindestens 2 Punkte übrig sind
        if len(new_coords) >= 2:
            clipped_roads.append(
                {"id": road["id"], "coords": new_coords, "name": road["name"]}
            )
            segment_count += len(coords) - len(new_coords)
        else:
            removed_count += 1

    if removed_count > 0 or segment_count > 0:
        print(
            f"  Clipping: {removed_count} Straßen entfernt, {segment_count} Punkte außerhalb des Grids entfernt"
        )

    return clipped_roads


def get_road_polygons(roads, bbox, height_points, height_elevations):
    """Extrahiert Straßen-Polygone mit ihren Koordinaten und Höhen (OPTIMIERT)."""
    road_polygons = []

    # Sammle alle Koordinaten für Batch-Verarbeitung
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

    # Batch-Elevation-Lookup
    print(f"  Lade Elevations für {len(all_coords)} Straßen-Punkte...")
    all_elevations = get_elevations_for_points(
        all_coords, bbox, height_points, height_elevations
    )

    # Batch-UTM-Transformation (vektorisiert)
    lats = np.array([c[0] for c in all_coords])
    lons = np.array([c[1] for c in all_coords])
    xs, ys = transformer_to_utm.transform(lons, lats)

    # Transformiere direkt in lokale Koordinaten
    from .. import config

    if config.LOCAL_OFFSET is not None:
        ox, oy, oz = config.LOCAL_OFFSET
        xs = xs - ox
        ys = ys - oy
        # Transformiere auch Z-Koordinaten!
        all_elevations = [e - oz for e in all_elevations]

    # Erstelle Straßen-Polygone (bereits in lokalen Koordinaten)
    for start_idx, end_idx, way in road_indices:
        utm_coords = [
            (xs[i], ys[i], all_elevations[i]) for i in range(start_idx, end_idx)
        ]

        road_polygons.append(
            {
                "id": way["id"],
                "coords": utm_coords,
                "name": way.get("tags", {}).get("name", f"road_{way['id']}"),
            }
        )

    # Snap & Merge naher Straßenenden (für Kreuzungen und nicht-verbundene OSM-Ways)
    if config.ENABLE_ROAD_EDGE_SNAPPING:
        print(f"  Snap & Merge naher Straßenenden...")
        road_polygons = snap_road_endpoints(road_polygons)
    else:
        print(
            f"  Snap naher Straßenenden SKIP (config.ENABLE_ROAD_EDGE_SNAPPING=False)"
        )

    # Glätte Straßen und füge bei scharfen Kurven mehr Punkte ein
    if config.ENABLE_ROAD_SMOOTHING:
        print(f"  Glätte Straßen mit Spline-Interpolation...")
        road_polygons = smooth_roads_with_spline(road_polygons)
    else:
        print(f"  Glättung SKIP (config.ENABLE_ROAD_SMOOTHING=False)")

    return road_polygons


def smooth_roads_with_spline(road_polygons):
    """
    Glättet Straßen mit Catmull-Rom Spline und gleichmäßiger Punkt-Verteilung.

    - Erstellt geglättete Centerline mit Spline
    - Verteilt Punkte gleichmäßig entlang der Kurve
    - Berechnet radiale Senkrechten (Tangenten) für saubere Ränder

    Returns:
        Modifizierte road_polygons mit geglätteten Koordinaten
    """
    from scipy import interpolate

    total_points_before = sum(len(road["coords"]) for road in road_polygons)
    total_points_after = 0

    for road in road_polygons:
        road_id = road.get("id")
        coords = road["coords"]

        if len(coords) < 3:
            total_points_after += len(coords)
            continue

        coords_array = np.array(coords)

        # Entferne Duplikate/sehr nahe Punkte (können Spline verwirren)
        unique_coords = [coords_array[0]]
        for i in range(1, len(coords_array)):
            dist = np.linalg.norm(coords_array[i, :2] - unique_coords[-1][:2])
            if dist > 0.1:  # Mind. 10cm Abstand
                unique_coords.append(coords_array[i])

        if len(unique_coords) < 3:
            total_points_after += len(coords)
            continue

        coords_array = np.array(unique_coords)

        # Berechne kumulative Distanz entlang der Centerline
        diffs = np.diff(coords_array[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_dist[-1]

        if total_length < 0.5:  # Zu kurze Straße
            total_points_after += len(coords)
            continue

        # Verwende PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
        # Robuster als UnivariateSpline, geht durch alle Punkte, keine Randbedingungsprobleme
        try:
            # PchipInterpolator ist monoton und vermeidet Überschwingen
            from scipy.interpolate import PchipInterpolator

            interp_x = PchipInterpolator(cumulative_dist, coords_array[:, 0])
            interp_y = PchipInterpolator(cumulative_dist, coords_array[:, 1])
            interp_z = PchipInterpolator(cumulative_dist, coords_array[:, 2])
        except Exception:
            # Fallback bei Problemen - behalte Original
            total_points_after += len(coords)
            continue

        # Sample gleichmäßig entlang der Kurve
        # Stelle sicher, dass die Segmentlänge <= ROAD_SMOOTH_MAX_SEGMENT bleibt
        # (n-1) Segmente pro Kurve, daher +1 Samples
        num_samples = max(
            len(coords_array),
            int(np.ceil(total_length / config.ROAD_SMOOTH_MAX_SEGMENT)) + 1,
            2,
        )

        # Gleichmäßig verteilte Parameter entlang der Kurve
        sample_params = np.linspace(0, total_length, num_samples)

        # Evaluiere Interpolation an Sample-Punkten
        smooth_x = interp_x(sample_params)
        smooth_y = interp_y(sample_params)
        smooth_z = interp_z(sample_params)

        # Erstelle geglättete Koordinaten
        smoothed_coords = list(zip(smooth_x, smooth_y, smooth_z))

        road["coords"] = smoothed_coords
        total_points_after += len(smoothed_coords)

    print(
        f"    → {total_points_before} Punkte → {total_points_after} Punkte ({total_points_after - total_points_before:+d})"
    )

    return road_polygons


def smooth_roads_adaptive(road_polygons):
    """
    Glättet Straßen durch adaptive Unterteilung: Fügt bei scharfen Kurven mehr Punkte ein.

    Nutzt Config-Werte:
        - ROAD_SMOOTH_ANGLE_THRESHOLD: Winkel in Grad - ab diesem Wert wird unterteilt
        - ROAD_SMOOTH_MAX_SEGMENT: Maximale Segmentlänge in Metern
        - ROAD_SMOOTH_MIN_SEGMENT: Minimale Segmentlänge in Metern

    Returns:
        Modifizierte road_polygons mit geglätteten Koordinaten
    """
    angle_threshold_rad = np.radians(config.ROAD_SMOOTH_ANGLE_THRESHOLD)
    max_segment_length = config.ROAD_SMOOTH_MAX_SEGMENT
    min_segment_length = config.ROAD_SMOOTH_MIN_SEGMENT
    total_points_before = sum(len(road["coords"]) for road in road_polygons)
    total_points_after = 0

    for road in road_polygons:
        coords = road["coords"]
        if len(coords) < 2:
            continue

        coords_array = np.array(coords)
        smoothed = [coords_array[0]]  # Startpunkt

        for i in range(len(coords_array) - 1):
            current = coords_array[i]
            next_point = coords_array[i + 1]

            # Berechne Segment-Länge
            segment_vec = next_point[:2] - current[:2]
            segment_length = np.linalg.norm(segment_vec)

            # Bestimme ob Kurve existiert
            if i < len(coords_array) - 2:
                # Berechne Winkel zwischen diesem und nächstem Segment
                next_segment_vec = coords_array[i + 2][:2] - next_point[:2]
                next_length = np.linalg.norm(next_segment_vec)

                if segment_length > 1e-6 and next_length > 1e-6:
                    # Normalisiere Vektoren
                    v1 = segment_vec / segment_length
                    v2 = next_segment_vec / next_length

                    # Berechne Winkel
                    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle = np.arccos(dot)

                    # Berechne Anzahl Unterteilungen basierend auf Winkel
                    if angle > angle_threshold_rad:
                        # Schärfere Kurve = mehr Punkte
                        subdivisions = int(np.ceil(angle / angle_threshold_rad))
                    else:
                        subdivisions = 1
                else:
                    subdivisions = 1
            else:
                subdivisions = 1

            # Zusätzliche Unterteilung bei langen Segmenten
            if segment_length > max_segment_length:
                length_subdivisions = int(np.ceil(segment_length / max_segment_length))
                subdivisions = max(subdivisions, length_subdivisions)

            # Verhindere zu viele Punkte bei kurzen Segmenten
            if segment_length < min_segment_length and subdivisions > 1:
                subdivisions = 1

            # Füge interpolierte Punkte ein
            if subdivisions > 1:
                for j in range(1, subdivisions + 1):
                    t = j / subdivisions
                    interpolated = current * (1 - t) + next_point * t
                    smoothed.append(interpolated)
            else:
                smoothed.append(next_point)

        # Konvertiere zurück zu Liste von Tupeln
        road["coords"] = [(p[0], p[1], p[2]) for p in smoothed]
        total_points_after += len(smoothed)

    print(
        f"    → {total_points_before} Punkte → {total_points_after} Punkte ({total_points_after - total_points_before:+d})"
    )

    return road_polygons


def snap_road_endpoints(road_polygons, snap_distance=5.0):
    """
    Findet Straßenenden, die nahe beieinander liegen und merged sie.

    Args:
        road_polygons: Liste von Straßen-Dictionaries mit 'coords'
        snap_distance: Maximale Distanz für Snap (in Metern)

    Returns:
        Modifizierte road_polygons mit gesnappten Endpunkten
    """
    from scipy.spatial import cKDTree

    if len(road_polygons) < 2:
        return road_polygons

    # Sammle alle Straßenenden (Anfang und Ende jeder Straße)
    endpoints = []
    endpoint_info = []  # (road_idx, is_start, point_xy)

    for road_idx, road in enumerate(road_polygons):
        coords = road["coords"]
        if len(coords) >= 2:
            # Anfangspunkt
            start_xy = (coords[0][0], coords[0][1])
            endpoints.append(start_xy)
            endpoint_info.append((road_idx, True, start_xy))

            # Endpunkt
            end_xy = (coords[-1][0], coords[-1][1])
            endpoints.append(end_xy)
            endpoint_info.append((road_idx, False, end_xy))

    if len(endpoints) < 2:
        return road_polygons

    # Baue KDTree für schnelle Nachbarschaftssuche
    endpoints_array = np.array(endpoints)
    kdtree = cKDTree(endpoints_array)

    # Finde Cluster von nahen Endpunkten
    pairs = kdtree.query_pairs(snap_distance)

    if len(pairs) == 0:
        print(f"    → Keine nahen Endpunkte gefunden")
        return road_polygons

    # Gruppiere zu Clustern (Union-Find)
    from collections import defaultdict

    parent = list(range(len(endpoints)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in pairs:
        union(i, j)

    # Sammle Cluster
    clusters = defaultdict(list)
    for i in range(len(endpoints)):
        clusters[find(i)].append(i)

    # Merge jeden Cluster zu einem gemeinsamen Punkt
    snapped_count = 0
    for cluster_indices in clusters.values():
        if len(cluster_indices) <= 1:
            continue  # Keine Nachbarn

        # Berechne Schwerpunkt des Clusters
        cluster_points = endpoints_array[cluster_indices]
        centroid = cluster_points.mean(axis=0)

        # Snap alle Punkte im Cluster zum Schwerpunkt
        for idx in cluster_indices:
            road_idx, is_start, _ = endpoint_info[idx]
            coords = road_polygons[road_idx]["coords"]

            if is_start:
                # Ändere Startpunkt
                coords[0] = (centroid[0], centroid[1], coords[0][2])
            else:
                # Ändere Endpunkt
                coords[-1] = (centroid[0], centroid[1], coords[-1][2])

            snapped_count += 1

    print(
        f"    → {len(pairs)} Endpunkt-Paare gemerged in {len([c for c in clusters.values() if len(c) > 1])} Cluster"
    )

    # Entferne doppelte aufeinanderfolgende Punkte (entstehen durch Snap)
    for road in road_polygons:
        coords = road["coords"]
        if len(coords) < 2:
            continue

        # Filtere aufeinanderfolgende Duplikate
        cleaned = [coords[0]]
        for i in range(1, len(coords)):
            # Prüfe ob Punkt sich vom vorherigen unterscheidet (mind. 0.01m)
            dx = coords[i][0] - cleaned[-1][0]
            dy = coords[i][1] - cleaned[-1][1]
            dist_sq = dx * dx + dy * dy

            if dist_sq > 0.0001:  # > 1cm Abstand
                cleaned.append(coords[i])

        # Nur updaten wenn sich was geändert hat
        if len(cleaned) != len(coords):
            if len(cleaned) >= 2:
                road["coords"] = cleaned
            # Wenn zu kurz: coords bleibt, wird später gefiltert

    return road_polygons


def get_road_centerline_robust(road_poly):
    """
    Berechne die Mittellinie eines Straßen-Polygons mittels PCA und Mittelwertsbildung.

    Args:
        road_poly: Shapely Polygon der Straße

    Returns:
        centerline: (N, 2) NumPy Array mit Mittellinie-Koordinaten
    """
    coords = np.array(road_poly.exterior.coords[:-1])  # ohne Wiederholung des Endpunkts

    if len(coords) < 4:
        return coords

    # Berechne Schwerpunkt
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # PCA: Finde Hauptrichtung (längste Achse = Straßenrichtung)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Hauptrichtung (Eigenvector mit größtem Eigenwert)
    main_direction = eigvecs[:, -1]

    # Senkrechte Richtung
    perp_direction = np.array([-main_direction[1], main_direction[0]])

    # Projiziere alle Punkte auf beide Richtungen
    proj_along = centered @ main_direction  # Entlang der Straße
    proj_perp = centered @ perp_direction  # Quer zur Straße (linke/rechte Seite)

    # Teile Punkte in zwei Seiten: links und rechts des Mittels
    median_perp = np.median(proj_perp)

    left_mask = proj_perp <= median_perp
    right_mask = proj_perp > median_perp

    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]

    if len(left_indices) < 2 or len(right_indices) < 2:
        return coords  # Fallback wenn Teilung nicht funktioniert

    # Sortiere beide Seiten nach der Längsprojektion (entlang der Straße)
    left_indices = left_indices[np.argsort(proj_along[left_indices])]
    right_indices = right_indices[np.argsort(proj_along[right_indices])]

    # Interpoliere beide Seiten auf die gleiche Anzahl von Punkten
    # So können wir sie direkt miteinander vergleichen
    n_samples = max(len(left_indices), len(right_indices))

    # Interpoliere linke Seite
    left_coords = coords[left_indices]
    left_interp_x = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(left_indices)),
        left_coords[:, 0],
    )
    left_interp_y = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(left_indices)),
        left_coords[:, 1],
    )

    # Interpoliere rechte Seite
    right_coords = coords[right_indices]
    right_interp_x = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(right_indices)),
        right_coords[:, 0],
    )
    right_interp_y = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(right_indices)),
        right_coords[:, 1],
    )

    # Berechne Mittelpunkte zwischen linker und rechter Seite
    centerline_x = (left_interp_x + right_interp_x) / 2
    centerline_y = (left_interp_y + right_interp_y) / 2

    centerline = np.column_stack([centerline_x, centerline_y])
    return centerline
