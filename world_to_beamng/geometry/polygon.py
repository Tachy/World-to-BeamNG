"""
Polygon-Operationen und Strassen-Extraktion.
"""

import numpy as np
from shapely.geometry import Polygon

from ..terrain.elevation import get_elevations_for_points
from ..geometry.coordinates import transformer_to_utm
from .. import config


def clip_road_polygons(road_polygons, grid_bounds_local, margin=3.0):
    """
    Clippt Strassen-Polygone am Grid-Rand mit Margin.

    Args:
        road_polygons: Liste von Strassen-Dictionaries mit 'coords'
        grid_bounds_local: (min_x, max_x, min_y, max_y) in lokalen Koordinaten
        margin: Abstand vom Grid-Rand in Metern (default 3.0)

    Returns:
        Geclippte road_polygons (Strassen die komplett ausserhalb liegen werden entfernt)
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
            # Hard Clip: Punkt ausserhalb -> verwerfen
            if clip_min_x <= x <= clip_max_x and clip_min_y <= y <= clip_max_y:
                new_coords.append((x, y, z))

        # Strasse behalten wenn mindestens 2 Punkte uebrig sind
        if len(new_coords) >= 2:
            # Unterteile lange Segmente nach Clipping (um grosse Luecken zu fuellen)
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

                    # Wenn Segment länger als MAX_SEGMENT, interpoliere Zwischenpunkte
                    max_seg = getattr(config, "ROAD_SMOOTH_MAX_SEGMENT", 5.0) or 5.0
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

            clipped_roads.append(
                {
                    "id": road["id"],
                    "coords": final_coords,
                    "name": road["name"],
                    "osm_tags": road.get("osm_tags", {}),  # OSM-Tags durchreichen
                }
            )
            segment_count += len(coords) - len(final_coords)
        else:
            removed_count += 1

    if removed_count > 0 or segment_count > 0:
        print(f"  Clipping: {removed_count} Strassen entfernt, {segment_count} Punkte ausserhalb des Grids entfernt")

    return clipped_roads


def get_road_polygons(roads, bbox, height_points, height_elevations):
    """Extrahiert Strassen-Polygone mit ihren Koordinaten und Hoehen (OPTIMIERT)."""
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

    # Batch-Elevation-Lookup
    print(f"  Lade Elevations fuer {len(all_coords)} Strassen-Punkte...")
    all_elevations = get_elevations_for_points(all_coords, bbox, height_points, height_elevations)

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
        # Z-Koordinaten sind bereits normalisiert (aus get_elevations_for_points)!

    # Erstelle Strassen-Polygone (bereits in lokalen Koordinaten)
    for start_idx, end_idx, way in road_indices:
        utm_coords = [(xs[i], ys[i], all_elevations[i]) for i in range(start_idx, end_idx)]

        road_polygons.append(
            {
                "id": way["id"],
                "coords": utm_coords,
                "name": way.get("tags", {}).get("name", f"road_{way['id']}"),
                "osm_tags": way.get("tags", {}),  # Alle OSM-Tags speichern
            }
        )

    # DEPRECATED: Strassenenden-Snapping wird nicht mehr verwendet (Junction-Erkennung in Schritt 6a)

    # Glätte Strassen und fuege bei scharfen Kurven mehr Punkte ein
    if config.ENABLE_ROAD_SMOOTHING:
        print(f"  Glaette Strassen mit Spline-Interpolation...")
        road_polygons = smooth_roads_with_spline(road_polygons)
    else:
        print(f"  Glaettung SKIP (config.ENABLE_ROAD_SMOOTHING=False)")

    return road_polygons


def smooth_roads_with_spline(road_polygons):
    """
    Glättet Strassen mit Catmull-Rom Spline und gleichmässiger Punkt-Verteilung.

    - Erstellt geglättete Centerline mit Spline
    - Verteilt Punkte gleichmässig entlang der Kurve
    - Unterteilt lange gerade Segmente (auch Strassen mit nur 2 Punkten)
    - WICHTIG: Behält Start/End-Punkte bei (wichtig fuer Junctions!)

    Returns:
        Modifizierte road_polygons mit geglätteten Koordinaten
    """
    from scipy import interpolate

    total_points_before = sum(len(road["coords"]) for road in road_polygons)
    total_points_after = 0

    for road in road_polygons:
        road_id = road.get("id")
        coords = road["coords"]

        if len(coords) < 2:
            total_points_after += len(coords)
            continue

        coords_array = np.array(coords)

        # Speichere Original-Start und End (fuer Junction-Snapping!)
        original_start = coords_array[0].copy()
        original_end = coords_array[-1].copy()

        # Entferne Duplikate/sehr nahe Punkte (koennen Spline verwirren)
        unique_coords = [coords_array[0]]
        for i in range(1, len(coords_array)):
            dist = np.linalg.norm(coords_array[i, :2] - unique_coords[-1][:2])
            if dist > 0.1:  # Mind. 10cm Abstand
                unique_coords.append(coords_array[i])

        if len(unique_coords) < 2:
            total_points_after += len(coords)
            continue

        coords_array = np.array(unique_coords)

        # Berechne kumulative Distanz entlang der Centerline
        diffs = np.diff(coords_array[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_dist[-1]

        if total_length < 0.5:  # Zu kurze Strasse
            total_points_after += len(coords)
            continue

        # Fuer Strassen mit nur 2 Punkten: Unterteile das lange Segment
        if len(unique_coords) == 2:
            max_segment = config.ROAD_SMOOTH_MAX_SEGMENT
            if total_length > max_segment:
                # Interpoliere linear zwischen den zwei Punkten
                num_samples = int(np.ceil(total_length / max_segment)) + 1
                sample_params = np.linspace(0, total_length, num_samples)

                smooth_x = coords_array[0, 0] + (coords_array[1, 0] - coords_array[0, 0]) * (
                    sample_params / total_length
                )
                smooth_y = coords_array[0, 1] + (coords_array[1, 1] - coords_array[0, 1]) * (
                    sample_params / total_length
                )
                smooth_z = coords_array[0, 2] + (coords_array[1, 2] - coords_array[0, 2]) * (
                    sample_params / total_length
                )

                smoothed_coords = list(zip(smooth_x, smooth_y, smooth_z))

                # Stelle sicher, dass Start und End exakt die Original-Werte haben
                if len(smoothed_coords) > 0:
                    smoothed_coords[0] = tuple(original_start)
                    smoothed_coords[-1] = tuple(original_end)

                road["coords"] = smoothed_coords
                total_points_after += len(road["coords"])
            else:
                total_points_after += len(coords)
            continue

        # Verwende PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) fuer 3+ Punkte
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

        # Sample gleichmässig entlang der Kurve
        # Stelle sicher, dass die Segmentlänge <= ROAD_SMOOTH_MAX_SEGMENT bleibt
        # (n-1) Segmente pro Kurve, daher +1 Samples
        num_samples = max(
            len(coords_array),
            int(np.ceil(total_length / config.ROAD_SMOOTH_MAX_SEGMENT)) + 1,
            2,
        )

        # Gleichmässig verteilte Parameter entlang der Kurve
        sample_params = np.linspace(0, total_length, num_samples)

        # Evaluiere Interpolation an Sample-Punkten
        smooth_x = interp_x(sample_params)
        smooth_y = interp_y(sample_params)
        smooth_z = interp_z(sample_params)

        # Erstelle geglättete Koordinaten
        smoothed_coords = list(zip(smooth_x, smooth_y, smooth_z))

        # Stelle sicher, dass Start und End exakt die Original-Werte haben (fuer Junctions!)
        if len(smoothed_coords) > 0:
            smoothed_coords[0] = tuple(original_start)
            smoothed_coords[-1] = tuple(original_end)

        road["coords"] = smoothed_coords
        total_points_after += len(smoothed_coords)

    print(
        f"    -> {total_points_before} Punkte -> {total_points_after} Punkte ({total_points_after - total_points_before:+d})"
    )

    return road_polygons


def smooth_roads_adaptive(road_polygons):
    """
    Glättet Strassen durch adaptive Unterteilung: Fuegt bei scharfen Kurven mehr Punkte ein.

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

            # Fuege interpolierte Punkte ein
            if subdivisions > 1:
                for j in range(1, subdivisions + 1):
                    t = j / subdivisions
                    interpolated = current * (1 - t) + next_point * t
                    smoothed.append(interpolated)
            else:
                smoothed.append(next_point)

        # Konvertiere zurueck zu Liste von Tupeln
        smoothed_coords = [(p[0], p[1], p[2]) for p in smoothed]

        # Optional: gleichmäßiges Resampling auf definierten Zielabstand
        resample_spacing = getattr(config, "ROAD_RESAMPLE_SPACING", None)
        if resample_spacing and len(smoothed_coords) >= 2:
            coords_arr = np.array(smoothed_coords)
            diffs = np.diff(coords_arr[:, :2], axis=0)
            seg_len = np.linalg.norm(diffs, axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            total_len = cum[-1]
            if total_len > 0:
                num_samples = max(2, int(np.ceil(total_len / resample_spacing)) + 1)
                t = np.linspace(0.0, total_len, num_samples)
                # Interpoliere x,y,z linear entlang der Bogenlänge
                x = np.interp(t, cum, coords_arr[:, 0])
                y = np.interp(t, cum, coords_arr[:, 1])
                z = np.interp(t, cum, coords_arr[:, 2])
                # Stelle sicher, dass Endpunkte exakt bleiben
                x[0], y[0], z[0] = coords_arr[0, 0], coords_arr[0, 1], coords_arr[0, 2]
                x[-1], y[-1], z[-1] = (
                    coords_arr[-1, 0],
                    coords_arr[-1, 1],
                    coords_arr[-1, 2],
                )
                smoothed_coords = list(zip(x, y, z))

        road["coords"] = smoothed_coords
        total_points_after += len(smoothed_coords)

    print(
        f"    -> {total_points_before} Punkte -> {total_points_after} Punkte ({total_points_after - total_points_before:+d})"
    )

    return road_polygons


def get_road_centerline_robust(road_poly):
    """
    Berechne die Mittellinie eines Strassen-Polygons mittels PCA und Mittelwertsbildung.

    Args:
        road_poly: Shapely Polygon der Strasse

    Returns:
        centerline: (N, 2) NumPy Array mit Mittellinie-Koordinaten
    """
    coords = np.array(road_poly.exterior.coords[:-1])  # ohne Wiederholung des Endpunkts

    if len(coords) < 4:
        return coords

    # Berechne Schwerpunkt
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # PCA: Finde Hauptrichtung (längste Achse = Strassenrichtung)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Hauptrichtung (Eigenvector mit groesstem Eigenwert)
    main_direction = eigvecs[:, -1]

    # Senkrechte Richtung
    perp_direction = np.array([-main_direction[1], main_direction[0]])

    # Projiziere alle Punkte auf beide Richtungen
    proj_along = centered @ main_direction  # Entlang der Strasse
    proj_perp = centered @ perp_direction  # Quer zur Strasse (linke/rechte Seite)

    # Teile Punkte in zwei Seiten: links und rechts des Mittels
    median_perp = np.median(proj_perp)

    left_mask = proj_perp <= median_perp
    right_mask = proj_perp > median_perp

    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]

    if len(left_indices) < 2 or len(right_indices) < 2:
        return coords  # Fallback wenn Teilung nicht funktioniert

    # Sortiere beide Seiten nach der Längsprojektion (entlang der Strasse)
    left_indices = left_indices[np.argsort(proj_along[left_indices])]
    right_indices = right_indices[np.argsort(proj_along[right_indices])]

    # Interpoliere beide Seiten auf die gleiche Anzahl von Punkten
    # So koennen wir sie direkt miteinander vergleichen
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
