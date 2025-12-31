"""
Analyse-Tool zur Erkennung von ueberlappenden Strassen in OSM-Daten.
"""

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree


def analyze_road_overlaps(road_polygons, threshold_distance=5.0):
    """
    Analysiert, ob sich Strassen ueberlappen oder sehr nah beieinander liegen.

    Args:
        road_polygons: Liste der Strassen-Polygone mit coords
        threshold_distance: Maximaler Abstand in Metern fuer "Überlappung"

    Returns:
        dict mit Statistiken und Liste der ueberlappenden Strassenpaare
    """
    print(f"\n[Analyse] Pruefe {len(road_polygons)} Strassen auf Überlappungen...")

    # Erstelle LineStrings aus den Strassenkoordinaten
    linestrings = []
    valid_roads = []

    for road in road_polygons:
        coords = road["coords"]
        if len(coords) < 2:
            continue

        # Nur X-Y Koordinaten (ohne Z)
        coords_2d = [(x, y) for x, y, z in coords]
        try:
            line = LineString(coords_2d)
            if line.is_valid and line.length > 0:
                linestrings.append(line)
                valid_roads.append(road)
        except Exception:
            continue

    if len(linestrings) < 2:
        print("  [!] Zu wenige gueltige Strassen fuer Analyse")
        return {"total_roads": len(road_polygons), "overlaps": []}

    print(f"  Baue STRtree fuer {len(linestrings)} Strassen...")
    tree = STRtree(linestrings)
    # Mapping fuer robustes Index-Lookup (falls STRtree Indexe statt Geometrien liefert)
    geom_to_index = {id(geom): idx for idx, geom in enumerate(linestrings)}

    overlapping_pairs = []
    duplicate_candidates = []
    near_duplicates = []

    print(f"  Suche Überlappungen...")

    for i, line in enumerate(linestrings):
        # Finde alle Strassen in der Nähe (KD-Query via Buffer)
        candidates = tree.query(line.buffer(threshold_distance))

        for candidate in candidates:
            # STRtree kann je nach Shapely-Version Geometrien ODER Indizes liefern
            if isinstance(candidate, (int, np.integer)):
                j = int(candidate)
                candidate_line = linestrings[j]
            else:
                j = geom_to_index.get(id(candidate))
                if j is None:
                    # Fallback: langsame Suche, sollte selten sein
                    try:
                        j = linestrings.index(candidate)
                    except ValueError:
                        continue
                candidate_line = candidate

            # Vermeide Selbst-Vergleich und Duplikate (i,j) vs (j,i)
            if i >= j:
                continue

            # Berechne Abstand zwischen den Linien
            distance = line.distance(candidate_line)

            if distance < threshold_distance:
                road_i = valid_roads[i]
                road_j = valid_roads[j]

                # Berechne Überlappungsgrad
                intersection = line.intersection(
                    candidate_line.buffer(threshold_distance)
                )
                overlap_ratio = 0
                if hasattr(intersection, "length") and intersection.length > 0:
                    overlap_ratio = intersection.length / min(
                        line.length, candidate_line.length
                    )

                overlap_info = {
                    "road_1_id": road_i["id"],
                    "road_1_name": road_i["name"],
                    "road_2_id": road_j["id"],
                    "road_2_name": road_j["name"],
                    "distance": distance,
                    "overlap_ratio": overlap_ratio,
                    "length_1": line.length,
                    "length_2": candidate_line.length,
                }

                overlapping_pairs.append(overlap_info)

                # Klassifiziere Überlappungen
                if (
                    overlap_ratio > 0.9
                    and abs(line.length - candidate_line.length)
                    / max(line.length, candidate_line.length)
                    < 0.1
                ):
                    duplicate_candidates.append(overlap_info)
                elif overlap_ratio > 0.5:
                    near_duplicates.append(overlap_info)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(linestrings)} Strassen geprueft...")

    # Ausgabe der Ergebnisse
    print(f"\n  ERGEBNISSE:")
    print(f"    • Gepruefte Strassen: {len(linestrings)}")
    print(f"    • Überlappungen gesamt: {len(overlapping_pairs)}")
    print(
        f"    • Wahrscheinliche Duplikate (>90% Überlappung): {len(duplicate_candidates)}"
    )
    print(f"    • Teilweise Überlappungen (>50%): {len(near_duplicates)}")

    if duplicate_candidates:
        print(f"\n  TOP 10 DUPLIKAT-KANDIDATEN:")
        for i, dup in enumerate(
            sorted(
                duplicate_candidates, key=lambda x: x["overlap_ratio"], reverse=True
            )[:10]
        ):
            print(
                f"    {i+1}. Road {dup['road_1_id']} ({dup['road_1_name']}) ↔ Road {dup['road_2_id']} ({dup['road_2_name']})"
            )
            print(
                f"       Überlappung: {dup['overlap_ratio']*100:.1f}%, Abstand: {dup['distance']:.2f}m"
            )

    return {
        "total_roads": len(road_polygons),
        "valid_roads": len(linestrings),
        "total_overlaps": len(overlapping_pairs),
        "duplicates": len(duplicate_candidates),
        "near_duplicates": len(near_duplicates),
        "overlapping_pairs": overlapping_pairs,
        "duplicate_candidates": duplicate_candidates,
        "near_duplicate_candidates": near_duplicates,
    }


def get_road_statistics(road_polygons):
    """
    Zeigt grundlegende Statistiken ueber die Strassen.
    """
    print(f"\n[Statistik] Strassen-Eigenschaften:")

    lengths = []
    point_counts = []
    names = {}

    for road in road_polygons:
        coords = road["coords"]
        if len(coords) < 2:
            continue

        # Berechne Länge
        coords_array = np.array(coords)
        diffs = np.diff(coords_array[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)

        lengths.append(total_length)
        point_counts.append(len(coords))

        # Zähle Strassennamen
        name = road.get("name", "unbenannt")
        names[name] = names.get(name, 0) + 1

    if lengths:
        print(f"  Strassenlaengen:")
        print(f"    • Min: {np.min(lengths):.1f}m")
        print(f"    • Max: {np.max(lengths):.1f}m")
        print(f"    • Durchschnitt: {np.mean(lengths):.1f}m")
        print(f"    • Median: {np.median(lengths):.1f}m")

        print(f"\n  Punkte pro Strasse:")
        print(f"    • Min: {np.min(point_counts)}")
        print(f"    • Max: {np.max(point_counts)}")
        print(f"    • Durchschnitt: {np.mean(point_counts):.1f}")

        # Häufigste Strassennamen (koennte auf Duplikate hinweisen)
        sorted_names = sorted(names.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Haeufigste Strassennamen:")
        for name, count in sorted_names[:10]:
            if count > 1:
                print(f"    • '{name}': {count}x")

    return {
        "total_roads": len(road_polygons),
        "lengths": lengths,
        "point_counts": point_counts,
        "unique_names": len(names),
        "name_counts": names,
    }
