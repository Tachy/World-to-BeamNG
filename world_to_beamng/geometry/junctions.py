"""
Erkennung und Handling von Straßen-Junctions direkt in Centerlines.

Diese Module erkennt Kreuzungen und T-Junctions direkt aus den Centerline-Koordinaten
(wo Straßen-Punkte verbunden sind), um diese bei der Mesh-Generierung sauber zu meshen.
"""

import numpy as np
from scipy.spatial import cKDTree


def detect_junctions_in_centerlines(road_polygons):
    """
    Erkennt Junctions (Kreuzungen/Einmündungen) direkt in den Centerlines.

    Eine Junction ist ein Punkt, wo mehrere Straßen zusammenkommen.
    OSM-Daten haben Straßenenden als Punkte, die verbunden sind → natürliche Junctions.

    Args:
        road_polygons: Liste von Straßen-Dicts mit 'coords', 'id', 'name'

    Returns:
        List of junctions with:
        {
            'position': (x, y, z),         # 3D-Position der Junction
            'road_indices': [i, j, ...],   # Indices der beteiligten Straßen
            'connection_types': [...]      # 'end' oder 'start' pro Straße
        }
    """
    if not road_polygons:
        return []

    # Sammle alle Straßenenden (Anfang und Ende)
    endpoints = []  # (x, y, z, road_idx, is_start)

    for road_idx, road in enumerate(road_polygons):
        coords = road["coords"]
        if len(coords) >= 2:
            # Anfangspunkt
            start = coords[0]
            endpoints.append((start[0], start[1], start[2], road_idx, True))

            # Endpunkt
            end = coords[-1]
            endpoints.append((end[0], end[1], end[2], road_idx, False))

    if len(endpoints) < 2:
        return []

    # Baue KDTree für schnelle Nachbarschaftssuche (nur XY-Koordinaten)
    endpoints_xy = np.array([(p[0], p[1]) for p in endpoints])
    kdtree = cKDTree(endpoints_xy)

    # Finde alle Punkte, die sehr nah beieinander liegen (< 0.1m toleranz)
    junction_pairs = kdtree.query_pairs(r=0.1)

    if not junction_pairs:
        return []

    # Gruppiere Endpoints zu Junctions (Union-Find)
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

    for i, j in junction_pairs:
        union(i, j)

    # Sammle Cluster
    clusters = defaultdict(list)
    for i, endpoint in enumerate(endpoints):
        root = find(i)
        clusters[root].append(i)

    # Baue Junctions aus Clustern
    junctions = []

    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue  # Keine echte Junction

        # Berechne Schwerpunkt der Junction (xyz-Mittelwert)
        cluster_points = [endpoints[i] for i in cluster_indices]
        avg_x = np.mean([p[0] for p in cluster_points])
        avg_y = np.mean([p[1] for p in cluster_points])
        avg_z = np.mean([p[2] for p in cluster_points])
        junction_pos_2d = np.array([avg_x, avg_y])

        # Sammle beteiligten Straßen UND deren Fahrtrichtungen bei der Junction
        junction_roads = {}  # road_idx → 'start' oder 'end'
        direction_vectors = {}  # road_idx → [direction_x, direction_y]

        for idx in cluster_indices:
            ep = endpoints[idx]
            road_idx = ep[3]
            is_start = ep[4]
            connection_type = "start" if is_start else "end"

            if road_idx not in junction_roads:
                junction_roads[road_idx] = []
            junction_roads[road_idx].append(connection_type)

            # Berechne Fahrtrichtung dieser Straße bei der Junction
            road = road_polygons[road_idx]
            coords = road["coords"]

            if is_start and len(coords) >= 2:
                # Straße beginnt: Richtung von coords[0] zu coords[1]
                p1 = np.array(coords[0][:2])
                p2 = np.array(coords[1][:2])
                direction = p2 - p1
            elif not is_start and len(coords) >= 2:
                # Straße endet: Richtung von coords[-2] zu coords[-1]
                p1 = np.array(coords[-2][:2])
                p2 = np.array(coords[-1][:2])
                direction = p2 - p1
            else:
                direction = np.array([1.0, 0.0])

            # Normalisiere
            norm = np.linalg.norm(direction)
            if norm > 0.01:
                direction = direction / norm

            direction_vectors[road_idx] = direction

        # Filtere: Junction muss mindestens 2 verschiedene Straßen haben
        road_list = sorted(list(junction_roads.keys()))

        if len(road_list) >= 2:
            junctions.append(
                {
                    "position": (avg_x, avg_y, avg_z),
                    "road_indices": road_list,
                    "connection_types": {r: junction_roads[r] for r in road_list},
                    "direction_vectors": direction_vectors,  # NEU: Echte Fahrtrichtungen!
                    "num_connections": len(cluster_indices),
                }
            )

    return junctions


def mark_junction_endpoints(road_polygons, junctions):
    """
    Markiert Straßen-Endpoints, die zu Junctions gehören.

    Dies wird später bei der Mesh-Generierung verwendet, um diese Punkte
    beim Stitchen zu identifizieren.

    Args:
        road_polygons: Liste von Straßen (wird modifiziert)
        junctions: Liste von Junctions aus detect_junctions_in_centerlines()

    Returns:
        Modifizierte road_polygons mit 'junction_indices' attribute
    """
    # Markiere jeden Endpoint
    for road_idx, road in enumerate(road_polygons):
        road["junction_indices"] = {"start": None, "end": None}

    for junction_idx, junction in enumerate(junctions):
        for road_idx in junction["road_indices"]:
            if road_idx < len(road_polygons):
                conn_types = junction["connection_types"].get(road_idx, [])
                for conn_type in conn_types:
                    if road_idx < len(road_polygons):
                        road_polygons[road_idx]["junction_indices"][
                            conn_type
                        ] = junction_idx

    return road_polygons


def analyze_junction_types(junctions):
    """
    Analysiert Junctions um deren Typ zu bestimmen (T, X, Einfädlung etc.).

    Args:
        junctions: Liste von Junctions

    Returns:
        Dict mit Statistiken:
        {
            'T_junctions': count,      # 3 Straßen
            'X_junctions': count,      # 4+ Straßen
            'endpoints': count,        # 2 Straßen (z.B. Einfädlung)
        }
    """
    stats = {"T_junctions": 0, "X_junctions": 0, "endpoints": 0}

    for junction in junctions:
        num_roads = len(junction["road_indices"])
        if num_roads == 3:
            stats["T_junctions"] += 1
        elif num_roads >= 4:
            stats["X_junctions"] += 1
        elif num_roads == 2:
            stats["endpoints"] += 1

    return stats


def debug_junctions(junctions, road_polygons):
    """
    Gibt Debug-Info über erkannte Junctions aus.

    Args:
        junctions: Liste von Junctions
        road_polygons: Straßen (für Kontext)
    """
    if not junctions:
        print("  ℹ Keine Junctions erkannt")
        return

    stats = analyze_junction_types(junctions)
    total_roads = len(road_polygons)

    print(f"  ℹ {len(junctions)} Junctions erkannt:")
    print(f"      - T-Kreuzungen (3 Straßen):    {stats['T_junctions']}")
    print(f"      - Kreuzungen (4+ Straßen):     {stats['X_junctions']}")
    print(f"      - Einfädelungen (2 Straßen):   {stats['endpoints']}")

    # Statistik über Straßenbeteiligung
    road_participation = {}
    for junction_idx, junction in enumerate(junctions):
        for road_idx in junction["road_indices"]:
            if road_idx not in road_participation:
                road_participation[road_idx] = 0
            road_participation[road_idx] += 1

    if road_participation:
        max_junctions = max(road_participation.values())
        multi_junction_roads = sum(1 for v in road_participation.values() if v > 1)
        print(f"      - Straßen mit >1 Junction:    {multi_junction_roads}")
        print(f"      - Max. Junctions pro Straße:  {max_junctions}")


def snap_road_endpoints_to_junctions(road_polygons, junctions):
    """
    Snapptt Straßen-Endpunkte auf die exakten Junction-Positionen.

    WICHTIG: Dies muss VOR der Glättung (smooth_roads_with_spline) geschehen,
    damit die Spline die korrekten End-Positionen hat.

    Args:
        road_polygons: Liste von Straßen (wird modifiziert)
        junctions: Liste von Junctions mit 'position' und 'road_indices'

    Returns:
        Modifizierte road_polygons mit gesnappten Endpunkten
    """
    if not junctions:
        return road_polygons

    snapped_count = 0

    for junction_idx, junction in enumerate(junctions):
        junction_pos = junction["position"]  # (x, y, z)

        for road_idx in junction["road_indices"]:
            if road_idx >= len(road_polygons):
                continue

            road = road_polygons[road_idx]
            coords = road["coords"]

            if len(coords) < 2:
                continue

            # Prüfe Connection-Typ für diese Straße an dieser Junction
            conn_types = junction["connection_types"].get(road_idx, [])

            # Snapp den Start-Punkt wenn diese Straße am Start in dieser Junction anfängt
            if "start" in conn_types:
                # Ersetze den Anfangspunkt
                coords[0] = junction_pos
                snapped_count += 1

            # Snapp den End-Punkt wenn diese Straße am End in dieser Junction endet
            if "end" in conn_types:
                # Ersetze den Endpunkt
                coords[-1] = junction_pos
                snapped_count += 1

    if snapped_count > 0:
        print(f"  ℹ {snapped_count} Straßen-Endpunkte auf Junctions gesnappt")

    return road_polygons
