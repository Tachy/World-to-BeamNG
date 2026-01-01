"""
Erkennung und Handling von Strassen-Junctions direkt in Centerlines.

Diese Module erkennt Kreuzungen und T-Junctions direkt aus den Centerline-Koordinaten
(wo Strassen-Punkte verbunden sind), um diese bei der Mesh-Generierung sauber zu meshen.
"""

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection


def detect_junctions_in_centerlines(road_polygons):
    """
    Erkennt Junctions (Kreuzungen/Einmuendungen) direkt in den Centerlines.

    Eine Junction ist ein Punkt, wo mehrere Strassen zusammenkommen.
    OSM-Daten haben Strassenenden als Punkte, die verbunden sind -> natuerliche Junctions.

    Args:
        road_polygons: Liste von Strassen-Dicts mit 'coords', 'id', 'name'

    Returns:
        List of junctions with:
        {
            'position': (x, y, z),         # 3D-Position der Junction
            'road_indices': [i, j, ...],   # Indices der beteiligten Strassen
            'connection_types': [...]      # 'end' oder 'start' pro Strasse
        }
    """
    if not road_polygons:
        return []

    # Sammle alle Strassenenden (Anfang und Ende)
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

    # Baue KDTree fuer schnelle Nachbarschaftssuche (nur XY-Koordinaten)
    endpoints_xy = np.array([(p[0], p[1]) for p in endpoints])
    kdtree = cKDTree(endpoints_xy)

    # Finde alle Punkte, die nah beieinander liegen (Toleranz 0.5 m = 50cm)
    endpoint_merge_tol = 0.5
    junction_pairs = kdtree.query_pairs(r=endpoint_merge_tol)

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

    def _direction_at_endpoint(coords, is_start):
        if is_start and len(coords) >= 2:
            p1 = np.array(coords[0][:2])
            p2 = np.array(coords[1][:2])
        elif not is_start and len(coords) >= 2:
            p1 = np.array(coords[-2][:2])
            p2 = np.array(coords[-1][:2])
        else:
            return np.array([1.0, 0.0])
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm > 0.01:
            direction = direction / norm
        return direction

    def _get_direction_at_point(road_idx, proj_xy):
        """Berechnet Straßenrichtung an einem beliebigen Punkt (NumPy-vektorisiert)."""
        coords = road_polygons[road_idx].get("coords", [])
        if len(coords) < 2:
            return np.array([1.0, 0.0])

        # Vektorisierte Segment-Berechnung
        coords_xy = np.array(
            (
                coords[:, :2]
                if isinstance(coords, np.ndarray)
                else [c[:2] for c in coords]
            ),
            dtype=np.float64,
        )

        # Alle Segmente: p1 = coords[0:-1], p2 = coords[1:]
        p1 = coords_xy[:-1]  # (n_segs, 2)
        p2 = coords_xy[1:]  # (n_segs, 2)
        seg = p2 - p1  # (n_segs, 2)
        seg_len_sq = np.sum(seg**2, axis=1, keepdims=True)  # (n_segs, 1)

        # Filtere zu kurze Segmente
        valid = seg_len_sq.ravel() > 1e-12
        if not np.any(valid):
            return np.array([1.0, 0.0])

        p1_valid = p1[valid]
        p2_valid = p2[valid]
        seg_valid = seg[valid]
        seg_len_sq_valid = seg_len_sq[valid]

        # Projiziere Punkt auf alle gültigen Segmente (Broadcasting)
        vec_to_point = np.array(proj_xy) - p1_valid  # (n_valid, 2)
        t = (
            np.sum(vec_to_point * seg_valid, axis=1, keepdims=True) / seg_len_sq_valid
        )  # (n_valid, 1)
        t = np.clip(t, 0, 1)

        # Nächste Punkte auf Segmenten
        proj_pts = p1_valid + t * seg_valid  # (n_valid, 2)

        # Abstände zum Punkt (Broadcasting)
        dists = np.linalg.norm(np.array(proj_xy) - proj_pts, axis=1)  # (n_valid,)
        best_idx = np.argmin(dists)

        # Richtung des besten Segments
        best_seg = seg_valid[best_idx]
        norm = np.linalg.norm(best_seg)

        return best_seg / norm if norm > 0.01 else np.array([1.0, 0.0])

    def _get_z_at_point(road_idx, xy_point):
        """Interpoliert Z-Koordinate an einem XY-Punkt auf der Straße (NumPy-vektorisiert)."""
        coords = road_polygons[road_idx].get("coords", [])
        if not coords or len(coords) < 2:
            return 0.0 if not coords else coords[0][2]

        # Konvertiere zu NumPy Arrays
        coords_array = np.array(coords, dtype=np.float64)
        coords_xy = coords_array[:, :2]
        coords_z = coords_array[:, 2]

        # Vektorisierte Segment-Berechnung
        p1 = coords_xy[:-1]  # (n_segs, 2)
        p2 = coords_xy[1:]  # (n_segs, 2)
        z1 = coords_z[:-1]  # (n_segs,)
        z2 = coords_z[1:]  # (n_segs,)

        # Mittelpunkte aller Segmente
        seg_mids = (p1 + p2) / 2  # (n_segs, 2)

        # Abstände aller Mittelpunkte zum Punkt (Broadcasting)
        dists = np.linalg.norm(seg_mids - np.array(xy_point), axis=1)  # (n_segs,)
        best_idx = np.argmin(dists)

        # Z-Wert des nächsten Segments
        return (z1[best_idx] + z2[best_idx]) / 2

    def _add_junction(position_xyz, cluster_indices, extra_connections=None):
        cluster_points = [endpoints[i] for i in cluster_indices]
        avg_x, avg_y, avg_z = position_xyz

        junction_roads = {}
        direction_vectors = {}

        for idx in cluster_indices:
            ep = endpoints[idx]
            road_idx = ep[3]
            is_start = ep[4]
            connection_type = "start" if is_start else "end"

            junction_roads.setdefault(road_idx, []).append(connection_type)
            coords = road_polygons[road_idx]["coords"]
            direction_vectors[road_idx] = _direction_at_endpoint(coords, is_start)

        if extra_connections:
            for road_idx, conn_type, direction in extra_connections:
                junction_roads.setdefault(road_idx, []).append(conn_type)
                if direction is not None:
                    direction_vectors[road_idx] = direction

        road_list = sorted(list(junction_roads.keys()))
        if len(road_list) >= 2:
            junctions.append(
                {
                    "position": (avg_x, avg_y, avg_z),
                    "road_indices": road_list,
                    "connection_types": {r: junction_roads[r] for r in road_list},
                    "direction_vectors": direction_vectors,
                    "num_connections": len(cluster_indices)
                    + (len(extra_connections) if extra_connections else 0),
                }
            )

    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue  # Keine echte Junction

        # Vektorisierte Mittelwert-Berechnung
        cluster_points = np.array(
            [endpoints[i][:3] for i in cluster_indices], dtype=np.float64
        )
        avg_point = np.mean(cluster_points, axis=0)

        _add_junction(tuple(avg_point), cluster_indices)

    # ---- Zusätzliche Erkennung: Endpoint auf durchgehender Centerline (T-Junction) ----
    # Erkennt Punkte, bei denen ein Endpoint auf der Mittellinie einer anderen Straße endet
    # (typisch für T-Junctions mit durchgehender Straße).
    t_search_radius = 10.0  # Meter - breite Vorauswahl mit KDTree
    t_line_tol = 0.5  # Meter - Toleranz für Punkt-zu-Linie Entfernung (50cm)
    merge_tol = 1.0  # Zusammenführungs-Toleranz zu bestehenden Junctions (1.0m)
    t_count = 0

    # Sammmle alle Linienpunkte mit ihrem Straßen-Index
    all_line_points = []  # [(x, y, road_idx, point_idx)]
    line_points_xy = []

    for road_idx, road in enumerate(road_polygons):
        coords = road.get("coords", [])
        for pt_idx, coord in enumerate(coords):
            all_line_points.append((coord[0], coord[1], road_idx, pt_idx))
            line_points_xy.append([coord[0], coord[1]])

    if not line_points_xy:
        pass
    else:
        line_kdtree = cKDTree(np.array(line_points_xy))

        t_count = 0

        for ep_x, ep_y, ep_z, road_idx, is_start in endpoints:
            # STUFE 1: KDTree-Vorauswahl - finde Linienpunkte in 10m Umkreis (Performance!)
            nearby_dists, nearby_indices = line_kdtree.query(
                [ep_x, ep_y], k=50, distance_upper_bound=t_search_radius
            )

            # nearby_indices can be a single index or array depending on k value
            if np.isscalar(nearby_indices):
                nearby_indices = [nearby_indices]
                nearby_dists = [nearby_dists]
            else:
                nearby_indices = [
                    i
                    for i, d in zip(nearby_indices, nearby_dists)
                    if d <= t_search_radius
                ]

            for line_pt_idx in nearby_indices:
                if line_pt_idx >= len(all_line_points):
                    continue

                lx, ly, other_road_idx, pt_idx_on_line = all_line_points[line_pt_idx]

                if other_road_idx == road_idx:
                    continue  # Gleiche Straße

                # STUFE 2: Projiziere auf die gesamte Linie und prüfe Punkt-zu-Linie Entfernung
                other_coords = road_polygons[other_road_idx].get("coords", [])
                if len(other_coords) < 2:
                    continue

                other_line = LineString([(c[0], c[1]) for c in other_coords])
                proj_dist = other_line.project(Point(ep_x, ep_y))
                proj_pt = other_line.interpolate(proj_dist)
                proj_xy = (proj_pt.x, proj_pt.y)

                # Berechne echte Punkt-zu-Linie Entfernung
                point_to_line_dist = np.linalg.norm(
                    np.array([ep_x - proj_pt.x, ep_y - proj_pt.y])
                )

                # STUFE 3: Nur akzeptieren, wenn Punkt wirklich nah bei der Linie liegt (10cm)
                if point_to_line_dist > t_line_tol:
                    continue  # Punkt ist zu weit weg von der Linie

                proj_z = ep_z

                # Prüfe ob bereits eine Junction an DIESER Position existiert (positionsbasiert!)
                # Erlaubt mehrere Junctions zwischen denselben Straßen an verschiedenen Positionen
                already_exists_here = False
                for j in junctions:
                    j_pos = np.array(j["position"][:2])
                    if (
                        np.linalg.norm(j_pos - np.array([proj_xy[0], proj_xy[1]]))
                        <= merge_tol
                    ):
                        # Junction existiert bereits an dieser Position
                        already_exists_here = True
                        break

                if already_exists_here:
                    continue  # Junction an dieser Position bereits vorhanden

                # Richtung auf der Linie an der Projektionsstelle
                through_dir = _get_direction_at_point(other_road_idx, proj_xy)
                new_junc_pos = (proj_xy[0], proj_xy[1], proj_z)

                # Versuche mit bestehender Junction zu mergen
                merged = False
                for j in junctions:
                    j_pos = np.array(j["position"][:2])
                    if np.linalg.norm(j_pos - np.array(new_junc_pos[:2])) <= merge_tol:
                        if road_idx not in j["road_indices"]:
                            j["road_indices"].append(road_idx)
                            j["connection_types"][road_idx] = [
                                "start" if is_start else "end"
                            ]
                            j["direction_vectors"][road_idx] = _direction_at_endpoint(
                                road_polygons[road_idx]["coords"], is_start
                            )
                        if other_road_idx not in j["road_indices"]:
                            j["road_indices"].append(other_road_idx)
                            j["connection_types"][other_road_idx] = ["mid"]
                            j["direction_vectors"][other_road_idx] = through_dir
                        merged = True
                        t_count += 1
                        break

                if not merged:
                    extra_conns = [
                        (
                            road_idx,
                            "start" if is_start else "end",
                            _direction_at_endpoint(
                                road_polygons[road_idx]["coords"], is_start
                            ),
                        ),
                        (other_road_idx, "mid", through_dir),
                    ]
                    _add_junction(new_junc_pos, [], extra_connections=extra_conns)
                    t_count += 1

        # Statistik-Ausgabe erfolgt zentral über junction_stats

    # ---- Dritte Erkennung: Line-on-Line Kreuzungen (X-Junctions ohne Endpoint-Match) ----
    # Erkennt Kreuzungen, wo zwei Straßen sich kreuzen, aber die Endpunkte nicht exakt aufeinander treffen

    ll_search_radius = 10.0  # Meter - Vorauswahl mit KDTree
    ll_line_tol = 0.5  # Meter - Toleranz für Line-zu-Line Entfernung (50cm)
    ll_count = 0

    # Verwende den bereits erstellten KDTree der Linienpunkte für Performance
    if line_points_xy:
        # Iteriere über alle Road-Paare

        for road_idx, road in enumerate(road_polygons):
            coords = road.get("coords", [])
            if len(coords) < 2:
                continue

            # Erstelle LineString für diese Straße
            road_line = LineString([(c[0], c[1]) for c in coords])
            road_bounds = road_line.bounds  # (minx, miny, maxx, maxy)

            # Finde Linienpunkte anderer Straßen in der Nähe (KDTree-Vorauswahl)
            # Nutze Mittelpunkt der Straße als Suchpunkt
            center_x = (road_bounds[0] + road_bounds[2]) / 2
            center_y = (road_bounds[1] + road_bounds[3]) / 2
            search_dist = (
                max(road_bounds[2] - road_bounds[0], road_bounds[3] - road_bounds[1])
                / 2
                + ll_search_radius
            )

            nearby_indices = line_kdtree.query_ball_point(
                [center_x, center_y], r=search_dist
            )

            # Extrahiere die Straßen-IDs der nahen Punkte
            nearby_road_ids = set()
            for idx in nearby_indices:
                if idx < len(all_line_points):
                    _, _, other_road_idx, _ = all_line_points[idx]
                    if other_road_idx != road_idx:
                        nearby_road_ids.add(other_road_idx)

            # Prüfe jede Straße in der Nähe
            for other_road_idx in nearby_road_ids:
                # Prüfe nur aufsteigende Paare um Duplikate zu vermeiden (road_idx < other_road_idx)
                if road_idx >= other_road_idx:
                    continue

                # Berechne Line-zu-Line Distanz mit Shapely
                other_coords = road_polygons[other_road_idx].get("coords", [])
                if len(other_coords) < 2:
                    continue

                other_line = LineString([(c[0], c[1]) for c in other_coords])
                line_dist = road_line.distance(other_line)

                if line_dist <= ll_line_tol:
                    # Die Linien sind sehr nah beieinander oder kreuzen sich!
                    # Finde den Kreuzungspunkt oder den nächsten Punkt
                    intersection = road_line.intersection(other_line)

                    if intersection.is_empty:
                        # Kein Schnittpunkt, aber sehr nah - finde nächsten Punkt
                        # Nutze den Punkt auf Line1 der am nächsten zu Line2 ist
                        # und den Punkt auf Line2 der am nächsten zu Line1 ist

                        # Finde den Punkt auf road_line am nächsten zu other_line
                        nearest_dist = float("inf")
                        nearest_pt1 = None
                        for coord in coords:
                            dist = other_line.distance(Point(coord[0], coord[1]))
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_pt1 = Point(coord[0], coord[1])

                        # Finde den Punkt auf other_line am nächsten zu road_line
                        nearest_dist = float("inf")
                        nearest_pt2 = None
                        for coord in other_coords:
                            dist = road_line.distance(Point(coord[0], coord[1]))
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_pt2 = Point(coord[0], coord[1])

                        if nearest_pt1 and nearest_pt2:
                            # Mittelpunkt zwischen den zwei nächsten Punkten
                            cross_x = (nearest_pt1.x + nearest_pt2.x) / 2
                            cross_y = (nearest_pt1.y + nearest_pt2.y) / 2
                        else:
                            continue  # Kann keinen gültigen Punkt finden
                    else:
                        # Schnittpunkt gefunden
                        from shapely.geometry import MultiPoint, GeometryCollection

                        if isinstance(intersection, Point):
                            cross_x, cross_y = intersection.x, intersection.y
                        elif isinstance(intersection, (MultiPoint, GeometryCollection)):
                            # Mehrere Schnittpunkte - nutze den ersten
                            geoms = (
                                list(intersection.geoms)
                                if hasattr(intersection, "geoms")
                                else [intersection]
                            )
                            if geoms and isinstance(geoms[0], Point):
                                cross_x, cross_y = geoms[0].x, geoms[0].y
                            else:
                                continue  # Kann nicht als Punkt interpretiert werden
                        else:
                            continue  # LineString oder Polygon - zu komplex

                    # FRÜHE Duplikatsprüfung BEVOR teure Z-Berechnung
                    already_exists_here = False
                    for j in junctions:
                        j_pos = np.array(j["position"][:2])
                        if (
                            np.linalg.norm(j_pos - np.array([cross_x, cross_y]))
                            <= merge_tol
                        ):
                            already_exists_here = True
                            break

                    if already_exists_here:
                        continue  # Junction an dieser Position bereits vorhanden

                    # Interpoliere Z-Koordinate aus beiden Straßen (Durchschnitt)
                    best_z1 = _get_z_at_point(road_idx, (cross_x, cross_y))
                    best_z2 = _get_z_at_point(other_road_idx, (cross_x, cross_y))
                    cross_z = (best_z1 + best_z2) / 2

                    # Berechne Richtungen an der Kreuzungsstelle für beide Straßen
                    dir1 = _get_direction_at_point(road_idx, (cross_x, cross_y))
                    dir2 = _get_direction_at_point(other_road_idx, (cross_x, cross_y))

                    new_junc_pos = (cross_x, cross_y, cross_z)

                    # Versuche mit bestehender Junction zu mergen
                    merged = False
                    for j in junctions:
                        j_pos = np.array(j["position"][:2])
                        if (
                            np.linalg.norm(j_pos - np.array(new_junc_pos[:2]))
                            <= merge_tol
                        ):
                            # Füge beide Straßen als "mid" hinzu (Kreuzung, nicht Endpoint)
                            if road_idx not in j["road_indices"]:
                                j["road_indices"].append(road_idx)
                                j["connection_types"][road_idx] = ["mid"]
                                j["direction_vectors"][road_idx] = dir1
                            if other_road_idx not in j["road_indices"]:
                                j["road_indices"].append(other_road_idx)
                                j["connection_types"][other_road_idx] = ["mid"]
                                j["direction_vectors"][other_road_idx] = dir2
                            merged = True
                            ll_count += 1
                            break

                    if not merged:
                        # Neue Junction erstellen - beide Straßen als "mid" (durchgehend)
                        extra_conns = [
                            (road_idx, "mid", dir1),
                            (other_road_idx, "mid", dir2),
                        ]
                        _add_junction(new_junc_pos, [], extra_connections=extra_conns)
                        ll_count += 1

        # Statistik-Ausgabe erfolgt zentral über junction_stats

    return junctions


def mark_junction_endpoints(road_polygons, junctions):
    """
    Markiert Strassen-Endpoints, die zu Junctions gehoeren.

    Dies wird später bei der Mesh-Generierung verwendet, um diese Punkte
    beim Stitchen zu identifizieren.

    Args:
        road_polygons: Liste von Strassen (wird modifiziert)
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
    Klassifiziert nach Anzahl der ankommenden Straßen.

    Args:
        junctions: Liste von Junctions

    Returns:
        Dict mit Statistiken und Klassifizierung:
        {
            'two_roads': count,        # 2 Strassen (Endpunkte treffen sich)
            'three_roads': count,      # 3 Strassen (T-Junctions)
            'four_roads': count,       # 4 Strassen (X-Junctions)
            'five_plus': count,        # 5+ Strassen
            'junction_details': [...]  # Details pro Junctions mit >4 Strassen
        }
    """
    stats = {
        "two_roads": 0,
        "three_roads": 0,
        "four_roads": 0,
        "five_plus": 0,
        "five_plus_details": [],
    }

    # Klassifiziere jede Junction
    for junction_idx, junction in enumerate(junctions):
        num_roads = len(junction["road_indices"])

        if num_roads == 2:
            stats["two_roads"] += 1
        elif num_roads == 3:
            stats["three_roads"] += 1
        elif num_roads == 4:
            stats["four_roads"] += 1
        elif num_roads >= 5:
            stats["five_plus"] += 1

            # Sammle Details für 5+ Straßen Junctions
            connection_info = []
            for road_idx in junction["road_indices"]:
                conn_types = junction["connection_types"].get(road_idx, [])
                connection_info.append((road_idx, conn_types))

            stats["five_plus_details"].append(
                {
                    "idx": junction_idx,
                    "num_roads": num_roads,
                    "position": junction["position"],
                    "road_connections": connection_info,
                }
            )

    return stats


def junction_stats(junctions, road_polygons):
    """Gibt Statistik über erkannte Junctions aus (Produktiv-Einsatz)."""
    if not junctions:
        print("  [i] Keine Junctions erkannt")
        return

    stats = analyze_junction_types(junctions)
    total_roads = len(road_polygons)

    print(f"  [i] {len(junctions)} Junctions erkannt:")
    print(f"      - 2 Strassen (Endpunkte treffen):  {stats['two_roads']}")
    print(f"      - 3 Strassen (T-Junctions):        {stats['three_roads']}")
    print(f"      - 4 Strassen (X-Junctions):        {stats['four_roads']}")
    print(f"      - 5+ Strassen:                     {stats['five_plus']}")

    # Statistik ueber Strassenbeteiligung
    road_participation = {}
    for junction_idx, junction in enumerate(junctions):
        for road_idx in junction["road_indices"]:
            if road_idx not in road_participation:
                road_participation[road_idx] = 0
            road_participation[road_idx] += 1

    if road_participation:
        max_junctions = max(road_participation.values())
        multi_junction_roads = sum(1 for v in road_participation.values() if v > 1)
        print(f"\n      Straßen-Beteiligung:")
        print(f"      - Strassen mit >1 Junction:        {multi_junction_roads}")
        print(f"      - Max. Junctions pro Strasse:      {max_junctions}")
