"""
Erkennung und Handling von Strassen-Junctions direkt in Centerlines.

Diese Module erkennt Kreuzungen und T-Junctions direkt aus den Centerline-Koordinaten
(wo Strassen-Punkte verbunden sind), um diese bei der Mesh-Generierung sauber zu meshen.
"""

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection, box
from shapely.strtree import STRtree


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

    # Sammle alle Strassenenden (Anfang und Ende) und precompute Segmentdaten
    endpoints = []  # (x, y, z, road_idx, is_start)
    road_cache = []  # pro Straße: vorberechnete Segmentdaten für Direction/Interpolation

    for road_idx, road in enumerate(road_polygons):
        coords = road["coords"]

        if len(coords) >= 2:
            # Anfangspunkt
            start = coords[0]
            endpoints.append((start[0], start[1], start[2], road_idx, True))

            # Endpunkt
            end = coords[-1]
            endpoints.append((end[0], end[1], end[2], road_idx, False))

            coords_xy = coords[:, :2] if isinstance(coords, np.ndarray) else np.array(coords)[:, :2]
            coords_z = coords[:, 2] if isinstance(coords, np.ndarray) else np.array(coords)[:, 2]

            p1 = coords_xy[:-1]
            p2 = coords_xy[1:]
            seg = p2 - p1
            seg_len_sq = np.sum(seg * seg, axis=1)
            valid = seg_len_sq > 1e-12

            seg_valid = seg[valid]
            seg_len_sq_valid = seg_len_sq[valid]
            p1_valid = p1[valid]

            # Vorbereitete Start/End-Richtungen (normalisiert) für _direction_at_endpoint
            start_dir = seg_valid[0] if len(seg_valid) else np.array([1.0, 0.0])
            end_dir = seg_valid[-1] if len(seg_valid) else np.array([1.0, 0.0])
            sd_norm = np.sqrt(start_dir.dot(start_dir))
            ed_norm = np.sqrt(end_dir.dot(end_dir))
            if sd_norm > 0.01:
                start_dir = start_dir / sd_norm
            else:
                start_dir = np.array([1.0, 0.0])
            if ed_norm > 0.01:
                end_dir = end_dir / ed_norm
            else:
                end_dir = np.array([1.0, 0.0])

            # Für Z-Interpolation: Segment-Mittelpunkte und mittleres Z pro Segment
            seg_mids = (p1 + p2) / 2.0
            z_mid = (coords_z[:-1] + coords_z[1:]) / 2.0

            road_cache.append(
                {
                    "coords": coords,
                    "coords_xy": coords_xy,
                    "coords_z": coords_z,
                    "p1_valid": p1_valid,
                    "seg_valid": seg_valid,
                    "seg_len_sq_valid": seg_len_sq_valid,
                    "seg_mids": seg_mids,
                    "z_mid": z_mid,
                    "start_dir": start_dir,
                    "end_dir": end_dir,
                }
            )
        else:
            road_cache.append(
                {
                    "coords": coords,
                    "coords_xy": (np.array(coords)[:, :2] if len(coords) else np.empty((0, 2))),
                    "coords_z": (np.array(coords)[:, 2] if len(coords) else np.empty((0,))),
                    "p1_valid": np.empty((0, 2)),
                    "seg_valid": np.empty((0, 2)),
                    "seg_len_sq_valid": np.empty((0,)),
                    "seg_mids": np.empty((0, 2)),
                    "z_mid": np.empty((0,)),
                    "start_dir": np.array([1.0, 0.0]),
                    "end_dir": np.array([1.0, 0.0]),
                }
            )

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

    def _direction_at_endpoint(road_idx, is_start):
        cache = road_cache[road_idx]
        return cache["start_dir"] if is_start else cache["end_dir"]

    def _get_direction_at_point(road_idx, proj_xy):
        """Berechnet Straßenrichtung an einem beliebigen Punkt (vektorisiert mit reduce)."""
        cache = road_cache[road_idx]
        seg_valid = cache["seg_valid"]
        seg_len_sq_valid = cache["seg_len_sq_valid"]
        p1_valid = cache["p1_valid"]

        if seg_valid.size == 0:
            return np.array([1.0, 0.0])

        # OPTIMIZATION: Nutze dot() statt sum() für Vektorprodukte
        vec_to_point = proj_xy - p1_valid  # Broadcasting: (n_valid, 2)
        # Dot-Product ohne reshape: sum(vec * seg) = sum along axis 1
        t = np.einsum("ij,ij->i", vec_to_point, seg_valid) / seg_len_sq_valid
        t = np.clip(t, 0, 1)

        # Nächste Punkte auf Segmenten
        proj_pts = p1_valid + t[:, None] * seg_valid  # (n_valid, 2)

        # OPTIMIZATION: Nutze einsum für dot-product (schneller als sum+*+*)
        diff = proj_pts - proj_xy
        dists_sq = np.einsum("ij,ij->i", diff, diff)
        best_idx = int(np.argmin(dists_sq))

        # Richtung des besten Segments
        best_seg = seg_valid[best_idx]
        seg_norm = np.sqrt(seg_len_sq_valid[best_idx])
        return best_seg / seg_norm if seg_norm > 0.01 else np.array([1.0, 0.0])

    def _get_z_at_point(road_idx, xy_point):
        """Interpoliert Z-Koordinate an einem XY-Punkt auf der Straße (vektorisiert mit einsum)."""
        cache = road_cache[road_idx]
        coords = cache["coords"]
        if not coords or len(coords) < 2:
            return 0.0 if not coords else coords[0][2]

        seg_mids = cache["seg_mids"]
        z_mid = cache["z_mid"]

        if seg_mids.size == 0:
            return coords[0][2]

        # OPTIMIZATION: Nutze einsum statt sum() für Distanzberechnung
        diff = seg_mids - np.array(xy_point)
        dists_sq = np.einsum("ij,ij->i", diff, diff)
        best_idx = int(np.argmin(dists_sq))

        return z_mid[best_idx]

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
            direction_vectors[road_idx] = _direction_at_endpoint(road_idx, is_start)

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
                    "num_connections": len(cluster_indices) + (len(extra_connections) if extra_connections else 0),
                }
            )

    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue  # Keine echte Junction

        # Vektorisierte Mittelwert-Berechnung
        cluster_points = np.array([endpoints[i][:3] for i in cluster_indices], dtype=np.float64)
        avg_point = np.mean(cluster_points, axis=0)

        _add_junction(tuple(avg_point), cluster_indices)

    # ---- Zusätzliche Erkennung: Endpoint auf durchgehender Centerline (T-Junction) ----
    # Erkennt Punkte, bei denen ein Endpoint auf der Mittellinie einer anderen Straße endet
    # (typisch für T-Junctions mit durchgehender Straße).
    t_search_radius = 10.0  # Meter - breite Vorauswahl mit KDTree
    t_line_tol = 0.5  # Meter - Toleranz für Punkt-zu-Linie Entfernung (50cm)
    t_line_tol_sq = t_line_tol * t_line_tol
    merge_tol = 1.0  # Zusammenführungs-Toleranz zu bestehenden Junctions (1.0m)
    merge_tol_sq = merge_tol * merge_tol
    t_count = 0

    # Sammle alle Linienpunkte mit ihrem Straßen-Index und baue LineStrings/Indexe
    all_line_points = []  # [(x, y, road_idx, point_idx)]
    line_points_xy = []
    line_strings = []  # index-aligniert zu road_polygons (None für zu kurze Straßen)
    indexed_geoms = []  # Geometrien, die in den STRtree kommen
    geom_to_idx = {}  # STRtree-Geometrie → road_idx

    for road_idx, road in enumerate(road_polygons):
        coords = road.get("coords", [])

        # Punkte sammeln (für KDTree aus T-Erkennung)
        for pt_idx, coord in enumerate(coords):
            all_line_points.append((coord[0], coord[1], road_idx, pt_idx))
            line_points_xy.append([coord[0], coord[1]])

        # LineString einmalig bauen
        if len(coords) >= 2:
            ls = LineString([(c[0], c[1]) for c in coords])
            line_strings.append(ls)
            indexed_geoms.append(ls)
            geom_to_idx[ls] = road_idx
        else:
            line_strings.append(None)

    if not line_points_xy:
        pass
    else:
        line_kdtree = cKDTree(np.array(line_points_xy))

        t_count = 0

        for ep_x, ep_y, ep_z, road_idx, is_start in endpoints:
            # STUFE 1: KDTree-Vorauswahl - finde Linienpunkte in 10m Umkreis (Performance!)
            nearby_dists, nearby_indices = line_kdtree.query([ep_x, ep_y], k=50, distance_upper_bound=t_search_radius)

            # nearby_indices can be a single index or array depending on k value
            if np.isscalar(nearby_indices):
                nearby_indices = [nearby_indices]
                nearby_dists = [nearby_dists]
            else:
                nearby_indices = [i for i, d in zip(nearby_indices, nearby_dists) if d <= t_search_radius]

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
                # STUFE 3: Nur akzeptieren, wenn Punkt wirklich nah bei der Linie liegt (10cm)
                dist_vec = np.array([ep_x - proj_pt.x, ep_y - proj_pt.y])
                point_to_line_dist_sq = np.sum(dist_vec * dist_vec)
                if point_to_line_dist_sq > t_line_tol_sq:
                    continue  # Punkt ist zu weit weg von der Linie

                proj_z = ep_z

                # Prüfe ob bereits eine Junction an DIESER Position existiert (positionsbasiert!)
                # Erlaubt mehrere Junctions zwischen denselben Straßen an verschiedenen Positionen
                existing_junction = None
                proj_xy_vec = np.array([proj_xy[0], proj_xy[1]])
                for j in junctions:
                    j_pos = np.array(j["position"][:2])
                    if np.sum((j_pos - proj_xy_vec) ** 2) <= merge_tol_sq:
                        # Junction existiert bereits an dieser Position
                        existing_junction = j
                        break

                # Wenn Junction existiert, versuche Straßen hinzuzufügen statt zu überspringen
                if existing_junction is not None:
                    # Versuche Straßen zu dieser existierenden Junction hinzuzufügen
                    through_dir = _get_direction_at_point(other_road_idx, proj_xy)

                    # Stelle sicher dass connection_types initialisiert ist
                    if "connection_types" not in existing_junction:
                        existing_junction["connection_types"] = {}
                    if "direction_vectors" not in existing_junction:
                        existing_junction["direction_vectors"] = {}

                    # Füge erste Straße hinzu oder aktualisiere connection_type
                    if road_idx not in existing_junction["road_indices"]:
                        existing_junction["road_indices"].append(road_idx)

                    conn_type = "start" if is_start else "end"
                    if conn_type not in existing_junction["connection_types"].get(road_idx, []):
                        if road_idx not in existing_junction["connection_types"]:
                            existing_junction["connection_types"][road_idx] = []
                        existing_junction["connection_types"][road_idx].append(conn_type)
                    existing_junction["direction_vectors"][road_idx] = _direction_at_endpoint(road_idx, is_start)

                    # Füge zweite Straße hinzu oder aktualisiere connection_type
                    if other_road_idx not in existing_junction["road_indices"]:
                        existing_junction["road_indices"].append(other_road_idx)

                    if "mid" not in existing_junction["connection_types"].get(other_road_idx, []):
                        if other_road_idx not in existing_junction["connection_types"]:
                            existing_junction["connection_types"][other_road_idx] = []
                        existing_junction["connection_types"][other_road_idx].append("mid")
                    existing_junction["direction_vectors"][other_road_idx] = through_dir

                    continue  # Fertig mit dieser Prüfung

                # Richtung auf der Linie an der Projektionsstelle
                through_dir = _get_direction_at_point(other_road_idx, proj_xy)
                new_junc_pos = (proj_xy[0], proj_xy[1], proj_z)

                # Versuche mit bestehender Junction zu mergen
                merged = False
                new_junc_xy = np.array(new_junc_pos[:2])
                for j in junctions:
                    j_pos = np.array(j["position"][:2])
                    if np.sum((j_pos - new_junc_xy) ** 2) <= merge_tol_sq:
                        if road_idx not in j["road_indices"]:
                            j["road_indices"].append(road_idx)
                            j["connection_types"][road_idx] = ["start" if is_start else "end"]
                            j["direction_vectors"][road_idx] = _direction_at_endpoint(road_idx, is_start)
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
                            _direction_at_endpoint(road_idx, is_start),
                        ),
                        (other_road_idx, "mid", through_dir),
                    ]
                    _add_junction(new_junc_pos, [], extra_connections=extra_conns)
                    t_count += 1

        # Statistik-Ausgabe erfolgt zentral über junction_stats

    # ---- Dritte Erkennung: Line-on-Line Kreuzungen (X-Junctions ohne Endpoint-Match) ----
    # Erkennt Kreuzungen, wo zwei Straßen sich kreuzen, aber die Endpunkte nicht exakt aufeinander treffen

    ll_search_radius = 10.0  # Meter - Vorauswahl mit KDTree
    ll_line_tol = 1.0  # Meter - Toleranz für Line-zu-Line Entfernung (1m)
    ll_count = 0

    # Verwende den bereits erstellten KDTree der Linienpunkte für Performance
    if line_points_xy and indexed_geoms:
        tree = STRtree(indexed_geoms)

        for road_idx, road in enumerate(road_polygons):
            road_line = line_strings[road_idx]
            if road_line is None:
                continue

            road_bounds = road_line.bounds  # (minx, miny, maxx, maxy)
            pad = ll_search_radius + ll_line_tol
            query_geom = box(
                road_bounds[0] - pad,
                road_bounds[1] - pad,
                road_bounds[2] + pad,
                road_bounds[3] + pad,
            )

            candidates = tree.query(query_geom)

            for other_geom in candidates:
                other_road_idx = geom_to_idx.get(other_geom)
                if other_road_idx is None or other_road_idx <= road_idx:
                    continue

                other_line = other_geom
                line_dist = road_line.distance(other_line)

                if line_dist > ll_line_tol:
                    continue

                intersection = road_line.intersection(other_line)

                if intersection.is_empty:
                    nearest_dist = float("inf")
                    nearest_pt1 = None
                    for coord in road.get("coords", []):
                        dist = other_line.distance(Point(coord[0], coord[1]))
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_pt1 = Point(coord[0], coord[1])

                    nearest_dist = float("inf")
                    nearest_pt2 = None
                    other_coords = road_polygons[other_road_idx].get("coords", [])
                    for coord in other_coords:
                        dist = road_line.distance(Point(coord[0], coord[1]))
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_pt2 = Point(coord[0], coord[1])

                    if nearest_pt1 and nearest_pt2:
                        cross_x = (nearest_pt1.x + nearest_pt2.x) / 2
                        cross_y = (nearest_pt1.y + nearest_pt2.y) / 2
                    else:
                        continue
                else:
                    if isinstance(intersection, Point):
                        cross_x, cross_y = intersection.x, intersection.y
                    elif isinstance(intersection, (MultiPoint, GeometryCollection)):
                        geoms = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
                        if geoms and isinstance(geoms[0], Point):
                            cross_x, cross_y = geoms[0].x, geoms[0].y
                        else:
                            continue
                    else:
                        continue

                already_exists_here = False
                cross_xy = np.array([cross_x, cross_y])
                for j in junctions:
                    j_pos = np.array(j["position"][:2])
                    if np.sum((j_pos - cross_xy) ** 2) <= merge_tol_sq:
                        already_exists_here = True
                        break

                if already_exists_here:
                    continue

                best_z1 = _get_z_at_point(road_idx, (cross_x, cross_y))
                best_z2 = _get_z_at_point(other_road_idx, (cross_x, cross_y))
                cross_z = (best_z1 + best_z2) / 2

                dir1 = _get_direction_at_point(road_idx, (cross_x, cross_y))
                dir2 = _get_direction_at_point(other_road_idx, (cross_x, cross_y))

                new_junc_pos = (cross_x, cross_y, cross_z)

                merged = False
                new_ll_xy = np.array(new_junc_pos[:2])
                for j in junctions:
                    j_pos = np.array(j["position"][:2])
                    if np.sum((j_pos - new_ll_xy) ** 2) <= merge_tol_sq:
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
                        road_polygons[road_idx]["junction_indices"][conn_type] = junction_idx

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


def split_roads_at_mid_junctions(road_polygons, junctions, merge_tol=0.5):
    """
    Splittet Strassen an Junction-Punkten, die als "mid" erkannt wurden.

    Ergebnis: neue Road-Liste, in der jeder Abschnitt eine eigene Strasse mit
    Start/End-Junction besitzt. Alle Eigenschaften der Originalstrasse werden
    kopiert, die IDs werden um einen Teil-Suffix erweitert.
    """

    if not junctions or not road_polygons:
        return road_polygons, junctions

    junction_positions = [np.asarray(j["position"], dtype=float) for j in junctions]

    def _dir_start(coords_arr):
        if len(coords_arr) < 2:
            return np.array([1.0, 0.0])
        v = coords_arr[1, :2] - coords_arr[0, :2]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _dir_end(coords_arr):
        if len(coords_arr) < 2:
            return np.array([1.0, 0.0])
        v = coords_arr[-1, :2] - coords_arr[-2, :2]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _new_id(base_id, part_idx):
        if isinstance(base_id, int):
            return base_id * 1000 + part_idx
        return f"{base_id}_p{part_idx}"

    new_roads = []
    old_to_new_map = {}  # old road idx -> list of new road indices

    for road_idx, road in enumerate(road_polygons):
        coords = np.asarray(road.get("coords", []), dtype=float)
        if len(coords) < 2:
            # Unveraendert uebernehmen
            new_road = road.copy()
            new_road.setdefault("junction_indices", {"start": None, "end": None})
            new_road["start_junction_id"] = None
            new_road["end_junction_id"] = None
            new_roads.append(new_road)
            # Mapping setzen, damit Junctions ihre Verbindungen behalten
            old_to_new_map[road_idx] = [len(new_roads) - 1]
            continue

        # Sammle bekannte Start/End-Junctions aus originalen connection_types
        start_junc_id = None
        end_junc_id = None
        for j_idx, j in enumerate(junctions):
            conn = j.get("connection_types", {}).get(road_idx, [])
            if "start" in conn:
                start_junc_id = j_idx
            if "end" in conn:
                end_junc_id = j_idx

        # Sammle alle mid-Junctions fuer diese Strasse
        cut_marks = []
        for j_idx, j in enumerate(junctions):
            conn = j.get("connection_types", {}).get(road_idx, [])
            if "mid" not in conn:
                continue
            pos = junction_positions[j_idx]
            p = coords[:, :2]
            seg = p[1:] - p[:-1]
            seg_len = np.linalg.norm(seg, axis=1)
            valid = seg_len > 1e-6
            if not np.any(valid):
                continue
            seg_len_sq = np.maximum(seg_len * seg_len, 1e-12)
            vec = pos[:2] - p[:-1]
            t = np.sum(vec * seg, axis=1) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = p[:-1] + seg * t[:, None]
            dist_sq = np.sum((proj - pos[:2]) ** 2, axis=1)
            best = int(np.argmin(dist_sq))
            best_t = float(t[best])
            best_proj = proj[best]
            best_z = coords[best, 2] + best_t * (coords[best + 1, 2] - coords[best, 2])
            # Arc-Position als segmentindex + t
            cut_marks.append((best + best_t, best_proj[0], best_proj[1], best_z, j_idx))

        if not cut_marks:
            new_road = road.copy()
            new_road.setdefault("junction_indices", {"start": None, "end": None})
            new_road["start_junction_id"] = None
            new_road["end_junction_id"] = None
            new_roads.append(new_road)
            # Mapping setzen, damit unveraenderte Strassen in Junctions verbleiben
            old_to_new_map[road_idx] = [len(new_roads) - 1]
            continue

        # Doppelte Schnitte (nahe beieinander) zusammenfassen
        cut_marks.sort(key=lambda x: x[0])
        merged_cuts = []
        for c in cut_marks:
            if not merged_cuts:
                merged_cuts.append(c)
                continue
            last = merged_cuts[-1]
            if abs(c[0] - last[0]) <= 1e-4:
                merged_cuts[-1] = c  # ersetze mit letzter (gleiches Segment)
            else:
                merged_cuts.append(c)

        # Map Segment -> Cuts
        cuts_by_seg = {}
        for c in merged_cuts:
            s_pos = c[0]
            seg_idx = int(np.floor(s_pos))
            t_seg = s_pos - seg_idx
            cuts_by_seg.setdefault(seg_idx, []).append((t_seg, c[1], c[2], c[3], c[4]))

        current_coords = [coords[0]]
        current_start_j = start_junc_id
        parts = []

        for seg_idx in range(len(coords) - 1):
            if seg_idx in cuts_by_seg:
                seg_cuts = sorted(cuts_by_seg[seg_idx], key=lambda x: x[0])
                for t_seg, x_cut, y_cut, z_cut, j_idx in seg_cuts:
                    cut_pt = np.array([x_cut, y_cut, z_cut])
                    current_coords.append(cut_pt)
                    parts.append((current_coords, current_start_j, j_idx))
                    current_coords = [cut_pt]
                    current_start_j = j_idx
            # füge Ende des Segments hinzu, falls kein Cut dort endet
            next_pt = coords[seg_idx + 1]
            current_coords.append(next_pt)

        # letztes Teilstück
        parts.append((current_coords, current_start_j, end_junc_id))

        # Baue neue Roads aus Parts
        base_id = road.get("id", f"road{road_idx}")
        new_ids_for_this = []
        for idx, (coords_part, start_j, end_j) in enumerate(parts, 1):
            coords_arr = np.asarray(coords_part)
            if len(coords_arr) < 2:
                continue
            new_road = road.copy()
            new_road["id"] = _new_id(base_id, idx)
            new_road["coords"] = coords_arr.tolist()
            new_road["start_junction_id"] = start_j
            new_road["end_junction_id"] = end_j
            new_road["junction_indices"] = {"start": start_j, "end": end_j}
            new_roads.append(new_road)
            new_ids_for_this.append(len(new_roads) - 1)

        if new_ids_for_this:
            old_to_new_map[road_idx] = new_ids_for_this
        else:
            old_to_new_map[road_idx] = []

    # Rebaue Junction-Liste basierend auf den neuen Roads, bewahre Junction-Zahl
    new_junctions = []
    for j_idx, j in enumerate(junctions):
        pos = junction_positions[j_idx]
        roads_here = []
        conn_types = {}
        dir_vectors = {}

        for old_ridx, conn_list in j.get("connection_types", {}).items():
            mapped = old_to_new_map.get(old_ridx, [])
            for new_ridx in mapped:
                road = new_roads[new_ridx]
                coords_arr = np.asarray(road.get("coords", []), dtype=float)

                # Setze fehlende Start/End IDs falls nötig
                if "start" in conn_list and road.get("start_junction_id") is None:
                    road["start_junction_id"] = j_idx
                    road["junction_indices"]["start"] = j_idx
                if "end" in conn_list and road.get("end_junction_id") is None:
                    road["end_junction_id"] = j_idx
                    road["junction_indices"]["end"] = j_idx

                # Nur zählen, wenn das Teilstück tatsächlich an dieser Junction startet/endet
                if road.get("start_junction_id") == j_idx:
                    roads_here.append(new_ridx)
                    conn_types.setdefault(new_ridx, []).append("start")
                    dir_vectors[new_ridx] = _dir_start(coords_arr)
                if road.get("end_junction_id") == j_idx:
                    roads_here.append(new_ridx)
                    conn_types.setdefault(new_ridx, []).append("end")
                    dir_vectors[new_ridx] = _dir_end(coords_arr)

        roads_unique = sorted(set(roads_here))
        if len(roads_unique) >= 2:
            new_junctions.append(
                {
                    "position": tuple(pos.tolist()),
                    "road_indices": roads_unique,
                    "connection_types": {r: conn_types.get(r, []) for r in roads_unique},
                    "direction_vectors": {r: dir_vectors.get(r, np.array([1.0, 0.0])) for r in roads_unique},
                    "num_connections": sum(len(conn_types.get(r, [])) for r in roads_unique),
                }
            )

    return new_roads, new_junctions


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
