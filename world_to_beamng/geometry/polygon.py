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
                    "osm_way_id": road.get("osm_way_id"),
                    "osm_nodes": road.get("osm_nodes"),
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


def _endpoint_matches(pt_a, pt_b, tol=1e-4):
    return abs(pt_a[0] - pt_b[0]) < tol and abs(pt_a[1] - pt_b[1]) < tol and abs(pt_a[2] - pt_b[2]) < tol


def _find_unique_touching_road(road_polygons, point, exclude_id, predicate=None):
    """Findet die eine Straße in road_polygons, deren Anfangs- oder Endpunkt mit `point` übereinstimmt
    (ausser `exclude_id`, optional gefiltert über `predicate(osm_tags)`); bei Mehrdeutigkeit (0 oder 2+
    Treffer, z.B. an einer echten Mehrwege-Kreuzung) wird None zurückgegeben - dort muss die scharfe Kante
    für die Junction-Logik erhalten bleiben.

    Returns:
        (road, touching_at_start) oder None
    """
    found = []
    for road in road_polygons:
        if road["id"] == exclude_id:
            continue
        if predicate and not predicate(road.get("osm_tags", {})):
            continue
        coords = road["coords"]
        if len(coords) < 2:
            continue
        if _endpoint_matches(coords[0], point):
            found.append((road, True))
        elif _endpoint_matches(coords[-1], point):
            found.append((road, False))
    return found[0] if len(found) == 1 else None


def _walk_bridge_approach(ordered, slope_threshold, max_extension):
    """Läuft `ordered` ab Index 0 (Berührpunkt zur Brücke) entlang, solange das Gefälle des jeweils
    nächsten Segments >= slope_threshold bleibt (noch Teil der Hangflanke, die die zu kurze Brücke nicht
    abdeckt) und die kumulierte Distanz max_extension nicht überschreitet; ist der Nachbar selbst kürzer,
    stoppt der Lauf an dessen eigenem Ende (keine dritte Straße wird einbezogen).

    Returns:
        (extension_points, remaining_ordered) - extension_points sind die neuen Brücken-Punkte in
        Richtung vom Berührpunkt weg (ohne den Berührpunkt selbst); remaining_ordered ist der beim
        Nachbarn verbleibende Rest (beginnend mit dem neuen, gemeinsamen Grenzpunkt).
    """
    idx = 0
    cum = 0.0
    n = len(ordered)
    while idx + 1 < n:
        a, b = ordered[idx], ordered[idx + 1]
        seg_dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if seg_dist < 1e-9:
            idx += 1
            continue
        slope = abs(b[2] - a[2]) / seg_dist
        if slope < slope_threshold:
            break
        if cum + seg_dist > max_extension + 1e-9:
            remaining = max_extension - cum
            if remaining > 1e-9:
                t = remaining / seg_dist
                point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))
                ordered = ordered[: idx + 1] + [point] + ordered[idx + 1 :]
                idx += 1
            break
        cum += seg_dist
        idx += 1

    return ordered[1 : idx + 1], ordered[idx:]


def extend_short_bridges_to_natural_grade(road_polygons, slope_threshold=None, max_extension=None):
    """Verlängert zu kurz getaggte Brücken in ihre angrenzende Oberflächenstraße hinein, bis dort wieder
    normales Gefälle herrscht: manche OSM-Brücken beginnen bereits mitten in der Hanglage statt auf
    Straßenniveau, wodurch die anschließende lineare Höheninterpolation (apply_structure_elevation_profiles)
    eine unrealistisch steile Rampe ergibt. Läuft VOR apply_structure_elevation_profiles, damit diese auf
    dem bereits verlängerten Verlauf arbeitet.

    Verlängert wird nur, wenn an einem Brücken-Ende genau EINE Oberflächenstraße anliegt (eindeutiger
    Anschluss); Tunnel/Galerien als "Nachbar" werden ignoriert (deren Höhenprofil ist kein echtes Gelände).
    Config: BRIDGE_APPROACH_SLOPE_THRESHOLD, BRIDGE_APPROACH_MAX_EXTENSION.
    """
    if slope_threshold is None:
        slope_threshold = config.BRIDGE_APPROACH_SLOPE_THRESHOLD
    if max_extension is None:
        max_extension = config.BRIDGE_APPROACH_MAX_EXTENSION

    is_surface = lambda tags: classify_structure(tags) == "surface"

    for bridge in [r for r in road_polygons if classify_structure(r.get("osm_tags", {})) == "bridge"]:
        coords = bridge["coords"]
        if len(coords) < 2:
            continue

        for at_start in (True, False):
            touch_point = coords[0] if at_start else coords[-1]
            found = _find_unique_touching_road(road_polygons, touch_point, bridge["id"], predicate=is_surface)
            if not found:
                continue
            neighbor, touching_at_start = found
            ordered = neighbor["coords"] if touching_at_start else list(reversed(neighbor["coords"]))

            extension_points, remaining_ordered = _walk_bridge_approach(ordered, slope_threshold, max_extension)
            if not extension_points:
                continue

            coords = (list(reversed(extension_points)) + coords) if at_start else (coords + extension_points)
            neighbor["coords"] = remaining_ordered if touching_at_start else list(reversed(remaining_ordered))

        bridge["coords"] = coords

    return road_polygons


def _stable_grade_start(ordered, slope_threshold, stable_length, max_distance):
    """Läuft `ordered` ab Index 0 (Portal) nach außen und sucht den ersten Punkt, ab dem die Steigung auf den
    nächsten `stable_length` Metern durchgehend unter `slope_threshold` bleibt.

    Returns:
        (index, cum) - Index und Bogenlänge dieses Punkts plus die kumulierten Bogenlängen aller Punkte,
        oder None, wenn innerhalb von `max_distance` kein stabiler Abschnitt beginnt (oder die Straße endet).
    """
    arr = np.asarray(ordered, dtype=float)
    seg_len = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(seg_len > 1e-9, np.abs(np.diff(arr[:, 2])) / seg_len, 0.0)

    for idx in range(len(arr) - 1):
        if cum[idx] > max_distance:
            return None
        window_end = cum[idx] + stable_length
        if window_end > cum[-1]:
            return None  # Straße zu kurz, um eine stabile Steigung zu belegen
        in_window = (cum[:-1] >= cum[idx]) & (cum[:-1] < window_end)
        if np.all(slope[in_window] < slope_threshold):
            return idx, cum
    return None


def settle_tunnel_portals_to_approach_grade(road_polygons, slope_threshold=None, stable_length=None, max_distance=None):
    """Bringt Tunnelportale und Galerie-Enden auf die Höhe der Zufahrtsstraße (an einer Galerie zeigt das DGM das Dach
    samt Erdüberwurf, nicht die Fahrbahn - dasselbe Problem wie am Tunnelportal).

    Der OSM-Tunnelanfang liegt oft schon im Hang: das DGM zeigt dort die Portal-Böschung bzw. den Hang über dem
    Portal statt des Straßenniveaus, die Zufahrt steigt auf den letzten Metern steil an und der Tunnel beginnt
    (lineares Profil ab diesem Punkt, siehe apply_structure_elevation_profiles) mehrere Meter zu hoch.

    Pro Tunnel-Ende mit genau einer anschließenden Oberflächenstraße wird die Zufahrt vom Portal weg abgetastet,
    bis die Steigung über `stable_length` Meter stabil unter `slope_threshold` bleibt. Die dort gemessene
    Steigung wird bis zum Portal verlängert: Portalpunkt und die Zufahrtspunkte davor liegen danach auf dieser
    Geraden. Läuft VOR apply_structure_elevation_profiles, damit das Tunnelprofil schon von der neuen
    Portalhöhe ausgeht. Config: TUNNEL_APPROACH_SLOPE_THRESHOLD/_STABLE_LENGTH/_MAX_DISTANCE.
    """
    if slope_threshold is None:
        slope_threshold = config.TUNNEL_APPROACH_SLOPE_THRESHOLD
    if stable_length is None:
        stable_length = config.TUNNEL_APPROACH_STABLE_LENGTH
    if max_distance is None:
        max_distance = config.TUNNEL_APPROACH_MAX_DISTANCE

    is_surface = lambda tags: classify_structure(tags) == "surface"

    for tunnel in [r for r in road_polygons if classify_structure(r.get("osm_tags", {})) in ("tunnel", "gallery")]:
        for at_start in (True, False):
            coords = tunnel["coords"]
            if len(coords) < 2:
                break
            portal = coords[0] if at_start else coords[-1]
            # Die Zufahrt muss eindeutig sein - an einer echten Kreuzung direkt am Portal bleibt alles, wie es ist.
            if _find_unique_touching_road(road_polygons, portal, tunnel["id"]) is None:
                continue
            found = _find_unique_touching_road(road_polygons, portal, tunnel["id"], predicate=is_surface)
            if not found:
                continue
            neighbor, touching_at_start = found
            ordered = list(neighbor["coords"]) if touching_at_start else list(reversed(neighbor["coords"]))

            stable = _stable_grade_start(ordered, slope_threshold, stable_length, max_distance)
            if stable is None:
                continue
            idx, cum = stable
            if idx == 0:
                continue  # Zufahrt ist schon bis zum Portal stabil

            arr = np.asarray(ordered, dtype=float)
            z_stable = arr[idx, 2]
            z_window_end = float(np.interp(cum[idx] + stable_length, cum, arr[:, 2]))
            grade = (z_window_end - z_stable) / stable_length  # Steigung nach außen (vom Portal weg)
            for j in range(idx):
                x, y, _ = ordered[j]
                ordered[j] = (x, y, float(z_stable - grade * (cum[idx] - cum[j])))

            neighbor["coords"] = ordered if touching_at_start else list(reversed(ordered))
            new_portal = ordered[0]
            tunnel["coords"] = [new_portal] + list(coords[1:]) if at_start else list(coords[:-1]) + [new_portal]
            logger.debug(
                f"  Tunnel {tunnel['id']}: Portal {'Anfang' if at_start else 'Ende'} von {portal[2]:.2f} auf "
                f"{new_portal[2]:.2f} m (Zufahrt {neighbor['id']} ab {cum[idx]:.1f} m stabil)"
            )

    return road_polygons


def _xy_match(a, b, tol=1e-3) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def structure_chains(road_polygons):
    """
    Ketten aus Tunnel-/Galerie-Ways, die Ende an Ende EINDEUTIG aneinanderstoßen (genau ein Bauwerks-Partner und
    keine weitere Straße am Stoß). Returns: Liste von Ketten, je Kette [(road, reversed), ...] in Laufrichtung.
    """
    members = [
        r for r in road_polygons
        if classify_structure(r.get("osm_tags", {})) in ("tunnel", "gallery") and len(r["coords"]) >= 2
    ]

    def touches(road, point):
        return _xy_match(road["coords"][0], point) or _xy_match(road["coords"][-1], point)

    def partner(road, point):
        found = [o for o in members if o is not road and touches(o, point)]
        others = [o for o in road_polygons if o is not road and len(o["coords"]) >= 2 and touches(o, point)]
        if len(found) != 1 or len(others) != 1:
            return None
        return found[0], _xy_match(found[0]["coords"][0], point)

    chains, seen = [], set()
    for road in members:
        if id(road) in seen:
            continue
        # zum Kettenanfang zurücklaufen (Ring-Schutz über die Mitgliederzahl)
        cur, rev = road, False
        for _ in range(len(members)):
            found = partner(cur, cur["coords"][-1] if rev else cur["coords"][0])
            if found is None or found[0] is road:
                break
            cur, rev = found[0], found[1]  # Vorgänger endet am Stoß: passt sein Anfang, wird er rückwärts gelaufen
        chain = []
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            chain.append((cur, rev))
            found = partner(cur, cur["coords"][0] if rev else cur["coords"][-1])
            cur, rev = (found[0], not found[1]) if found else (None, False)
        chains.append(chain)
    return chains


def outside_map(point, bounds, edge_margin) -> bool:
    """Punkt liegt außerhalb der Karte oder höchstens edge_margin vor ihrer Kante (bounds: min_x, max_x, min_y, max_y)."""
    min_x, max_x, min_y, max_y = bounds
    x, y = point[0], point[1]
    return not (min_x + edge_margin < x < max_x - edge_margin and min_y + edge_margin < y < max_y - edge_margin)


def _chain_profile(chain, roof_offset, min_end_distance, min_spacing, window, bounds=None, edge_margin=0.0, entrances=()):
    """Höhenprofil einer Kette, in-place: stückweise linear nach Bogenlänge durch die beiden Außenenden und die
    Dach-Stützpunkte der Galerien (Modellhöhe - roof_offset; mindestens min_end_distance von beiden Kettenenden und
    min_spacing voneinander, Median über +-window). Stoßpunkte zwischen zwei Bauwerken sind nie Stützpunkte.
    Reicht die Kette mit genau einem Ende über die Kartengrenze (bounds) und ist das andere Ende eine Einfahrt (liegt
    auf einem Endpunkt aus `entrances`, d.h. eine Oberflächenstraße schließt an), liegt sie ganz auf Einfahrtshöhe
    (Steigung 0) - die Einfahrt ist dort gesperrt, siehe tunnels/roadblock.py."""
    points, is_roof_sample, owners = [], [], []  # owners: (road, Index in road["coords"], Index in points)
    for k, (road, rev) in enumerate(chain):
        coords = road["coords"]
        order = range(len(coords) - 1, -1, -1) if rev else range(len(coords))
        gallery = classify_structure(road.get("osm_tags", {})) == "gallery"
        for j, idx in enumerate(order):
            if k > 0 and j == 0:  # Stoß: derselbe Punkt wie das Ende des Vorgängers
                is_roof_sample[-1] = False
                owners.append((road, idx, len(points) - 1))
                continue
            points.append(coords[idx])
            is_roof_sample.append(gallery)  # Stöße werden oben beim Nachfolger ausgeschlossen, Kettenenden über min_end_distance
            owners.append((road, idx, len(points) - 1))
    arr = np.asarray(points, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])))])
    total = cum[-1]
    if total < 1e-9:
        return
    roof = np.asarray(is_roof_sample)

    if bounds is not None:
        start_out, end_out = outside_map(arr[0], bounds, edge_margin), outside_map(arr[-1], bounds, edge_margin)
        inner = arr[-1] if start_out else arr[0]
        if start_out != end_out and any(_xy_match(inner, e) for e in entrances):
            flat = float(inner[2])
            for road, idx, _ in owners:
                x, y = road["coords"][idx][0], road["coords"][idx][1]
                road["coords"][idx] = (float(x), float(y), flat)
            return

    stations, heights = [0.0], [float(arr[0, 2])]
    last = -np.inf
    for i in np.nonzero(roof)[0]:
        station = cum[i]
        if station < min_end_distance or station > total - min_end_distance or station - last < min_spacing:
            continue
        near = roof & (np.abs(cum - station) <= window)
        stations.append(float(station))
        heights.append(float(np.median(arr[near, 2])) - roof_offset)
        last = station
    stations.append(float(total))
    heights.append(float(arr[-1, 2]))

    z = np.interp(cum, stations, heights)
    for road, idx, point_idx in owners:
        x, y = road["coords"][idx][0], road["coords"][idx][1]
        road["coords"][idx] = (float(x), float(y), float(z[point_idx]))


def apply_structure_elevation_profiles(
    road_polygons, roof_offset=None, min_end_distance=None, min_spacing=None, window=None, bounds=None, edge_margin=None
):
    """
    Höhenprofile für Bauwerke statt der rohen DGM-Höhe an jedem Punkt (siehe Design-Spec Abschnitt 2; z.B. der
    16,9 km lange Gotthard-Straßentunnel bekäme sonst die Bergrücken-Höhe darüber):

    - Brücken: linear zwischen ihren Endpunkten.
    - Tunnel und Galerien: je KETTE aneinanderstoßender Bauwerke (structure_chains) ein Profil durch die beiden
      Außenenden (dort schließt die Straße an, siehe settle_tunnel_portals_to_approach_grade) und durch Dach-
      Stützpunkte der Galerien: an einer Galerie zeigt das DGM das Dach, die Fahrbahn liegt roof_offset darunter.
      Stützpunkte mindestens min_end_distance von den Kettenenden und min_spacing voneinander entfernt, Median über
      +-window. Stöße Tunnel <-> Galerie lesen nie das Gelände (dort liegt Berg bzw. Dach).
      Siehe docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md.
    - Reicht eine Tunnel/Galerie-Kette mit genau einem Ende über die Kartengrenze (`bounds` = min_x, max_x, min_y,
      max_y; höchstens edge_margin vor der Kante zählt schon als draußen) und schließt am anderen Ende eine
      Oberflächenstraße an (Einfahrt), liegt sie ganz auf Einfahrtshöhe.
    """
    if roof_offset is None:
        roof_offset = config.GALLERY_HEIGHT + config.GALLERY_ROOF_THICKNESS
    if min_end_distance is None:
        min_end_distance = config.GALLERY_ROOF_SAMPLE_END_DISTANCE
    if min_spacing is None:
        min_spacing = config.GALLERY_ROOF_SAMPLE_SPACING
    if window is None:
        window = config.GALLERY_ROOF_SAMPLE_WINDOW
    if edge_margin is None:
        edge_margin = config.MAP_EDGE_TUNNEL_MARGIN

    for road in road_polygons:
        if classify_structure(road.get("osm_tags", {})) == "bridge" and len(road["coords"]) >= 2:
            road["coords"] = _linear_elevation_profile(road["coords"])
    entrances = [
        point
        for road in road_polygons
        if classify_structure(road.get("osm_tags", {})) == "surface" and len(road["coords"]) >= 2
        for point in (road["coords"][0], road["coords"][-1])
    ]
    for chain in structure_chains(road_polygons):
        _chain_profile(chain, roof_offset, min_end_distance, min_spacing, window, bounds, edge_margin, entrances)
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
        # OSM-Knoten (ID + lokale Lage) vor dem Resampling festhalten: nur ein gemeinsamer Knoten zweier Ways ist
        # eine echte Verbindung (siehe geometry/junctions.py::build_junction_network())
        node_ids = way.get("nodes") or []
        osm_nodes = (
            [(int(n), float(x), float(y)) for n, (x, y) in zip(node_ids, xy_coords)] if len(node_ids) == len(xy_coords) else None
        )
        temp_roads_xy.append(
            {
                "id": way["id"],
                "xy_coords": xy_coords,
                "name": osm_tags.get("name", f"road_{way['id']}"),
                "osm_tags": osm_tags,
                "osm_nodes": osm_nodes,
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
                "osm_way_id": road["id"],
                "osm_nodes": road["osm_nodes"],
            }
        )

    # SCHRITT 3a: zu kurz getaggte Brücken werden in ihre Nachbarstraße hinein verlängert, bis dort wieder
    # normales Gefälle herrscht - VOR dem linearen Höhenprofil, damit dieses auf dem bereits verlängerten
    # Verlauf arbeitet (siehe extend_short_bridges_to_natural_grade).
    road_polygons = extend_short_bridges_to_natural_grade(road_polygons)

    # SCHRITT 3a': Tunnelportale, deren OSM-Endpunkt schon im Hang liegt, auf die stabile Steigung der
    # Zufahrt bringen - ebenfalls VOR dem linearen Profil, das dann von der korrigierten Portalhöhe ausgeht.
    road_polygons = settle_tunnel_portals_to_approach_grade(road_polygons)

    # SCHRITT 3b: Brücken/Tunnel/Galerien bekommen ein lineares Höhenprofil statt der rohen DGM-Abtastung
    # (siehe Design-Spec Abschnitt 2) - VOR dem Smoothing, damit dieses auf dem bereits korrekten Profil arbeitet.
    map_bounds = (
        float(np.min(height_points[:, 0])), float(np.max(height_points[:, 0])),
        float(np.min(height_points[:, 1])), float(np.max(height_points[:, 1])),
    )
    road_polygons = apply_structure_elevation_profiles(road_polygons, bounds=map_bounds)

    # SCHRITT 4: Optional - mildes XY-Smoothing (Z bleibt erhalten oder nur leicht geglättet)
    if config.ENABLE_ROAD_SMOOTHING:
        logger.info(f"  Mildes XY-Smoothing...")
        road_polygons = smooth_roads_xy_only(road_polygons)

        # SCHRITT 4b: an eindeutigen Brücken/Tunnel/Galerie-Übergängen wird der Knick, den das unabhängige
        # Smoothing pro Straße hinterlässt, zusätzlich weggeglättet (siehe smooth_structure_transitions).
        road_polygons = smooth_structure_transitions(road_polygons)
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


def _blend_structure_boundary(road_a, at_start_a, road_b, at_start_b, window, iterations, weight_center):
    """Glättet die je bis zu `window` Punkte beidseits des gemeinsamen Grenzpunkts von road_a/road_b
    GEMEINSAM (ein Chaikin-Lauf über die zusammengesetzte Fenster-Sequenz), sodass der Grenzpunkt in
    beiden Straßen identisch bleibt; die fernen Fensterenden dienen als fixe Anker."""
    weight_neighbor = (1.0 - weight_center) / 2.0
    coords_a = road_a["coords"]
    coords_b = road_b["coords"]
    wa = min(window, len(coords_a) - 1)
    wb = min(window, len(coords_b) - 1)

    a_slice = list(reversed(coords_a[: wa + 1])) if at_start_a else list(coords_a[-(wa + 1) :])
    b_slice = list(coords_b[: wb + 1]) if at_start_b else list(reversed(coords_b[-(wb + 1) :]))

    # a_slice endet mit dem Grenzpunkt, b_slice beginnt mit dem Grenzpunkt - einmal zusammenführen
    window_seq = a_slice[:-1] + b_slice
    n = len(window_seq)
    if n < 3:
        return

    arr = np.array(window_seq, dtype=float)
    smoothed = arr.copy()
    for _ in range(max(1, iterations)):
        temp = smoothed.copy()
        for i in range(1, n - 1):
            temp[i] = weight_center * smoothed[i] + weight_neighbor * smoothed[i - 1] + weight_neighbor * smoothed[i + 1]
        smoothed = temp
    smoothed[0] = arr[0]  # ferne Fensterenden bleiben fix (Anker)
    smoothed[-1] = arr[-1]

    new_a_slice = [tuple(float(v) for v in p) for p in smoothed[: wa + 1]]
    new_b_slice = [tuple(float(v) for v in p) for p in smoothed[wa:]]

    if at_start_a:
        coords_a[: wa + 1] = list(reversed(new_a_slice))
    else:
        coords_a[-(wa + 1) :] = new_a_slice

    if at_start_b:
        coords_b[: wb + 1] = new_b_slice
    else:
        coords_b[-(wb + 1) :] = list(reversed(new_b_slice))


def smooth_structure_transitions(road_polygons, window=3, iterations=None, weight_center=None):
    """Weicht den Knick an eindeutigen Brücken/Tunnel/Galerie-Übergängen auf.

    smooth_roads_xy_only glättet jede Straße unabhängig und hält dabei ihre Endpunkte exakt fest, wodurch
    am gemeinsamen Übergang zu einer Struktur ein sichtbarer Knick entstehen kann - u.a. weil die Struktur
    ein lineares statt das natürliche DGM-Höhenprofil bekommt (apply_structure_elevation_profiles). Dieser
    Schritt läuft NACH smooth_roads_xy_only und glättet die paar Punkte beidseits eines eindeutigen
    Struktur-Übergangs gemeinsam (XYZ), sodass der Grenzpunkt in beiden Straßen identisch bleibt (kein
    Spalt) und die Straße "aus einem Guss" wirkt.

    Nur eindeutige 2-Wege-Übergänge (Struktur <-> eine andere Straße, egal ob Oberfläche oder wieder eine
    Struktur) werden geglättet; echte Mehrwege-Kreuzungen (3+ Straßen an einem Punkt) bleiben unangetastet,
    da dort die scharfe Kante für die Junction-Logik nötig ist (siehe _find_unique_touching_road).
    """
    iterations = config.ROAD_SMOOTH_ITERATIONS if iterations is None else iterations
    weight_center = config.ROAD_SMOOTH_WEIGHT if weight_center is None else weight_center

    processed = set()
    for road in road_polygons:
        if classify_structure(road.get("osm_tags", {})) == "surface":
            continue
        coords = road["coords"]
        if len(coords) < 2:
            continue

        for at_start in (True, False):
            key = (road["id"], at_start)
            if key in processed:
                continue
            touch_point = coords[0] if at_start else coords[-1]
            found = _find_unique_touching_road(road_polygons, touch_point, road["id"])
            if not found:
                continue
            neighbor, neighbor_at_start = found

            _blend_structure_boundary(road, at_start, neighbor, neighbor_at_start, window, iterations, weight_center)
            processed.add(key)
            processed.add((neighbor["id"], neighbor_at_start))

    return road_polygons
