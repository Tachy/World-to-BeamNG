"""
Gelände an Tunneln - nur unmittelbar lokal an Röhre und Portal (siehe tunnels/tunnel_mesh.py, tunnels/tunnel_portal.py).

Die Röhre ist ein Zylinder mit Außenschale (tunnel_mesh.shell_cross_section()); sie darf frei stehen und muss nicht
vom Gelände versteckt werden. Die Heightmap ist aber eine einzige Fläche je Rasterzelle - sie darf nicht quer durch
das Röhreninnere laufen. Deshalb:

1. Überdeckung: An jeder Station, an der das Gelände innerhalb der Schale über den Röhrenboden ragt
   (ENTER_TOLERANCE), liegt im Grundriss der Schale Erde `cover` über dem runden Außenquerschnitt. Liegt das
   Gelände schon höher, bleibt es; liegt es unter der Röhre, bleibt es auch (die Röhre steht dort frei). Keine
   seitlichen Böschungen, keine Dämme ins Tal.
2. Portal-Zone (je offenem Portal): zwischen Portalebene und flat_depth liegt das Gelände knapp unter dem
   Röhrenboden (dort verdeckt es der Boden der Röhre); dahinter wird der Hang im Grundriss des Portalbauwerks auf
   dessen Außenkontur abgetragen (runder Kragen bzw. Oberkante der Galerie-Stirnwand) - das Bauwerk wächst nicht mit
   dem Hang.
3. Löcher: eine Rasterzelle über der Röhre (Abstand <= Radius + HOLE_BAND_MARGIN), deren Ecken teils auf/unter dem
   Röhrenboden und teils darüber liegen, liefe als schräge Fläche durch die Röhre - sie wird Terrain-Loch. Das
   passiert an der Portalstufe (verdeckt vom Kragen) und dort, wo die Röhre aus dem Gelände austritt (verdeckt von
   der Schale).

Oberflächenstraßen (z.B. ein Weg, der über den Tunnel führt, oder die Zufahrt) bleiben unangetastet - ihre Höhe
bestimmt die Straßen-Einbettung.

Ein Tunnel-Ende ist nur dann ein Portal, wenn davor offenes Gelände liegt (siehe _portal_is_open()): endet eine
Kette mitten im Berg (z.B. am Kartenrand abgeschnitten oder an einer mehrdeutigen Stoßstelle), bleibt das Gelände
dort unberührt und es entsteht kein Portalbauwerk. Übergänge in eine Galerie (portal["kind"] == "gallery") sind
immer Portale.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy

from ..tunnels.tunnel_portal import portal_local_coords

DENSE_STEP = 0.5  # Abtastung der Centerline für die Abstandsberechnung, in Metern
WINDOW = 100.0  # Centerline-Abschnitt je Verarbeitungsfenster, in Metern
ENTER_TOLERANCE = 0.3  # so weit darf das Gelände über den Röhrenboden ragen, ohne als "in der Röhre" zu gelten
HOLE_BAND_MARGIN = 0.3  # Loch-Zellen reichen so weit seitlich über den Röhrenradius hinaus, in Metern
FLOOR_CLEARANCE = 0.05  # so weit liegt das Gelände in der Portal-Zone unter dem Röhrenboden, in Metern
STRUCTURE_CLEARANCE = 0.1  # so weit bleibt der abgetragene Hang unter der Außenkontur des Portalbauwerks, in Metern
OPEN_PROBE_DIST = 3.0  # Abstand vor der Portalebene, an dem offenes Gelände geprüft wird, in Metern
APRON_LENGTH = 1.5  # so weit vor der Portalebene wird das Gelände höchstens auf Bodenhöhe gehalten, in Metern


def _portal_is_open(heights, origin_x, origin_y, square_size, portal) -> bool:
    """Offenes Portal: kurz vor der Portalebene liegt das Gelände (nach der Straßen-Einbettung) höchstens auf
    halber Kronenhöhe über dem Röhrenboden - dort kommt man tatsächlich von außen in die Röhre."""
    from .road_embedding import sample_heightmap_bilinear

    px, py = portal["xy"]
    ux, uy = portal["axis"]
    probe = np.array([[px - ux * OPEN_PROBE_DIST, py - uy * OPEN_PROBE_DIST]])
    ground = float(sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, probe)[0])
    return ground < portal["floor_z"] + 0.5 * portal["crown"]


def _dense_centerline(coords) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(coords, dtype=float)
    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, cum[-1], max(2, int(np.ceil(cum[-1] / DENSE_STEP)) + 1))
    xy = np.column_stack([np.interp(samples, cum, arr[:, 0]), np.interp(samples, cum, arr[:, 1])])
    return xy, samples, np.interp(samples, cum, arr[:, 2])


def _grid_window(heights, origin_x, origin_y, square_size, min_x, max_x, min_y, max_y):
    size_y, size_x = heights.shape
    col0 = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col1 = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row0 = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row1 = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))
    if col0 > col1 or row0 > row1:
        return None
    gx, gy = np.meshgrid(origin_x + np.arange(col0, col1 + 1) * square_size, origin_y + np.arange(row0, row1 + 1) * square_size)
    return (slice(row0, row1 + 1), slice(col0, col1 + 1)), gx, gy


def _unprotected(protected, gx, gy) -> np.ndarray:
    if protected is None:
        return np.ones(gx.shape, dtype=bool)
    return ~intersects_xy(protected, gx, gy)


def _tube_windows(heights, origin_x, origin_y, square_size, xy, reach):
    """(Fenster-Slice, gx, gy, Abstand zur Centerline, nächster Stationsindex) je WINDOW-Abschnitt der Röhre."""
    tree = cKDTree(xy)
    per_window = max(2, int(WINDOW / DENSE_STEP))
    for start in range(0, len(xy), per_window):
        part = xy[start : start + per_window + 1]
        window = _grid_window(
            heights, origin_x, origin_y, square_size,
            part[:, 0].min() - reach, part[:, 0].max() + reach, part[:, 1].min() - reach, part[:, 1].max() + reach,
        )
        if window is None:
            continue
        view_slice, gx, gy = window
        dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        yield view_slice, gx, gy, dist.reshape(gx.shape), idx.reshape(gx.shape)


def _cover_tube(heights, origin_x, origin_y, square_size, plan, cover, protected) -> None:
    """Schritt 1 (siehe Moduldocstring), in-place."""
    xy, _, floor_z = _dense_centerline(plan["coords"])
    radius = plan["radius"]
    outer = radius + plan.get("shell", 0.0)
    last = len(xy) - 1

    def footprint(dist, idx):
        # Nur Zellen seitlich der Röhre (nicht vor den Enden: dort ist der nächste Punkt ein Endpunkt)
        return (dist <= outer) & (idx > 0) & (idx < last)

    windows = list(_tube_windows(heights, origin_x, origin_y, square_size, xy, outer + square_size))
    enters = np.zeros(len(xy), dtype=bool)
    for view_slice, _, _, dist, idx in windows:
        inside = footprint(dist, idx) & (heights[view_slice] > floor_z[idx] + ENTER_TOLERANCE)
        enters[idx[inside]] = True

    for view_slice, gx, gy, dist, idx in windows:
        valid = footprint(dist, idx) & enters[idx] & _unprotected(protected, gx, gy)
        required = floor_z[idx] + radius / 2.0 + np.sqrt(np.maximum(outer**2 - dist**2, 0.0)) + cover
        view = heights[view_slice]
        view[valid] = np.maximum(view[valid], required[valid])


def _shape_portal(heights, origin_x, origin_y, square_size, portal, protected) -> None:
    """Schritt 2 (siehe Moduldocstring) für ein offenes Portal, in-place."""
    radius, half_width = portal["radius"], portal["half_width"]
    length, flat_depth = portal["length"], portal["flat_depth"]
    floor_z = portal["floor_z"]

    px, py = portal["xy"]
    extent = length + half_width + 3.0
    window = _grid_window(heights, origin_x, origin_y, square_size, px - extent, px + extent, py - extent, py + extent)
    if window is None:
        return
    view_slice, gx, gy = window
    along, across = portal_local_coords(portal, gx, gy)
    free = _unprotected(protected, gx, gy)
    view = heights[view_slice]

    in_structure = np.abs(across) < half_width
    flat = free & in_structure & (along >= 0.0) & (along < flat_depth)
    view[flat] = floor_z - FLOOR_CLEARANCE
    apron = free & (np.abs(across) <= radius + 1.0) & (along >= -APRON_LENGTH) & (along < 0.0)
    view[apron] = np.minimum(view[apron], floor_z)

    # Hang im Grundriss des Bauwerks auf dessen Außenkontur abtragen (runder Kragen bzw. Stirnwand-Oberkante)
    behind = free & in_structure & (along >= flat_depth) & (along <= length)
    if portal.get("kind") == "gallery":
        limit = np.full(gx.shape, portal["top_z"] - STRUCTURE_CLEARANCE)
    else:
        limit = floor_z + radius / 2.0 + np.sqrt(np.maximum(half_width**2 - across**2, 0.0)) - STRUCTURE_CLEARANCE
    view[behind] = np.minimum(view[behind], limit[behind])


def _mark_holes(heights, holes, origin_x, origin_y, square_size, plan) -> None:
    """Schritt 3 (siehe Moduldocstring): Loch-Zellen über der Röhre, in-place in `holes`."""
    xy, _, floor_z = _dense_centerline(plan["coords"])
    band = plan["radius"] + HOLE_BAND_MARGIN
    last = len(xy) - 1
    for view_slice, _, _, dist, idx in _tube_windows(heights, origin_x, origin_y, square_size, xy, band + square_size):
        view = heights[view_slice]
        in_band = (dist <= band) & (idx > 0) & (idx < last)
        high = view > floor_z[idx] + ENTER_TOLERANCE

        def corners(a):
            return np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]])

        cell_high = corners(high)
        crossing = corners(in_band).any(axis=0) & cell_high.any(axis=0) & ~cell_high.all(axis=0)
        rows, cols = np.nonzero(crossing)
        holes[rows + view_slice[0].start, cols + view_slice[1].start] = True


def shape_terrain_for_tunnels(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    plans: List[Dict],
    cover: float,
    protected=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Überdeckung, Portal-Zonen und Löcher für alle Tunnel-Pläne (siehe tunnels/tunnel_portal.py::plan_tunnels()).

    Args:
        cover: Erdschicht über der Röhrenschale, wo das Gelände in die Röhre ragt, in Metern
        protected: shapely-Geometrie der Oberflächenstraßen (oder None) - dort bleibt das Gelände unverändert

    Returns:
        (neue Heightmap, Loch-Maske (bool, gleiche Shape; True = Rasterzelle wird Terrain-Loch))
        Die Portale in `plans` bekommen dabei "open" (siehe _portal_is_open()).
    """
    result = heights.copy()
    holes = np.zeros(heights.shape, dtype=bool)
    for plan in plans:
        for portal in plan["portals"]:
            # Übergang in eine Galerie: davor liegt immer ein Bauwerk - immer ein Portal
            portal["open"] = portal.get("kind") == "gallery" or _portal_is_open(heights, origin_x, origin_y, square_size, portal)
    for plan in plans:
        _cover_tube(result, origin_x, origin_y, square_size, plan, cover, protected)
    for plan in plans:
        for portal in [p for p in plan["portals"] if p["open"]]:
            _shape_portal(result, origin_x, origin_y, square_size, portal, protected)
    for plan in plans:
        _mark_holes(result, holes, origin_x, origin_y, square_size, plan)
    return result, holes
