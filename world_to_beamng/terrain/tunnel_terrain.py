"""
Gelände an Tunneln: Überdeckung über der Röhre und die Portal-Zone (siehe tunnels/tunnel_portal.py).

Die Heightmap ist eine einzige Fläche je Rasterzelle - sie kann nicht gleichzeitig über der Röhre (Berg) und in der
Röhre (Luft) liegen. Deshalb:

1. Überdeckung: entlang der Röhre liegt das Gelände mindestens TUNNEL_COVER über der Krone, seitlich mit
   TUNNEL_COVER_SLOPE ans natürliche Gelände angeböscht. Wo der Tunnel flach unter dem Hang liegt (typisch die
   ersten Meter hinter dem Portal), ragte die Röhre sonst aus dem Gelände bzw. das Gelände in die Röhre. Nur wo
   die Röhre mindestens zur Hälfte im Gelände steckt (Gelände an der Mittellinie mindestens halbe Kronenhöhe über
   dem Boden): verläuft das Tunnelprofil über einer Senke durch die Luft (unplausible OSM-Daten) oder ist der
   "Tunnel" im DGM eigentlich offene Straße (z.B. covered=yes-Galerie am Hang), entstünde sonst ein Damm.
2. Portal-Zone: zwischen Portalebene und TUNNEL_PORTAL_FLAT_DEPTH liegt das Gelände knapp unter dem Röhrenboden
   (dort verdeckt es der Boden der Röhre), dahinter auf Überdeckungshöhe. Die Zellen am Übergang, die in den
   Röhrenquerschnitt reichen, werden Terrain-Löcher; sie liegen komplett im Portalblock, dessen Ober-/Unterkante
   (portal["top_z"]/["bottom_z"]) hier an ihre Eckhöhen angepasst wird.

3. Lücken: liegt die Röhre hinter einem offenen Portal erst nach einer kurzen Strecke (<= cover_gap_max) im
   Gelände, wird diese Strecke mit überdeckt - sonst fiele das Gelände hinter dem Portalblock ab und schnitte
   als Erdwand durch die Röhre. Übergänge in eine Galerie (portal["kind"] == "gallery") sind immer Portale.

Oberflächenstraßen (z.B. ein Weg, der über den Tunnel führt, oder die Zufahrt) bleiben unangetastet - ihre
Höhe bestimmt die Straßen-Einbettung.

Ein Tunnel-Ende ist nur dann ein Portal, wenn davor offenes Gelände liegt (siehe _portal_is_open()): endet eine
Kette mitten im Berg (z.B. am Kartenrand abgeschnitten oder an einer mehrdeutigen Stoßstelle), bleibt das Gelände
dort unberührt und es entsteht kein Portalblock.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy

from ..tunnels.tunnel_portal import portal_local_coords

DENSE_STEP = 0.5  # Abtastung der Centerline für die Abstandsberechnung, in Metern
WINDOW = 100.0  # Centerline-Abschnitt je Verarbeitungsfenster, in Metern
MAX_COVER_REACH = 20.0  # so weit seitlich über den Portalblock hinaus wirkt die Überdeckungs-Böschung höchstens
HOLE_BAND_MARGIN = 0.3  # Loch-Zellen reichen so weit seitlich über den Röhrenradius hinaus, in Metern
FLOOR_CLEARANCE = 0.05  # so weit liegt das Gelände in der Portal-Zone unter dem Röhrenboden, in Metern
OPEN_PROBE_DIST = 3.0  # Abstand vor der Portalebene, an dem offenes Gelände geprüft wird, in Metern


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


def _fill_portal_gaps(buried: np.ndarray, s: np.ndarray, start_open: bool, end_open: bool, max_gap: float) -> np.ndarray:
    """`buried` plus die nicht eingegrabenen Strecken, die an einem offenen Portal beginnen und höchstens max_gap
    lang sind, bevor die Röhre im Gelände steckt. Ohne diese Füllung fiele das Gelände hinter der Portal-
    Überdeckung wieder auf Fahrbahnhöhe - die Geländefläche liefe als Erdwand quer durch die Röhre (Nordportal
    Tunnel Fieud: ~9 m flach hinter dem Portal). Längere Strecken bleiben offen (sonst Dämme)."""
    filled = buried.copy()
    if max_gap <= 0.0 or not buried.any():
        return filled
    first = int(np.argmax(buried))
    if start_open and first > 0 and s[first] - s[0] <= max_gap:
        filled[:first] = True
    last = len(buried) - 1 - int(np.argmax(buried[::-1]))
    if end_open and last < len(buried) - 1 and s[-1] - s[last] <= max_gap:
        filled[last + 1 :] = True
    return filled


def _raise_cover(heights, origin_x, origin_y, square_size, plan, cover, cover_slope, protected, cover_gap_max=0.0) -> None:
    """Schritt 1 (siehe Moduldocstring): Überdeckung entlang der ganzen Röhre, in-place."""
    xy, s, floor_z = _dense_centerline(plan["coords"])
    from .road_embedding import sample_heightmap_bilinear

    buried = sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, xy) >= floor_z + 0.5 * plan["crown"]
    tree = cKDTree(xy)
    total = s[-1]
    start_portal, end_portal = plan["portals"]
    buried = _fill_portal_gaps(buried, s, start_portal["open"], end_portal["open"], cover_gap_max)
    half_width = start_portal["half_width"]
    flat_depth = start_portal["flat_depth"]
    reach = half_width + MAX_COVER_REACH
    top_offset = plan["crown"] + cover
    last = len(xy) - 1

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
        dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), distance_upper_bound=reach)
        dist, idx = dist.reshape(gx.shape), idx.reshape(gx.shape)
        # Nur Zellen seitlich der Röhre (nicht vor den Portalen: dort nächster Punkt = Endpunkt)
        valid = np.isfinite(dist) & (idx > 0) & (idx < last)
        idx = np.where(valid, idx, 0)
        valid &= buried[idx]
        along = s[idx]
        # Vor der Portal-Zone fällt die Überdeckung zur Portalebene hin ab; innerhalb der Blockbreite regelt
        # dort die Portal-Zone (Schritt 2) das Gelände.
        portal_excess = np.zeros_like(along)
        if start_portal["open"]:
            portal_excess += np.maximum(0.0, flat_depth - along)
        if end_portal["open"]:
            portal_excess += np.maximum(0.0, flat_depth - (total - along))
        valid &= ~((dist <= half_width) & (portal_excess > 0))
        valid &= _unprotected(protected, gx, gy)
        required = floor_z[idx] + top_offset - (np.maximum(0.0, dist - half_width) + portal_excess) / cover_slope
        view = heights[view_slice]
        view[valid] = np.maximum(view[valid], required[valid])


def _shape_portal(heights, origin_x, origin_y, square_size, portal, cover, protected) -> Optional[np.ndarray]:
    """Schritt 2 (siehe Moduldocstring) für ein Portal, in-place. Gibt die Loch-Zellen als (rows, cols) zurück
    und passt portal["top_z"]/["bottom_z"] an deren Eckhöhen an."""
    radius, half_width = portal["radius"], portal["half_width"]
    length, flat_depth = portal["length"], portal["flat_depth"]
    floor_z = portal["floor_z"]
    top_level = floor_z + portal["crown"] + cover

    px, py = portal["xy"]
    extent = length + half_width + 3.0
    window = _grid_window(heights, origin_x, origin_y, square_size, px - extent, px + extent, py - extent, py + extent)
    if window is None:
        return None
    view_slice, gx, gy = window
    along, across = portal_local_coords(portal, gx, gy)
    free = _unprotected(protected, gx, gy)
    view = heights[view_slice]

    in_block = np.abs(across) <= half_width
    flat = free & in_block & (along >= 0.0) & (along < flat_depth)
    view[flat] = floor_z - FLOOR_CLEARANCE
    apron = free & (np.abs(across) <= radius + 1.0) & (along >= -1.5) & (along < 0.0)
    view[apron] = np.minimum(view[apron], floor_z)
    covered = free & in_block & (along >= flat_depth) & (along <= length + 1.0)
    view[covered] = np.maximum(view[covered], top_level)

    # Loch-Zellen: Rasterzelle (row, col) = Quadrat mit linker unterer Ecke im Vertex (row, col). Ein Loch, wo
    # das Quadrat die Stufe Portal-Zone -> Überdeckung überspannt und in den Röhrenquerschnitt reicht.
    behind = along >= flat_depth
    corners_behind = [behind[:-1, :-1], behind[:-1, 1:], behind[1:, :-1], behind[1:, 1:]]
    mixed = np.any(corners_behind, axis=0) & ~np.all(corners_behind, axis=0)
    band = radius + HOLE_BAND_MARGIN
    corner_across = np.stack([across[:-1, :-1], across[:-1, 1:], across[1:, :-1], across[1:, 1:]])
    in_band = ~(np.all(corner_across > band, axis=0) | np.all(corner_across < -band, axis=0))
    near = np.all(np.stack([along[:-1, :-1], along[1:, 1:], along[:-1, 1:], along[1:, :-1]]) < length, axis=0)
    hole = mixed & in_band & near
    if not np.any(hole):
        return None

    corner_heights = np.stack([view[:-1, :-1], view[:-1, 1:], view[1:, :-1], view[1:, 1:]])[:, hole]
    portal["top_z"] = max(portal["top_z"], float(corner_heights.max()) + 0.1)
    portal["bottom_z"] = min(portal["bottom_z"], float(corner_heights.min()) - 0.2)

    rows, cols = np.nonzero(hole)
    return rows + view_slice[0].start, cols + view_slice[1].start


def shape_terrain_for_tunnels(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    plans: List[Dict],
    cover: float,
    cover_slope: float,
    protected=None,
    cover_gap_max: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Überdeckung + Portal-Zonen für alle Tunnel-Pläne (siehe tunnels/tunnel_portal.py::plan_tunnels()).

    Args:
        protected: shapely-Geometrie der Oberflächenstraßen (oder None) - dort bleibt das Gelände unverändert
        cover_gap_max: so lange Lücke zwischen offenem Portal und eingegrabener Röhre wird noch überdeckt (0 = aus)

    Returns:
        (neue Heightmap, Loch-Maske (bool, gleiche Shape; True = Rasterzelle wird Terrain-Loch))
        Die Portale in `plans` bekommen dabei "open" (siehe _portal_is_open()) und ihre endgültige
        "top_z"/"bottom_z".
    """
    result = heights.copy()
    holes = np.zeros(heights.shape, dtype=bool)
    for plan in plans:
        for portal in plan["portals"]:
            # Übergang in eine Galerie: davor liegt immer ein Bauwerk - immer ein Portal
            portal["open"] = portal.get("kind") == "gallery" or _portal_is_open(heights, origin_x, origin_y, square_size, portal)
    for plan in plans:
        _raise_cover(result, origin_x, origin_y, square_size, plan, cover, cover_slope, protected, cover_gap_max)
    for plan in plans:
        for portal in [p for p in plan["portals"] if p["open"]]:
            cells = _shape_portal(result, origin_x, origin_y, square_size, portal, cover, protected)
            if cells is not None:
                holes[cells] = True
    return result, holes
