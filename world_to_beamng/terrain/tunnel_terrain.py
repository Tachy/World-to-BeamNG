"""
Terrain at tunnels - only immediately local to the tube and portal (see tunnels/tunnel_mesh.py, tunnels/tunnel_portal.py).

The tube is a cylinder with an outer shell (tunnel_mesh.shell_cross_section()); it may stand free and does not have to
be hidden by the terrain. The heightmap, however, is a single surface per raster cell - it must not run straight through
the tube interior. Therefore:

1. Cover: At every station where the terrain inside the shell rises above the tube floor
   (ENTER_TOLERANCE), the footprint of the shell holds earth `cover` above the round outer cross-section. If the
   terrain is already higher, it stays; if it lies below the tube, it also stays (the tube stands free there). No
   lateral embankments, no fills into the valley.
2. Portal zone (per open portal): between the portal plane and flat_depth the terrain lies just below the
   tube floor (there the tube floor hides it); behind that, the slope within the portal footprint is cut down to its
   round outer contour (collar or tube shell) - the portal does not grow with the slope.
3. Holes: a raster cell above the tube (distance <= radius + HOLE_BAND_MARGIN) whose corners lie partly on/below the
   tube floor and partly above it would run as a sloped face through the tube - it becomes a terrain hole. This
   happens at the portal step (hidden by the collar) and where the tube emerges from the terrain (hidden by
   the shell).

Surface roads (e.g. a path that runs over the tunnel, or the access road) remain untouched - their height
determines the road embedding.

A tunnel end is only a portal if open terrain lies in front of it (see _portal_is_open()): if a
chain ends in the middle of the mountain (e.g. cut off at the map edge or at an ambiguous joint), the terrain
there stays untouched and no portal structure is created. Transitions into a gallery (portal["kind"] == "gallery") are
always portals.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy

from ..tunnels.tunnel_portal import portal_local_coords

DENSE_STEP = 0.5  # Sampling of the centerline for the distance computation, in meters
WINDOW = 100.0  # Centerline section per processing window, in meters
ENTER_TOLERANCE = 0.3  # how far the terrain may rise above the tube floor without counting as "inside the tube"
HOLE_BAND_MARGIN = 0.0  # Hole only if a cell corner lies within the tube radius (otherwise the face does not cut the interior)
FLOOR_CLEARANCE = 0.05  # how far the terrain lies below the tube floor in the portal zone, in meters
STRUCTURE_CLEARANCE = 0.1  # how far the cut-down slope stays below the outer contour of the portal structure, in meters
OPEN_PROBE_DIST = 3.0  # Distance in front of the portal plane at which open terrain is checked, in meters
APRON_LENGTH = 1.5  # how far in front of the portal plane the terrain is held at most at floor height, in meters


def _portal_is_open(heights, origin_x, origin_y, square_size, portal) -> bool:
    """Open portal: shortly before the portal plane the terrain (after the road embedding) lies at most at
    half the crown height above the tube floor - there one actually enters the tube from outside."""
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
    """(window slice, gx, gy, distance to the centerline, nearest station index) per WINDOW section of the tube."""
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
    """Step 1 (see module docstring), in-place."""
    xy, _, floor_z = _dense_centerline(plan["coords"])
    radius = plan["radius"]
    outer = radius + plan.get("shell", 0.0)
    last = len(xy) - 1

    def footprint(dist, idx):
        # Only cells beside the tube (not in front of the ends: there the nearest point is an end point)
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
    """Step 2 (see module docstring) for an open portal, in-place."""
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

    # Cut the slope within the portal footprint down to its outer contour: top edge of the rectangular collar or,
    # without a collar, the round tube shell
    behind = free & in_structure & (along >= flat_depth) & (along <= length)
    if portal.get("collar", 0.0) > 0.0:
        limit = np.full(gx.shape, portal["top_z"] - STRUCTURE_CLEARANCE)
    else:
        limit = floor_z + radius / 2.0 + np.sqrt(np.maximum(half_width**2 - across**2, 0.0)) - STRUCTURE_CLEARANCE
    view[behind] = np.minimum(view[behind], limit[behind])


def _mark_holes(heights, holes, origin_x, origin_y, square_size, plan) -> None:
    """Step 3 (see module docstring): hole cells above the tube, in-place in `holes`."""
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
    Cover, portal zones and holes for all tunnel plans (see tunnels/tunnel_portal.py::plan_tunnels()).

    Args:
        cover: Earth layer above the tube shell where the terrain rises into the tube, in meters
        protected: shapely geometry of the surface roads (or None) - the terrain stays unchanged there

    Returns:
        (new heightmap, hole mask (bool, same shape; True = raster cell becomes a terrain hole))
        In the process, the portals in `plans` get "open" (see _portal_is_open()).
    """
    result = heights.copy()
    holes = np.zeros(heights.shape, dtype=bool)
    for plan in plans:
        for portal in plan["portals"]:
            # Transition into a gallery: a structure always lies in front of it - always a portal
            portal["open"] = portal.get("kind") == "gallery" or _portal_is_open(heights, origin_x, origin_y, square_size, portal)
    for plan in plans:
        _cover_tube(result, origin_x, origin_y, square_size, plan, cover, protected)
    for plan in plans:
        for portal in [p for p in plan["portals"] if p["open"]]:
            _shape_portal(result, origin_x, origin_y, square_size, portal, protected)
    for plan in plans:
        _mark_holes(result, holes, origin_x, origin_y, square_size, plan)
    return result, holes
