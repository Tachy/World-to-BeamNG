"""
Horizont-Mesh mit exakt passendem Loch für den Terrain-Block.

Das Terrain ist ein TerrainBlock (Heightmap), kein Mesh - der Horizont kann deshalb nicht
mehr mit Terrain-Vertices "vernäht" werden. Stattdessen wird die Horizont-Geometrie direkt
aus der Heightmap abgeleitet:

1. Das 200-m-Raster ist an den Kanten des Terrain-Lochs verankert (anchored_axis), das Loch
   ist also exakt N Quads breit und nicht mehr "irgendwo zwischen zwei Rasterlinien".
2. Die Höhen des Horizonts werden zum Loch hin auf die Terrainhöhe an der Kante gezogen
   (blend_to_terrain) und gehen nach außen sanft in die DGM30-Höhen über. So gibt es keine
   Stufe zwischen Terrain und Horizont (DGM30 ist ein grobes Oberflächenmodell und weicht
   vom DGM1 um bis zu ~50 m ab).
3. Direkt am Loch sitzt ein feiner Randring (seam_step, Standard 10 m) mit den exakten
   Terrain-Randhöhen, der per Dreiecksfächer an das grobe Raster anschließt. Ein schmaler
   "Flansch" unter dem Terrain (flange_inset/flange_sink) deckt Restrisse ab, weil das
   Terrain feiner ist als der Randring.
"""

from typing import Callable, Tuple

import numpy as np
from scipy.spatial import cKDTree

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]
Hole = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


def anchored_axis(lo: float, hi: float, hole_lo: float, hole_hi: float, spacing: float) -> np.ndarray:
    """
    Rasterlinien-Koordinaten, die hole_lo und hole_hi exakt enthalten.

    Das Loch wird in round((hole_hi - hole_lo) / spacing) gleich große Schritte geteilt
    (Schrittweite <= ca. spacing); nach außen läuft das Raster mit derselben Schrittweite
    bis zum Datenbereich [lo, hi].
    """
    cells = max(1, int(round((hole_hi - hole_lo) / spacing)))
    step = (hole_hi - hole_lo) / cells
    k_min = int(np.ceil((lo - hole_lo) / step - 1e-9))
    k_max = int(np.floor((hi - hole_lo) / step + 1e-9))
    k_min = min(k_min, -1)
    k_max = max(k_max, cells + 1)
    axis = hole_lo + step * np.arange(k_min, k_max + 1)
    axis[np.argmin(np.abs(axis - hole_lo))] = hole_lo  # Rundungsfehler an den Ankern vermeiden
    axis[np.argmin(np.abs(axis - hole_hi))] = hole_hi
    return axis


def blend_to_terrain(
    x: np.ndarray,
    y: np.ndarray,
    dgm_z: np.ndarray,
    hole: Hole,
    terrain_height_at: HeightAt,
    blend_distance: float,
) -> np.ndarray:
    """
    Horizont-Höhe: am Loch die Terrainhöhe der nächstgelegenen Terrainkante, ab
    blend_distance Abstand die reine DGM30-Höhe, dazwischen Smoothstep.
    """
    x0, y0, x1, y1 = hole
    cx, cy = np.clip(x, x0, x1), np.clip(y, y0, y1)
    distance = np.hypot(x - cx, y - cy)
    t = np.clip(distance / blend_distance, 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    reference = np.asarray(terrain_height_at(cx, cy), dtype=float)
    return reference * (1.0 - weight) + dgm_z * weight


def _fan_triangles(coarse: np.ndarray, fine: np.ndarray, per_cell: int) -> list:
    """
    Dreiecksfächer zwischen einer groben Polylinie (n+1 Punkte) und einer feinen (n*per_cell+1
    Punkte, deren Eckpunkte mit den groben zusammenfallen können): pro grober Zelle fächert die
    erste Hälfte vom vorderen, die zweite vom hinteren groben Punkt, ein Dreieck schließt die Mitte.
    """
    triangles = []
    mid = per_cell // 2
    for cell in range(len(coarse) - 1):
        c0, c1 = coarse[cell], coarse[cell + 1]
        base = cell * per_cell
        for m in range(per_cell):
            triangles.append((c0 if m < mid else c1, fine[base + m], fine[base + m + 1]))
        triangles.append((c0, fine[base + mid], c1))
    return triangles


def build_horizon_geometry(
    points: np.ndarray,
    elevations: np.ndarray,
    hole: Hole,
    terrain_height_at: HeightAt,
    spacing: float = 200.0,
    seam_step: float = 10.0,
    blend_distance: float = 1000.0,
    flange_inset: float = 5.0,
    flange_sink: float = 15.0,
):
    """
    Baut die Horizont-Geometrie um ein rechteckiges Terrain-Loch.

    Args:
        points: (N, 2) DGM30-Stützpunkte (lokale Koordinaten)
        elevations: (N,) zugehörige Höhen
        hole: (x_min, y_min, x_max, y_max) des Terrain-Bereichs (echte Daten, ohne Padding)
        terrain_height_at: Höhenabfrage der fertigen Terrain-Heightmap
        spacing: Rasterabstand des groben Horizonts
        seam_step: Punktabstand des feinen Randrings am Loch
        blend_distance: Länge des Höhenübergangs Terrain -> DGM30
        flange_inset, flange_sink: Breite/Tiefe des Flansches unter dem Terrain (0 = kein Flansch)

    Returns:
        (vertices (V,3), faces (F,3) nach oben orientiert, nx, ny des groben Rasters)
    """
    x0, y0, x1, y1 = hole
    xs = anchored_axis(points[:, 0].min(), points[:, 0].max(), x0, x1, spacing)
    ys = anchored_axis(points[:, 1].min(), points[:, 1].max(), y0, y1, spacing)
    nx, ny = len(xs), len(ys)

    grid_x, grid_y = np.meshgrid(xs, ys)  # (ny, nx), y-major
    flat_x, flat_y = grid_x.ravel(), grid_y.ravel()
    _, nearest = cKDTree(points).query(np.column_stack([flat_x, flat_y]))
    z = blend_to_terrain(flat_x, flat_y, np.asarray(elevations, dtype=float)[nearest], hole, terrain_height_at, blend_distance)
    vertices = [np.column_stack([flat_x, flat_y, z])]
    next_index = len(z)

    ix0, ix1 = int(np.argmin(np.abs(xs - x0))), int(np.argmin(np.abs(xs - x1)))
    jy0, jy1 = int(np.argmin(np.abs(ys - y0))), int(np.argmin(np.abs(ys - y1)))

    def vid(i, j):
        return j * nx + i

    # --- grobe Quads: Loch und der direkt angrenzende Ring (den übernimmt der Randring) entfallen
    keep = np.ones((ny - 1, nx - 1), dtype=bool)
    keep[jy0 - 1 : jy1 + 1, ix0 : ix1] = False  # Süd-Ring, Loch, Nord-Ring (Spalten über dem Loch)
    keep[jy0:jy1, ix0 - 1 : ix1 + 1] = False  # West-Ring, Loch, Ost-Ring (Zeilen neben dem Loch)
    quad_j, quad_i = np.nonzero(keep)
    v0 = quad_j * nx + quad_i
    v1, v2, v3 = v0 + 1, v0 + nx, v0 + nx + 1
    faces = [np.column_stack([v0, v1, v2]), np.column_stack([v1, v3, v2])]

    # --- feiner Randring: pro Seite feine Punkte auf der Lochkante, Fächer zur groben Außenspalte
    def side_points(coarse_along):
        """Feine Punkte entlang einer Kante (auf- oder absteigend); Zellgrenzen = grobe Rasterlinien."""
        per_cell = max(1, int(round(abs(coarse_along[1] - coarse_along[0]) / seam_step)))
        pts = []
        for c in range(len(coarse_along) - 1):
            lo, hi = coarse_along[c], coarse_along[c + 1]
            pts.extend(lo + (hi - lo) * m / per_cell for m in range(per_cell))
        pts.append(coarse_along[-1])
        return np.array(pts), per_cell

    ring = []  # geordnete Schleife der Randring-Vertex-Indizes (gegen den Uhrzeigersinn)
    new_vertices = []

    def add_vertex(px, py):
        nonlocal next_index
        pz = float(np.asarray(terrain_height_at(np.array([px]), np.array([py])))[0])
        new_vertices.append((px, py, pz))
        next_index += 1
        return next_index - 1

    sides = [
        # (feste Achse, fester Wert, laufende Koordinaten, Index-Funktion der Ecke, Außenspalte)
        ("x", x0, ys[jy0 : jy1 + 1], lambda j: vid(ix0, jy0 + j), [vid(ix0 - 1, jy0 + j) for j in range(jy1 - jy0 + 1)]),
        ("y", y1, xs[ix0 : ix1 + 1], lambda i: vid(ix0 + i, jy1), [vid(ix0 + i, jy1 + 1) for i in range(ix1 - ix0 + 1)]),
        ("x", x1, ys[jy0 : jy1 + 1][::-1], lambda j: vid(ix1, jy1 - j), [vid(ix1 + 1, jy1 - j) for j in range(jy1 - jy0 + 1)]),
        ("y", y0, xs[ix0 : ix1 + 1][::-1], lambda i: vid(ix1 - i, jy0), [vid(ix1 - i, jy0 - 1) for i in range(ix1 - ix0 + 1)]),
    ]
    triangles = []
    for axis, fixed, along, corner_vid, coarse_ids in sides:
        pts, per_cell = side_points(along)
        fine_ids = []
        for n, p in enumerate(pts):
            # Ecken (erster/letzter Punkt der Seite) sind bestehende Rasterknoten
            if n == 0:
                fine_ids.append(corner_vid(0))
            elif n == len(pts) - 1:
                fine_ids.append(corner_vid(len(along) - 1))
            else:
                fine_ids.append(add_vertex(fixed, p) if axis == "x" else add_vertex(p, fixed))
        triangles.extend(_fan_triangles(coarse_ids, fine_ids, per_cell))
        ring.extend(fine_ids[:-1])  # letzter Punkt = erste Ecke der nächsten Seite

    # --- Flansch unter dem Terrain
    if flange_inset > 0 and flange_sink > 0:
        all_vertices_so_far = np.vstack([vertices[0], np.array(new_vertices)]) if new_vertices else vertices[0]
        inner_by_position = {}  # Randpunkte nahe einer Ecke landen auf demselben Innenpunkt -> ein Vertex
        inner_ids = []
        for idx in ring:
            px, py, _ = all_vertices_so_far[idx]
            ix = float(np.clip(px, x0 + flange_inset, x1 - flange_inset))
            iy = float(np.clip(py, y0 + flange_inset, y1 - flange_inset))
            key = (round(ix, 6), round(iy, 6))
            if key not in inner_by_position:
                iz = float(np.asarray(terrain_height_at(np.array([ix]), np.array([iy])))[0]) - flange_sink
                new_vertices.append((ix, iy, iz))
                next_index += 1
                inner_by_position[key] = next_index - 1
            inner_ids.append(inner_by_position[key])
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            a_in, b_in = inner_ids[k], inner_ids[(k + 1) % len(ring)]
            triangles.append((a, b, b_in))
            triangles.append((a, b_in, a_in))
        # Fällt ein Innenpunkt zusammen, entartet eines der beiden Dreiecke -> weglassen
        triangles = [t for t in triangles if len(set(t)) == 3]

    if new_vertices:
        vertices.append(np.array(new_vertices, dtype=float))
    vertices = np.vstack(vertices)
    faces_array = np.vstack(faces + [np.array(triangles, dtype=int).reshape(-1, 3)]).astype(int)

    # Einheitlich nach oben orientieren (Fächer/Flansch je Seite sonst gemischt)
    a, b, c = (vertices[faces_array[:, k], :2] for k in range(3))
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area < 0
    faces_array[flip] = faces_array[flip][:, [0, 2, 1]]
    return vertices, faces_array, nx, ny
