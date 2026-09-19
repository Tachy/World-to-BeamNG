"""Tests für world_to_beamng.terrain.horizon_seam (Horizont-Mesh mit exakt passendem Terrain-Loch)."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.terrain.horizon_seam import anchored_axis, blend_to_terrain, build_horizon_geometry

HOLE = (-1000.0, -1000.0, 1000.0, 1000.0)


def _domain_points(half=5000.0, step=200.0, offset=(-6.25, -45.33)):
    """Unregelmäßig zum Loch ausgerichtetes DGM30-artiges Punktraster (wie in echt)."""
    xs = np.arange(-half, half + 1, step) + offset[0]
    ys = np.arange(-half, half + 1, step) + offset[1]
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _build(dgm_z=500.0, terrain=lambda x, y: np.full_like(np.asarray(x, float), 100.0), **kw):
    points = _domain_points()
    elevations = np.full(len(points), dgm_z) if np.isscalar(dgm_z) else dgm_z(points)
    return build_horizon_geometry(points, elevations, HOLE, terrain, **kw)


def _signed_area_xy(vertices, faces):
    a, b, c = (vertices[faces[:, k], :2] for k in range(3))
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def test_anchored_axis_contains_both_hole_edges_exactly():
    axis = anchored_axis(-5006.25, 4993.75, -1000.0, 1000.0, 200.0)

    assert -1000.0 in axis and 1000.0 in axis
    assert np.diff(axis) == pytest.approx(np.full(len(axis) - 1, 200.0))
    assert axis.min() <= -4800.0 and axis.max() >= 4800.0  # deckt den Datenbereich ab


def test_anchored_axis_adapts_spacing_when_hole_is_not_a_multiple():
    axis = anchored_axis(-3000.0, 3000.0, -1000.0, 1050.0, 200.0)  # 2050 m Loch

    assert -1000.0 in axis and 1050.0 in axis
    steps = np.diff(axis)
    assert steps == pytest.approx(np.full(len(steps), steps[0]))
    assert 190.0 < steps[0] <= 205.0


def test_blend_matches_terrain_at_hole_edge_and_dgm_far_away():
    x = np.array([1000.0, 1000.0 + 200.0, 1000.0 + 2000.0])
    y = np.zeros(3)
    terrain = lambda px, py: np.full_like(np.asarray(px, float), 100.0)

    z = blend_to_terrain(x, y, np.array([500.0, 500.0, 500.0]), HOLE, terrain, blend_distance=1000.0)

    assert z[0] == pytest.approx(100.0)  # am Rand: Terrainhöhe
    assert 100.0 < z[1] < 500.0  # dazwischen: sanfter Übergang
    assert z[2] == pytest.approx(500.0)  # weit weg: reine DGM30-Höhe


def test_every_triangle_faces_up():
    vertices, faces, _, _ = _build()

    assert (_signed_area_xy(vertices, faces) > 0).all()


def test_no_horizon_triangle_covers_the_terrain_except_the_hidden_flange():
    inset = 5.0
    vertices, faces, _, _ = _build(flange_inset=inset)

    cx = vertices[faces][:, :, 0].mean(1)
    cy = vertices[faces][:, :, 1].mean(1)
    x0, y0, x1, y1 = HOLE
    inside = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
    deep = (cx > x0 + inset) & (cx < x1 - inset) & (cy > y0 + inset) & (cy < y1 - inset)
    assert inside.any()  # der Flansch reicht ein Stück unter das Terrain
    assert not deep.any()  # aber nichts liegt tiefer im Terrainbereich


def test_seam_vertices_sit_exactly_on_terrain_edge_heights():
    terrain = lambda x, y: 300.0 + 0.05 * np.asarray(x, float) + 20.0 * np.sin(np.asarray(y, float) / 90.0)
    vertices, faces, _, _ = _build(terrain=terrain)

    x0, y0, x1, y1 = HOLE
    used = np.unique(faces)
    v = vertices[used]
    on_edge = (
        (np.isclose(v[:, 0], x0) | np.isclose(v[:, 0], x1)) & (v[:, 1] >= y0 - 1e-6) & (v[:, 1] <= y1 + 1e-6)
    ) | ((np.isclose(v[:, 1], y0) | np.isclose(v[:, 1], y1)) & (v[:, 0] >= x0 - 1e-6) & (v[:, 0] <= x1 + 1e-6))
    assert on_edge.sum() >= 4 * 190  # feine Randpunkte (nicht nur das 200-m-Raster)
    assert v[on_edge, 2] == pytest.approx(terrain(v[on_edge, 0], v[on_edge, 1]), abs=0.02)


def test_mesh_is_watertight_no_cracks_between_seam_and_grid():
    vertices, faces, _, _ = _build(flange_inset=5.0)

    directed = Counter()
    for a, b, c in faces:
        directed[(a, b)] += 1
        directed[(b, c)] += 1
        directed[(c, a)] += 1
    # Jede innere Kante wird genau einmal je Richtung benutzt (konsistent orientiert)...
    assert max(directed.values()) == 1
    # ...und hat ihr Gegenstück; offen bleiben nur Außenrand und Flansch-Innenkante.
    open_edges = [(a, b) for (a, b) in directed if (b, a) not in directed]
    x0, y0, x1, y1 = HOLE
    tol = 1e-6
    ext = vertices[:, :2]
    xmin, xmax, ymin, ymax = ext[:, 0].min(), ext[:, 0].max(), ext[:, 1].min(), ext[:, 1].max()
    stray = []
    for a, b in open_edges:
        pa, pb = vertices[a, :2], vertices[b, :2]
        on_outer = (
            (abs(pa[0] - xmin) < tol and abs(pb[0] - xmin) < tol)
            or (abs(pa[0] - xmax) < tol and abs(pb[0] - xmax) < tol)
            or (abs(pa[1] - ymin) < tol and abs(pb[1] - ymin) < tol)
            or (abs(pa[1] - ymax) < tol and abs(pb[1] - ymax) < tol)
        )
        on_flange_rim = all(
            x0 + 5.0 - tol <= p[0] <= x1 - 5.0 + tol and y0 + 5.0 - tol <= p[1] <= y1 - 5.0 + tol for p in (pa, pb)
        )
        if not (on_outer or on_flange_rim):
            stray.append((pa.tolist(), pb.tolist()))
    assert stray == []


def test_flange_lies_below_the_terrain_so_it_stays_hidden():
    terrain = lambda x, y: np.full_like(np.asarray(x, float), 100.0)
    vertices, faces, _, _ = _build(terrain=terrain, flange_inset=5.0, flange_sink=15.0)

    x0, y0, x1, y1 = HOLE
    v = vertices[np.unique(faces)]
    inner = (v[:, 0] > x0 + 1e-6) & (v[:, 0] < x1 - 1e-6) & (v[:, 1] > y0 + 1e-6) & (v[:, 1] < y1 - 1e-6)
    assert inner.any()
    assert (v[inner, 2] < 100.0 - 14.0).all()


def test_grid_dimensions_are_returned_for_texturing():
    vertices, faces, nx, ny = _build()

    assert nx > 40 and ny > 40
    assert len(vertices) >= nx * ny


def test_seam_at_heightmap_resolution_follows_the_terrain_edge_exactly():
    """
    Randring im Raster der Heightmap (1 m): die Horizont-Kante liegt auf jedem DGM1-Punkt exakt
    auf dem Terrain und ist dazwischen wie die Terrainfläche linear -> kein Riss, DGM1 bleibt 1:1.
    """
    from world_to_beamng.terrain.road_embedding import sample_heightmap_bilinear

    rng = np.random.RandomState(3)
    heights = 300.0 + np.cumsum(rng.randn(2049, 2049), axis=1) * 0.7  # raue, feine Terraindetails
    terrain = lambda x, y: sample_heightmap_bilinear(
        heights, -1000.0, -1000.0, 1.0, np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
    )
    vertices, faces, _, _ = _build(terrain=terrain, seam_step=1.0)

    x0, y0, x1, y1 = HOLE
    used = vertices[np.unique(faces)]
    west = used[np.isclose(used[:, 0], x0) & (used[:, 1] >= y0) & (used[:, 1] <= y1)]
    west = west[np.argsort(west[:, 1])]
    assert len(west) >= 2001
    mid_xy = 0.5 * (west[:-1, :2] + west[1:, :2])  # Punkte zwischen zwei Randvertices
    horizon_mid = 0.5 * (west[:-1, 2] + west[1:, 2])
    assert np.abs(horizon_mid - terrain(mid_xy[:, 0], mid_xy[:, 1])).max() < 1e-3


@pytest.mark.parametrize("seam_step", [1.0, 2.0, 10.0])
def test_no_degenerate_triangles_or_duplicate_edges_at_any_seam_resolution(seam_step):
    # Bei feinem Randring (< Flanschbreite) fallen an den Ecken mehrere Randpunkte auf denselben
    # Flansch-Innenpunkt - das darf keine Nulldreiecke oder doppelten Kanten erzeugen.
    vertices, faces, _, _ = _build(seam_step=seam_step, flange_inset=5.0)

    assert (_signed_area_xy(vertices, faces) > 1e-6).all()
    directed = Counter()
    for a, b, c in faces:
        directed[(a, b)] += 1
        directed[(b, c)] += 1
        directed[(c, a)] += 1
    assert max(directed.values()) == 1
