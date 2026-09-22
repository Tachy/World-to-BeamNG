"""Tests für world_to_beamng.tunnels.gallery_mesh: talseitig offene Lawinengalerie (Dach + Stützen)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.gallery_mesh import build_gallery_mesh, build_galleries, valley_side

FLOOR, ROOF = "asphalt_road_standard", "tunnel_concrete"


def _straight_coords(length=60.0, z=500.0, n=13):
    return [(x, 0.0, z) for x in np.linspace(0.0, length, n)]


def test_valley_side_picks_the_lower_natural_terrain():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    # Gelände fällt nach +y ab: bei Laufrichtung +x ist +y die LINKE Seite (Standard-Konvention wie
    # offset_points(): links = Richtung um +90° CCW gedreht = (-dy,dx); für direction=(1,0) ist das (0,1) = +y).
    # +y ist also die Talseite -> links talwärts -> side < 0.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)

    side = valley_side(xy, ground_at, half_width=4.0)

    assert np.all(side < 0)


def test_valley_side_flips_when_the_slope_is_mirrored():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    ground_at = lambda x, y: 500.0 + 2.0 * np.asarray(y, float)  # steigt nach +y -> -y (rechts) ist die Talseite

    side = valley_side(xy, ground_at, half_width=4.0)

    assert np.all(side > 0)


def test_roof_and_floor_are_flat_at_the_given_heights():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF, roof_thickness=0.35)
    v = mesh["vertices"]

    assert v[:, 2].min() == pytest.approx(500.0)
    assert v[:, 2].max() == pytest.approx(505.35)  # Boden(500) + Höhe(5) + Dachdicke(0.35)


def test_faces_are_split_by_material():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    assert set(mesh["faces"]) == {FLOOR, ROOF}
    assert len(mesh["faces"][FLOOR]) > 0 and len(mesh["faces"][ROOF]) > 0


def test_columns_are_on_the_valley_side_not_the_mountain_side():
    # Boden/Dach spannen immer beide Kanten (y=+4 und y=-4) - reine Vertex-Präsenz an einer Kante unterscheidet
    # also NICHT, wo die Stützen sitzen. Stattdessen: Stützen fügen an ihrer Kante zusätzliche Vertices ein
    # (4 Seitenflächen je Stütze), an der Bergseite (nur die Wandfläche) nicht - die Talseite muss daher
    # spürbar mehr Vertices nahe ihrer Kante haben als die Bergseite.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # fällt nach +y -> +y ist die Talseite (links)
    mesh = build_gallery_mesh(_straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF, column_spacing=10.0)

    v = np.array(mesh["vertices"])
    near_valley_edge = np.sum(np.abs(v[:, 1] - 4.0) < 0.3)  # +y = Talseite in diesem Szenario
    near_mountain_edge = np.sum(np.abs(v[:, 1] + 4.0) < 0.3)
    assert near_valley_edge > near_mountain_edge


def test_build_galleries_returns_one_mesh_per_gallery():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    galleries = [{"id": 1, "coords": _straight_coords(z=500.0), "width": 8.0, "floor_material": FLOOR}]

    meshes = build_galleries(galleries, ground_at, roof_material=ROOF, height=5.0)

    assert [m["id"] for m in meshes] == ["gallery_1"]


def test_build_galleries_skips_degenerate_galleries():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)
    galleries = [{"id": 1, "coords": [(0.0, 0.0, 500.0)], "width": 8.0, "floor_material": FLOOR}]

    assert build_galleries(galleries, ground_at, roof_material=ROOF) == []
