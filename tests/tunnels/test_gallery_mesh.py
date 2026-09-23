"""Tests für world_to_beamng.tunnels.gallery_mesh: talseitig offene Lawinengalerie (Dach + Stützen)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.gallery_mesh import build_gallery_mesh, build_galleries, resolve_open_side, valley_side

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
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, roof_thickness=0.35, floor_thickness=3.0,
    )
    v = mesh["vertices"]

    assert v[:, 2].min() == pytest.approx(497.0)  # Boden(500) - Bodendicke(3)
    assert v[:, 2].max() == pytest.approx(505.35)  # Boden(500) + Höhe(5) + Dachdicke(0.35)


def test_faces_are_split_by_material():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    assert set(mesh["faces"]) == {FLOOR, ROOF}
    assert len(mesh["faces"][FLOOR]) > 0 and len(mesh["faces"][ROOF]) > 0


def _off_grid_vertices_near(vertices, y, grid_step=5.0, tol=0.3):
    """Vertices nahe y, deren X NICHT auf dem Centerline-Punktraster (Vielfache von grid_step) liegt - Boden/
    Dach/Wand-Flächen haben nur Vertices auf dem Punktraster, nur Stützen fügen Vertices dazwischen ein."""
    x, y_coord = vertices[:, 0], vertices[:, 1]
    on_grid = np.abs((x / grid_step) - np.round(x / grid_step)) < 0.01
    return np.sum((~on_grid) & (np.abs(y_coord - y) < tol))


def test_columns_are_on_the_valley_side_not_the_mountain_side():
    # Boden/Dach/Wand spannen immer beide Kanten (y=+4 und y=-4) und liegen nur auf dem 5m-Centerline-
    # Punktraster - nur Stützen (hier bei x=5,15,...,55, exakt auf dem Raster in diesem Szenario deckungs-
    # gleich mit column_spacing=10) fügen zusätzliche Vertices EXAKT an ihrer x-Position ein. Um das von
    # Boden/Dach/Wand-Vertices (ebenfalls bei Vielfachen von 5) zu unterscheiden, column_spacing hier bewusst
    # NICHT auf dem 5m-Raster wählen.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # fällt nach +y -> +y ist die Talseite (links)
    mesh = build_gallery_mesh(_straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0)

    v = np.array(mesh["vertices"])
    assert _off_grid_vertices_near(v, y=4.0) > 0  # Talseite: Stützen-Vertices abseits des Punktrasters
    assert _off_grid_vertices_near(v, y=-4.0) == 0  # Bergseite: keine Stützen


def test_wall_extends_wall_thickness_into_the_mountain():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # Bergseite ist -y (rechts)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, wall_thickness=3.0, column_spacing=1000.0,  # keine Stützen (verfälschen die Kante sonst)
    )

    v = np.array(mesh["vertices"])
    # Bergseite (y<0): die Wand-Außenfläche reicht bis width/2 + wall_thickness = 4 + 3 = 7 m von der Achse.
    assert v[:, 1].min() == pytest.approx(-7.0)
    # Talseite (y>0) bleibt bei der reinen Fahrbahnbreite, width/2 = 4 m.
    assert v[:, 1].max() == pytest.approx(4.0)


def test_wall_is_flush_with_the_roof_top():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # Bergseite ist -y (rechts)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, roof_thickness=0.5, column_spacing=1000.0,  # keine Stützen (verfälschen die Kante sonst)
    )

    v = np.array(mesh["vertices"])
    mountain_side = np.abs(v[:, 1] + 4.0) < 6.0  # gesamte Bergseite (Wand reicht bis y=-9 bei wall_thickness=5)
    assert v[mountain_side][:, 2].max() == pytest.approx(505.5)  # Boden(500) + Höhe(5) + Dachdicke(0.5)


def test_curb_is_on_the_open_side_only():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # Talseite (offen) ist +y
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, curb_height=0.5, curb_width=0.25, column_spacing=1000.0,
    )

    v = np.array(mesh["vertices"])
    at_valley_edge = np.abs(v[:, 1] - 4.0) < 0.01
    at_mountain_edge = np.abs(v[:, 1] + 4.0) < 0.01
    # Sockel-Oberkante (500.5 = Boden 500 + Sockelhöhe 0.5) nur auf der Talseite, nicht auf der Bergseite.
    assert np.any(np.isclose(v[at_valley_edge][:, 2], 500.5))
    assert not np.any(np.isclose(v[at_mountain_edge][:, 2], 500.5))


def test_curb_does_not_widen_the_gallery_footprint():
    # Sockel liegt curb_width INNERHALB der Fahrbahnkante, ragt also nicht über die bisherige Breite hinaus.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, curb_height=0.5, curb_width=0.25, wall_thickness=0.0, column_spacing=1000.0,
    )

    v = np.array(mesh["vertices"])
    assert v[:, 1].max() == pytest.approx(4.0)  # width / 2


def test_columns_sit_flush_on_top_of_the_curb_not_in_the_floor():
    """Regression: Stützen steckten bisher vom Boden-Niveau an im Sockel (Z-Überlappung) - die Basis muss
    jetzt auf der Sockel-Oberkante sitzen, die Oberkante bleibt unverändert bei der Dach-Unterkante (die
    Stütze wird dadurch um curb_height kürzer)."""
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # +y ist Talseite (offen)
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at,
        floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0,
        curb_height=0.5, curb_width=0.4, column_size=0.4,
    )
    v = np.array(mesh["vertices"])

    # Stützen-Vertices: abseits des 5m-Punktrasters (x) UND nahe der Talseiten-Kante (y=4, Stützen-Außenkante).
    off_grid = np.abs((v[:, 0] / 5.0) - np.round(v[:, 0] / 5.0)) >= 0.01
    near_valley_edge = np.abs(v[:, 1] - 4.0) < 0.5
    column_vertices = v[off_grid & near_valley_edge]

    assert len(column_vertices) > 0
    assert column_vertices[:, 2].min() == pytest.approx(500.5)  # Sockel-Oberkante (Boden 500 + 0.5), nicht 500
    assert column_vertices[:, 2].max() == pytest.approx(505.0)  # unverändert: Boden(500) + Höhe(5)


def test_columns_footprint_is_centered_on_the_curb_and_flush_with_the_roof_edge():
    """Regression: Stützen standen bisher auf der Fahrbahnkante zentriert (Dachkante schnitt durch die
    Stützenmitte). Jetzt auf der Sockel-Mittellinie zentriert - bei curb_width == column_size fällt die
    Stützen-Außenkante exakt mit der (unveränderten) Dach-/Fahrbahnkante zusammen."""
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at,
        floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0,
        curb_height=0.5, curb_width=0.4, column_size=0.4,
    )
    v = np.array(mesh["vertices"])
    off_grid = np.abs((v[:, 0] / 5.0) - np.round(v[:, 0] / 5.0)) >= 0.01
    column_vertices = v[off_grid]

    assert len(column_vertices) > 0
    assert column_vertices[:, 1].max() == pytest.approx(4.0)  # = width/2 = Dach-/Fahrbahnkante, kein Überstand
    assert column_vertices[:, 1].min() == pytest.approx(3.6)  # Sockel-Mitte (3.8) - halbe Stützenbreite (0.2)


def test_ends_are_capped_with_outward_facing_faces():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    v, n = np.array(mesh["vertices"]), np.array(mesh["normals"])
    # Stirnflächen am Anfang (x=0, Normale -x) und Ende (x=60, Normale +x).
    start_faces = np.abs(v[:, 0]) < 1e-6
    end_faces = np.abs(v[:, 0] - 60.0) < 1e-6
    assert np.any(start_faces & (n[:, 0] < -0.99))
    assert np.any(end_faces & (n[:, 0] > 0.99))


def test_build_galleries_returns_one_mesh_per_gallery():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    galleries = [{"id": 1, "coords": _straight_coords(z=500.0), "width": 8.0, "floor_material": FLOOR}]

    meshes = build_galleries(galleries, ground_at, roof_material=ROOF, height=5.0)

    assert [m["id"] for m in meshes] == ["gallery_1"]


def test_build_galleries_skips_degenerate_galleries():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)
    galleries = [{"id": 1, "coords": [(0.0, 0.0, 500.0)], "width": 8.0, "floor_material": FLOOR}]

    assert build_galleries(galleries, ground_at, roof_material=ROOF) == []


# --- resolve_open_side / open_side-Override ----------------------------------------------------------------------


def test_resolve_open_side_reads_the_avalanche_protector_tag():
    assert resolve_open_side({"avalanche_protector:left": "open"}) == "left"
    assert resolve_open_side({"avalanche_protector:right": "open"}) == "right"


def test_resolve_open_side_is_none_without_a_reliable_tag():
    assert resolve_open_side({}) is None
    assert resolve_open_side({"avalanche_protector:left": "no"}) is None


def test_open_side_override_ignores_ground_at_even_when_it_disagrees():
    # ground_at würde die Talseite auf +y (links) legen (siehe test_valley_side_picks_the_lower_natural_terrain) -
    # der Tag muss trotzdem gewinnen, das DGM zeigt an einer bestehenden Galerie ja das Bauwerk selbst.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at,
        floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0, open_side="right",
    )

    v = np.array(mesh["vertices"])
    assert _off_grid_vertices_near(v, y=-4.0) > 0  # "right" = -y offen -> Stützen dort
    assert _off_grid_vertices_near(v, y=4.0) == 0


def test_build_galleries_uses_the_osm_tag_when_present():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # würde ohne Tag "left" liefern
    galleries = [{
        "id": 1, "coords": _straight_coords(length=60.0, z=500.0), "width": 8.0, "floor_material": FLOOR,
        "osm_tags": {"avalanche_protector:right": "open"},
    }]

    meshes = build_galleries(galleries, ground_at, roof_material=ROOF, column_spacing=12.0)

    v = np.array(meshes[0]["vertices"])
    assert _off_grid_vertices_near(v, y=-4.0) > 0
    assert _off_grid_vertices_near(v, y=4.0) == 0
