"""Höhenprofil über Ketten aus Tunneln und Galerien (world_to_beamng.geometry.polygon.apply_structure_elevation_profiles).

Hintergrund (Nuova strada del San Gottardo, 2026-09-24): An Stößen Tunnel <-> Galerie liefert das Höhenmodell keine
brauchbare Fahrbahnhöhe (dort liegt Berg bzw. das Galeriedach). Die Kette bekommt deshalb ein Profil durch ihre beiden
Außenenden und durch Stützpunkte aus dem Galeriedach (Modellhöhe - lichte Höhe - Dachdicke), die mindestens 100 m von
den Kettenenden und 200 m voneinander entfernt liegen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.polygon import apply_structure_elevation_profiles, settle_tunnel_portals_to_approach_grade

ROOF = 5.5  # GALLERY_HEIGHT + GALLERY_ROOF_THICKNESS
KW = dict(roof_offset=ROOF, min_end_distance=100.0, min_spacing=200.0, window=10.0)


def _way(road_id, xs, z_of_x, **tags):
    return {"id": road_id, "coords": [(float(x), 0.0, float(z_of_x(x))) for x in xs], "name": "r", "osm_tags": tags}


def _z_at(roads, x):
    for road in roads:
        for px, _, pz in road["coords"]:
            if abs(px - x) < 1e-9:
                return pz
    raise AssertionError(f"kein Punkt bei x={x}")


def test_inner_joint_between_tunnels_ignores_the_terrain_height():
    junk = lambda x: 999.0 if x == 300 else (100.0 if x == 0 else 130.0)
    roads = [
        _way(1, range(0, 301, 10), junk, highway="primary", tunnel="yes"),
        _way(2, range(300, 601, 10), junk, highway="primary", tunnel="yes"),
    ]

    apply_structure_elevation_profiles(roads, **KW)

    assert roads[0]["coords"][-1][2] == pytest.approx(115.0)  # auf der Geraden 100 -> 130, nicht 999
    assert roads[1]["coords"][0][2] == pytest.approx(115.0)
    assert _z_at(roads, 150) == pytest.approx(107.5)


def test_chain_works_when_members_are_digitised_in_opposite_directions():
    junk = lambda x: 999.0 if x == 300 else (100.0 if x == 0 else 130.0)
    second = _way(2, range(300, 601, 10), junk, highway="primary", tunnel="yes")
    second["coords"] = second["coords"][::-1]
    roads = [_way(1, range(0, 301, 10), junk, highway="primary", tunnel="yes"), second]

    apply_structure_elevation_profiles(roads, **KW)

    assert _z_at(roads, 300) == pytest.approx(115.0)
    assert _z_at(roads, 450) == pytest.approx(122.5)


def test_gallery_roof_gives_support_points_away_from_the_chain_ends():
    # Kette: Tunnel 0..150, Galerie 150..750, Tunnel 750..900. Wahre Fahrbahn in der Galerie 8 m über der Geraden
    # zwischen den Außenenden (100 -> 118); das Höhenmodell zeigt dort das Dach (Fahrbahn + 5,5 m).
    line = lambda x: 100.0 + 0.02 * x
    roads = [
        _way(1, range(0, 151, 10), lambda x: 100.0 if x == 0 else 500.0, highway="primary", tunnel="yes"),
        _way(2, range(150, 751, 10), lambda x: line(x) + 8.0 + ROOF, highway="primary", covered="yes", layer="-1"),
        _way(3, range(750, 901, 10), lambda x: 118.0 if x == 900 else 500.0, highway="primary", tunnel="yes"),
    ]

    apply_structure_elevation_profiles(roads, **KW)

    # Stützpunkte bei 160, 360, 560: erster Galeriepunkt >= 100 m vom Ende (der Stoßpunkt 150 selbst zählt nie -
    # genau dort ist das Höhenmodell unzuverlässig), dann alle 200 m; 760 läge schon im Tunnel
    for x in (160, 360, 560):
        # Median über +-10 m: am ersten Stützpunkt ist das Fenster am Stoß einseitig -> auf 2 % Steigung bis 0,1 m
        assert _z_at(roads, x) == pytest.approx(line(x) + 8.0, abs=0.15)
    assert _z_at(roads, 0) == pytest.approx(100.0) and _z_at(roads, 900) == pytest.approx(118.0)
    assert _z_at(roads, 80) == pytest.approx(100.0 + (line(160) + 8.0 - 100.0) * 80 / 160, abs=0.1)  # zum Ende angeglichen
    assert _z_at(roads, 150) < 500.0 and _z_at(roads, 750) < 500.0  # Stöße lesen nie das Gelände


def test_gallery_closer_than_the_end_distance_gives_no_support_point():
    roads = [_way(1, range(0, 151, 10), lambda x: 200.0 if x in (0, 150) else 999.0, highway="primary", covered="yes", layer="-1")]

    apply_structure_elevation_profiles(roads, **KW)

    assert _z_at(roads, 70) == pytest.approx(200.0)  # nur linear zwischen den Enden


def test_standalone_gallery_gets_roof_support_points_too():
    # Einzelne Galerie 0..320: Stützpunkt nur bei 100 (300 läge näher als 100 m am Ende)
    roads = [_way(1, range(0, 321, 10), lambda x: 200.0 if x in (0, 320) else 205.0 + ROOF, highway="primary", covered="yes", layer="-1")]

    apply_structure_elevation_profiles(roads, **KW)

    assert _z_at(roads, 100) == pytest.approx(205.0)
    assert _z_at(roads, 210) == pytest.approx(205.0 - 5.0 * 110 / 220)


def test_roof_support_point_uses_the_median_against_outliers():
    def roof(x):
        if x in (0, 320):
            return 200.0
        return 999.0 if x == 100 else 205.0 + ROOF  # ein Ausreißer genau am Stützpunkt

    roads = [_way(1, range(0, 321, 5), roof, highway="primary", covered="yes", layer="-1")]

    apply_structure_elevation_profiles(roads, **KW)

    assert _z_at(roads, 100) == pytest.approx(205.0)


def test_bridges_keep_their_own_linear_profile():
    roads = [
        _way(1, range(0, 301, 10), lambda x: 100.0 if x == 0 else (130.0 if x == 300 else 50.0), highway="primary", bridge="yes"),
        _way(2, range(300, 601, 10), lambda x: 130.0 if x == 300 else (100.0 if x == 600 else 999.0), highway="primary", tunnel="yes"),
    ]

    apply_structure_elevation_profiles(roads, **KW)

    assert _z_at(roads, 150) == pytest.approx(115.0)  # Brücke: linear 100 -> 130
    assert _z_at(roads, 450) == pytest.approx(115.0)  # Tunnel für sich: linear 130 -> 100


def test_gallery_end_is_levelled_to_the_approach_like_a_tunnel_end():
    # Zufahrt x=0..30 mit stabilen 5 %, die letzten 8 m steigen steil an (Hang/Dach über dem Galerie-Ende)
    approach = []
    for x in range(0, 31, 2):
        z = 100.0 + 0.05 * x + (max(0, x - 22) * 0.5)
        approach.append((float(x), 0.0, z))
    gallery = [approach[-1], (60.0, 0.0, 120.0), (100.0, 0.0, 125.0)]
    roads = [
        {"id": 1, "coords": approach, "name": "r", "osm_tags": {"highway": "primary"}},
        {"id": 2, "coords": gallery, "name": "r", "osm_tags": {"highway": "primary", "covered": "yes", "layer": "-1"}},
    ]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=0.10, stable_length=6.0, max_distance=40.0)

    assert roads[1]["coords"][0] == pytest.approx((30.0, 0.0, 101.5))


# --- Kette, die über die Kartengrenze reicht: Röhre mit Steigung 0 auf Einfahrtshöhe ---

BOUNDS = (-1000.0, 1000.0, -1000.0, 1000.0)


def _approach_to(x_end, z):
    return _way(99, range(int(x_end) - 50, int(x_end) + 1, 10), lambda x: z, highway="primary")


def test_chain_leaving_the_map_gets_a_flat_tube_at_the_entrance_height():
    roads = [
        _way(1, range(800, 1201, 10), lambda x: 100.0 if x == 800 else 150.0 + 0.05 * x, highway="primary", tunnel="yes"),
        _approach_to(800, 100.0),
    ]

    apply_structure_elevation_profiles(roads, bounds=BOUNDS, edge_margin=25.0, **KW)

    assert all(z == pytest.approx(100.0) for _, _, z in roads[0]["coords"])


def test_chain_ending_just_inside_the_edge_counts_as_leaving_the_map():
    roads = [_way(1, range(700, 991, 10), lambda x: 100.0 if x == 700 else 170.0, highway="primary", tunnel="yes"), _approach_to(700, 100.0)]

    apply_structure_elevation_profiles(roads, bounds=BOUNDS, edge_margin=25.0, **KW)

    assert roads[0]["coords"][-1][2] == pytest.approx(100.0)  # x=990 liegt nur 10 m vor der Kante


def test_chain_leaving_on_both_sides_keeps_its_linear_profile():
    roads = [_way(1, range(-1200, 1201, 50), lambda x: 100.0 if x == -1200 else (200.0 if x == 1200 else 999.0), highway="trunk", tunnel="yes")]

    apply_structure_elevation_profiles(roads, bounds=BOUNDS, edge_margin=25.0, **KW)

    assert _z_at(roads, 0) == pytest.approx(150.0)


def test_chain_inside_the_map_is_not_flattened():
    roads = [_way(1, range(0, 301, 10), lambda x: 100.0 if x == 0 else (130.0 if x == 300 else 999.0), highway="primary", tunnel="yes")]

    apply_structure_elevation_profiles(roads, bounds=BOUNDS, edge_margin=25.0, **KW)

    assert _z_at(roads, 150) == pytest.approx(115.0)


def test_chain_leaving_the_map_without_an_approach_road_is_not_flattened():
    # z.B. Festungsstollen: das Ende in der Karte ist eine Verzweigung im Berg, keine Einfahrt
    roads = [_way(1, range(800, 1201, 10), lambda x: 100.0 if x == 800 else (160.0 if x == 1200 else 999.0), highway="path", tunnel="yes")]

    apply_structure_elevation_profiles(roads, bounds=BOUNDS, edge_margin=25.0, **KW)

    assert _z_at(roads, 1000) == pytest.approx(130.0)  # linear, nicht flach
