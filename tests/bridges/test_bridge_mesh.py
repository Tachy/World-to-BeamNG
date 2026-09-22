"""Tests für world_to_beamng.bridges.bridge_mesh: Brücken-Deck (Fahrbahn+Bordstein+Geländer) + Stützpfeiler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.bridges.bridge_mesh import build_bridge_mesh, build_bridges

DECK, PIER, RAIL = "asphalt_road_standard", "bridge_concrete", "bridge_railing"


def _flat_ground(z=150.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _coords(length=60.0, z=200.0, n=7):
    return [(x, 5.0, z) for x in np.linspace(0.0, length, n)]


def _zs(mesh, material):
    faces = mesh["faces"][material]
    return np.array([mesh["vertices"][i][2] for face in faces for i in face])


def test_deck_top_is_flat_at_the_given_height_and_carriageway_width():
    # pier_spacing groesser als die Spannweite: isoliert den Test auf die reine Deck-Geometrie (kein Pfeiler-Vertex
    # in "vertices", der die min()-Annahme unten verfaelschen wuerde - siehe test_piers_reach_down_... fuer die
    # Pfeiler-Faelle mit dem Standard-pier_spacing).
    mesh = build_bridge_mesh(
        _coords(z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, deck_thickness=0.6, pier_spacing=1000.0,
    )
    deck_zs = _zs(mesh, DECK)

    assert deck_zs.max() == pytest.approx(200.0)  # Fahrbahn-Oberkante = Höhenprofil, folgt NICHT dem Gelände
    assert deck_zs.min() == pytest.approx(200.0 - 0.6)  # Deck-Unterkante


def test_deck_faces_use_the_road_material_not_the_pier_material():
    mesh = build_bridge_mesh(_coords(n=3), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL)

    assert DECK in mesh["faces"] and len(mesh["faces"][DECK]) > 0


def test_curb_sits_on_top_of_the_deck_and_narrows_the_carriageway():
    mesh = build_bridge_mesh(
        _coords(n=3, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, curb_width=0.25, curb_height=0.15, pier_spacing=1000.0,
    )
    deck_zs, pier_zs = _zs(mesh, DECK), _zs(mesh, PIER)

    assert deck_zs.max() == pytest.approx(200.0)  # Fahrbahn bleibt auf Deck-Niveau
    assert pier_zs.max() == pytest.approx(200.0 + 0.15)  # Bordstein-Oberkante = Deck + curb_height
    # Bordstein-Vertices liegen ausserhalb der halben Fahrbahnbreite (8/2 - 0.25 = 3.75 m von der Achse)
    xy_at_curb_top = np.array(
        [mesh["vertices"][i][1] for face in mesh["faces"][PIER] for i in face if mesh["vertices"][i][2] == pytest.approx(200.0 + 0.15)]
    )
    assert np.any(np.abs(np.abs(xy_at_curb_top - 5.0) - 4.0) < 1e-6)  # äußere Bordsteinkante bei voller Breite (4 m)


def test_railing_posts_and_handrail_sit_above_the_curb():
    mesh = build_bridge_mesh(
        _coords(n=3, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, curb_height=0.15, railing_height=0.9, railing_post_size=0.08, pier_spacing=1000.0,
    )

    assert RAIL in mesh["faces"] and len(mesh["faces"][RAIL]) > 0
    rail_zs = _zs(mesh, RAIL)
    assert rail_zs.min() == pytest.approx(200.0 + 0.15)  # Pfosten beginnen auf der Bordstein-Oberkante
    assert rail_zs.max() == pytest.approx(200.0 + 0.15 + 0.9 + 0.08 / 2.0)  # Handlauf-Oberkante


def test_piers_reach_down_to_the_natural_ground_below_a_deep_span():
    mesh = build_bridge_mesh(_coords(length=60.0, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL, pier_spacing=25.0)

    assert len(mesh["faces"][PIER]) > 0
    assert _zs(mesh, PIER).min() == pytest.approx(150.0)  # (mindestens) ein Pfeiler reicht bis zum natürlichen Gelände


def test_no_piers_when_clearance_is_too_small():
    mesh = build_bridge_mesh(_coords(length=60.0, z=151.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL, deck_thickness=0.2, min_pier_clearance=1.0)

    # PIER-Material enthält noch die Bordstein-Flächen, aber keinen Pfeiler, der das Gelände erreicht
    assert _zs(mesh, PIER).min() == pytest.approx(151.0)


def test_build_bridges_returns_one_mesh_per_bridge_with_its_own_deck_material():
    bridges = [
        {"id": 1, "coords": _coords(z=200.0), "width": 8.0, "deck_material": "asphalt_road_standard"},
        {"id": 2, "coords": _coords(z=210.0), "width": 6.0, "deck_material": "concrete"},
    ]

    meshes = build_bridges(bridges, _flat_ground(150.0), pier_material=PIER, railing_material=RAIL)

    assert [m["id"] for m in meshes] == ["bridge_1", "bridge_2"]
    assert "asphalt_road_standard" in meshes[0]["faces"] and "concrete" in meshes[1]["faces"]


def test_build_bridges_skips_degenerate_bridges():
    bridges = [{"id": 1, "coords": [(0.0, 0.0, 200.0)], "width": 8.0, "deck_material": DECK}]

    assert build_bridges(bridges, _flat_ground(150.0), pier_material=PIER, railing_material=RAIL) == []
