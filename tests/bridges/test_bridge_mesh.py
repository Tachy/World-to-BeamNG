"""Tests für world_to_beamng.bridges.bridge_mesh: Brücken-Deck + Stützpfeiler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.bridges.bridge_mesh import build_bridge_mesh, build_bridges

DECK, PIER = "asphalt_road_standard", "bridge_concrete"


def _flat_ground(z=150.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _coords(length=60.0, z=200.0, n=7):
    return [(x, 5.0, z) for x in np.linspace(0.0, length, n)]


def test_deck_top_is_flat_at_the_given_height_and_road_width():
    # pier_spacing groesser als die Spannweite: isoliert den Test auf die reine Deck-Geometrie (kein Pfeiler-Vertex
    # in "vertices", der die min()-Annahme unten verfaelschen wuerde - siehe test_piers_reach_down_... fuer die
    # Pfeiler-Faelle mit dem Standard-pier_spacing).
    mesh = build_bridge_mesh(_coords(z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, deck_thickness=0.6, pier_spacing=1000.0)
    v = mesh["vertices"]

    assert v[:, 2].max() == pytest.approx(200.0)  # Deck-Oberkante = Höhenprofil, folgt NICHT dem Gelände
    assert v[:, 2].min() == pytest.approx(200.0 - 0.6)  # Deck-Unterkante minus Pfeiler-Vertices


def test_deck_faces_use_the_road_material_not_the_pier_material():
    mesh = build_bridge_mesh(_coords(n=3), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER)

    assert DECK in mesh["faces"] and len(mesh["faces"][DECK]) > 0


def test_piers_reach_down_to_the_natural_ground_below_a_deep_span():
    mesh = build_bridge_mesh(_coords(length=60.0, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, pier_spacing=25.0)

    assert len(mesh["faces"][PIER]) > 0
    pier_face = mesh["faces"][PIER][0]
    pier_z = np.array([mesh["vertices"][i][2] for i in pier_face])
    assert pier_z.min() == pytest.approx(150.0)  # Pfeiler reicht bis zum natürlichen Gelände


def test_no_piers_when_clearance_is_too_small():
    mesh = build_bridge_mesh(_coords(length=60.0, z=151.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, deck_thickness=0.2, min_pier_clearance=1.0)

    assert mesh["faces"].get(PIER, []) == []


def test_build_bridges_returns_one_mesh_per_bridge_with_its_own_deck_material():
    bridges = [
        {"id": 1, "coords": _coords(z=200.0), "width": 8.0, "deck_material": "asphalt_road_standard"},
        {"id": 2, "coords": _coords(z=210.0), "width": 6.0, "deck_material": "concrete"},
    ]

    meshes = build_bridges(bridges, _flat_ground(150.0), pier_material=PIER)

    assert [m["id"] for m in meshes] == ["bridge_1", "bridge_2"]
    assert "asphalt_road_standard" in meshes[0]["faces"] and "concrete" in meshes[1]["faces"]


def test_build_bridges_skips_degenerate_bridges():
    bridges = [{"id": 1, "coords": [(0.0, 0.0, 200.0)], "width": 8.0, "deck_material": DECK}]

    assert build_bridges(bridges, _flat_ground(150.0), pier_material=PIER) == []
