"""Tests für world_to_beamng.geometry.junctions.split_roads_at_mid_junctions: Straßen an Mid-Junctions
aufteilen. Regression: eine Mid-Junction, deren Projektion exakt (oder numerisch fast) auf einen bereits
vorhandenen Centerline-Punkt fällt, darf keinen doppelten Endpunkt erzeugen (0-Länge-Segment -> NaN bei
nachgelagerter Richtungs-Normalisierung, z.B. in Brücken-/Tunnel-/Galerie-Meshes, die keine eigene
Duplikat-Bereinigung wie drop_close_nodes() durchlaufen)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.junctions import split_roads_at_mid_junctions


def _road(road_id, points, z=0.0, **tags):
    return {
        "id": road_id,
        "coords": np.array([[float(x), float(y), z] for x, y in points]),
        "name": "r",
        "osm_tags": tags,
    }


def _min_gap(coords):
    xy = np.asarray(coords)[:, :2]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).min())


def test_mid_junction_landing_exactly_on_an_existing_point_does_not_duplicate_it():
    road = _road(1, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    junctions = [{"position": (2.0, 0.0, 0.0), "connection_types": {0: ["mid"]}}]

    new_roads, _ = split_roads_at_mid_junctions([road], junctions)

    assert len(new_roads) == 2
    for r in new_roads:
        assert _min_gap(r["coords"]) > 0.0, f"road {r['id']} has a zero-length (duplicated-point) segment: {r['coords']}"


def test_mid_junction_landing_very_close_to_an_existing_point_does_not_produce_a_near_zero_segment():
    # Projektion faellt numerisch fast (aber nicht exakt) auf den Punkt bei x=2 - derselbe Fehlermodus kann
    # auch ohne exakte Gleichheit auftreten (Rundung in der Projektionsrechnung).
    road = _road(1, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    junctions = [{"position": (2.0 + 1e-9, 0.0, 0.0), "connection_types": {0: ["mid"]}}]

    new_roads, _ = split_roads_at_mid_junctions([road], junctions)

    for r in new_roads:
        assert _min_gap(r["coords"]) > 1e-6, f"road {r['id']} has a near-zero segment: {r['coords']}"


def test_mid_junction_in_the_middle_of_a_segment_still_splits_normally():
    road = _road(1, [(0.0, 0.0), (10.0, 0.0)])
    junctions = [{"position": (5.0, 0.0, 0.0), "connection_types": {0: ["mid"]}}]

    new_roads, _ = split_roads_at_mid_junctions([road], junctions)

    assert len(new_roads) == 2
    coords_a = np.asarray(new_roads[0]["coords"])
    coords_b = np.asarray(new_roads[1]["coords"])
    assert coords_a[-1][0] == pytest.approx(5.0) and coords_a[-1][1] == pytest.approx(0.0)
    assert coords_b[0][0] == pytest.approx(5.0) and coords_b[0][1] == pytest.approx(0.0)
    assert _min_gap(coords_a) == pytest.approx(5.0)
    assert _min_gap(coords_b) == pytest.approx(5.0)


def test_road_without_mid_junctions_is_returned_unchanged():
    road = _road(1, [(0.0, 0.0), (10.0, 0.0)])

    new_roads, _ = split_roads_at_mid_junctions([road], [])

    assert len(new_roads) == 1
    assert np.array_equal(np.asarray(new_roads[0]["coords"]), np.asarray(road["coords"]))
