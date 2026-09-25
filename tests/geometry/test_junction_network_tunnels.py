"""Tests for world_to_beamng.geometry.junctions.build_junction_network: tunnels lie on a different level and
never form junctions - a tunnel is neither split at a path crossing above it, nor vice versa."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.geometry.junctions import build_junction_network


def _road(road_id, points, z=100.0, **tags):
    # as from get_road_polygons(): list of (x, y, z), densely (1 m) sampled
    coords = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        n = max(1, int(round(np.hypot(x1 - x0, y1 - y0))))
        coords += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, z) for k in range(n)]
    coords.append((float(points[-1][0]), float(points[-1][1]), z))
    return {"id": road_id, "coords": coords, "name": "r", "osm_tags": tags}


def test_a_path_crossing_over_a_tunnel_splits_neither_road():
    # (tunnel and path each form their own network without shared endpoints - no junction arises at all)
    tunnel = _road(1, [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], z=50.0, highway="primary", tunnel="yes")
    path = _road(2, [(50.0, -30.0), (50.0, 0.0), (50.0, 30.0)], highway="path")

    roads, junctions = build_junction_network([tunnel, path])

    assert sorted(r["id"] for r in roads) == [1, 2]
    assert junctions == []
    tunnel_out = next(r for r in roads if r["id"] == 1)
    assert tunnel_out["junction_indices"] == {"start": None, "end": None}
    assert np.allclose(np.asarray(tunnel_out["coords"]), tunnel["coords"])


def test_surface_junctions_are_still_detected_and_split():
    main = _road(1, [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], highway="primary")
    side = _road(2, [(50.0, 0.0), (50.0, 30.0)], highway="residential")
    tunnel = _road(3, [(100.0, 0.0), (200.0, 0.0)], z=50.0, highway="primary", tunnel="yes")
    branch = _road(4, [(100.0, 0.0), (100.0, -30.0)], highway="track")

    roads, junctions = build_junction_network([main, side, tunnel, branch])

    ids = [r["id"] for r in roads]
    assert 1001 in ids and 1002 in ids and 1 not in ids  # main road split at the T-junction
    assert ids[-1] == 3  # tunnel appended unchanged at the end
    assert len(junctions) == 2  # T-junction + endpoint at x=100 (without the tunnel)
    assert all(3 not in [roads[i]["id"] for i in j["road_indices"]] for j in junctions)


def test_tunnel_branching_off_a_tunnel_is_still_a_junction():
    main = _road(1, [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], z=50.0, highway="primary", tunnel="yes")
    branch = _road(2, [(50.0, 0.0), (50.0, 40.0)], z=50.0, highway="primary", tunnel="yes")
    other = _road(3, [(100.0, 0.0), (100.0, -40.0)], z=50.0, highway="primary", tunnel="yes")
    path_over = _road(4, [(20.0, -30.0), (20.0, 30.0)], highway="path")
    seed_a = _road(5, [(500.0, 0.0), (550.0, 0.0)], highway="track")
    seed_b = _road(6, [(550.0, 0.0), (550.0, 30.0)], highway="track")

    roads, junctions = build_junction_network([main, branch, other, path_over, seed_a, seed_b])
    ids = [r["id"] for r in roads]

    assert 1001 in ids and 1002 in ids and 1 not in ids  # main tunnel split at the branch ...
    tunnel_parts = [r for r in roads if r["id"] in (1001, 1002)]
    assert sorted(round(float(np.asarray(r["coords"])[:, 0].max())) for r in tunnel_parts) == [50, 100]  # ... only there, not at the path at x=20
    assert 4 in ids  # the path above stays whole
    tunnel_junctions = [j for j in junctions if any(roads[i]["id"] in (1001, 1002, 2, 3) for i in j["road_indices"])]
    assert len(tunnel_junctions) == 2  # branch at x=50 + joint at x=100
    for j in junctions:
        kinds = {roads[i]["osm_tags"].get("tunnel") == "yes" for i in j["road_indices"]}
        assert len(kinds) == 1  # never tunnel and surface in the same junction
    for road in roads:
        for end in ("start", "end"):
            index = road["junction_indices"][end]
            if index is not None:
                assert any(roads[i] is road for i in junctions[index]["road_indices"])  # indices match the overall list


def _with_nodes(road, nodes):
    """OSM nodes as from get_road_polygons(): [(node_id, x, y), ...] plus the way ID."""
    road["osm_way_id"] = road["id"]
    road["osm_nodes"] = nodes
    return road


def _tunnel_crossing_setup(shared_node: bool):
    main = _with_nodes(
        _road(1, [(0.0, 0.0), (100.0, 0.0)], z=50.0, highway="primary", tunnel="yes"),
        [(10, 0.0, 0.0), (11, 50.0, 0.0), (12, 100.0, 0.0)],
    )
    crossing_node = 11 if shared_node else 21
    other = _with_nodes(
        _road(2, [(50.0, -40.0), (50.0, 40.0)], z=80.0, highway="path", tunnel="yes"),
        [(20, 50.0, -40.0), (crossing_node, 50.0, 0.0), (22, 50.0, 40.0)],
    )
    seed_a = _with_nodes(_road(5, [(500.0, 0.0), (550.0, 0.0)], z=50.0, highway="primary", tunnel="yes"), [(30, 500.0, 0.0), (31, 550.0, 0.0)])
    seed_b = _with_nodes(_road(6, [(550.0, 0.0), (550.0, 30.0)], z=50.0, highway="primary", tunnel="yes"), [(31, 550.0, 0.0), (32, 550.0, 30.0)])
    return [main, other, seed_a, seed_b]


def test_tunnels_crossing_at_different_depths_without_a_shared_osm_node_are_not_split():
    roads, junctions = build_junction_network(_tunnel_crossing_setup(shared_node=False))
    ids = [r["id"] for r in roads]

    assert 1 in ids and 2 in ids  # neither of the two tunnels split
    assert len(junctions) == 1  # only the real joint of the seed ways (shared node 31)


def test_tunnels_sharing_an_osm_node_form_a_real_junction():
    roads, junctions = build_junction_network(_tunnel_crossing_setup(shared_node=True))
    ids = [r["id"] for r in roads]

    assert 1001 in ids and 1002 in ids and 2001 in ids and 2002 in ids
    assert len(junctions) == 2
