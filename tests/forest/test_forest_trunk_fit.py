"""Trunks (not just the origin) respect the exclusion zones and stand on the ground."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import box

from world_to_beamng.forest.tree_footprints import TrunkFitter, read_trunk_feet
from world_to_beamng.forest.vineyard_generator import make_height_sampler

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# 90° about Z; BeamNG reads the model axes as rows: model X points to +y, model Y to -x
ROT_90 = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# Group with two trunks (origin and 6 m along model X), single tree with one trunk at the origin
FEET = {
    "group": np.array([[0.0, 0.0, -1.0], [6.0, 0.0, -1.0]]),
    "single": np.array([[0.0, 0.0, -0.5]]),
}
POOL = {"group": 0.5, "single": 0.5}


def _instance(tree_type, x, y, z=0.0, matrix=IDENTITY, scale=1.0):
    return {"type": tree_type, "pos": [x, y, z], "rotationMatrix": list(matrix), "scale": scale}


def _flat(z=0.0):
    return make_height_sampler(np.full((60, 60), z), 0.0, 0.0, 1.0)


def _downslope(gradient):
    """Slope that falls by `gradient` meters of elevation per meter in the +x direction (ground at x = -gradient * x)."""
    return make_height_sampler(np.tile(-np.arange(60, dtype=float) * gradient, (60, 1)), 0.0, 0.0, 1.0)


def _fitter(**kwargs):
    kwargs.setdefault("height_at", _flat())
    return TrunkFitter(FEET, rng=np.random.default_rng(0), **kwargs)


def test_instances_that_fit_are_left_untouched():
    instance = _instance("group", 10.0, 10.0, 0.0)
    result = _fitter(exclusion=box(40, 40, 50, 50)).fit([instance], POOL)
    assert result == [instance]


def test_trunk_inside_the_exclusion_zone_swaps_the_type_without_moving_the_tree():
    # Origin free (x=10), but the second trunk stands on the path at x=16
    instance = _instance("group", 10.0, 10.0, 0.0)
    result = _fitter(exclusion=box(14, 0, 20, 30)).fit([instance], POOL)
    assert len(result) == 1
    assert result[0]["type"] == "single"
    assert result[0]["pos"] == instance["pos"]
    assert result[0]["rotationMatrix"] == instance["rotationMatrix"]
    assert result[0]["scale"] == instance["scale"]


def test_trunk_offsets_follow_rotation_and_scale():
    # Rotated by 90° the second trunk points to +y: at (10, 10 + 6 * 2) = (10, 22) with scale 2
    instance = _instance("group", 10.0, 10.0, 0.0, matrix=ROT_90, scale=2.0)
    blocked_at_y22 = _fitter(exclusion=box(5, 20, 15, 25)).fit([instance], POOL)
    assert blocked_at_y22[0]["type"] == "single"
    free_at_x22 = _fitter(exclusion=box(20, 5, 25, 15)).fit([instance], POOL)  # the trunk would lie there without rotation
    assert free_at_x22 == [instance]


def test_instance_is_dropped_when_no_pool_type_fits():
    instance = _instance("group", 10.0, 10.0, 0.0)
    result = _fitter(exclusion=box(14, 0, 20, 30)).fit([instance], {"group": 1.0})
    assert result == []


def test_floating_trunk_sinks_the_tree_by_the_excess():
    # Slope 0.5 m/m falls toward +x: the ground below the second trunk (x=16) is 3 m lower than at the origin.
    # Base point at -1 m => 2 m above the ground; max_float 0.5 => 1.5 m lowering needed, 2.0 m allowed
    instance = _instance("group", 10.0, 10.0, -5.0)
    result = _fitter(height_at=_downslope(0.5), max_float=0.5, max_sink=2.0).fit([instance], POOL)
    assert result[0]["type"] == "group"
    assert result[0]["pos"][:2] == [10.0, 10.0]
    assert result[0]["pos"][2] == pytest.approx(-5.0 - 1.5)


def test_tree_on_too_steep_ground_is_swapped_when_sinking_is_not_enough():
    instance = _instance("group", 10.0, 10.0, -5.0)
    result = _fitter(height_at=_downslope(0.5), max_float=0.5, max_sink=1.0).fit([instance], POOL)
    assert result[0]["type"] == "single"
    assert result[0]["pos"] == [10.0, 10.0, -5.0]  # single tree sits on the ground unchanged


def test_buried_trunks_are_not_touched_on_flat_ground():
    instance = _instance("group", 10.0, 10.0, 0.0)
    assert _fitter().fit([instance], POOL) == [instance]


def test_types_without_footprint_are_treated_as_single_trunk():
    instance = _instance("unknown", 10.0, 10.0, 0.0)
    assert _fitter(exclusion=box(40, 40, 50, 50)).fit([instance], POOL) == [instance]


def test_row_trees_use_the_row_exclusion_not_the_forest_exclusion():
    instance = _instance("single", 10.0, 10.0, 0.0)
    fitter = _fitter(exclusion=box(5, 5, 15, 15), row_exclusion=box(40, 40, 50, 50))
    assert fitter.fit([instance], {"single": 1.0}, row=True) == [instance]
    assert fitter.fit([instance], {"single": 1.0}, row=False) == []


def test_without_height_sampler_only_the_exclusion_is_checked():
    instance = _instance("group", 10.0, 10.0, 99.0)
    fitter = TrunkFitter(FEET, exclusion=None, height_at=None, rng=np.random.default_rng(0))
    assert fitter.fit([instance], POOL) == [instance]


def test_empty_input_returns_empty_list():
    assert _fitter().fit([], POOL) == []


# --- Trunk bases from the collision model ---------------------------------------------------------------

DAE_TEMPLATE = """<?xml version="1.0"?>
<COLLADA><library_geometries>
<geometry id="Crown-mesh" name="Crown"><mesh><source id="Crown-mesh-positions">
<float_array id="Crown-mesh-positions-array" count="6">0 0 15 9 9 20</float_array></source></mesh></geometry>
<geometry id="Col-mesh" name="Col"><mesh><source id="Col-mesh-positions">
<float_array id="Col-mesh-positions-array" count="{count}">{values}</float_array></source></mesh></geometry>
</library_geometries><library_visual_scenes><visual_scene id="Scene">
<node id="c" name="Colmesh_tree-1"><instance_geometry url="#Col-mesh" name="Colmesh_tree-1"/></node>
<node id="k" name="tree_a350"><instance_geometry url="#Crown-mesh" name="tree_a350"/></node>
</visual_scene></library_visual_scenes></COLLADA>"""


def _write_dae(tmp_path, points):
    flat = " ".join(str(v) for p in points for v in p)
    path = tmp_path / "tree.dae"
    path.write_text(DAE_TEMPLATE.format(count=len(points) * 3, values=flat), encoding="utf-8")
    return path


def test_read_trunk_feet_returns_the_lowest_point_of_every_trunk(tmp_path):
    # two trunks (base + tip each) at (0,0) and (5,-3); the crown in the other geometry does not count
    path = _write_dae(tmp_path, [(0, 0, -1.5), (0.1, 0, 8), (5, -3, -1.2), (5.2, -3, 9)])
    feet = read_trunk_feet(path)
    assert sorted(map(tuple, np.round(feet, 1).tolist())) == [(0.0, 0.0, -1.5), (5.0, -3.0, -1.2)]


def test_read_trunk_feet_ignores_high_collision_pieces_and_merges_duplicates(tmp_path):
    # Vertices of the same trunk in neighboring cells count once; pieces high above the ground are not bases
    path = _write_dae(tmp_path, [(0.9, 0, -1.0), (1.1, 0, -0.9), (3, 3, 6.0)])
    feet = read_trunk_feet(path)
    assert len(feet) == 1
    assert feet[0][2] == pytest.approx(-1.0)


def test_read_trunk_feet_falls_back_to_the_origin_without_collision_mesh(tmp_path):
    path = tmp_path / "plain.dae"
    path.write_text("<COLLADA></COLLADA>", encoding="utf-8")
    assert read_trunk_feet(path).tolist() == [[0.0, 0.0, 0.0]]
    assert read_trunk_feet(tmp_path / "missing.dae").tolist() == [[0.0, 0.0, 0.0]]


# --- Connection to the instance generator ---------------------------------------------------------


class _RecordingFitter:
    def __init__(self):
        self.calls = []

    def fit(self, instances, pool, row=False):
        self.calls.append((len(instances), dict(pool), row))
        return instances[:1]  # simulates discarded instances


def test_instance_generator_passes_pool_and_row_flag_to_the_fitter():
    from world_to_beamng.forest.forest_instance_generator import ForestInstanceGenerator

    fitter = _RecordingFitter()
    properties = {
        "forest": {"preferred_trees": {"group": 1.0}, "average_height": [20.0, 20.0]},
        "row": {"preferred_trees": {"single": 1.0}, "average_height": [20.0, 20.0], "row_spacing": 8.0},
    }
    points = {0: [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)], 1: [(0.0, 9.0, 0.0), (8.0, 9.0, 0.0)]}
    forests = [{"type": "forest"}, {"type": "row"}]

    result = ForestInstanceGenerator().generate_instances_for_forests(points, forests, properties, fitter=fitter)

    assert fitter.calls == [(2, {"group": 1.0}, False), (2, {"single": 1.0}, True)]
    assert len(result) == 2  # only what the fitter returns


def test_instance_generator_without_fitter_is_unchanged():
    from world_to_beamng.forest.forest_instance_generator import ForestInstanceGenerator

    properties = {"forest": {"preferred_trees": {"group": 1.0}, "average_height": [20.0, 20.0]}}
    result = ForestInstanceGenerator().generate_instances_for_forests(
        {0: [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]}, [{"type": "forest"}], properties
    )
    assert len(result) == 2
