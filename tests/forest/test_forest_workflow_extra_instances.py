"""Tests for ForestWorkflow.add_instances() (additional instances such as vineyard vines)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.forest.forest_json_writer import ForestJSONWriter
from world_to_beamng.workflow.forest_workflow import ForestWorkflow

VINE = {"type": "grape_vine", "pos": [1.0, 2.0, 3.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 1.0}
TREE = {"type": "oak", "pos": [5.0, 6.0, 7.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 1.2}


def test_add_instances_appends_to_the_tree_instances():
    workflow = ForestWorkflow(config)
    workflow.all_tree_instances = [TREE]

    added = workflow.add_instances([VINE, dict(VINE)])

    assert added == 2
    assert [i["type"] for i in workflow.all_tree_instances] == ["oak", "grape_vine", "grape_vine"]


def test_added_instances_end_up_in_forest_json_with_trees(tmp_path):
    workflow = ForestWorkflow(config)
    workflow.json_writer = ForestJSONWriter(tmp_path)
    workflow.all_tree_instances = [dict(TREE)]
    workflow.add_instances([dict(VINE)])

    result = workflow.finalize_forest_export()

    assert result["status"] == "success"
    lines = (tmp_path / "forest.forest4.json").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["type"] for line in lines) == ["grape_vine", "oak"]


def test_add_instances_with_empty_list_is_a_noop():
    workflow = ForestWorkflow(config)

    assert workflow.add_instances([]) == 0
    assert workflow.all_tree_instances == []
