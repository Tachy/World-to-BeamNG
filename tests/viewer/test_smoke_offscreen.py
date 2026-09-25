"""Off-screen smoke test of the level viewer on the synthetic level: all layers build, the terrain texture is
oriented with the image top at the north edge, and a pick in the window center reports the item under it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from tests.viewer.conftest import ORIGIN, PADDING, ROAD_Z, TER_SIZE
from tools.level_viewer.cli import ViewerOptions


def _viewer(level_dir, **overrides):
    from tools.level_viewer.app import LevelViewer

    try:
        return LevelViewer(ViewerOptions(level_dir=level_dir, terrain_step=2, layer_overrides=overrides), off_screen=True)
    except Exception as exc:  # no OpenGL context available (headless CI)
        pytest.skip(f"off-screen rendering not available: {exc}")


def test_all_layers_build_and_report_a_summary(level_dir):
    viewer = _viewer(level_dir, **{key: True for key in "gamobhczn"})
    try:
        summaries = {layer.title: layer.summary for layer in viewer.layers if layer.built}
        assert summaries["Roads"] == "1 in 1 materials"
        assert summaries["Markings"] == "1 in 1 materials"
        assert summaries["Water"] == "1 rivers, 1 water blocks"
        assert summaries["Forest"] == "3 trees, 2 types"
        assert summaries["Zones + spawns"] == "1 zones, 0 portals, 1 spawns"
        assert "not resolvable" in summaries["Horizon"]  # vanilla /assets path
    finally:
        viewer.plotter.close()


def test_terrain_photo_is_north_up_and_picking_reports_the_road(level_dir):
    viewer = _viewer(level_dir, a=True, m=False, o=False, z=False, c=False)
    try:
        solid = TER_SIZE - PADDING
        cx, cy = ORIGIN[0] + solid / 2, ORIGIN[1] + solid / 2
        viewer.plotter.camera_position = [(cx, cy, ORIGIN[2] + 400), (cx, cy, ORIGIN[2]), (0, 1, 0)]
        viewer.plotter.remove_actor("panel")
        image = viewer.plotter.screenshot(return_img=True)

        # Minimap: top half (north) red, bottom half grey; look at the middle column just above/below the center
        height = image.shape[0]
        top = image[int(height * 0.3) : int(height * 0.4), image.shape[1] // 2]
        bottom = image[int(height * 0.6) : int(height * 0.7), image.shape[1] // 2]
        assert top[:, 0].mean() > 3 * top[:, 1].mean() + 20  # red (shaded) in the north
        assert abs(int(bottom[:, 0].mean()) - int(bottom[:, 1].mean())) < 40  # grey south part

        # Road "road_a" runs from (0, 0) to (40, 0): look straight down onto it
        viewer.plotter.camera_position = [(20, 0, ROAD_Z + 150), (20, 0, ROAD_Z), (0, 1, 0)]
        viewer.plotter.render()
        width, height = viewer.plotter.window_size
        info = viewer.pick(width // 2, height // 2)
        assert info is not None and info[0].startswith("road_a") and "[DecalRoad]" in info[0]
        assert any(line.startswith("terrain below:") for line in info)
    finally:
        viewer.plotter.close()
