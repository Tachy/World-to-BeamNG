"""Junction corners in the export: fill meshes as junctions.dae + TSStatic; sidewalks joined around corners."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng import config
from world_to_beamng.junctions.corners import find_junction_corners
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _Items:
    def __init__(self):
        self.roads, self.objects = {}, {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = nodes

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), **fields}


class _Materials:
    def __init__(self):
        self.materials, self.added = {}, {}

    def get_templates(self):
        return {}

    def add_building_material(self, name, **fields):
        self.added[name] = fields


class _Dae:
    def export_multi_mesh(self, output_path, meshes, with_uv=False, material_textures=None):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<COLLADA/>", encoding="utf-8")


def _stub():
    stub = SimpleNamespace(items=_Items(), materials=_Materials(), dae=_Dae(), _export_structure_road_assets=lambda lines: None)
    stub._register_structure_road_materials = lambda names, **kw: TerrainWorkflow._register_structure_road_materials(stub, names, **kw)
    return stub


def _t_mesh_data(sides_east=None, sides_north=None):
    def poly(rid, pts, width, sides):
        c = np.array([[x, y, 100.0] for x, y in pts])
        d = {"road_id": rid, "trimmed_centerline": c, "osm_tags": {"highway": "residential"}}
        if sides:
            d["sidewalk_sides"] = sides
        return d, {"road_id": rid, "coords": c, "half_widths": np.full(len(c), width / 2.0), "highway": "residential",
                   "surface": "asphalt_road_standard"}

    width = config.OSM_MAPPER.get_road_properties({"highway": "residential"})["width"]
    east = poly("east", [(x, 100.0) for x in np.arange(100.0, 160.5, 1.0)], width, sides_east)
    west = poly("west", [(x, 100.0) for x in np.arange(40.0, 100.5, 1.0)], width, None)
    north = poly("north", [(100.0, y) for y in np.arange(100.0, 160.5, 1.0)], width, sides_north)
    corners = find_junction_corners([east[1], west[1], north[1]], {"default_radius": 6.0}, 0.5, 160.0, rank={})
    return {"road_slope_polygons_2d": [east[0], west[0], north[0]], "junction_corners": corners,
            "heightmap": np.full((256, 256), 100.0), "terrain_origin_x": 0.0, "terrain_origin_y": 0.0}


def test_export_junctions_writes_dae_tsstatic_and_structure_material(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    stub = _stub()
    assert TerrainWorkflow.export_junctions(stub, _t_mesh_data()) == 1
    assert (tmp_path / "shapes" / "junctions" / "junctions.dae").is_file()
    item = stub.items.objects["junctions"]
    assert item["collisionType"] == "None" and "rotation" not in item  # visual only: vehicles drive on the terrain
    assert "asphalt_road_standard_junction" in stub.materials.added


def test_export_junctions_removes_leftovers_without_corners(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    old = tmp_path / "shapes" / "junctions" / "junctions.dae"
    old.parent.mkdir(parents=True)
    old.write_text("old", encoding="utf-8")
    assert TerrainWorkflow.export_junctions(_stub(), {"junction_corners": []}) == 0
    assert not old.exists()


def test_sidewalks_are_joined_around_the_corner_in_export_decal_roads():
    mesh_data = _t_mesh_data(sides_east={"left": "asphalt_road_standard"}, sides_north={"right": "asphalt_road_standard"})
    TerrainWorkflow.export_decal_roads(_stub(), mesh_data)
    assert len(mesh_data["sidewalk_meshes"]) == 1


def test_gravel_fills_are_translucent_like_the_gravel_decal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    mesh_data = _t_mesh_data()
    for corner in mesh_data["junction_corners"]:
        corner["surface"] = "gravel_road"
    stub = _stub()
    TerrainWorkflow.export_junctions(stub, mesh_data)
    gravel = stub.materials.added["gravel_road_junction"]
    opacity = config.OSM_MAPPER.config["surface_types"]["gravel_road"]["opacityFactor"]
    assert gravel["stage_properties"]["opacityFactor"] == opacity and gravel["translucent"] is True


def test_opaque_surfaces_stay_opaque(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    stub = _stub()
    TerrainWorkflow.export_junctions(stub, _t_mesh_data())
    asphalt = stub.materials.added["asphalt_road_standard_junction"]
    assert "translucent" not in asphalt and not (asphalt.get("stage_properties") or {}).get("opacityFactor")


def test_fill_materials_do_not_share_names_with_bridge_decks(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    stub = _stub()
    TerrainWorkflow.export_junctions(stub, _t_mesh_data())
    assert not any(name.endswith("_structure") for name in stub.materials.added)


def test_corners_of_roads_without_decal_are_not_drawn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    mesh_data = _t_mesh_data()
    for corner in mesh_data["junction_corners"]:
        corner["surface"] = "dirt_road"  # a dirt track joins: terrain only, nothing drawn
    stub = _stub()
    assert TerrainWorkflow.export_junctions(stub, mesh_data) == 0
    assert not (tmp_path / "shapes" / "junctions" / "junctions.dae").exists()
    assert "junctions" not in stub.items.objects
