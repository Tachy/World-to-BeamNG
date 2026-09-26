"""Tests for TerrainWorkflow._build_tunnels() and export_tunnels(): tunnels (tube + portals) and galleries
(roof + supports) as one shared DAE with one TSStatic."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.textures import registry
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow, _structure_items


class _Items:
    def __init__(self):
        self.objects = {}

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), **fields}


class _Materials:
    def __init__(self):
        self.added = {}

    def get_templates(self):
        return {"buildings": {"wall": {"material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"}}}}

    def add_building_material(self, name, **fields):
        self.added[name] = fields


class _Dae:
    def __init__(self):
        self.calls = []

    def export_multi_mesh(self, output_path, meshes, with_uv=False, material_textures=None):
        self.calls.append((Path(output_path), meshes, with_uv))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<COLLADA/>", encoding="utf-8")
        return output_path


def _stub():
    return SimpleNamespace(items=_Items(), materials=_Materials(), dae=_Dae())


CONCRETE = {
    "baseColorMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_b.color.dds",
    "normalMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_nm.normal.dds",
    "roughnessMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_r.data.dds",
}


@pytest.fixture
def shapes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    monkeypatch.setattr(registry, "prepared_textures", lambda: {config.CONCRETE_TEXTURE_NAME: CONCRETE})
    return tmp_path / "shapes"


def _mesh(name):
    return {
        "id": name,
        "vertices": np.zeros((8, 3)),
        "uvs": np.zeros((8, 2)),
        "normals": np.tile([0.0, 0.0, 1.0], (8, 1)),
        "faces": {"asphalt_road_standard": [[0, 1, 2]], config.TUNNEL_MATERIAL_NAME: [[4, 5, 6]]},
    }


def _road(road_id, structure_type, tags):
    return {
        "road_id": road_id,
        "trimmed_centerline": np.array([[0.0, 0.0, 500.0], [200.0, 0.0, 500.0]]),
        "osm_tags": tags,
        "structure_type": structure_type,
    }


def test_export_tunnels_writes_one_dae_one_item_and_registers_floor_and_concrete_materials(shapes_dir):
    stub = _stub()
    roads = [_road(1, "tunnel", {"highway": "trunk", "tunnel": "yes"}), _road(2, "gallery", {"highway": "primary", "tunnel": "avalanche_protector"})]

    count = TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1"), _mesh("tunnel_1_portal_start"), _mesh("tunnel_1_portal_end"), _mesh("gallery_2")], "structure_road_polygons": roads})

    assert count == 4
    dae_path, meshes, with_uv = stub.dae.calls[0]
    assert dae_path == shapes_dir / "tunnels" / "tunnels.dae" and with_uv is True and len(meshes) == 4
    item = stub.items.objects["tunnels"]
    assert item["class"] == "TSStatic" and item["shape_name"] == "levels/world_to_beamng/art/shapes/tunnels/tunnels.dae"
    assert item["collisionType"] == "Visible Mesh Final"
    assert config.TUNNEL_MATERIAL_NAME in stub.materials.added
    assert "asphalt_road_standard_structure" in stub.materials.added
    assert stub.materials.added["asphalt_road_standard_structure"]["groundType"] == "ASPHALT"


def test_export_tunnels_takes_the_concrete_texture_from_the_registry_check_and_never_falls_back(shapes_dir, monkeypatch):
    def aborted():
        raise registry.MissingTexturesError("Beton-Textur fehlt")

    monkeypatch.setattr(registry, "prepared_textures", aborted)
    stub = _stub()

    with pytest.raises(registry.MissingTexturesError):
        TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1")], "structure_road_polygons": [_road(1, "tunnel", {"tunnel": "yes"})]})
    assert not stub.materials.added and not stub.dae.calls


def test_nothing_is_exported_and_stale_files_are_removed_without_tunnels(shapes_dir, monkeypatch):
    stale = shapes_dir / "tunnels"
    stale.mkdir(parents=True)
    (stale / "tunnels.dae").write_text("alt", encoding="utf-8")
    (stale / "tunnels.cdae").write_text("alt", encoding="utf-8")
    stub = _stub()

    assert TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [], "structure_road_polygons": []}) == 0
    assert not (stale / "tunnels.dae").exists() and not (stale / "tunnels.cdae").exists()

    monkeypatch.setattr(config, "TUNNELS_ENABLED", False)
    assert TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1")], "structure_road_polygons": [_road(1, "tunnel", {"tunnel": "yes"})]}) == 0
    assert not stub.dae.calls


def test_build_tunnels_creates_tube_plus_portals_for_a_tunnel_and_one_mesh_per_gallery():
    heights = np.full((50, 50), 495.0)
    tunnel_road = _road(1, "tunnel", {"highway": "trunk", "tunnel": "yes"})
    gallery_road = _road(2, "gallery", {"highway": "primary", "tunnel": "avalanche_protector"})

    plans = plan_tunnels(_structure_items([tunnel_road, gallery_road], "tunnel"), segment_step=10.0, collar_ratio=0.1, flat_depth=1.5, length=3.5)

    meshes = TerrainWorkflow._build_tunnels(SimpleNamespace(), [tunnel_road, gallery_road], plans, heights, 0.0, 0.0)

    ids = [m["id"] for m in meshes]
    assert "tunnel_1" in ids and "tunnel_1_portal_start" in ids and "tunnel_1_portal_end" in ids and "gallery_2" in ids


def test_build_tunnels_skips_surface_and_bridge_roads():
    heights = np.full((10, 10), 495.0)
    road = _road(1, "bridge", {"bridge": "yes"})

    assert _structure_items([road], "tunnel") == []
    assert TerrainWorkflow._build_tunnels(SimpleNamespace(), [road], [], heights, 0.0, 0.0) == []


from types import SimpleNamespace

from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow, _plan_tunnels


def _structure(road_id, coords, **tags):
    from world_to_beamng.geometry.road_structures import classify_structure

    tags = {"highway": "primary", "lanes": "2", **tags}
    return {"road_id": road_id, "trimmed_centerline": np.array(coords, dtype=float), "osm_tags": tags,
            "structure_type": classify_structure(tags)}


def _tunnel_and_gallery():
    return [
        _structure(1, [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)], tunnel="yes"),
        _structure(2, [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], covered="yes", layer="-1"),
    ]


def test_plan_tunnels_marks_the_portal_at_a_covered_gallery_as_transition():
    start, end = _plan_tunnels(_tunnel_and_gallery())[0]["portals"]
    assert start["kind"] == "gallery" and end["kind"] == "open"


def test_gallery_at_a_transition_keeps_its_end_cap_in_the_workflow():
    roads = _tunnel_and_gallery()
    plans = _plan_tunnels(roads)
    heights = np.full((300, 300), 500.0)

    meshes = TerrainWorkflow._build_tunnels(SimpleNamespace(), roads, plans, heights, -150.0, -150.0)

    gallery = next(m for m in meshes if m["id"] == "gallery_2")
    v, n = gallery["vertices"], gallery["normals"]
    faces = [f for fs in gallery["faces"].values() for f in fs]
    at_portal = [f for f in faces if np.allclose(v[f][:, 0], 0.0) and np.allclose(n[f[0]], [1.0, 0.0, 0.0])]
    assert at_portal  # no portal block anymore: the gallery closes its cross-section itself
    assert any(m["id"] == "tunnel_1_portal_start" for m in meshes)


def test_roadblocks_are_placed_on_the_ground_and_exported_as_barrier_statics():
    from world_to_beamng import config
    from world_to_beamng.workflow.terrain_workflow import _roadblock_items

    roads = [_structure(1, [(100.0, 0.0, 500.0), (400.0, 0.0, 500.0)], tunnel="yes")]  # end outside the map
    plans = _plan_tunnels(roads)
    heights = np.full((300, 300), 512.0)
    bounds = (-150.0, 149.0, -150.0, 149.0)

    approach = {"road_id": 9, "trimmed_centerline": np.array([(50.0, 0.0, 500.0), (100.0, 0.0, 500.0)]),
                "osm_tags": {"highway": "primary"}, "structure_type": "surface"}
    blocks = _roadblock_items(plans, heights, -150.0, -150.0, bounds, [approach])

    assert blocks and all(b["position"][2] == pytest.approx(512.0) for b in blocks)
    assert all(b["position"][0] == pytest.approx(95.0) for b in blocks)

    added = {}
    stub = SimpleNamespace(items=SimpleNamespace(add_item=lambda name, **kw: added.__setitem__(name, kw)))
    TerrainWorkflow.export_roadblocks(stub, {"roadblocks": blocks})

    assert set(added) == {b["name"] for b in blocks}
    first = added[blocks[0]["name"]]
    assert first["item_class"] == "TSStatic" and first["shape_name"] == config.ROADBLOCK_SHAPE
    assert first["rotation_matrix"] == blocks[0]["rotation_matrix"]


def test_tunnel_zones_are_planned_from_the_config_and_exported_as_zone_objects():
    from world_to_beamng.workflow.terrain_workflow import _tunnel_zone_items

    # 300 m: long enough for the entrance insets (config.TUNNEL_ZONE_ENTRANCE_INSET) at both open portals
    roads = [
        _structure(1, [(0.0, 0.0, 500.0), (300.0, 0.0, 500.0)], tunnel="yes"),
        _structure(2, [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], covered="yes", layer="-1"),
    ]
    plans = _plan_tunnels(roads)
    zones = _tunnel_zone_items(plans)

    assert zones and all(z["name"].startswith(("tunnel_zone_1_", "tunnel_zone_portal_1_")) for z in zones)  # only the tunnel

    added = {}
    stub = SimpleNamespace(items=SimpleNamespace(add_item=lambda name, **kw: added.__setitem__(name, kw)))
    count = TerrainWorkflow.export_tunnel_zones(stub, {"tunnel_zones": zones})

    assert count == len(zones) and set(added) == {z["name"] for z in zones}
    zone = next(z for z in zones if z["class"] == "Zone")
    first = added[zone["name"]]
    assert first["item_class"] == "Zone"
    assert any(kw["item_class"] == "Portal" for kw in added.values())
    assert first["scale"] == zone["scale"] and first["rotation_matrix"] == zone["rotation_matrix"]
    assert first["useAmbientLightColor"] is True and first["ambientLightColor"] == [0, 0, 0, 1]
    assert first["skyLightFactor"] == pytest.approx(0.05)


def test_untagged_gallery_embankment_uses_the_terrain_valley_side():
    from world_to_beamng import config
    from world_to_beamng.workflow.terrain_workflow import _gallery_embedding

    road = _structure(2, [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], covered="yes", layer="-1")
    ground_at = lambda x, y: 500.0 + 1.0 * np.asarray(y, float)  # travel direction +x: valley on the right (-y)

    override, flat_sides = _gallery_embedding(road, ground_at)

    assert road["open_side"] == "right"
    assert set(override) == {"left", "right"}
    assert np.allclose(override["right"], config.GALLERY_VALLEY_SLOPE_WIDTH)  # real terrain already at edge + 5 m
    assert override["left"] == config.GALLERY_MOUNTAIN_EMBED_MARGIN
    assert flat_sides == {"left"}
    assert _structure_items([road], "gallery")[0]["open_side"] == "right"  # same side for the gallery mesh


def test_gallery_valley_embankment_reaches_past_the_roof_the_dgm_still_shows():
    # Above the gallery the DGM shows its roof (~5 m above the road surface) up to ~9 m beside the axis. A fixed
    # reference 5 m behind the edge (8.25 m) would still lie on the roof -> terrain spike downhill (long gallery
    # Nuova strada). The reference is searched downhill until the DGM drops below the roof level.
    from world_to_beamng import config
    from world_to_beamng.workflow.terrain_workflow import _gallery_embedding

    road = _structure(2, [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], covered="yes", layer="-1")
    half = config.OSM_MAPPER.get_road_properties(road["osm_tags"])["width"] / 2.0

    def ground_at(x, y):
        y = np.asarray(y, float)
        valley = 500.0 - 5.0 + 0.5 * y  # valley on the right (-y), rising on the uphill side (+y)
        return np.where((y < 0.0) & (y > -9.0), 505.0, valley)  # gallery roof in the DGM up to 9 m on the valley side

    override, _ = _gallery_embedding(road, ground_at)

    widths = np.asarray(override["right"])
    assert np.all(half + widths > 9.0)  # reference point beyond the roof
    # first search point with lower terrain (search step 0.5 m) is taken directly, no surcharge
    assert np.all(half + widths <= 9.0 + config.GALLERY_VALLEY_SEARCH_STEP)


def test_gallery_valley_embankment_keeps_the_minimum_width_when_the_valley_side_is_higher():
    from world_to_beamng import config
    from world_to_beamng.workflow.terrain_workflow import _gallery_embedding

    road = _structure(2, [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], covered="yes", layer="-1")
    ground_at = lambda x, y: 500.0 + 30.0 + 0.1 * np.asarray(y, float)  # everything far above the road surface

    override, _ = _gallery_embedding(road, ground_at)

    assert np.allclose(override[road["open_side"]], config.GALLERY_VALLEY_SLOPE_WIDTH)


@pytest.mark.parametrize("highway", ["path", "footway", "steps", "bridleway", "pedestrian", "construction"])
def test_tunnels_of_footpaths_and_non_roads_are_not_built(highway):
    # In the mountains there are "tunnels" for paths (fortress galleries) - only roads and cycleways get a tunnel
    from world_to_beamng.workflow.terrain_workflow import _plan_tunnels

    roads = [_structure(1, [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)], tunnel="yes", highway=highway)]

    assert _plan_tunnels(roads) == []


@pytest.mark.parametrize("highway", ["primary", "track", "cycleway", "service"])
def test_tunnels_of_roads_and_cycleways_are_built(highway):
    from world_to_beamng.workflow.terrain_workflow import _plan_tunnels

    roads = [_structure(1, [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)], tunnel="yes", highway=highway)]

    assert len(_plan_tunnels(roads)) == 1
