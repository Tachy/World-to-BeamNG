"""
Material export of the LOD2 buildings: plaster per color, windows, beaver-tail tiles (unchanged), gravel, sheet metal rim, trim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.export.beamng_exporter import BeamNGExporter
from world_to_beamng import config
from world_to_beamng.facade import building_textures
from world_to_beamng.facade.facade_styles import PLASTER_COLORS
from world_to_beamng.facade.material_names import (
    FLAT_ROOF_MATERIAL,
    ROOF_EDGE_MATERIAL,
    ROOF_MATERIAL,
    ROOF_TRIM_MATERIAL,
    WALL_MATERIALS,
    WINDOW_MATERIAL,
)
from world_to_beamng.managers.material_manager import MaterialManager
from world_to_beamng.textures import registry

GENERATED = {
    **{f"plaster_color_{c.name}": f"levels/x/plaster_{c.name}_b.dds" for c in PLASTER_COLORS},
    "plaster_normal": "levels/x/plaster_nm.dds",
    "plaster_roughness": "levels/x/plaster_r.dds",
    "windows_color": "levels/x/windows_b.dds",
    "windows_normal": "levels/x/windows_nm.dds",
    "windows_roughness": "levels/x/windows_r.dds",
}

LIBRARY = {
    config.FLAT_ROOF_GRAVEL_TEXTURE: {
        "baseColorMap": "levels/x/gravel_b.dds",
        "normalMap": "levels/x/gravel_nm.dds",
        "roughnessMap": "levels/x/gravel_r.dds",
    }
}


def _materials(tmp_path, monkeypatch):
    monkeypatch.setattr(building_textures, "ensure_building_textures", lambda output_dir=None: GENERATED)
    monkeypatch.setattr(registry, "prepared_textures", lambda: LIBRARY)
    MaterialManager.reset_instance()
    manager = MaterialManager.get_instance(tmp_path)
    exporter = BeamNGExporter.__new__(BeamNGExporter)
    exporter.materials = manager
    exporter._add_lod2_materials()
    return manager.materials


def test_one_wall_material_per_plaster_colour_sharing_normal_and_roughness(tmp_path, monkeypatch):
    materials = _materials(tmp_path, monkeypatch)

    assert len(WALL_MATERIALS) == len(PLASTER_COLORS)
    for name, color in zip(WALL_MATERIALS, PLASTER_COLORS):
        stage = materials[name]["Stages"][0]
        assert stage["baseColorMap"] == GENERATED[f"plaster_color_{color.name}"]
        assert stage["normalMap"] == GENERATED["plaster_normal"]
        assert stage["roughnessMap"] == GENERATED["plaster_roughness"]
        assert stage["useAnisotropic"] is True
        assert "materialFactors" not in stage  # UVs are metric, no tiling scale
    MaterialManager.reset_instance()


def test_windows_use_the_sprite_atlas(tmp_path, monkeypatch):
    stage = _materials(tmp_path, monkeypatch)[WINDOW_MATERIAL]["Stages"][0]

    assert stage["baseColorMap"] == GENERATED["windows_color"]
    assert stage["normalMap"] == GENERATED["windows_normal"]
    MaterialManager.reset_instance()


def test_old_wall_materials_are_gone(tmp_path, monkeypatch):
    materials = _materials(tmp_path, monkeypatch)

    assert "lod2_wall_white" not in materials and "lod2_wall_facade" not in materials
    MaterialManager.reset_instance()


def test_tile_roof_keeps_its_texture_and_tint_but_loses_the_tiling_factor(tmp_path, monkeypatch):
    stage = _materials(tmp_path, monkeypatch)[ROOF_MATERIAL]["Stages"][0]

    assert "t_roof_slates_rounded_b.color.dds" in stage["baseColorMap"]
    assert stage["diffuseColor"] == [0.85, 0.45, 0.3, 1.0]
    assert "materialFactors" not in stage
    MaterialManager.reset_instance()


def test_flat_roof_uses_gravel_and_rim_and_trim_are_untextured(tmp_path, monkeypatch):
    materials = _materials(tmp_path, monkeypatch)

    gravel = materials[FLAT_ROOF_MATERIAL]["Stages"][0]
    assert gravel["baseColorMap"] == "levels/x/gravel_b.dds"
    assert gravel["normalMap"] == "levels/x/gravel_nm.dds" and gravel["roughnessMap"] == "levels/x/gravel_r.dds"
    for name in (ROOF_EDGE_MATERIAL, ROOF_TRIM_MATERIAL):
        stage = materials[name]["Stages"][0]
        assert "baseColorMap" not in stage and len(stage["baseColorFactor"]) == 4
        assert "roughnessFactor" in stage and "metallicFactor" in stage
    assert materials[ROOF_EDGE_MATERIAL]["Stages"][0]["metallicFactor"] > 0.5  # sheet metal
    assert materials[ROOF_TRIM_MATERIAL]["Stages"][0]["metallicFactor"] < 0.2  # wood
    MaterialManager.reset_instance()

