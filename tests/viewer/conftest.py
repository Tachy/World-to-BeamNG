"""Synthetic exported level for the level viewer tests (small .ter, a few items, materials, minimap, forest)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from PIL import Image

from world_to_beamng.managers.item_manager import ItemManager
from world_to_beamng.terrain.ter_writer import write_ter

LEVEL = "test_level"
TER_SIZE = 128
PADDING = 28  # east/north padding (holes), like the export's power-of-two padding
MAX_HEIGHT = 100.0
ORIGIN = (-50.0, -50.0, 1000.0)
ROAD_Z = ORIGIN[2] + 50 * 400 * MAX_HEIGHT / 65536  # terrain height at y = 0 (row 50, u16 = 50 * 400)


def _write_jsonl(path: Path, objects):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(o) for o in objects) + "\n", encoding="utf-8")


def make_level(root: Path) -> Path:
    level_dir = root / "levels" / LEVEL
    level_dir.mkdir(parents=True)

    # Terrain: height rises toward the north (row index), holes in the padded east/north strip
    rows = np.arange(TER_SIZE, dtype=np.float64)[:, None]
    heights = np.repeat(rows * 400.0, TER_SIZE, axis=1).clip(0, 65535).astype(np.uint16)
    layers = np.zeros((TER_SIZE, TER_SIZE), dtype=np.uint8)
    layers[TER_SIZE - PADDING :, :] = 255
    layers[:, TER_SIZE - PADDING :] = 255
    layers[10:12, 10:30] = 1
    write_ter(level_dir / f"{LEVEL}.ter", heights, layers, ["grass", "dirt"])

    rotation = ItemManager._heading_rotation_matrix(1.0, 0.0)  # facing east
    items = [
        {"name": "MissionGroup", "class": "SimGroup"},
        {"name": "theTerrain", "class": "TerrainBlock", "position": list(ORIGIN), "maxHeight": MAX_HEIGHT,
         "squareSize": 1.0, "terrainFile": f"/levels/{LEVEL}/{LEVEL}.ter"},
        {"name": "road_a", "class": "DecalRoad", "material": "asphalt_road_standard", "renderPriority": 12,
         "position": [0, 0, ROAD_Z], "nodes": [[0, 0, ROAD_Z, 6], [20, 0, ROAD_Z, 6], [40, 0, ROAD_Z, 6]]},
        {"name": "marking_road_a_edge", "class": "DecalRoad", "material": "line_edge_white", "renderPriority": 1,
         "drivability": -1, "position": [0, 2.8, ROAD_Z], "nodes": [[0, 2.8, ROAD_Z, 0.15], [40, 2.8, ROAD_Z, 0.15]]},
        {"name": "river_0", "class": "River", "position": [0, 20, 1000],
         "nodes": [[0, 20, 1000, 2.5, 1, 0, 0, 1], [10, 25, 1000, 2.5, 1, 0, 0, 1]]},
        {"name": "pond_0", "class": "WaterBlock", "position": [20, 20, 1005], "scale": [10, 4, 2]},
        {"name": "tunnel_zone_1", "class": "Zone", "position": [30, 30, 1010], "scale": [20, 6, 8],
         "rotationMatrix": rotation},
        {"name": "the_forest", "class": "Forest", "dataFile": f"levels/{LEVEL}/forest/forest.forest4.json"},
        {"name": "horizon", "class": "TSStatic", "position": [0, 0, 0], "shapeName": "/assets/vanilla/missing.dae"},
    ]
    _write_jsonl(level_dir / "main" / "MissionGroup" / "items.level.json", items)
    _write_jsonl(
        level_dir / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json",
        [{"name": "spawn_east", "class": "SpawnSphere", "position": [5, 5, 1010], "rotationMatrix": rotation}],
    )
    _write_jsonl(
        level_dir / "forest" / "forest.forest4.json",
        [{"type": "oak", "pos": [1, 2, 1003]}, {"type": "pine", "pos": [3, 4, 1005]}, {"type": "oak", "pos": [5, 6, 1007]}],
    )
    (level_dir / "main" / "materials.json").write_text(
        json.dumps({"bridge_concrete": {"class": "Material", "Stages": [{"diffuseColor": [0.5, 0.5, 0.5, 1]}]}}),
        encoding="utf-8",
    )

    # Minimap: 4x4 px, top half (north, y > ORIGIN.y + 64) red, bottom half grey; covers the whole padded grid
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    img[:2] = [255, 0, 0]
    (level_dir / "minimap").mkdir()
    Image.fromarray(img).save(level_dir / "minimap" / "terrain.png")
    (level_dir / "info.json").write_text(
        json.dumps({"minimap": [{"file": "minimap/terrain.png", "size": [TER_SIZE, TER_SIZE],
                                 "offset": [ORIGIN[0], ORIGIN[1] + TER_SIZE]}]}),
        encoding="utf-8",
    )
    return level_dir


@pytest.fixture
def level_dir(tmp_path):
    return make_level(tmp_path)
