"""
Reads an exported level from disk into plain Python/numpy structures (no pyvista here).

Everything the viewer shows comes from the files BeamNG itself loads:
- main/MissionGroup/items.level.json and main/MissionGroup/PlayerDropPoints/items.level.json (JSONL, one item per line)
- the .ter file referenced by the TerrainBlock item (heights + layer map)
- main/materials.json, info.json (minimap placement), art/shapes/textures/aerial_photo.json (photo tile bounds)
- forest/forest.forest4.json (JSONL tree instances)

Coordinates are level-local meters, Z up. A rotationMatrix stores the images of the local +X/+Y/+Z axes in its ROWS
(see ItemManager._heading_rotation_matrix), so a local point p maps to ``p @ R + position``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import numpy as np

ITEM_FILES = (
    Path("main") / "MissionGroup" / "items.level.json",
    Path("main") / "MissionGroup" / "PlayerDropPoints" / "items.level.json",
)
HOLE = 255  # layer map value of a terrain hole


@dataclass
class LevelItem:
    """One object from an items.level.json file; `index` is unique across all item files of the level."""

    index: int
    name: str
    cls: str
    position: np.ndarray
    scale: np.ndarray
    rotation: Optional[np.ndarray]  # (3, 3), rows = images of local +X/+Y/+Z; None = identity
    raw: dict
    source: str  # item file the line came from (relative to the level dir)

    @property
    def material(self) -> Optional[str]:
        return self.raw.get("material")


@dataclass
class PhotoTile:
    name: str
    path: Path
    bounds: tuple  # (x_min, x_max, y_min, y_max), image row 0 = north


@dataclass
class MinimapInfo:
    path: Path
    x_min: float
    y_max: float
    size_x: float
    size_y: float  # image row 0 = north (y_max)


@dataclass
class LevelData:
    level_dir: Path
    items: List[LevelItem]
    materials: Dict[str, dict]
    minimap: Optional[MinimapInfo] = None
    photos: List[PhotoTile] = field(default_factory=list)
    by_class: Dict[str, List[LevelItem]] = field(default_factory=dict)

    def of(self, *classes: str) -> List[LevelItem]:
        """All items of the given classes, in file order."""
        out: List[LevelItem] = []
        for cls in classes:
            out.extend(self.by_class.get(cls, []))
        return sorted(out, key=lambda item: item.index)


@dataclass
class TerrainGrid:
    """Downsampled terrain for display plus the full-resolution heights for exact readouts."""

    x: np.ndarray  # (nx,) world x of the sampled columns
    y: np.ndarray  # (ny,) world y of the sampled rows (row 0 = south)
    z: np.ndarray  # (ny, nx) float32 heights in meters
    hole: np.ndarray  # (ny, nx) bool, True where the sample is a terrain hole
    origin: tuple  # (x, y) of full-res cell (0, 0)
    square_size: float
    z_base: float
    height_scale: float  # meters per u16 step (maxHeight / 65536)
    heights_u16: np.ndarray  # full-resolution heightmap (row 0 = south)
    layer_map: np.ndarray  # full-resolution layer map
    material_names: List[str]

    def height_at(self, x: float, y: float) -> Optional[float]:
        """Bilinear height from the full-resolution heightmap, None outside the grid or on a hole."""
        col = (x - self.origin[0]) / self.square_size
        row = (y - self.origin[1]) / self.square_size
        size_y, size_x = self.heights_u16.shape
        if not (0 <= col <= size_x - 1 and 0 <= row <= size_y - 1):
            return None
        if self.layer_map[int(round(row)), int(round(col))] == HOLE:
            return None
        c0, r0 = min(int(col), size_x - 2), min(int(row), size_y - 2)
        fx, fy = col - c0, row - r0
        h = self.heights_u16[r0 : r0 + 2, c0 : c0 + 2].astype(np.float64)
        u16 = (h[0, 0] * (1 - fx) + h[0, 1] * fx) * (1 - fy) + (h[1, 0] * (1 - fx) + h[1, 1] * fx) * fy
        return float(self.z_base + u16 * self.height_scale)

    def material_at(self, x: float, y: float) -> Optional[str]:
        col = int(round((x - self.origin[0]) / self.square_size))
        row = int(round((y - self.origin[1]) / self.square_size))
        size_y, size_x = self.layer_map.shape
        if not (0 <= col < size_x and 0 <= row < size_y):
            return None
        index = int(self.layer_map[row, col])
        if index == HOLE:
            return "(hole)"
        return self.material_names[index] if index < len(self.material_names) else str(index)


@dataclass
class ForestInstances:
    types: List[str]  # distinct tree types, index = type_idx
    type_idx: np.ndarray  # (n,) int
    positions: np.ndarray  # (n, 3) float32


# ---------------------------------------------------------------------------------------------------------------------
# Paths


def default_level_dir() -> Path:
    from world_to_beamng import config

    return Path(config.BEAMNG_DIR)


def resolve_asset(level_dir: Path, virtual_path: str) -> Optional[Path]:
    """
    Maps a BeamNG virtual path ("levels/<level>/art/..." or "/levels/<level>/...") to a file on disk.

    Only files of the user levels folder (the parent of `level_dir`) can be resolved; vanilla paths such as
    "/assets/..." or other levels live in BeamNG's content zips and give None.
    """
    if not virtual_path:
        return None
    parts = PurePosixPath(virtual_path.replace("\\", "/").lstrip("/")).parts
    if len(parts) < 3 or parts[0].lower() != "levels":
        return None
    candidate = Path(level_dir).parent.joinpath(*parts[1:])
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------------------------------------------------
# Items and level metadata


def read_jsonl(path: Path) -> List[dict]:
    """Objects of a JSONL file (one JSON object per line, blank lines ignored)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _vec3(value, default) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    except (TypeError, ValueError):
        pass
    return np.asarray(default, dtype=float)


def _rotation(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr[:9].reshape(3, 3) if arr.size >= 9 else None


def load_level(level_dir: Path) -> LevelData:
    """Items of both item files (global index in file order), materials and texture placement."""
    level_dir = Path(level_dir)
    items: List[LevelItem] = []
    for rel in ITEM_FILES:
        path = level_dir / rel
        if not path.exists():
            continue
        for raw in read_jsonl(path):
            scale = raw.get("scale", [1, 1, 1])
            if isinstance(scale, (int, float)):
                scale = [scale] * 3
            items.append(
                LevelItem(
                    index=len(items),
                    name=str(raw.get("name", "")),
                    cls=str(raw.get("class", "")),
                    position=_vec3(raw.get("position"), (0.0, 0.0, 0.0)),
                    scale=_vec3(scale, (1.0, 1.0, 1.0)),
                    rotation=_rotation(raw.get("rotationMatrix")),
                    raw=raw,
                    source=rel.as_posix(),
                )
            )

    by_class: Dict[str, List[LevelItem]] = {}
    for item in items:
        by_class.setdefault(item.cls, []).append(item)

    materials_path = level_dir / "main" / "materials.json"
    materials = json.loads(materials_path.read_text(encoding="utf-8")) if materials_path.exists() else {}

    return LevelData(
        level_dir=level_dir,
        items=items,
        materials=materials,
        minimap=_load_minimap(level_dir),
        photos=_load_photos(level_dir),
        by_class=by_class,
    )


def _load_minimap(level_dir: Path) -> Optional[MinimapInfo]:
    info_path = level_dir / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    entries = info.get("minimap") or []
    if not entries:
        return None
    entry = entries[0]
    path = level_dir / entry.get("file", "")
    if not path.is_file():
        return None
    size = entry.get("size") or info.get("size") or [0, 0]
    offset = entry.get("offset") or [0, 0]
    return MinimapInfo(path=path, x_min=float(offset[0]), y_max=float(offset[1]), size_x=float(size[0]), size_y=float(size[1]))


def _load_photos(level_dir: Path) -> List[PhotoTile]:
    textures = level_dir / "art" / "shapes" / "textures"
    meta = textures / "aerial_photo.json"
    if not meta.exists():
        return []
    photos = []
    for photo in json.loads(meta.read_text(encoding="utf-8")).get("photos", []):
        path = textures / f"{photo['name']}.png"
        if path.exists():
            photos.append(PhotoTile(name=photo["name"], path=path, bounds=tuple(float(v) for v in photo["bounds"])))
    return photos


# ---------------------------------------------------------------------------------------------------------------------
# Terrain


def _sample_indices(start: int, stop: int, step: int) -> np.ndarray:
    """start..stop (inclusive) in steps of `step`, always ending exactly at stop."""
    idx = np.arange(start, stop + 1, max(1, step))
    if idx[-1] != stop:
        idx = np.append(idx, stop)
    return idx


def load_terrain(level: LevelData, step: int = 4) -> Optional[TerrainGrid]:
    """
    Heights of the TerrainBlock's .ter file, cropped to the non-hole area (the export pads the grid toward
    east/north with holes) and sampled every `step` cells. Height = position.z + u16 * maxHeight / 65536.
    """
    from world_to_beamng.terrain.ter_writer import read_ter

    blocks = level.of("TerrainBlock")
    if not blocks:
        return None
    block = blocks[0]
    ter_path = resolve_asset(level.level_dir, block.raw.get("terrainFile", ""))
    if ter_path is None:
        return None

    heights_u16, layer_map, material_names = read_ter(ter_path)
    square = float(block.raw.get("squareSize", 1.0))
    height_scale = float(block.raw.get("maxHeight", 0.0)) / 65536.0
    z_base = float(block.position[2])
    origin = (float(block.position[0]), float(block.position[1]))

    solid = layer_map != HOLE
    if not solid.any():
        return None
    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    row_idx = _sample_indices(int(rows[0]), int(rows[-1]), step)
    col_idx = _sample_indices(int(cols[0]), int(cols[-1]), step)

    sub = heights_u16[np.ix_(row_idx, col_idx)]
    return TerrainGrid(
        x=origin[0] + col_idx * square,
        y=origin[1] + row_idx * square,
        z=(z_base + sub.astype(np.float32) * height_scale).astype(np.float32),
        hole=layer_map[np.ix_(row_idx, col_idx)] == HOLE,
        origin=origin,
        square_size=square,
        z_base=z_base,
        height_scale=height_scale,
        heights_u16=heights_u16,
        layer_map=layer_map,
        material_names=material_names,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Forest and debug network


def load_forest(level: LevelData) -> Optional[ForestInstances]:
    """Tree instances of the Forest item's data file (JSONL lines with "type" and "pos")."""
    forests = level.of("Forest")
    path = resolve_asset(level.level_dir, forests[0].raw.get("dataFile", "")) if forests else None
    if path is None:
        fallback = level.level_dir / "forest" / "forest.forest4.json"
        path = fallback if fallback.exists() else None
    if path is None:
        return None
    types: Dict[str, int] = {}
    type_idx, positions = [], []
    for obj in read_jsonl(path):
        pos = obj.get("pos")
        if not pos or len(pos) < 3:
            continue
        type_idx.append(types.setdefault(str(obj.get("type", "?")), len(types)))
        positions.append(pos[:3])
    if not positions:
        return None
    return ForestInstances(
        types=list(types), type_idx=np.asarray(type_idx, dtype=np.int32), positions=np.asarray(positions, dtype=np.float32)
    )


def load_debug_network(path: Optional[Path]) -> dict:
    """cache/debug_network.json ({"primitives": [...], "grid_colors": {...}}); empty dict if missing/invalid."""
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
