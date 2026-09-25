"""
Vine assets for vineyards: copies the shapes (grape_vine, grape_vine_group) together with
their materials from BeamNG's italy level into our own level and registers them as
forest items in art/forest/managedItemData.json.

Runs on every export (idempotent), right after io/beamng_assets.py::ensure_tree_assets(), which writes
the tree items of the same managedItemData.json and keeps the vine entries.
"""

import json
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Set

from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

ITALY_SHAPE_DIR = "levels/italy/art/shapes/trees/trees_italy/"
ITALY_MATERIALS = ITALY_SHAPE_DIR + "main.materials.json"
# leaves_strong (foliage of the vines) is located in the tree materials of the original levels
LEAVES_MATERIALS_LEVEL = "east_coast_usa"
LEAVES_MATERIALS = "levels/east_coast_usa/art/shapes/trees/main.materials.json"

ITEM_NAMES = ("grape_vine", "grape_vine_group")
# Required: .dae; optional: compiled version and imposters (LOD at a distance)
SHAPE_SUFFIXES = (".dae", ".cdae", ".dae.imposter.dds", ".dae.imposter_normals.dds")
NEEDED_MATERIALS = {"grape": ITALY_MATERIALS, "olive_trunk": ITALY_MATERIALS, "leaves_strong": LEAVES_MATERIALS}

# Values as in BeamNG's italy level (managedItemData.json)
_ITEM_DEFAULTS = {
    "class": "TSForestItemData",
    "annotation": "NATURE",
    "branchAmp": 0.1,
    "dampingCoefficient": 3,
    "detailAmp": 5,
    "detailFreq": 0.07,
    "tightnessCoefficient": 1,
    "trunkBendScale": 0.05,
    "windScale": 0.5,
}


def _stable_id(level_name: str, name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"world_to_beamng/{level_name}/{name}"))


def _defined_material_names(level_dir: Path, exclude: Path) -> Set[str]:
    """Names of all materials already defined in the level (except in `exclude`)."""
    names: Set[str] = set()
    for path in (level_dir / "art").rglob("*.materials.json") if (level_dir / "art").is_dir() else []:
        if path == exclude:
            continue
        try:
            names.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return names


def _read_zip_materials(zip_path: Path, member: str) -> Dict:
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read(member).decode("utf-8", "ignore"))


def ensure_vineyard_assets(level_dir: Path, install_dir: Path, level_name: str = "world_to_beamng") -> Dict:
    """
    Args:
        level_dir: Level directory (config.BEAMNG_DIR)
        install_dir: BeamNG installation directory (contains content/levels/*.zip)
        level_name: Level name for the paths in managedItemData.json

    Returns:
        {"items": [...], "materials": [...], "copied": n}

    Raises:
        FileNotFoundError: if the italy level is not found
    """
    levels_dir = Path(install_dir) / "content" / "levels"
    italy_zip = levels_dir / "italy.zip"
    if not italy_zip.is_file():
        raise FileNotFoundError(f"BeamNG level 'italy' (source of the vine shapes) not found: {italy_zip}")

    shape_dir = Path(level_dir) / "art" / "shapes" / "vineyard"
    shape_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    with zipfile.ZipFile(italy_zip) as z:
        members = set(z.namelist())
        for stem in ITEM_NAMES:
            for suffix in SHAPE_SUFFIXES:
                member = f"{ITALY_SHAPE_DIR}{stem}{suffix}"
                target = shape_dir / f"{stem}{suffix}"
                if member not in members:
                    if suffix == ".dae":
                        raise FileNotFoundError(f"{member} missing in {italy_zip}")
                    continue
                if not target.exists():
                    target.write_bytes(z.read(member))
                    copied += 1

    # Only materials the level does not know yet (avoid duplicate definitions)
    materials_path = shape_dir / "main.materials.json"
    already_defined = _defined_material_names(Path(level_dir), exclude=materials_path)
    zip_by_member = {ITALY_MATERIALS: italy_zip, LEAVES_MATERIALS: levels_dir / f"{LEAVES_MATERIALS_LEVEL}.zip"}
    sources: Dict[str, Dict] = {}
    materials: Dict[str, Dict] = {}
    for name, member in NEEDED_MATERIALS.items():
        if name in already_defined:
            continue
        if member not in sources:
            sources[member] = _read_zip_materials(zip_by_member[member], member)
        entry = dict(sources[member][name])
        entry["persistentId"] = _stable_id(level_name, f"material/{name}")
        materials[name] = entry
    if materials:
        materials_path.write_text(json.dumps(materials, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif materials_path.exists():
        materials_path.unlink()

    # Add forest items to managedItemData.json (existing entries are kept)
    item_path = Path(level_dir) / "art" / "forest" / "managedItemData.json"
    item_path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = item_path.read_text(encoding="utf-8") if item_path.exists() else None
    data = json.loads(existing_text) if existing_text else {}
    for stem in ITEM_NAMES:
        data[stem] = {
            "name": stem,
            "internalName": stem,
            "persistentId": _stable_id(level_name, f"item/{stem}"),
            "shapeFile": f"levels/{level_name}/art/shapes/vineyard/{stem}.dae",
            **_ITEM_DEFAULTS,
        }
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    # Only write if something actually changes - this function runs on EVERY export
    # (see docstring above); an unconditional write would change the timestamp every time,
    # even though _stable_id() is deterministic and the content mostly stays identical. Other caches
    # (see workflow/forest_workflow.py::_forest_cache_key()) use exactly this timestamp to
    # detect whether the available tree species have changed - a cache hit would
    # otherwise never occur.
    if new_text != existing_text:
        item_path.write_text(new_text, encoding="utf-8")

    logger.info(f"  [OK] Vine assets: {copied} file(s) copied, materials: {sorted(materials) or 'already present'}")
    return {"items": list(ITEM_NAMES), "materials": sorted(materials), "copied": copied}
