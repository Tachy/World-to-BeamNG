"""
Assets taken over from the local BeamNG installation into the exported level, ensured on every export
(idempotent - nothing is copied or rewritten when it is already up to date):

- tree shapes + materials from east_coast_usa.zip into art/shapes/trees, registered as forest items in
  art/forest/managedItemData.json (ensure_tree_assets)
- stock textures that data/osm_to_beamng.json references under levels/<level>/art/shapes/assets/materials/
  (road decals, plaster, roof tiles, terrain detail textures) from content/assets/materials/*.zip
  (ensure_shared_textures)

The vine shapes for vineyards are handled the same way in io/vineyard_assets.py; it runs after
ensure_tree_assets() and adds its items to the same managedItemData.json.
"""

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

TREE_SOURCE_LEVEL = "east_coast_usa"
TREE_ZIP_PREFIX = f"levels/{TREE_SOURCE_LEVEL}/art/shapes/trees/"
TREE_SIGNATURE_FILENAME = ".tree_assets.json"
TREE_SIGNATURE_VERSION = 1

# Mapping from filename patterns to tree species (German names kept: they match German file names)
TREE_NAME_PATTERNS = {
    r"oak|eiche": "oak",
    r"pedunculate|sessile|quercus": "oak",
    r"beech|buche|fagus": "beech",
    r"birch|birke|betula": "birch",
    r"aspen|espe|tremuloides": "aspen",
    r"spruce|fichte|picea": "spruce",
    r"pine|scots|kiefer|sylvestris": "scots_pine",
    r"fir|tanne|abies": "fir",
    r"larch|lärche|larix": "larch",
    r"maple|ahorn": "maple",
    r"ash|esche|fraxinus": "ash",
    r"elm|ulme": "elm",
    r"poplar|pappel": "poplar",
    r"alder|erle|alnus": "alder",
    r"willow|weide|salix": "willow",
    r"rowan|eberesche|sorbus": "rowan",
    r"hazel|hasel|corylus": "hazel",
    r"elder|holunder|sambucus": "elder",
    r"cork|kork": "cork_oak",
    r"holm|steineiche": "holm_oak",
    r"olive|oliv": "olive",
}

# east_coast_usa's bonus folder "ECA_coast_bush" contains, besides generic filler bushes (generibush*), also
# Mediterranean species (cork oak, stone pine) that are out of place in a central European forest - see research
# 2026-09-18: cork_oak_bush_* and maritime_pine_bush had wrongly ended up in the forest mixes via the plain filename
# patterns. cork_oak_bush_* additionally references the broken "holm_oak_trunk" texture (no longer in the current
# east_coast_usa.zip) - the exclusion fixes both at once. generibush/generibush_small stay (climate-neutral fillers).
EXCLUDED_NON_NATIVE_SPECIES = ("cork_oak_bush_large", "cork_oak_bush_medium", "maritime_pine_bush")


def extract_tree_name_from_filename(filename: str) -> str:
    """Tree species of a DAE file name (e.g. "tree_beech_large_a.dae" -> "beech")."""
    name = Path(filename).stem
    name_lower = name.lower()
    for pattern, tree_name in TREE_NAME_PATTERNS.items():
        if re.search(pattern, name_lower):
            return tree_name
    cleaned = re.sub(r"[_\-\d]", " ", name).strip()
    if not cleaned or cleaned.lower() in ["tree", "model", "asset"]:
        return "tree"
    return cleaned.lower()


def _zip_signature(zip_path: Path) -> dict:
    stat = zip_path.stat()
    return {"zip": zip_path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _extract_trees(zip_path: Path, dest_dir: Path, level_name: str) -> List[str]:
    """
    Extracts the tree folder of the INSTALLED east_coast_usa.zip (not the old unpacked user folder, whose
    materials/textures are outdated - research 2026-09-18) and rewrites its "levels/east_coast_usa/..."
    references in .dae and materials.json files to the own level. .cdae files (compiled cache that embeds the old
    path and is not safely text-patchable) are skipped; BeamNG compiles them from the .dae on the next load.

    Returns:
        written files relative to dest_dir (posix)
    """
    old_ref = f"levels/{TREE_SOURCE_LEVEL}/art/shapes/trees"
    new_ref = f"levels/{level_name}/art/shapes/trees"
    written: List[str] = []
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            if entry.endswith("/") or not entry.startswith(TREE_ZIP_PREFIX) or entry.lower().endswith(".cdae"):
                continue
            filename = entry.rsplit("/", 1)[-1]
            if any(filename.lower().startswith(species) for species in EXCLUDED_NON_NATIVE_SPECIES):
                continue
            rel = entry[len(TREE_ZIP_PREFIX) :]
            data = z.read(entry)
            if filename.lower().endswith((".dae", "materials.json")):
                data = data.decode("utf-8").replace(old_ref, new_ref).encode("utf-8")
            out_path = dest_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            written.append(rel)
    for stale in dest_dir.rglob("*.cdae"):  # from older copies that still contained them
        stale.unlink()
    return sorted(written)


def tree_items(shape_dir: Path, level_dir: Path, level_name: str) -> Dict[str, dict]:
    """managedItemData entries for all tree DAEs below shape_dir (which must lie inside level_dir)."""
    items = {}
    for dae_file in sorted(Path(shape_dir).rglob("*.dae")):
        item_key = dae_file.stem
        rel = dae_file.relative_to(level_dir).as_posix()
        radius = 2.0 if extract_tree_name_from_filename(dae_file.name) in ("cork_oak", "holm_oak") else 1.5
        items[item_key] = {
            "name": item_key,
            "class": "ForestItemData",
            "internalName": item_key,
            "shapeFile": f"levels/{level_name}/{rel}",
            "collidable": True,
            "radius": radius,
        }
    return items


def ensure_tree_assets(level_dir: Path, install_dir: Path, level_name: str) -> Dict:
    """
    Makes sure the level has the tree shapes and registers them in art/forest/managedItemData.json.

    The shapes are extracted again only when the source zip changed (size/mtime), the signature is missing or a
    listed file is gone. managedItemData.json is rewritten only when its content changes (its mtime is part of the
    forest cache key, see ForestWorkflow), and entries that are not trees (vines) are kept.

    Returns:
        {"items": number of tree items, "extracted": files extracted in this run (0 = up to date)}

    Raises:
        FileNotFoundError: if east_coast_usa.zip is not part of the installation
    """
    level_dir = Path(level_dir)
    zip_path = Path(install_dir) / "content" / "levels" / f"{TREE_SOURCE_LEVEL}.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"BeamNG level '{TREE_SOURCE_LEVEL}' (source of the tree shapes) not found: {zip_path}")

    shape_dir = level_dir / "art" / "shapes" / "trees"
    signature_path = level_dir / "art" / "forest" / TREE_SIGNATURE_FILENAME
    expected = {"version": TREE_SIGNATURE_VERSION, "level": level_name, "excluded": list(EXCLUDED_NON_NATIVE_SPECIES), **_zip_signature(zip_path)}
    try:
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}

    extracted = 0
    files = previous.get("files", [])
    up_to_date = {k: previous.get(k) for k in expected} == expected and files and all((shape_dir / f).exists() for f in files)
    if not up_to_date:
        files = _extract_trees(zip_path, shape_dir, level_name)
        extracted = len(files)
        _write_if_changed(signature_path, json.dumps({**expected, "files": files}, indent=2))

    items = tree_items(shape_dir, level_dir, level_name)
    item_path = level_dir / "art" / "forest" / "managedItemData.json"
    try:
        existing = json.loads(item_path.read_text(encoding="utf-8")) if item_path.exists() else {}
    except ValueError:
        existing = {}
    tree_prefix = f"levels/{level_name}/art/shapes/trees/"
    # Keep the existing key order (vines added by ensure_vineyard_assets stay where they are), drop trees that no
    # longer exist, update/append the current ones - same format as vineyard_assets, so neither rewrites the other
    merged = {k: v for k, v in existing.items() if k in items or not str(v.get("shapeFile", "")).startswith(tree_prefix)}
    merged.update(items)
    _write_if_changed(item_path, json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    return {"items": len(items), "extracted": extracted}


# ---------------------------------------------------------------------------------------------------------------------


def find_texture_paths(obj, level_prefix: str, found: Set[str]) -> None:
    """Recursively collects all string values that start with level_prefix."""
    if isinstance(obj, dict):
        for value in obj.values():
            find_texture_paths(value, level_prefix, found)
    elif isinstance(obj, list):
        for value in obj:
            find_texture_paths(value, level_prefix, found)
    elif isinstance(obj, str) and obj.startswith(level_prefix):
        found.add(obj)


def _zip_index(materials_dir: Path) -> Dict[str, Path]:
    """{path inside the zip: zip file} over all content/assets/materials/*.zip."""
    index: Dict[str, Path] = {}
    for zip_path in sorted(materials_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as z:
                for name in z.namelist():
                    if not name.endswith("/"):
                        index[name] = zip_path
        except zipfile.BadZipFile:
            continue
    return index


def ensure_shared_textures(level_dir: Path, install_dir: Path, level_name: str, mapping_path: Path) -> Dict:
    """
    Copies the stock textures that the OSM mapping references as level files
    ("levels/<level>/art/shapes/assets/materials/...") from BeamNG's content/assets/materials/*.zip, if they are
    missing in the level. Without them BeamNG shows "no Texture" on roads and roofs.

    Returns:
        {"referenced": n, "copied": n, "missing": [level paths not found in the content zips]}
    """
    level_dir = Path(level_dir)
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    level_prefix = f"levels/{level_name}/art/shapes/assets/materials/"
    shapes_prefix = f"levels/{level_name}/art/shapes/"
    found: Set[str] = set()
    find_texture_paths(mapping, level_prefix, found)

    todo = [p for p in sorted(found) if not (level_dir / "art" / "shapes" / p[len(shapes_prefix) :]).exists()]
    result = {"referenced": len(found), "copied": 0, "missing": []}
    if not todo:
        return result

    materials_dir = Path(install_dir) / "content" / "assets" / "materials"
    index = _zip_index(materials_dir) if materials_dir.is_dir() else {}
    by_zip: Dict[Path, List[str]] = {}
    for level_path in todo:
        virtual = level_path[len(shapes_prefix) :]  # e.g. "assets/materials/decalroad/..."
        zip_path = index.get(virtual)
        if zip_path is None:
            result["missing"].append(level_path)
        else:
            by_zip.setdefault(zip_path, []).append(virtual)
    for zip_path, members in by_zip.items():
        with zipfile.ZipFile(zip_path) as z:
            for virtual in members:
                dest = level_dir / "art" / "shapes" / virtual
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(virtual))
                result["copied"] += 1
    return result


def install_dir_or_none() -> Optional[Path]:
    """BeamNG installation directory, or None (with a warning) if it cannot be determined."""
    from .beamng_install import get_beamng_install_dir

    try:
        return get_beamng_install_dir()
    except Exception as exc:
        logger.warning(f"BeamNG installation not found - stock assets cannot be copied: {exc}")
        return None
