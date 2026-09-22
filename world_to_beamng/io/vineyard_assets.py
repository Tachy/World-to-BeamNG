"""
Reben-Assets für Weinberge: kopiert die Shapes (grape_vine, grape_vine_group) samt
Materialien aus BeamNGs italy-Level in den eigenen Level und registriert sie als
Forest-Items in art/forest/managedItemData.json.

Läuft bei jedem Export (idempotent), damit ein erneutes Erzeugen der Forest-Assets
(tools/generate_forest_assets.py überschreibt managedItemData.json) die Reben nicht
verliert.
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
# leaves_strong (Laub der Reben) liegt in den Baum-Materialien der Original-Levels
LEAVES_MATERIALS_LEVEL = "east_coast_usa"
LEAVES_MATERIALS = "levels/east_coast_usa/art/shapes/trees/main.materials.json"

ITEM_NAMES = ("grape_vine", "grape_vine_group")
# Pflicht: .dae; optional: kompilierte Fassung und Imposter (LOD in der Ferne)
SHAPE_SUFFIXES = (".dae", ".cdae", ".dae.imposter.dds", ".dae.imposter_normals.dds")
NEEDED_MATERIALS = {"grape": ITALY_MATERIALS, "olive_trunk": ITALY_MATERIALS, "leaves_strong": LEAVES_MATERIALS}

# Werte wie in BeamNGs italy-Level (managedItemData.json)
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
    """Namen aller im Level bereits definierten Materialien (außer in `exclude`)."""
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
        level_dir: Level-Verzeichnis (config.BEAMNG_DIR)
        install_dir: BeamNG-Installationsverzeichnis (enthält content/levels/*.zip)
        level_name: Level-Name für die Pfade in managedItemData.json

    Returns:
        {"items": [...], "materials": [...], "copied": n}

    Raises:
        FileNotFoundError: wenn das italy-Level nicht gefunden wird
    """
    levels_dir = Path(install_dir) / "content" / "levels"
    italy_zip = levels_dir / "italy.zip"
    if not italy_zip.is_file():
        raise FileNotFoundError(f"BeamNG-Level 'italy' (Quelle der Reben-Shapes) nicht gefunden: {italy_zip}")

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
                        raise FileNotFoundError(f"{member} fehlt in {italy_zip}")
                    continue
                if not target.exists():
                    target.write_bytes(z.read(member))
                    copied += 1

    # Nur Materialien, die der Level noch nicht kennt (doppelte Definitionen vermeiden)
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

    # Forest-Items in managedItemData.json ergänzen (bestehende Einträge bleiben)
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
    # Nur schreiben, wenn sich tatsaechlich etwas aendert - diese Funktion laeuft bei JEDEM Export
    # (siehe Docstring oben); ein unbedingtes Schreiben wuerde den Zeitstempel jedes Mal aendern,
    # obwohl _stable_id() deterministisch ist und der Inhalt meist identisch bleibt. Andere Caches
    # (siehe workflow/forest_workflow.py::_forest_cache_key()) nutzen genau diesen Zeitstempel, um
    # zu erkennen, ob sich die verfuegbaren Baumarten geaendert haben - ein Cache-Treffer wuerde
    # sonst nie greifen.
    if new_text != existing_text:
        item_path.write_text(new_text, encoding="utf-8")

    logger.info(f"  [OK] Reben-Assets: {copied} Datei(en) kopiert, Materialien: {sorted(materials) or 'bereits vorhanden'}")
    return {"items": list(ITEM_NAMES), "materials": sorted(materials), "copied": copied}
