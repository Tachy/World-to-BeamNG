"""
Generate Forest Assets: forestItemData.json + forest_type_templates

Kombiniertes Script das:
1. DAE-Dateien scannt und forestItemData.json generiert
2. Aus forestItemData.json Waldtypen und Mappings generiert
3. osm_to_beamng.json aktualisiert
"""

from pathlib import Path
import json
import re
from collections import defaultdict
import sys

# Importiere config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config


# Mapping von Dateinamen-Patterns zu Baumarten
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


def extract_tree_name_from_filename(filename: str) -> str:
    """Extrahiere Baumnamen aus DAE-Dateiname."""
    name = Path(filename).stem
    name_lower = name.lower()

    for pattern, tree_name in TREE_NAME_PATTERNS.items():
        if re.search(pattern, name_lower):
            return tree_name

    cleaned = re.sub(r"[_\-\d]", " ", name).strip()
    if not cleaned or cleaned.lower() in ["tree", "model", "asset"]:
        return "tree"
    return cleaned.lower()


def scan_dae_files(dir_path: str, beamng_root: str) -> dict:
    """
    Scanne DAE-Dateien und generiere forestItemData.

    Returns:
        {tree_key: {name, class, shapeFile, collidable, radius}}
    """
    dir_path_obj = Path(dir_path)
    beamng_root_obj = Path(beamng_root)

    if not dir_path_obj.is_dir():
        print(f"[ERROR] Verzeichnis nicht gefunden: {dir_path}")
        return None

    dae_files = sorted(dir_path_obj.rglob("*.dae"))

    print(f"[INFO] Gefundene DAE-Dateien: {len(dae_files)}")
    if not dae_files:
        print("[ERROR] Keine DAE-Dateien gefunden!")
        return None

    forest_item_data = {}
    tree_type_counts = defaultdict(int)

    for idx, dae_file in enumerate(dae_files, 1):
        tree_type = extract_tree_name_from_filename(dae_file.name)
        item_key = dae_file.stem

        try:
            relative_dae = dae_file.relative_to(beamng_root_obj)
        except ValueError:
            relative_dae = dae_file

        full_path = str(relative_dae).replace("\\", "/")
        if "levels/" in full_path:
            shape_file_path = "levels/" + full_path.split("levels/")[1]
        else:
            shape_file_path = full_path

        radius = 2.0 if tree_type in ["cork_oak", "holm_oak"] else 1.5

        forest_item_data[item_key] = {
            "name": item_key,
            "class": "TSForestItemData",
            "shapeFile": shape_file_path,
            "collidable": True,
            "radius": radius,
        }

        tree_type_counts[tree_type] += 1
        if idx <= 10 or idx % 10 == 0:
            print(f"[{idx:3d}] {item_key:40s} → {tree_type:15s}")

    print(f"\n[INFO] Baum-Typen Übersicht:")
    for tree_type in sorted(tree_type_counts.keys()):
        print(f"       {tree_type:30s} : {tree_type_counts[tree_type]:3d}x")

    return forest_item_data


def categorize_trees(forest_item_data: dict) -> dict:
    """Kategorisiere Bäume nach Typ."""
    trees_by_type = defaultdict(list)
    for tree_key, tree_info in forest_item_data.items():
        tree_type = tree_info.get("name", "unknown")
        trees_by_type[tree_type].append(tree_key)
    return dict(trees_by_type)


def create_tree_distribution(preferred_trees: list) -> dict:
    """Erstelle tree_distribution Dictionary mit gleichmäßiger Verteilung."""
    if not preferred_trees:
        return {}
    probability = 1.0 / len(preferred_trees)
    return {tree: probability for tree in preferred_trees}


def generate_forest_types(trees_by_type: dict) -> dict:
    """Generiere sinnvolle Waldtypen für deutsche Wälder."""
    forest_types = {}
    all_tree_keys = []
    for tree_type, keys in trees_by_type.items():
        all_tree_keys.extend(keys)

    # 1. German Deciduous Dense
    deciduous_trees = [t for t in all_tree_keys if "forest" in t or "large" in t][:10]
    if deciduous_trees:
        forest_types["german_deciduous_dense"] = {
            "tree_density": 1.0,
            "average_height": [20.0, 30.0],
            "underground_material": "forest_floor",
            "lod_distance": 250.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(deciduous_trees),
            "comment": "Dichter Laubwald - Buchen und Eichen (klassischer deutscher Wald)",
        }

    # 2. German Mixed Forest
    mixed_trees = [t for t in all_tree_keys if "forest" in t or "group" in t][:10]
    if mixed_trees:
        forest_types["german_mixed_forest"] = {
            "tree_density": 0.85,
            "average_height": [18.0, 26.0],
            "underground_material": "forest_floor",
            "lod_distance": 220.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(mixed_trees),
            "comment": "Mischwald - Buchen, Eichen und Espen (vielfältiger Bestand)",
        }

    # 3. German Sparse Deciduous
    sparse_trees = [t for t in all_tree_keys if "bush" in t or ("small" in t and "forest" not in t)][:10]
    if sparse_trees:
        forest_types["german_sparse_deciduous"] = {
            "tree_density": 0.4,
            "average_height": [15.0, 23.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(sparse_trees),
            "comment": "Lichter Laubwald - überwiegend Busch- und kleinere Bäume",
        }

    # 4. Orchard Area
    orchard_trees = [t for t in all_tree_keys if "small" in t or "sml" in t][:10]
    if orchard_trees:
        forest_types["orchard_area"] = {
            "tree_density": 0.3,
            "average_height": [8.0, 15.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(orchard_trees),
            "comment": "Obstplantage - niedrige und kleine Bäume",
        }

    # 5. Hedgerow
    hedge_trees = [t for t in all_tree_keys if "wall" in t or ("small" in t and "bush" not in t)][:6]
    if hedge_trees:
        forest_types["hedgerow"] = {
            "tree_density": 0.2,
            "average_height": [10.0, 18.0],
            "underground_material": "grassland",
            "lod_distance": 120.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(hedge_trees),
            "comment": "Hecke/Feldgehölz - dünne, lineare Bestände",
        }

    # 6. Dead Forest
    dead_trees = [t for t in all_tree_keys if "dead" in t]
    if dead_trees:
        forest_types["dead_forest"] = {
            "tree_density": 0.3,
            "average_height": [15.0, 25.0],
            "underground_material": "forest_floor",
            "lod_distance": 200.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(dead_trees),
            "comment": "Totholz/Verfallender Wald - dürre, tote Bäume",
        }

    return forest_types


def generate_forest_mappings(forest_types: dict) -> dict:
    """Generiere forest_mappings basierend auf verfügbaren Waldtypen."""
    if not forest_types:
        return {
            "landuse": {"forest": "generic_forest", "wood": "generic_forest", "orchard": "generic_forest"},
            "natural": {
                "wood": "generic_forest",
                "forest": "generic_forest",
                "scrub": "generic_forest",
                "heath": "generic_forest",
                "tree_row": "generic_forest",
                "wetland": "generic_forest",
            },
            "leisure": {"nature_reserve": "generic_forest", "park": "generic_forest"},
            "tag_overrides": {
                "trees=conifer": "generic_forest",
                "trees=broadleaf": "generic_forest",
                "trees=mixed": "generic_forest",
                "leaf_type=needleleaf": "generic_forest",
                "leaf_type=broadleaved": "generic_forest",
                "leaf_type=mixed": "generic_forest",
            },
        }

    default_forest = "german_mixed_forest" if "german_mixed_forest" in forest_types else list(forest_types.keys())[0]

    return {
        "landuse": {
            "forest": default_forest,
            "wood": default_forest,
            "orchard": "orchard_area" if "orchard_area" in forest_types else default_forest,
        },
        "natural": {
            "wood": default_forest,
            "forest": default_forest,
            "scrub": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "heath": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "tree_row": "hedgerow" if "hedgerow" in forest_types else default_forest,
            "wetland": default_forest,
        },
        "leisure": {
            "nature_reserve": default_forest,
            "park": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
        },
        "tag_overrides": {
            "trees=conifer": default_forest,
            "trees=broadleaf": "german_deciduous_dense" if "german_deciduous_dense" in forest_types else default_forest,
            "trees=mixed": default_forest,
            "leaf_type=needleleaf": default_forest,
            "leaf_type=broadleaved": (
                "german_deciduous_dense" if "german_deciduous_dense" in forest_types else default_forest
            ),
            "leaf_type=mixed": default_forest,
        },
    }


def main():
    """Hauptfunktion: Generiere forestItemData.json und Waldtypen."""
    print("=" * 80)
    print("[START] Generiere Forest Assets (forestItemData + forest_type_templates)")
    print("=" * 80)

    # ===== PHASE 1: Scan DAE-Dateien =====
    print("\n[PHASE 1] Scanne DAE-Dateien und generiere forestItemData.json")
    print("-" * 80)

    search_dir = Path(
        r"C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\east_coast_usa\art\shapes\trees"
    )

    if not search_dir.is_dir():
        print(f"[ERROR] Suchverzeichnis nicht gefunden: {search_dir}")
        return

    forest_item_data = scan_dae_files(str(search_dir), str(config.BEAMNG_DIR))

    if not forest_item_data:
        print("[ERROR] Keine Forest-Items generiert")
        return

    # Speichere forestItemData.json
    output_dir = config.BEAMNG_DIR / "main"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "forestItemData.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(forest_item_data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] forestItemData.json erstellt: {output_file}")
    print(f"       {len(forest_item_data)} Tree-Items")

    # ===== PHASE 2: Generiere Waldtypen =====
    print("\n[PHASE 2] Generiere Waldtypen und Mappings")
    print("-" * 80)

    trees_by_type = categorize_trees(forest_item_data)

    print("\n[INFO] Baum-Kategorisierung:")
    for tree_type, trees in sorted(trees_by_type.items()):
        print(f"       {tree_type:20s} : {len(trees):2d}x")

    forest_types = generate_forest_types(trees_by_type)

    print(f"\n[INFO] Generierte Waldtypen:")
    for forest_type in sorted(forest_types.keys()):
        tree_count = len(forest_types[forest_type].get("preferred_trees", {}))
        print(f"       {forest_type:30s} : {tree_count:3d} trees")

    forest_mappings = generate_forest_mappings(forest_types)

    # ===== PHASE 3: Update osm_to_beamng.json =====
    print(f"\n[PHASE 3] Aktualisiere osm_to_beamng.json")
    print("-" * 80)

    config_path = Path("data/osm_to_beamng.json")
    with open(config_path, "r", encoding="utf-8") as f:
        osm_config = json.load(f)

    osm_config["forest_type_templates"] = forest_types
    osm_config["forest_mappings"] = forest_mappings

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(osm_config, f, indent=4, ensure_ascii=False)

    print(f"[DONE] osm_to_beamng.json aktualisiert")
    print(f"       - {len(forest_types)} forest_type_templates")
    print(f"       - forest_mappings aktualisiert")

    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("[✓] ERFOLGREICH ABGESCHLOSSEN")
    print("=" * 80)
    print(f"forestItemData.json: {len(forest_item_data)} Tree-Items")
    print(f"Forest-Typen:        {len(forest_types)}")
    print(f"  - german_deciduous_dense")
    print(f"  - german_mixed_forest")
    print(f"  - german_sparse_deciduous")
    print(f"  - orchard_area")
    print(f"  - hedgerow")
    print(f"  - dead_forest")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
