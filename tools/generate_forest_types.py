"""
Generate forest_type_templates from forestItemData.json and update osm_to_beamng.json.

Liest forestItemData.json und generiert sinnvolle Waldtypen für deutsche Wälder.
Schreibt die Waldtypen in osm_to_beamng.json unter forest_type_templates.
"""

from pathlib import Path
import json
import sys
from collections import defaultdict

# Importiere config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config


def load_forest_item_data(forest_data_path: Path) -> dict:
    """
    Lade forestItemData.json.

    Args:
        forest_data_path: Pfad zu forestItemData.json

    Returns:
        Dict mit {tree_key: {name, shapeFile, ...}}
    """
    if not forest_data_path.exists():
        print(f"[ERROR] forestItemData.json nicht gefunden: {forest_data_path}")
        return {}

    with open(forest_data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_trees(forest_item_data: dict) -> dict:
    """
    Kategorisiere Bäume nach Typ.

    Args:
        forest_item_data: forestItemData.json als Dict

    Returns:
        {tree_type: [tree_keys]}
    """
    trees_by_type = defaultdict(list)

    for tree_key, tree_info in forest_item_data.items():
        # Die "name" ist der Baumtyp (oak, beech, aspen, etc.)
        tree_type = tree_info.get("name", "unknown")
        trees_by_type[tree_type].append(tree_key)

    return dict(trees_by_type)


def _create_tree_distribution(preferred_trees: list) -> dict:
    """
    Erstelle tree_distribution Dictionary mit gleichmäßiger Verteilung.

    Args:
        preferred_trees: Liste von Baumnamen

    Returns:
        {tree_name: probability}
    """
    if not preferred_trees:
        return {}

    probability = 1.0 / len(preferred_trees)
    return {tree: probability for tree in preferred_trees}


def generate_forest_types(trees_by_type: dict) -> dict:
    """
    Generiere sinnvolle Waldtypen für deutsche Wälder.

    Args:
        trees_by_type: {tree_type: [tree_keys]}

    Returns:
        {forest_type_name: {forest_type_definition}}
    """
    forest_types = {}

    # Verfügbare Baumtypen
    available_trees = set(trees_by_type.keys())

    print(f"\n[INFO] Verfügbare Baumtypen: {sorted(available_trees)}")

    # Fallback: Wenn ALLE Bäume vorhanden sind (z.B. oak, beech, aspen)
    all_tree_keys = []
    for tree_type, keys in trees_by_type.items():
        all_tree_keys.extend(keys)

    # 1. German Deciduous Dense (Dichter Laubwald)
    deciduous_trees = [t for t in all_tree_keys if "forest" in t or "large" in t][:10]

    if deciduous_trees:
        forest_types["german_deciduous_dense"] = {
            "tree_density": 1.0,
            "average_height": [20.0, 30.0],
            "underground_material": "forest_floor",
            "lod_distance": 250.0,
            "collision_enabled": True,
            "preferred_trees": deciduous_trees,
            "tree_distribution": _create_tree_distribution(deciduous_trees),
            "comment": "Dichter Laubwald - Buchen und Eichen (klassischer deutscher Wald)",
        }

    # 2. German Mixed Forest (Mischwald)
    mixed_trees = [t for t in all_tree_keys if "forest" in t or "group" in t][:10]

    if mixed_trees:
        forest_types["german_mixed_forest"] = {
            "tree_density": 0.85,
            "average_height": [18.0, 26.0],
            "underground_material": "forest_floor",
            "lod_distance": 220.0,
            "collision_enabled": True,
            "preferred_trees": mixed_trees,
            "tree_distribution": _create_tree_distribution(mixed_trees),
            "comment": "Mischwald - Buchen, Eichen und Espen (vielfältiger Bestand)",
        }

    # 3. German Sparse Deciduous (Lichter Laubwald)
    sparse_trees = [t for t in all_tree_keys if "bush" in t or ("small" in t and "forest" not in t)][:10]

    if sparse_trees:
        forest_types["german_sparse_deciduous"] = {
            "tree_density": 0.4,
            "average_height": [15.0, 23.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": sparse_trees,
            "tree_distribution": _create_tree_distribution(sparse_trees),
            "comment": "Lichter Laubwald - überwiegend Busch- und kleinere Bäume",
        }

    # 4. Orchard Area (Obstplantage)
    orchard_trees = [t for t in all_tree_keys if "small" in t or "sml" in t][:10]

    if orchard_trees:
        forest_types["orchard_area"] = {
            "tree_density": 0.3,
            "average_height": [8.0, 15.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": orchard_trees,
            "tree_distribution": _create_tree_distribution(orchard_trees),
            "comment": "Obstplantage - niedrige und kleine Bäume",
        }

    # 5. Hedgerow (Hecke/Feldgehölz)
    hedge_trees = [t for t in all_tree_keys if "wall" in t or ("small" in t and "bush" not in t)][:6]

    if hedge_trees:
        forest_types["hedgerow"] = {
            "tree_density": 0.2,
            "average_height": [10.0, 18.0],
            "underground_material": "grassland",
            "lod_distance": 120.0,
            "collision_enabled": True,
            "preferred_trees": hedge_trees,
            "tree_distribution": _create_tree_distribution(hedge_trees),
            "comment": "Hecke/Feldgehölz - dünne, lineare Bestände",
        }

    # 6. Dead Forest (Totholz/Verfallender Wald)
    dead_trees = [t for t in all_tree_keys if "dead" in t]

    if dead_trees:
        forest_types["dead_forest"] = {
            "tree_density": 0.3,
            "average_height": [15.0, 25.0],
            "underground_material": "forest_floor",
            "lod_distance": 200.0,
            "collision_enabled": True,
            "preferred_trees": dead_trees,
            "tree_distribution": _create_tree_distribution(dead_trees),
            "comment": "Totholz/Verfallender Wald - dürre, tote Bäume",
        }

    # Fallback: Wenn keine speziellen Waldtypen, nutze alle Bäume
    if not forest_types and all_tree_keys:
        forest_types["generic_forest"] = {
            "tree_density": 0.7,
            "average_height": [15.0, 25.0],
            "underground_material": "forest_floor",
            "lod_distance": 200.0,
            "collision_enabled": True,
            "preferred_trees": all_tree_keys[:20],
            "tree_distribution": _create_tree_distribution(all_tree_keys[:20]),
            "comment": "Generischer Wald - alle verfügbaren Bäume",
        }

    return forest_types


def generate_forest_mappings(forest_types: dict) -> dict:
    """
    Generiere sinnvolle forest_mappings basierend auf verfügbaren Waldtypen.

    Args:
        forest_types: {forest_type_name: {...}}

    Returns:
        forest_mappings Dict
    """
    # Fallback auf generischen Wald wenn nötig
    if not forest_types:
        print("[WARNING] Keine Waldtypen verfügbar, nutze Fallback-Mappings")
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

    # Bestimme Standard-Waldtyp
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
    """Hauptfunktion."""
    print(f"[START] Generiere forest_type_templates")
    print(f"        Quelle: {config.BEAMNG_DIR / 'main' / 'forestItemData.json'}\n")

    # Lade forestItemData.json
    forest_data_path = config.BEAMNG_DIR / "main" / "forestItemData.json"
    forest_item_data = load_forest_item_data(forest_data_path)

    if not forest_item_data:
        print("[ERROR] Keine Forest-Items geladen")
        return

    print(f"[INFO] {len(forest_item_data)} Tree-Items geladen\n")

    # Kategorisiere Bäume
    trees_by_type = categorize_trees(forest_item_data)

    print("[INFO] Baum-Kategorisierung:")
    for tree_type, trees in sorted(trees_by_type.items()):
        print(f"       {tree_type:20s} : {len(trees):2d}x")

    # Generiere Waldtypen
    print(f"\n[INFO] Generiere Waldtypen...")
    forest_types = generate_forest_types(trees_by_type)

    print(f"\n[INFO] Generierte Waldtypen:")
    for forest_type in sorted(forest_types.keys()):
        tree_count = len(forest_types[forest_type].get("preferred_trees", []))
        print(f"       {forest_type:30s} : {tree_count:3d} preferred_trees")

    # Generiere forest_mappings
    print(f"\n[INFO] Generiere forest_mappings...")
    forest_mappings = generate_forest_mappings(forest_types)

    # Lade osm_to_beamng.json
    config_path = Path("data/osm_to_beamng.json")
    with open(config_path, "r", encoding="utf-8") as f:
        osm_config = json.load(f)

    # Ersetze forest_type_templates und forest_mappings
    osm_config["forest_type_templates"] = forest_types
    osm_config["forest_mappings"] = forest_mappings

    # Schreibe osm_to_beamng.json
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(osm_config, f, indent=4, ensure_ascii=False)

    print(f"\n" + "=" * 80)
    print(f"[DONE] osm_to_beamng.json aktualisiert")
    print(f"       - {len(forest_types)} forest_type_templates")
    print(f"       - forest_mappings angepasst")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
