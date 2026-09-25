"""
Forest type templates and OSM tag -> forest type mappings, generated from the available tree items.

Used by tools/generate_forest_types.py to (re)write "forest_type_templates" and "forest_mappings" in
data/osm_to_beamng.json (committed); the export itself only reads that JSON. Tree heights in the lists below were
measured on the BeamNG models.
"""

from collections import defaultdict


def categorize_trees(forest_item_data: dict) -> dict:
    """Categorize trees by type."""
    trees_by_type = defaultdict(list)
    for tree_key, tree_info in forest_item_data.items():
        tree_type = tree_info.get("name", "unknown")
        trees_by_type[tree_type].append(tree_key)
    return dict(trees_by_type)


def create_tree_distribution(preferred_trees: list) -> dict:
    """Create a tree_distribution dictionary with uniform distribution."""
    if not preferred_trees:
        return {}
    probability = 1.0 / len(preferred_trees)
    return {tree: probability for tree in preferred_trees}


# Low deciduous trees (measured model height 6-12.2 m). Explicit instead of via name pattern:
# "low" in the name is unreliable (tree_douglasfir_group_low is 22.6 m tall).
LOW_DECIDUOUS_TREES = [
    "tree_aspen_small_low",
    "tree_aspen_small_low_group",
    "tree_aspen_small_a",
    "tree_aspen_small_b",
    "tree_aspen_small_c",
    "tree_aspen_small_d",
    "tree_beech_small_b",
    "tree_beech_small_c",
    "tree_beech_small_d",
    "tree_oak_sml_a",
    "tree_oak_sml_b",
]


# Gardens/residential areas: small deciduous trees (fruit-tree size; BeamNG has no real fruit trees) and
# bushes (measured height 1.2-3.3 m). Explicit, because name patterns are misleading regarding heights.
GARDEN_TREES = [
    "tree_aspen_small_low",
    "tree_aspen_small_a",
    "tree_aspen_small_b",
    "tree_aspen_small_c",
    "tree_aspen_small_d",
    "tree_beech_small_b",
    "tree_beech_small_c",
    "tree_beech_small_d",
    "tree_oak_sml_a",
    "tree_oak_sml_b",
]
GARDEN_BUSHES = [
    "tree_beech_bush_a",
    "tree_oak_bush_a",
    "tree_aspen_bush_a",
    "tree_aspen_bush_b",
    "tree_aspen_bush_c",
    "tree_oak_bush_c",
    "tree_beech_bush_b",
    "generibush_small",
]
# Orchards (landuse=orchard): broad-crowned small deciduous trees (model height 8.5-9.9 m, crown approx. half as wide as tall).
# Aspens are too slim, beech_small_d would be too tall at 6-7 m, bushes are not trees.
ORCHARD_TREES = [
    "tree_oak_sml_a",
    "tree_oak_sml_b",
    "tree_beech_small_c",
]
# Tree rows (natural=tree_row): small deciduous trees
TREE_ROW_TREES = [
    "tree_aspen_small_a",
    "tree_aspen_small_b",
    "tree_aspen_small_c",
    "tree_aspen_small_d",
    "tree_beech_small_b",
    "tree_beech_small_c",
    "tree_beech_small_d",
    "tree_oak_sml_a",
    "tree_oak_sml_b",
]
# Single trees (natural=tree): large, broad-crowned deciduous trees (measured model height 13-21 m). The "large" aspens
# are only 10 m tall and narrow, the "*_forest_*" trees are slim forest trunks - neither suits free-standing trees.
LARGE_DECIDUOUS_TREES = [
    "tree_oak_large_a",
    "tree_oak_large_b",
    "tree_oak_large_c",
    "tree_beech_large_b",
    "tree_beech_large_c",
]


# Woods (natural=wood, landuse=wood): dense forest of large deciduous trees with undergrowth
BROADLEAF_UNDERGROWTH_TYPE = "german_broadleaf_undergrowth"
BROADLEAF_CANOPY_SHARE = 0.65  # Share of large deciduous trees, the rest is undergrowth (bushes)


def generate_forest_types(trees_by_type: dict) -> dict:
    """Generate sensible forest types for German forests."""
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
            "comment": "Dense deciduous forest - beech and oak (classic German forest)",
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
            "comment": "Mixed forest - beech, oak and aspen (diverse stand)",
        }

    # 2b. German Low Deciduous: everything with trees except landuse=forest and woods (natural/landuse=wood) gets only low deciduous trees.
    # average_height acts as scaling (target height / 20 m): 16-22 -> 0.8-1.1.
    low_trees = [t for t in LOW_DECIDUOUS_TREES if t in all_tree_keys]
    if low_trees:
        forest_types["german_low_deciduous"] = {
            "tree_density": 0.7,
            "average_height": [16.0, 22.0],
            "underground_material": "forest_floor",
            "lod_distance": 200.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(low_trees),
            "comment": "Low deciduous forest (6-13 m) - everything with trees except landuse=forest and woods "
                       "(natural/landuse=wood); average_height acts as a scale (target height/20 m)",
        }

    # 2c. Gardens/allotments, residential areas, single trees. Minimum spacing in the ForestWorkflow is 5 m:
    # spacing = 5 / sqrt(tree_density) -> 0.3 gives approx. 9 m (sparse planting).
    garden_trees = [t for t in GARDEN_TREES if t in all_tree_keys]
    garden_bushes = [t for t in GARDEN_BUSHES if t in all_tree_keys]
    if garden_trees and garden_bushes:
        weights = {t: 0.45 / len(garden_trees) for t in garden_trees}
        weights.update({t: 0.55 / len(garden_bushes) for t in garden_bushes})
        forest_types["garden_mixed"] = {
            "tree_density": 0.3,
            "average_height": [16.0, 22.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": weights,
            "comment": "Gardens/allotments - sparse small deciduous trees (fruit tree size) and bushes; BeamNG has "
                       "no real fruit tree assets",
        }
    if garden_bushes:
        forest_types["residential_green"] = {
            "tree_density": 0.3,
            "average_height": [16.0, 22.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(garden_bushes),
            "comment": "Residential areas - sparse bushes between the houses (roads/buildings are left out)",
        }
    single_trees = [t for t in LARGE_DECIDUOUS_TREES if t in all_tree_keys]
    if single_trees:
        forest_types["single_tree"] = {
            "tree_density": 1.0,
            "average_height": [17.0, 23.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(single_trees),
            "comment": "Single trees (OSM natural=tree as a point): large, broad-crowned deciduous trees (oak/beech,"
                       " 13-21 m); average_height acts as a scale (target height/20 m)",
        }
    # 2d. Dense forest of 100 % large, broad-crowned deciduous trees with undergrowth (natural=wood / landuse=wood).
    # Density 3.6 -> minimum spacing 5/sqrt(3.6) = 2.6 m (normal deciduous forest 0.7 = 6 m). Scaling 10-13 m / 20 = 0.5-0.65:
    # half tree size, large models (13-21 m) become 6.5-14 m tall (the first 1.0-1.3 had trunks that were too massive);
    # the bushes scale along (same range per polygon).
    if single_trees and garden_bushes:
        weights = {t: BROADLEAF_CANOPY_SHARE / len(single_trees) for t in single_trees}
        weights.update({t: (1.0 - BROADLEAF_CANOPY_SHARE) / len(garden_bushes) for t in garden_bushes})
        forest_types[BROADLEAF_UNDERGROWTH_TYPE] = {
            "tree_density": 3.6,
            "average_height": [10.0, 13.0],
            "underground_material": "forest_floor",
            "lod_distance": 220.0,
            "collision_enabled": True,
            "preferred_trees": weights,
            "comment": "Dense deciduous forest of large, broad-crowned oaks/beeches (no conifers) with undergrowth "
                       "(bushes, approx. 35 %) - for woods (natural=wood, landuse=wood)",
        }
    row_trees = [t for t in TREE_ROW_TREES if t in all_tree_keys]
    if row_trees:
        # Tree row: natural=tree_row is a LINE - trees at row_spacing distance along the line
        forest_types["tree_row"] = {
            "tree_density": 1.0,
            "row_spacing": 8.0,
            "average_height": [16.0, 24.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(row_trees),
            "comment": "Tree row (OSM natural=tree_row is a LINE): trees at spacing row_spacing along the line",
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
            "comment": "Sparse deciduous forest - mostly bushes and smaller trees",
        }

    # 4. Orchard Area: approx. 5 m tall, broad-crowned (roundish) deciduous trees. Scaling = target height / 20 m, at least 0.5:
    # models 8.5-9.9 m x 0.5-0.6 give 4.2-6 m. Explicit list (measured heights), no name heuristic.
    orchard_trees = [t for t in ORCHARD_TREES if t in all_tree_keys]
    if orchard_trees:
        forest_types["orchard_area"] = {
            "tree_density": 0.3,
            "average_height": [10.0, 12.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(orchard_trees),
            "comment": "Orchard - approx. 5 m tall, broad-crowned (roundish) deciduous trees at approx. 9 m spacing;"
                       " average_height acts as a scale (target height/20 m, min. 0.5): models 8.5-9.9 m x 0.5-0.6",
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
            "comment": "Hedgerow/field copse - thin, linear stands",
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
            "comment": "Dead wood/decaying forest - dry, dead trees",
        }

    return forest_types


def generate_forest_mappings(forest_types: dict) -> dict:
    """Generate forest_mappings based on the available forest types."""
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
    # Only landuse=forest gets the tall mixed forest, woods the dense deciduous forest, everything else with trees the low deciduous forest
    low_forest = "german_low_deciduous" if "german_low_deciduous" in forest_types else default_forest
    # Woods (natural=wood, landuse=wood): dense deciduous forest with undergrowth
    wood_forest = BROADLEAF_UNDERGROWTH_TYPE if BROADLEAF_UNDERGROWTH_TYPE in forest_types else low_forest

    garden = "garden_mixed" if "garden_mixed" in forest_types else None
    residential = "residential_green" if "residential_green" in forest_types else None

    mappings = {
        "landuse": {
            "forest": default_forest,
            "wood": wood_forest,
            "orchard": "orchard_area" if "orchard_area" in forest_types else default_forest,
            **({"allotments": garden} if garden else {}),
            **({"residential": residential} if residential else {}),
        },
        "natural": {
            "wood": wood_forest,
            "forest": low_forest,
            "scrub": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "heath": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "tree_row": "tree_row" if "tree_row" in forest_types else ("hedgerow" if "hedgerow" in forest_types else default_forest),
            "wetland": low_forest,
        },
        "leisure": {
            "nature_reserve": low_forest,
            "park": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            **({"garden": garden} if garden else {}),
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
        # Clearings (inner rings of forest relations) get only low deciduous trees
        # Clearings only in forest relations - a hole in a residential area is something else
        "clearings": {
            "forest_type": low_forest,
            "only_for": ["landuse=forest", "landuse=wood", "natural=wood", "natural=forest"],
        },
        # Overrides (trees=conifer, ...) only refine landuse=forest - otherwise e.g. a
        # natural=wood with a conifer tag would be redirected to a tall forest type
        "tag_overrides_only_for": ["landuse=forest"],
    }
    if "single_tree" in forest_types:
        mappings["single_trees"] = {"forest_type": "single_tree"}  # OSM natural=tree (points)
    return mappings
