"""
Generate Forest Assets: managedItemData.json + forest_type_templates

Combined script that:
1. Scans DAE files and generates managedItemData.json (art/forest/, BeamNG item registry)
2. Generates forest types and mappings from the scanned items
3. Updates osm_to_beamng.json
"""

from pathlib import Path
import json
import re
import zipfile
from collections import defaultdict
import sys

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config
from world_to_beamng.io.beamng_install import get_beamng_install_dir


# Mapping from filename patterns to tree species
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
    """Extract the tree name from a DAE filename."""
    name = Path(filename).stem
    name_lower = name.lower()

    for pattern, tree_name in TREE_NAME_PATTERNS.items():
        if re.search(pattern, name_lower):
            return tree_name

    cleaned = re.sub(r"[_\-\d]", " ", name).strip()
    if not cleaned or cleaned.lower() in ["tree", "model", "asset"]:
        return "tree"
    return cleaned.lower()


# east_coast_usa's bonus folder "ECA_coast_bush" contains, besides generic
# filler bushes (generibush*), also Mediterranean species (cork oak, stone pine) that
# are out of place in a German forest (Black Forest/Freiburg) - see
# research 2026-09-18: cork_oak_bush_* and maritime_pine_bush had wrongly ended up in
# the "German" forest mixes via the plain filename patterns in
# extract_tree_name_from_filename(). cork_oak_bush_* additionally references
# the broken "holm_oak_trunk" texture (no longer exists in the current
# east_coast_usa.zip) - the exclusion fixes both at once.
# generibush/generibush_small stay (climate-neutral filler bushes, no
# dependency on the excluded species).
EXCLUDED_NON_NATIVE_SPECIES = ("cork_oak_bush_large", "cork_oak_bush_medium", "maritime_pine_bush")


def extract_tree_assets_from_zip(dest_dir: Path, install_dir: Path) -> int:
    """
    Extracts the tree asset folder (DAE, compiled .cdae, imposter DDS,
    materials.json) directly from the CURRENTLY INSTALLED east_coast_usa.zip.

    Reason why it is not copied from the unpacked user folder (AppData/.../levels/
    east_coast_usa) as before: that folder is a decades-old leftover unpack
    (files dated 2013) that Steam updates never touch - its materials.json/.dae
    reference material/texture names (e.g. "m_fir_merged_foliage" for Douglas fir/aspen)
    that were long since renamed or removed in the currently installed content
    pack (see research 2026-09-18: Douglas fir now uses "m_fir_leaves_distant" +
    generated .imposter.dds instead of the old merged_foliage texture). The
    content zip on the other hand is ALWAYS exactly as up to date as the installed
    game version and internally consistent (DAE, material and textures are
    shipped together) - hence no .link resolution is needed anymore, the current
    zip does not contain any .link placeholders at all.

    EXCLUDED_NON_NATIVE_SPECIES is skipped (see there).

    Returns:
        Number of extracted files
    """
    zip_path = install_dir / "content" / "levels" / "east_coast_usa.zip"
    if not zip_path.is_file():
        print(f"[ERROR] east_coast_usa.zip not found: {zip_path}")
        return 0

    prefix = "levels/east_coast_usa/art/shapes/trees/"
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    excluded = 0
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            if entry.endswith("/") or not entry.startswith(prefix):
                continue
            filename = entry.rsplit("/", 1)[-1]
            if any(filename.lower().startswith(species) for species in EXCLUDED_NON_NATIVE_SPECIES):
                excluded += 1
                continue
            out_path = dest_dir / entry[len(prefix) :]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(z.read(entry))
            count += 1

    print(f"[INFO] {count} tree asset files extracted from {zip_path.name} -> {dest_dir}")
    if excluded:
        print(f"[INFO] {excluded} files of non-native species skipped ({', '.join(EXCLUDED_NON_NATIVE_SPECIES)})")
    return count


def copy_tree_assets(dest_dir: Path, install_dir: Path) -> int:
    """
    Extracts the complete tree asset folder into the level itself, so that
    world_to_beamng no longer depends on a foreign level (east_coast_usa)
    (see extract_tree_assets_from_zip()).

    Does NOT replace the meshes with BeamNG's "canonical" trees_library anymore
    (content/assets/meshes.zip) - research 2026-09-18 showed that its
    beech/oak/birch meshes internally reference "m_ind_beech_leaves"/"m_ind_birch_leaves_01"
    materials that are defined in NO installed content pack
    (dead end, not the old "stale path" problem: the same research
    that fixed east_coast_usa as the source showed that the trees_library itself
    is broken). east_coast_usa's own meshes, fully defined via main.materials.json
    (e.g. "m_birch_leaves_distant" instead of "m_ind_birch_leaves_01"),
    are therefore now the more reliable source in EVERY case.

    Afterwards rewrites "levels/east_coast_usa/..." path references in all
    TEXT files (materials.json AND .dae, COLLADA embedded texture refs) to the
    own level. .cdae files are a compiled binary cache format that
    also embeds the old path but is NOT safely text-patchable
    (length-prefixed strings) - they are deleted instead, so that BeamNG
    recompiles them fresh from the .dae automatically on the next load.

    Returns:
        Number of extracted files (0 on failure)
    """
    extracted = extract_tree_assets_from_zip(dest_dir, install_dir)
    if extracted == 0:
        return 0

    old_ref = "levels/east_coast_usa/art/shapes/trees"
    new_ref = f"levels/{config.LEVEL_NAME}/art/shapes/trees"

    fixed = 0
    text_files = list(dest_dir.rglob("*materials.json")) + list(dest_dir.rglob("*.dae"))
    for text_file in text_files:
        text = text_file.read_text(encoding="utf-8")
        if old_ref in text:
            text_file.write_text(text.replace(old_ref, new_ref), encoding="utf-8")
            fixed += 1

    # .cdae is a compiled binary cache and embeds the old path.
    # Delete -> BeamNG recompiles fresh from the .dae automatically on the next load.
    cdae_files = list(dest_dir.rglob("*.cdae"))
    for cdae_file in cdae_files:
        cdae_file.unlink()
    if cdae_files:
        print(f"[INFO] {len(cdae_files)} .cdae cache files deleted (recompiled on load)")

    print(f"[INFO] {fixed} text files (.dae/.materials.json) rewritten to the own level path")
    return extracted


def scan_dae_files(dir_path: str, beamng_root: str) -> dict:
    """
    Scan DAE files and generate managedItemData.

    IMPORTANT: dir_path MUST already lie inside beamng_root (see copy_tree_assets),
    so that shapeFile points to the own level instead of the source of the assets.

    Returns:
        {tree_key: {name, class, shapeFile, collidable, radius}}
    """
    dir_path_obj = Path(dir_path)
    beamng_root_obj = Path(beamng_root)

    if not dir_path_obj.is_dir():
        print(f"[ERROR] Directory not found: {dir_path}")
        return None

    dae_files = sorted(dir_path_obj.rglob("*.dae"))

    print(f"[INFO] DAE files found: {len(dae_files)}")
    if not dae_files:
        print("[ERROR] No DAE files found!")
        return None

    forest_item_data = {}
    tree_type_counts = defaultdict(int)

    for idx, dae_file in enumerate(dae_files, 1):
        tree_type = extract_tree_name_from_filename(dae_file.name)
        item_key = dae_file.stem

        # dae_file lies under beamng_root_obj (world_to_beamng) -> "levels/<level>/<rel>"
        relative_dae = dae_file.relative_to(beamng_root_obj)
        shape_file_path = f"levels/{config.LEVEL_NAME}/" + str(relative_dae).replace("\\", "/")

        radius = 2.0 if tree_type in ["cork_oak", "holm_oak"] else 1.5

        forest_item_data[item_key] = {
            "name": item_key,
            "class": "ForestItemData",
            "internalName": item_key,
            "shapeFile": shape_file_path,
            "collidable": True,
            "radius": radius,
        }

        tree_type_counts[tree_type] += 1
        if idx <= 10 or idx % 10 == 0:
            print(f"[{idx:3d}] {item_key:40s} → {tree_type:15s}")

    print(f"\n[INFO] Tree type overview:")
    for tree_type in sorted(tree_type_counts.keys()):
        print(f"       {tree_type:30s} : {tree_type_counts[tree_type]:3d}x")

    return forest_item_data


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


def main():
    """Main function: generate managedItemData.json and forest types."""
    print("=" * 80)
    print("[START] Generating forest assets (managedItemData + forest_type_templates)")
    print("=" * 80)

    install_dir = get_beamng_install_dir()

    # ===== PHASE 0: Copy tree assets into the level itself =====
    print("\n[PHASE 0] Copying tree assets into the own level (makes world_to_beamng self-contained)")
    print("-" * 80)

    # east_coast_usa.zip serves as the fallback source for geometry + material definitions
    # for species that do not exist in BeamNG's canonical trees_library (see
    # extract_tree_assets_from_zip()).
    dest_dir = config.BEAMNG_DIR / "art" / "shapes" / "trees"
    if copy_tree_assets(dest_dir, install_dir) == 0:
        return

    # ===== PHASE 1: Scan DAE files (in the own copy!) =====
    print("\n[PHASE 1] Scanning DAE files and generating managedItemData.json")
    print("-" * 80)

    forest_item_data = scan_dae_files(str(dest_dir), str(config.BEAMNG_DIR))

    if not forest_item_data:
        print("[ERROR] No forest items generated")
        return

    # Save managedItemData.json (BeamNG expects the item registry under art/forest/)
    output_dir = config.BEAMNG_DIR / "art" / "forest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "managedItemData.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(forest_item_data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] managedItemData.json created: {output_file}")
    print(f"       {len(forest_item_data)} Tree-Items")

    # ===== PHASE 2: Generate forest types =====
    print("\n[PHASE 2] Generating forest types and mappings")
    print("-" * 80)

    trees_by_type = categorize_trees(forest_item_data)

    print("\n[INFO] Tree categorization:")
    for tree_type, trees in sorted(trees_by_type.items()):
        print(f"       {tree_type:20s} : {len(trees):2d}x")

    forest_types = generate_forest_types(trees_by_type)

    print(f"\n[INFO] Generated forest types:")
    for forest_type in sorted(forest_types.keys()):
        tree_count = len(forest_types[forest_type].get("preferred_trees", {}))
        print(f"       {forest_type:30s} : {tree_count:3d} trees")

    forest_mappings = generate_forest_mappings(forest_types)

    # ===== PHASE 3: Update osm_to_beamng.json =====
    print(f"\n[PHASE 3] Updating osm_to_beamng.json")
    print("-" * 80)

    config_path = Path("data/osm_to_beamng.json")
    with open(config_path, "r", encoding="utf-8") as f:
        osm_config = json.load(f)

    osm_config["forest_type_templates"] = forest_types
    osm_config["forest_mappings"] = forest_mappings

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(osm_config, f, indent=4, ensure_ascii=False)

    print(f"[DONE] osm_to_beamng.json updated")
    print(f"       - {len(forest_types)} forest_type_templates")
    print(f"       - forest_mappings updated")

    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("[✓] COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"managedItemData.json: {len(forest_item_data)} Tree-Items")
    print(f"Forest types:        {len(forest_types)}")
    print(f"  - german_deciduous_dense")
    print(f"  - german_mixed_forest")
    print(f"  - german_sparse_deciduous")
    print(f"  - orchard_area")
    print(f"  - hedgerow")
    print(f"  - dead_forest")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
