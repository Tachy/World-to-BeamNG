#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit test for the integrity of all exported BeamNG data.

Checks:
- DAE files (terrain + buildings)
- JSON files (materials + items)
- Textures
- Material/shape references
- XML validity
- Texture mapping (tile names to texture files)
"""

import sys
import io
from pathlib import Path, PurePosixPath

# Set UTF-8 encoding for stdout
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json
import xml.etree.ElementTree as ET
from lxml import etree as lxml_etree

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config
import logging
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()

# ============================================================================
# GLOBAL CONSTANTS FOR EXPORT INTEGRITY TESTS
# ============================================================================

# Directory names (relative to the BeamNG directory)
MAIN_DIR = "main"
SHAPES_DIR = "art/shapes"
TEXTURES_DIR = "art/shapes/textures"
BUILDINGS_DIR = "buildings"

# File names
MATERIALS_JSON = "materials.json"
ITEMS_JSON_FILENAME = "items.json"

# DAE filenames and patterns
TERRAIN_DAE_PATTERN = "terrain_*.dae"
TERRAIN_HORIZON_DAE = "terrain_horizon.dae"

# Material names and prefixes
LOD2_WALL_MATERIAL_PREFIX = "lod2_wall_plaster_"
LOD2_ROOF_MATERIAL = "lod2_roof_red"
HORIZON_MATERIAL = "horizon_terrain"

# Material prefixes for categorization
MATERIAL_PREFIX_TERRAIN_TILE = "tile_"
MATERIAL_PREFIX_ROAD = "italy_"
MATERIAL_PREFIX_LOD2 = "lod2_"
MATERIAL_PREFIX_HORIZON = "horizon_"
MATERIAL_PREFIX_JUNCTION = "junction"
MATERIAL_PREFIX_CENTERLINE = "centerline"
MATERIAL_PREFIX_BOUNDARY = "boundary"

# Texture prefixes
TEXTURE_PREFIX_TILE = "tile_"

# Item names and prefixes
HORIZON_ITEM_NAME = "Horizon"
BUILDING_ITEMS_PREFIX = "buildings_tile_"
TERRAIN_ITEMS_PREFIX = "terrain_"

# Configuration
TEXTURE_MAX_SIZE_MB = 50  # Warning for textures > 50 MB
BUILDING_Z_TOLERANCE_M = 5.0  # Tolerance for buildings below terrain (m)
UV_RANGE_MIN = -0.5
UV_RANGE_MAX = 1.5
Z_OUTLIER_SIGMA = 10  # Standard deviations for outlier detection

# ============================================================================


class ExportIntegrityTest:
    """Test suite for export integrity."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.dae_materials = set()  # Aus library_materials
        self.triangle_materials = set()  # Aus triangles material-Attributen
        self.beamng_dir = Path(config.BEAMNG_DIR)
        self.shapes_dir = Path(config.BEAMNG_DIR_SHAPES)
        self.textures_dir = Path(config.BEAMNG_DIR_TEXTURES)
        self.buildings_dir = Path(config.BEAMNG_DIR_BUILDINGS)
        self.cache_dir = Path(config.CACHE_DIR)

        # Load items.json as the root directory
        self.items = {}
        self.has_building_items = False
        self.has_horizon_item = False
        self.items_json_path = self.beamng_dir / Path(config.ITEMS_JSON)

        if self.items_json_path.exists():
            try:
                with open(self.items_json_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if "name" in item:
                            self.items[item["name"]] = item

                # Check for building items and horizon item
                self.has_building_item = any(k.startswith(BUILDING_ITEMS_PREFIX) for k in self.items.keys())
                self.has_horizon_item = HORIZON_ITEM_NAME in self.items
            except Exception as e:
                self.warning(f"Could not load items.json: {e}")

    def _resolve_relative_path(self, relative_path: str) -> Path:
        r"""
        Converts a relative BeamNG path to an absolute filesystem path.

        Args:
            relative_path: Path as it appears in BeamNG configuration files
                           (materials.json, items.json).
                           May contain POSIX or Windows separators.

        Returns:
            Absolute Path object of the referenced element.
        """
        # Convert the input path to a PurePosixPath for consistent handling
        # BeamNG internal paths are mostly POSIX style, even on Windows
        path_posix = PurePosixPath(relative_path.replace("\\", "/"))

        # config.BEAMNG_DIR is the absolute path to the level root directory
        # e.g. C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\world_to_beamng

        # 1. Path starts with 'levels/world_to_beamng/' (config.RELATIVE_DIR is 'levels/world_to_beamng')
        # In this case the part after 'levels/world_to_beamng/' is the relative path from the level root
        if path_posix.is_relative_to(config.RELATIVE_DIR):
            return self.beamng_dir / path_posix.relative_to(config.RELATIVE_DIR)

        # 2. Path starts with 'art/', 'main/' or is directly a filename
        # These are implicitly relative to the level root
        # Example: 'art/shapes/textures/file.dds'
        # Example: 'main/materials.json'
        # Example: 'info.json'
        return self.beamng_dir / path_posix

    def error(self, msg):
        """Register a critical error."""
        self.errors.append(f"❌ {msg}")
        logger.info(f"  ❌ {msg}")

    def warning(self, msg):
        """Register a warning."""
        self.warnings.append(f"⚠️ {msg}")
        logger.info(f"  ⚠️ {msg}")

    def success(self, msg):
        """Register a success."""
        logger.info(f"  ✅ {msg}")

    def test_terrain_dae(self):
        """Test terrain.dae integrity."""
        logger.info("\n[1] Testing terrain.dae...")

        # Search dynamically for terrain_<x>_<y>.dae
        terrain_daes = list(self.shapes_dir.glob("terrain_*.dae"))

        if not terrain_daes:
            self.error(f"No terrain_*.dae found in: {self.shapes_dir}")
            return

        terrain_dae = terrain_daes[0]
        self.success(f"terrain.dae found: {terrain_dae.name}")

        # Parse XML
        try:
            tree = ET.parse(terrain_dae)
            root = tree.getroot()

            # Check the COLLADA root
            if root.tag != "{http://www.collada.org/2005/11/COLLADASchema}COLLADA":
                self.error("No valid COLLADA root element")
                return

            self.success("Valid COLLADA XML")

            # Count geometries
            geometries = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}geometry")
            if len(geometries) == 0:
                self.error("No geometries found")
            else:
                self.success(f"{len(geometries)} geometries found")

            # Count materials
            materials = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}material")
            if len(materials) == 0:
                self.warning("No materials found in DAE")
            else:
                self.success(f"{len(materials)} materials found")

            material_ids = [m.get("id") for m in materials if m.get("id")]
            self.dae_materials.update(material_ids)

            # Collect material references from triangles
            triangles = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}triangles")
            tri_materials = [t.get("material") for t in triangles if t.get("material")]
            if tri_materials:
                self.triangle_materials.update(tri_materials)
                self.success(f"{len(set(tri_materials))} material references in triangles")

            # Check vertices/faces
            float_arrays = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}float_array")
            if float_arrays:
                total_floats = sum(int(fa.get("count", 0)) for fa in float_arrays)
                self.success(f"{len(float_arrays)} float arrays, {total_floats:,} values total")

        except ET.ParseError as e:
            self.error(f"XML parse error: {e}")
        except Exception as e:
            self.error(f"Error during test: {e}")

    def test_terrain_face_materials(self):
        """Test that EVERY face in terrain_x_y.dae has a material."""
        logger.info("\n[1b] Testing terrain face materials...")

        # Search dynamically for terrain_<x>_<y>.dae
        terrain_daes = list(self.shapes_dir.glob(TERRAIN_DAE_PATTERN))

        if not terrain_daes:
            self.warning(f"No {TERRAIN_DAE_PATTERN} found - skipping face material test")
            return

        terrain_dae = terrain_daes[0]
        self.success(f"Testing face materials in: {terrain_dae.name}")

        try:
            tree = ET.parse(terrain_dae)
            root = tree.getroot()
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Collect all defined materials
            materials = root.findall(".//collada:material", ns)
            defined_materials = set()
            for mat in materials:
                mat_id = mat.get("id")
                if mat_id:
                    defined_materials.add(mat_id)

            self.success(f"{len(defined_materials)} materials defined in library_materials: {defined_materials}")

            # Check all triangles and polylist primitives
            triangles_list = root.findall(".//collada:triangles", ns)
            polylists = root.findall(".//collada:polylist", ns)

            faces_without_material = []
            faces_with_undefined_material = []
            material_face_count = {}  # material -> number of faces

            # Check triangles
            for tri_idx, tri in enumerate(triangles_list):
                material = tri.get("material")
                count = int(tri.get("count", 0))

                # Count faces per material
                if material:
                    material_face_count[material] = material_face_count.get(material, 0) + count

                if material is None or material == "":
                    faces_without_material.append(f"triangles[{tri_idx}]: material attribute MISSING")
                elif material not in defined_materials:
                    faces_with_undefined_material.append(
                        f"triangles[{tri_idx}]: material='{material}' not defined in library_materials"
                    )

            # Check polylist
            for poly_idx, poly in enumerate(polylists):
                material = poly.get("material")
                count = int(poly.get("count", 0))

                # Count faces per material
                if material:
                    material_face_count[material] = material_face_count.get(material, 0) + count

                if material is None or material == "":
                    faces_without_material.append(f"polylist[{poly_idx}]: material attribute MISSING")
                elif material not in defined_materials:
                    faces_with_undefined_material.append(
                        f"polylist[{poly_idx}]: material='{material}' not defined in library_materials"
                    )

            # Report errors
            if faces_without_material:
                self.error(f"{len(faces_without_material)} faces without material attribute:")
                for err in faces_without_material[:10]:
                    self.error(f"  - {err}")
                if len(faces_without_material) > 10:
                    self.error(f"  ... and {len(faces_without_material) - 10} more")

            if faces_with_undefined_material:
                self.error(f"{len(faces_with_undefined_material)} faces with undefined material:")
                for err in faces_with_undefined_material[:10]:
                    self.error(f"  - {err}")
                if len(faces_with_undefined_material) > 10:
                    self.error(f"  ... and {len(faces_with_undefined_material) - 10} more")

            if not faces_without_material and not faces_with_undefined_material:
                total_faces = len(triangles_list) + len(polylists)
                self.success(f"✓ All {total_faces} faces have defined materials")

            # === Material statistics ===
            if material_face_count:
                logger.info("\n  [Material distribution] Triangles per material:")

                # Sort by count (descending)
                sorted_materials = sorted(material_face_count.items(), key=lambda x: x[1], reverse=True)

                for material_name, face_count in sorted_materials:
                    # Distinguish between tile materials and road materials
                    if material_name.startswith(MATERIAL_PREFIX_TERRAIN_TILE):
                        mat_type = "Terrain tile"
                    elif material_name.startswith(MATERIAL_PREFIX_ROAD):
                        mat_type = "Road"
                    else:
                        mat_type = "Other"

                    logger.info(f"    • {material_name:30s} ({mat_type:15s}): {face_count:6d} faces")

                # Summary
                total_faces_counted = sum(material_face_count.values())
                terrain_tile_faces = sum(
                    c for m, c in material_face_count.items() if m.startswith(MATERIAL_PREFIX_TERRAIN_TILE)
                )
                road_faces = sum(c for m, c in material_face_count.items() if m.startswith(MATERIAL_PREFIX_ROAD))

                logger.info(f"\n  [Summary]")
                logger.info(f"    Total:      {total_faces_counted:6d} faces")
                logger.info(
                    f"    Terrain:    {terrain_tile_faces:6d} faces ({100*terrain_tile_faces/total_faces_counted:.1f}%)"
                )
                logger.info(f"    Roads:      {road_faces:6d} faces ({100*road_faces/total_faces_counted:.1f}%)")

        except ET.ParseError as e:
            self.error(f"XML parse error: {e}")
        except Exception as e:
            self.error(f"Error during face material test: {e}")

    def test_building_daes(self):
        """Test buildings/*.dae integrity (only if building items exist in items.json)."""
        logger.info("\n[2] Testing buildings/*.dae...")

        # Skip the test if no building items are registered
        if not self.has_building_items:
            self.warning("No building items found in items.json - skipping building DAE test")
            return

        if not self.buildings_dir.exists():
            self.warning(f"Buildings directory not found: {self.buildings_dir}")
            return

        building_daes = list(self.buildings_dir.glob("*.dae"))

        if len(building_daes) == 0:
            self.warning("No building DAEs found")
            return

        self.success(f"{len(building_daes)} building DAEs found")

        errors = 0
        total_geometries = 0

        for dae_file in building_daes:
            try:
                tree = ET.parse(dae_file)
                root = tree.getroot()

                # Count geometries
                geometries = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}geometry")
                total_geometries += len(geometries)

                # Check whether LoD2 materials are present
                materials = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}material")
                material_ids = [m.get("id") for m in materials if m.get("id")]
                self.dae_materials.update(material_ids)

                has_wall = any(name.startswith(LOD2_WALL_MATERIAL_PREFIX) for name in material_ids)
                has_roof = LOD2_ROOF_MATERIAL in material_ids

                if not has_wall or not has_roof:
                    self.warning(f"{dae_file.name}: Missing materials (wall={has_wall}, roof={has_roof})")

            except Exception as e:
                errors += 1
                self.error(f"{dae_file.name}: Parse error - {e}")

        if errors == 0:
            self.success(f"All {len(building_daes)} DAEs parsed successfully")
            self.success(f"{total_geometries} building geometries total")
        else:
            self.error(f"{errors}/{len(building_daes)} DAEs faulty")

    def test_materials_json(self):
        """Test materials.json integrity."""
        logger.info("\n[3] Testing materials.json...")

        materials_json = self.beamng_dir / MAIN_DIR / MATERIALS_JSON

        if not materials_json.exists():
            self.error(f"materials.json not found: {materials_json}")
            return

        self.success(f"materials.json found")

        try:
            with open(materials_json, "r", encoding="utf-8") as f:
                materials = json.load(f)

            self.success(f"{len(materials)} materials defined")

            # Check LoD2 materials
            has_wall = any(name.startswith(LOD2_WALL_MATERIAL_PREFIX) for name in materials)
            has_roof = LOD2_ROOF_MATERIAL in materials

            if has_wall and has_roof:
                self.success("LoD2 materials present (wall + roof)")
            else:
                self.warning(f"LoD2 materials missing (wall={has_wall}, roof={has_roof})")

            # Check material structure
            invalid_materials = []
            for mat_name, mat_data in materials.items():
                if "mapTo" not in mat_data:
                    invalid_materials.append(f"{mat_name}: missing 'mapTo'")
                if "class" not in mat_data or mat_data["class"] != "Material":
                    invalid_materials.append(f"{mat_name}: missing/wrong 'class'")
                if "Stages" not in mat_data or not isinstance(mat_data["Stages"], list):
                    invalid_materials.append(f"{mat_name}: missing/wrong 'Stages'")

            if invalid_materials:
                for err in invalid_materials[:5]:  # Show at most 5 errors
                    self.error(err)
                if len(invalid_materials) > 5:
                    self.error(f"... and {len(invalid_materials)-5} more")
            else:
                self.success("All materials have valid structure")

            # Check texture references
            texture_refs = []
            for mat_name, mat_data in materials.items():
                if "Stages" in mat_data and mat_data["Stages"]:
                    stage = mat_data["Stages"][0]
                    if "baseColorMap" in stage:
                        texture_refs.append(stage["baseColorMap"])

            if texture_refs:
                self.success(f"{len(texture_refs)} texture references found")

                # Check whether referenced textures exist (with relative paths)
                missing_textures = []
                for tex_path in texture_refs:
                    # Convert relative path to absolute Windows path
                    full_path = self._resolve_relative_path(tex_path)

                    if not full_path.exists():
                        missing_textures.append(f"{tex_path} -> {full_path}")

                if missing_textures:
                    self.warning(f"{len(missing_textures)} referenced textures missing:")
                    for tex in missing_textures[:5]:
                        self.warning(f"  - {tex}")
                    if len(missing_textures) > 5:
                        self.warning(f"  ... and {len(missing_textures)-5} more")
                else:
                    self.success("All referenced textures exist")

                # Bidirectional comparison DAE <-> materials.json
                # Materials referenced in DAEs must exist here
                used_materials = set(self.dae_materials) | set(self.triangle_materials)
                json_material_names = set(materials.keys())
                json_mapto = {m.get("mapTo") for m in materials.values() if isinstance(m, dict) and "mapTo" in m}

                missing_in_json = [m for m in used_materials if m not in json_material_names and m not in json_mapto]
                if missing_in_json:
                    for m in missing_in_json[:5]:
                        self.error(f"Material referenced in DAE but not in materials.json: {m}")
                    if len(missing_in_json) > 5:
                        self.error(f"... and {len(missing_in_json)-5} more")
                else:
                    self.success("All DAE material references present in materials.json")

                # Optional warning for unused materials in JSON
                # Ignore tile materials (tile_<x>_<y>), terrain base materials and horizon materials
                unused = [m for m in json_material_names if m not in used_materials]
                unused = [m for m in unused if m not in ("unknown",)]  # allow fallback
                # Filter out tile materials (normal for the BeamNG material system)
                unused = [m for m in unused if not m.startswith(MATERIAL_PREFIX_TERRAIN_TILE)]
                # Filter out horizon materials (used directly by the item, not by geometries)
                unused = [m for m in unused if not m.startswith(MATERIAL_PREFIX_HORIZON)]
                if unused:
                    self.warning(f"{len(unused)} road materials in materials.json unused (e.g. {unused[:3]})")

        except json.JSONDecodeError as e:
            self.error(f"JSON parse error: {e}")
        except Exception as e:
            self.error(f"Error during test: {e}")

    def test_material_chain_integrity(self):
        """Test EXACT material integrity of the whole chain: DAE → materials.json → output directory."""
        logger.info("\n[3b] Testing material chain integrity (DAE → JSON → output)...")

        materials_json = self.beamng_dir / MAIN_DIR / MATERIALS_JSON

        if not materials_json.exists():
            self.error(f"materials.json not found: {materials_json}")
            return

        try:
            with open(materials_json, "r", encoding="utf-8") as f:
                materials = json.load(f)

            # === PHASE 1: Collect all material references from DAE files ===
            logger.info("\n  [Phase 1] Collecting material references from DAEs...")

            # Terrain DAE
            terrain_daes = list(self.shapes_dir.glob(TERRAIN_DAE_PATTERN))
            dae_material_refs = {}  # material_name -> [dae_files, count]

            for terrain_dae in terrain_daes:
                try:
                    tree = ET.parse(terrain_dae)
                    root = tree.getroot()
                    ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

                    # Extract material references from triangles
                    triangles = root.findall(".//collada:triangles", ns)
                    polylists = root.findall(".//collada:polylist", ns)

                    for tri in triangles:
                        mat_ref = tri.get("material")
                        if mat_ref:
                            if mat_ref not in dae_material_refs:
                                dae_material_refs[mat_ref] = {"daes": set(), "count": 0}
                            dae_material_refs[mat_ref]["daes"].add(terrain_dae.name)
                            count = int(tri.get("count", 0))
                            dae_material_refs[mat_ref]["count"] += count

                    for poly in polylists:
                        mat_ref = poly.get("material")
                        if mat_ref:
                            if mat_ref not in dae_material_refs:
                                dae_material_refs[mat_ref] = {"daes": set(), "count": 0}
                            dae_material_refs[mat_ref]["daes"].add(terrain_dae.name)
                            count = int(poly.get("count", 0))
                            dae_material_refs[mat_ref]["count"] += count
                except Exception as e:
                    self.warning(f"Could not parse {terrain_dae.name}: {e}")
                    continue

            # Building DAEs
            building_daes = list(self.buildings_dir.glob("*.dae"))
            for building_dae in building_daes:
                try:
                    tree = ET.parse(building_dae)
                    root = tree.getroot()
                    ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

                    # Extract material references
                    triangles = root.findall(".//collada:triangles", ns)
                    polylists = root.findall(".//collada:polylist", ns)

                    for tri in triangles:
                        mat_ref = tri.get("material")
                        if mat_ref:
                            if mat_ref not in dae_material_refs:
                                dae_material_refs[mat_ref] = {"daes": set(), "count": 0}
                            dae_material_refs[mat_ref]["daes"].add(building_dae.name)
                            count = int(tri.get("count", 0))
                            dae_material_refs[mat_ref]["count"] += count

                    for poly in polylists:
                        mat_ref = poly.get("material")
                        if mat_ref:
                            if mat_ref not in dae_material_refs:
                                dae_material_refs[mat_ref] = {"daes": set(), "count": 0}
                            dae_material_refs[mat_ref]["daes"].add(building_dae.name)
                            count = int(poly.get("count", 0))
                            dae_material_refs[mat_ref]["count"] += count
                except Exception as e:
                    self.warning(f"Could not parse {building_dae.name}: {e}")
                    continue

            if dae_material_refs:
                self.success(f"{len(dae_material_refs)} unique material references found in DAEs")
            else:
                self.warning("No material references found in DAEs")
                return

            # === PHASE 2: Validate that all DAE material references exist in JSON ===
            logger.info("\n  [Phase 2] Validating DAE material references in JSON...")

            missing_in_json = []
            material_counts = {}

            for mat_ref, ref_data in dae_material_refs.items():
                if mat_ref not in materials:
                    missing_in_json.append(
                        {"material": mat_ref, "daes": sorted(ref_data["daes"]), "faces": ref_data["count"]}
                    )
                else:
                    material_counts[mat_ref] = ref_data["count"]

            if missing_in_json:
                self.error(f"{len(missing_in_json)} material references from DAE not in materials.json:")
                for missing in missing_in_json[:10]:
                    self.error(
                        f"  - '{missing['material']}' ({missing['faces']} faces in {', '.join(missing['daes'][:2])})"
                    )
                if len(missing_in_json) > 10:
                    self.error(f"  ... and {len(missing_in_json)-10} more")
            else:
                self.success("✓ All DAE material references exist in materials.json")

            # === PHASE 3: Check material definitions ===
            logger.info("\n  [Phase 3] Validating material definitions...")

            invalid_definitions = []

            for mat_name, mat_def in materials.items():
                # Only check materials that are referenced in DAEs
                if (
                    mat_name not in dae_material_refs
                    and not mat_name.startswith("tile_")
                    and not mat_name.startswith("horizon_")
                ):
                    continue

                checks = {
                    "name": ("name" in mat_def),
                    "mapTo": ("mapTo" in mat_def),
                    "class": (mat_def.get("class") == "Material"),
                    "Stages": (
                        "Stages" in mat_def and isinstance(mat_def["Stages"], list) and len(mat_def["Stages"]) > 0
                    ),
                }

                missing = [k for k, v in checks.items() if not v]
                if missing:
                    invalid_definitions.append({"material": mat_name, "missing": missing})

            if invalid_definitions:
                self.error(f"{len(invalid_definitions)} material definitions are invalid:")
                for invalid in invalid_definitions[:5]:
                    self.error(f"  - '{invalid['material']}': missing {invalid['missing']}")
                if len(invalid_definitions) > 5:
                    self.error(f"  ... and {len(invalid_definitions)-5} more")
            else:
                self.success("✓ All material definitions are valid")

            # === PHASE 4: Check texture references in the output directory ===
            logger.info("\n  [Phase 4] Validating texture references in the output directory...")

            missing_textures = []
            texture_coverage = {}

            for mat_name, mat_def in materials.items():
                if "Stages" not in mat_def or not mat_def["Stages"]:
                    continue

                stage = mat_def["Stages"][0]

                # Check baseColorMap (BaseColorMap/Diffuse)
                if "baseColorMap" in stage:
                    tex_path = stage["baseColorMap"]
                    resolved_path = self._resolve_relative_path(tex_path)

                    if not resolved_path.exists():
                        missing_textures.append(
                            {
                                "material": mat_name,
                                "texture_type": "baseColorMap",
                                "path": tex_path,
                                "resolved": str(resolved_path),
                            }
                        )
                    else:
                        texture_coverage[f"{mat_name}_baseColorMap"] = {
                            "path": str(resolved_path),
                            "size_kb": resolved_path.stat().st_size / 1024,
                        }

                # Check normalMap
                if "normalMap" in stage:
                    tex_path = stage["normalMap"]
                    resolved_path = self._resolve_relative_path(tex_path)

                    if not resolved_path.exists():
                        missing_textures.append(
                            {
                                "material": mat_name,
                                "texture_type": "normalMap",
                                "path": tex_path,
                                "resolved": str(resolved_path),
                            }
                        )
                    else:
                        texture_coverage[f"{mat_name}_normalMap"] = {
                            "path": str(resolved_path),
                            "size_kb": resolved_path.stat().st_size / 1024,
                        }

                # Check roughnessMap
                if "roughnessMap" in stage:
                    tex_path = stage["roughnessMap"]
                    resolved_path = self._resolve_relative_path(tex_path)

                    if not resolved_path.exists():
                        missing_textures.append(
                            {
                                "material": mat_name,
                                "texture_type": "roughnessMap",
                                "path": tex_path,
                                "resolved": str(resolved_path),
                            }
                        )
                    else:
                        texture_coverage[f"{mat_name}_roughnessMap"] = {
                            "path": str(resolved_path),
                            "size_kb": resolved_path.stat().st_size / 1024,
                        }

                # Check ambientOcclusionMap
                if "ambientOcclusionMap" in stage:
                    tex_path = stage["ambientOcclusionMap"]
                    resolved_path = self._resolve_relative_path(tex_path)

                    if not resolved_path.exists():
                        missing_textures.append(
                            {
                                "material": mat_name,
                                "texture_type": "ambientOcclusionMap",
                                "path": tex_path,
                                "resolved": str(resolved_path),
                            }
                        )
                    else:
                        texture_coverage[f"{mat_name}_ambientOcclusionMap"] = {
                            "path": str(resolved_path),
                            "size_kb": resolved_path.stat().st_size / 1024,
                        }

            if missing_textures:
                self.error(f"{len(missing_textures)} referenced textures missing:")
                for missing in missing_textures[:5]:
                    self.error(f"  - Material '{missing['material']}' ({missing['texture_type']}): {missing['path']}")
                if len(missing_textures) > 5:
                    self.error(f"  ... and {len(missing_textures)-5} more")
            else:
                self.success(f"✓ All {len(texture_coverage)} texture references exist in the output directory")

            # === PHASE 5: Statistics and summary ===
            logger.info("\n  [Phase 5] Material chain statistics:")

            # Group materials by type
            tile_materials = {m: c for m, c in material_counts.items() if m.startswith(MATERIAL_PREFIX_TERRAIN_TILE)}
            terrain_materials = {m: c for m, c in material_counts.items() if m.startswith(MATERIAL_PREFIX_ROAD)}
            lod2_materials = {m: c for m, c in material_counts.items() if m.startswith(MATERIAL_PREFIX_LOD2)}
            other_materials = {
                m: c
                for m, c in material_counts.items()
                if m not in tile_materials and m not in terrain_materials and m not in lod2_materials
            }

            if tile_materials:
                total_tile_faces = sum(tile_materials.values())
                logger.info(f"    • Terrain tiles: {len(tile_materials)} materials, {total_tile_faces:,} faces")

            if terrain_materials:
                total_street_faces = sum(terrain_materials.values())
                logger.info(f"    • Roads: {len(terrain_materials)} materials, {total_street_faces:,} faces")

            if lod2_materials:
                total_lod2_faces = sum(lod2_materials.values())
                logger.info(f"    • Buildings (LoD2): {len(lod2_materials)} materials, {total_lod2_faces:,} faces")

            if other_materials:
                total_other_faces = sum(other_materials.values())
                logger.info(f"    • Other: {len(other_materials)} materials, {total_other_faces:,} faces")

            total_faces = sum(material_counts.values())
            total_textures = len(texture_coverage)
            logger.info(
                f"\n    [SUMMARY] {len(material_counts
)} referenced materials, {total_faces:,} faces, {total_textures} textures"
            )

            # === FINAL: Determine integrity status ===
            if not missing_in_json and not missing_textures and not invalid_definitions:
                self.success("✓✓✓ Material chain is FULLY INTACT")
            elif not missing_in_json and not invalid_definitions:
                self.warning(f"Material chain is SATISFACTORY ({len(missing_textures)} texture issues)")
            else:
                self.error("Material chain has CRITICAL ERRORS")

        except json.JSONDecodeError as e:
            self.error(f"JSON parse error in materials.json: {e}")
        except Exception as e:
            self.error(f"Error during material chain test: {e}")

    def test_road_materials_debug(self):
        """Debug road/other materials - analyze what is used in the DAE vs. materials.json."""
        logger.debug("\n[3c] DEBUG: Material rendering analysis...")

        materials_json = self.beamng_dir / MAIN_DIR / MATERIALS_JSON

        if not materials_json.exists():
            self.warning(f"materials.json not found: {materials_json}")
            return

        try:
            with open(materials_json, "r", encoding="utf-8") as f:
                materials = json.load(f)

            # === PHASE 1: Collect material categories from DAE ===
            logger.info("\n  [DAE material categorization]")

            terrain_mats = set(
                m for m in self.dae_materials | self.triangle_materials if m.startswith(MATERIAL_PREFIX_TERRAIN_TILE)
            )
            lod2_mats = set(
                m for m in self.dae_materials | self.triangle_materials if m.startswith(MATERIAL_PREFIX_LOD2)
            )

            # All remaining materials = roads/miscellaneous
            other_mats = (self.dae_materials | self.triangle_materials) - terrain_mats - lod2_mats

            self.success(f"{len(terrain_mats)} terrain tile materials")
            self.success(f"{len(lod2_mats)} LoD2 building materials")
            self.success(f"{len(other_mats)} other materials (roads/misc)")

            if not other_mats:
                self.warning("No road/other materials found")
                return

            # === PHASE 2: Analyze the "other" materials ===
            logger.info("\n  [Other material details (e.g. roads)]")

            for mat_name in sorted(other_mats)[:10]:  # Show first 10
                if mat_name not in materials:
                    self.error(f"Material '{mat_name}' used in DAE but NOT in materials.json!")
                    continue

                mat_def = materials[mat_name]
                logger.info(f"\n    Material: {mat_name}")
                logger.info(f"      mapTo: {mat_def.get('mapTo', 'MISSING')}")
                logger.info(f"      class: {mat_def.get('class', 'MISSING')}")

                # Check stages
                stages = mat_def.get("Stages", [])
                if stages:
                    stage = stages[0]
                    logger.info(f"      Stages[0]:")

                    # Check all texture fields (colorMap/baseColorMap are interchangeable)
                    tex_fields = ["colorMap", "baseColorMap", "normalMap", "metallicMap", "roughnessMap"]
                    has_textures = False
                    for tex_field in tex_fields:
                        if tex_field in stage:
                            tex_path = stage[tex_field]
                            resolved = self._resolve_relative_path(tex_path)
                            exists = "✓" if resolved.exists() else "✗"
                            logger.info(f"        {tex_field}: {exists} {tex_path}")
                            if resolved.exists():
                                has_textures = True
                            if not resolved.exists():
                                logger.info(f"                 ✗ MISSING: {resolved}")

                    if not has_textures:
                        self.error(f"    ⚠️  Material '{mat_name}': NO textures defined!")
                else:
                    self.error(f"    ⚠️  Material '{mat_name}': No stages defined!")

            if len(other_mats) > 10:
                logger.info(f"\n    ... and {len(other_mats)-10} more")

            # === PHASE 3: Texture path validation ===
            logger.info("\n  [Texture path validation]")

            texture_issues = []
            valid_textures = 0

            for mat_name in other_mats:
                if mat_name not in materials:
                    continue

                mat_def = materials[mat_name]
                stages = mat_def.get("Stages", [])
                if not stages:
                    continue

                stage = stages[0]

                # Check colorMap or baseColorMap (CRITICAL - needed for rendering!)
                # BeamNG accepts both names for the surface color texture
                color_map = stage.get("colorMap", "") or stage.get("baseColorMap", "")
                if not color_map:
                    # colorMap or baseColorMap is REQUIRED for visible materials
                    texture_issues.append(
                        f"{mat_name}: ❌ CRITICAL: colorMap/baseColorMap MISSING - road will not be visible!"
                    )
                else:
                    resolved = self._resolve_relative_path(color_map)
                    if resolved.exists():
                        valid_textures += 1
                    else:
                        # Check: is it a valid path format (starts with /levels/ or \levels/ or levels/)?
                        if color_map.startswith(("\\levels\\", "/levels/", "levels/")):
                            # Valid path format - but the file does not exist
                            # _resolve_relative_path should handle the path resolution correctly
                            texture_issues.append(f"{mat_name}: Path format OK, but FILE MISSING: {resolved}")
                        else:
                            texture_issues.append(f"{mat_name}: Invalid path format: {color_map}")

            if texture_issues:
                self.error(f"{len(texture_issues)} texture problems in road/other materials:")
                for issue in texture_issues[:10]:
                    self.error(f"  - {issue}")
                if len(texture_issues) > 10:
                    self.error(f"  ... and {len(texture_issues)-10} more")

                # Analysis: are ALL colorMaps missing?
                colormap_missing_count = sum(
                    1
                    for issue in texture_issues
                    if "colorMap MISSING" in issue or "colorMap" in issue and "MISSING" in issue
                )
                if colormap_missing_count == len(texture_issues):
                    self.error("\n  ⚠️  ROOT CAUSE FOUND:")
                    self.error("      All road materials are defined WITHOUT colorMap!")
                    self.error("      → normalMap and roughnessMap are present")
                    self.error("      → BUT: colorMap (BaseColor) is missing entirely")
                    self.error("      → BeamNG cannot render a surface color without colorMap")
                    self.error("      → Hence the roads look wrong/grey")
                    self.error("\n      SOLUTION: All road materials must get a colorMap!")
            else:
                self.success(f"✓ All {valid_textures} road material textures are available")

        except json.JSONDecodeError as e:
            self.error(f"JSON parse error: {e}")
        except Exception as e:
            self.error(f"Error during material debug: {e}")

    def test_dds_metadata_and_pbr_shader(self):
        """Test DDS metadata and PBR shader definitions for road materials."""
        logger.info("\n[DDS & PBR] Testing DDS metadata and shader definitions...")

        materials_json = self.beamng_dir / MAIN_DIR / MATERIALS_JSON
        if not materials_json.exists():
            self.error(f"materials.json not found: {materials_json}")
            return

        try:
            with open(materials_json, "r", encoding="utf-8") as f:
                materials = json.load(f)

            logger.info(f"\n  [Analyzing {len(materials)} materials]")

            dds_issues = []
            pbr_issues = []
            valid_dds = []
            valid_pbr = []

            for mat_name, mat_def in materials.items():
                # Filter only road materials
                if mat_name.startswith(MATERIAL_PREFIX_TERRAIN_TILE) or mat_name.startswith(MATERIAL_PREFIX_LOD2):
                    continue

                logger.info(f"\n    Material: {mat_name}")

                # === CHECK PBR SHADER ===
                shader_type = mat_def.get("shader", "unknown")
                logger.info(f"      Shader: {shader_type}", end="")

                if shader_type == "PBR":
                    logger.info(" ✓")
                    valid_pbr.append(mat_name)
                else:
                    logger.info(" ✗ (should be PBR!)")
                    pbr_issues.append(f"{mat_name}: Shader is '{shader_type}' (should be PBR)")

                # === CHECK DDS METADATA ===
                if "Stages" not in mat_def or not mat_def["Stages"]:
                    continue

                stage = mat_def["Stages"][0]

                # Check colorMap
                if "colorMap" in stage:
                    tex_path = stage["colorMap"]
                    resolved_path = self._resolve_relative_path(tex_path)

                    if resolved_path.exists() and resolved_path.suffix.lower() == ".dds":
                        try:
                            dds_info = self._read_dds_header(resolved_path)
                            logger.info(f"        colorMap: {resolved_path.name}")
                            logger.info(f"          Format: {dds_info.get('format', 'unknown')}")
                            logger.info(f"          Size: {dds_info.get('width', '?')}x{dds_info.get('height', '?')}")
                            logger.info(f"          Mipmaps: {dds_info.get('mipmaps', 0)}")

                            # Validate format
                            if dds_info.get("format") in ["DXT1", "DXT3", "DXT5", "BC4", "BC5", "BC6H", "BC7"]:
                                valid_dds.append(f"{mat_name}_colorMap")
                                logger.info(f"          ✓ Format OK")
                            else:
                                dds_issues.append(
                                    f"{mat_name}: colorMap has unexpected format '{dds_info.get('format')}'"
                                )
                                logger.info(f"          ⚠️  Unexpected format!")

                        except Exception as e:
                            dds_issues.append(f"{mat_name}: Error reading colorMap - {str(e)}")
                            logger.error(f"          ✗ Error: {e}")

                # Check normalMap
                if "normalMap" in stage:
                    tex_path = stage["normalMap"]
                    resolved_path = self._resolve_relative_path(tex_path)

                    if resolved_path.exists() and resolved_path.suffix.lower() == ".dds":
                        try:
                            dds_info = self._read_dds_header(resolved_path)
                            logger.info(f"        normalMap: {resolved_path.name}")
                            logger.info(f"          Format: {dds_info.get('format', 'unknown')}")
                            logger.info(f"          Size: {dds_info.get('width', '?')}x{dds_info.get('height', '?')}")
                            logger.info(f"          ✓ Format OK")
                            valid_dds.append(f"{mat_name}_normalMap")
                        except Exception as e:
                            dds_issues.append(f"{mat_name}: Error reading normalMap - {str(e)}")
                            logger.error(f"          ✗ Error: {e}")

            # === SUMMARY ===
            logger.info(f"\n  [Summary]")
            logger.info(f"    ✓ {len(valid_pbr)} materials with PBR shader")
            logger.info(f"    ✓ {len(valid_dds)} DDS textures validated")

            if pbr_issues:
                self.error(f"  ✗ {len(pbr_issues)} PBR shader problems:")
                for issue in pbr_issues[:5]:
                    self.error(f"      - {issue}")
                if len(pbr_issues) > 5:
                    self.error(f"      ... and {len(pbr_issues)-5} more")

            if dds_issues:
                self.error(f"  ✗ {len(dds_issues)} DDS metadata problems:")
                for issue in dds_issues[:5]:
                    self.error(f"      - {issue}")
                if len(dds_issues) > 5:
                    self.error(f"      ... and {len(dds_issues)-5} more")

            if not pbr_issues and not dds_issues:
                self.success(f"  ✓ All DDS metadata and PBR definitions are correct!")

        except json.JSONDecodeError as e:
            self.error(f"JSON parse error: {e}")
        except Exception as e:
            self.error(f"Error during DDS/PBR test: {e}")

    def _read_dds_header(self, dds_path: Path):
        """Read the DDS header and extract metadata."""
        with open(dds_path, "rb") as f:
            # DDS-Magic "DDS "
            magic = f.read(4)
            if magic != b"DDS ":
                raise ValueError("Invalid DDS format (magic is not 'DDS ')")

            # DWORD dwSize (always 124)
            dword_size = int.from_bytes(f.read(4), "little")

            # Flags
            f.seek(8)
            flags = int.from_bytes(f.read(4), "little")

            # Height and width
            height = int.from_bytes(f.read(4), "little")
            width = int.from_bytes(f.read(4), "little")

            # Pitch
            pitch = int.from_bytes(f.read(4), "little")

            # Depth
            depth = int.from_bytes(f.read(4), "little")

            # Mipmaps
            mipmaps = int.from_bytes(f.read(4), "little")

            # Skip reserved bytes
            f.seek(32 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 10 * 4)

            # Read PixelFormat struct
            pf_size = int.from_bytes(f.read(4), "little")
            pf_flags = int.from_bytes(f.read(4), "little")
            fourcc = f.read(4)

            # Determine format from FourCC
            format_map = {
                b"DXT1": "DXT1",
                b"DXT3": "DXT3",
                b"DXT5": "DXT5",
                b"BC4U": "BC4",
                b"BC4S": "BC4",
                b"BC5U": "BC5",
                b"BC5S": "BC5",
                b"BC6H": "BC6H",
                b"BC7 ": "BC7",
            }

            dds_format = format_map.get(fourcc, fourcc.decode("utf-8", errors="ignore").strip())

            return {
                "width": width,
                "height": height,
                "mipmaps": mipmaps,
                "format": dds_format,
                "depth": depth,
            }

    def test_road_material_binding_uv(self):
        """Test material binding and UV mapping for road geometries in the DAE."""
        logger.debug("\n[3d] DEBUG: Road material binding and UV mapping...")

        # Find terrain DAEs
        terrain_daes = list(self.shapes_dir.glob(TERRAIN_DAE_PATTERN))

        if not terrain_daes:
            self.warning(f"No {TERRAIN_DAE_PATTERN} found")
            return

        logger.info(f"\n  [Analyze {len(terrain_daes)} terrain DAEs]")

        for terrain_dae in terrain_daes:
            try:
                tree = ET.parse(terrain_dae)
                root = tree.getroot()
                ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

                # === PHASE 1: Extract geometries and their material references ===
                geometries = root.findall(".//collada:geometry", ns)

                road_geometries = []  # Geometries with road materials

                for geom in geometries:
                    geom_name = geom.get("name", "unknown")
                    geom_id = geom.get("id", "unknown")

                    # Find primitives (triangles/polylist) in this geometry
                    primitives = geom.findall(".//collada:triangles", ns) + geom.findall(".//collada:polylist", ns)

                    for prim in primitives:
                        mat_ref = prim.get("material")

                        # Filter only road materials (not tile_* and not lod2_*)
                        if (
                            mat_ref
                            and not mat_ref.startswith(MATERIAL_PREFIX_TERRAIN_TILE)
                            and not mat_ref.startswith(MATERIAL_PREFIX_LOD2)
                        ):
                            # This is a road or another material
                            road_geometries.append(
                                {
                                    "geom_name": geom_name,
                                    "geom_id": geom_id,
                                    "material": mat_ref,
                                    "primitive": prim,
                                    "count": int(prim.get("count", 0)),
                                }
                            )

                if not road_geometries:
                    self.warning(f"  {terrain_dae.name}: No road geometries found")
                    continue

                self.success(f"  {terrain_dae.name}: {len(road_geometries)} road geometries found")

                # === PHASE 2: Check material binding and UV mapping ===
                logger.info(f"\n    [Material binding analysis] {len(road_geometries)} road geometries:")

                binding_issues = []
                uv_issues = []

                for road_geom in road_geometries[:5]:  # Show first 5
                    mat_name = road_geom["material"]
                    geom_name = road_geom["geom_name"]

                    logger.info(f"\n      Geometrie: {geom_name}")
                    logger.info(f"        Material ref: {mat_name}")
                    logger.info(f"        Faces: {road_geom['count']}")

                    # Check whether the material exists in materials.json
                    materials_json = self.beamng_dir / MAIN_DIR / MATERIALS_JSON
                    if materials_json.exists():
                        with open(materials_json, "r", encoding="utf-8") as f:
                            materials = json.load(f)

                        if mat_name in materials:
                            logger.info(f"        ✓ Material '{mat_name}' found in materials.json")
                        else:
                            binding_issues.append(f"{geom_name}: Material '{mat_name}' NOT in materials.json")
                            logger.info(f"        ✗ Material '{mat_name}' NOT in materials.json!")

                    # === Check UV mapping in this geometry ===
                    prim = road_geom["primitive"]

                    # Search inputs for this primitive
                    inputs = prim.findall(".//collada:input", ns)
                    input_sources = {}

                    for inp in inputs:
                        semantic = inp.get("semantic")
                        source_ref = inp.get("source", "").lstrip("#")
                        input_sources[semantic] = source_ref

                    has_position = "POSITION" in input_sources
                    has_normal = "NORMAL" in input_sources
                    has_texcoord = "TEXCOORD" in input_sources or any("TEXCOORD" in s for s in input_sources.keys())

                    logger.info(
                        f"        Inputs: POSITION={has_position}, NORMAL={has_normal}, TEXCOORD={has_texcoord}"
                    )

                    if not has_texcoord:
                        uv_issues.append(f"{geom_name}: ✗ NO texture coordinates (TEXCOORD) defined!")
                        logger.info(f"        ⚠️  WARNING: No UV coordinates present!")
                    else:
                        # Try to extract texcoord data
                        texcoord_source = next(
                            (s for s in input_sources.values() if "TEXCOORD" in s or "uv" in s.lower()), None
                        )
                        if texcoord_source:
                            logger.info(f"        ✓ Texcoord source: {texcoord_source}")

                if len(road_geometries) > 5:
                    logger.info(f"\n      ... and {len(road_geometries)-5} more")

                # === PHASE 3: Summary ===
                logger.info(f"\n    [Binding Issues] {len(binding_issues)}")
                for issue in binding_issues:
                    self.error(f"      - {issue}")

                if uv_issues:
                    self.error(f"    [UV-Mapping Issues] {len(uv_issues)}:")
                    for issue in uv_issues:
                        self.error(f"      - {issue}")
                        self.error("        → Reason: textures cannot be mapped correctly!")
                else:
                    self.success(f"    ✓ All road geometries have UV coordinates")

                if not binding_issues and not uv_issues:
                    self.success(f"    ✓ Material binding and UV mapping OK for {len(road_geometries)} geometries")

            except ET.ParseError as e:
                self.error(f"  {terrain_dae.name}: XML parse error - {e}")
            except Exception as e:
                self.error(f"  {terrain_dae.name}: Error - {e}")

    def test_items_json(self):
        """Test items.json integrity."""
        logger.info("\n[4] Testing items.json...")

        items_json = self.beamng_dir / Path(config.ITEMS_JSON)

        if not items_json.exists():
            self.error(f"items.json not found: {items_json}")
            return

        self.success(f"items.json found")

        try:
            with open(items_json, "r", encoding="utf-8") as f:
                items = json.load(f)

            self.success(f"{len(items)} items defined")

            # Check terrain item (dynamic: "terrain_<x>_<y>")
            terrain_items = [k for k in items.keys() if k.startswith(TERRAIN_ITEMS_PREFIX)]
            if terrain_items:
                terrain_item_name = terrain_items[0]
                terrain_item = items[terrain_item_name]
                self.success(f"Terrain mesh item present: {terrain_item_name}")
                if terrain_item.get("class") != "TSStatic":
                    self.error(f"Terrain item {terrain_item_name}: class != TSStatic")
                if terrain_item.get("collisionType") != "Visible Mesh Final":
                    self.warning(f"Terrain item {terrain_item_name}: collisionType != Visible Mesh Final")
            else:
                self.warning(f"No {TERRAIN_ITEMS_PREFIX}<x>_<y> item found")

            # Count building items
            building_items = [k for k in items.keys() if k.startswith(BUILDING_ITEMS_PREFIX)]
            if building_items:
                self.success(f"{len(building_items)} building tile items")
            else:
                self.warning("No building items found")

            # Check item structure and shape references
            # Only test real items: terrain items, building items, horizon item
            # Ignore meta fields such as "name", "class", "__metadata" etc. (fields that do not start with _ and are only lowercase)
            invalid_items = []
            missing_shapes = []

            # Collect real item names
            real_item_names = set()
            real_item_names.update(terrain_items)
            real_item_names.update(building_items)
            if HORIZON_ITEM_NAME in items:
                real_item_names.add(HORIZON_ITEM_NAME)

            # Only test real items
            for item_name, item_data in items.items():
                # Skip meta fields (fields without an uppercase letter at the start or special prefixes)
                if not isinstance(item_data, dict):
                    continue

                # Only test items with relevant names
                if item_name not in real_item_names:
                    # Also skip items with a lowercase letter at the start (meta fields)
                    if item_name[0].islower():
                        continue

                # Check required fields for real items
                if "__name" not in item_data:
                    invalid_items.append(f"{item_name}: missing '__name'")
                if "class" not in item_data:
                    invalid_items.append(f"{item_name}: missing 'class'")

                # shapeName is optional for some items (e.g. items without geometry)
                # but if present, then validate it
                if "shapeName" in item_data:
                    shape_name = item_data["shapeName"]
                    # Convert relative path to absolute Windows path
                    shape_file = self._resolve_relative_path(shape_name)

                    if not shape_file.exists():
                        missing_shapes.append(f"{item_name} → {shape_name} ({shape_file})")
                elif item_name in real_item_names and item_name != HORIZON_ITEM_NAME:
                    # Building and terrain items should have shapeName
                    if item_name.startswith(TERRAIN_ITEMS_PREFIX) or item_name.startswith(BUILDING_ITEMS_PREFIX):
                        invalid_items.append(f"{item_name}: missing 'shapeName'")

            if invalid_items:
                for err in invalid_items[:5]:
                    self.error(err)
                if len(invalid_items) > 5:
                    self.error(f"... and {len(invalid_items)-5} more")
            else:
                self.success("All items have valid structure")

            if missing_shapes:
                self.error(f"{len(missing_shapes)} referenced shapes missing:")
                for shape in missing_shapes[:5]:
                    self.error(f"  - {shape}")
                if len(missing_shapes) > 5:
                    self.error(f"  ... and {len(missing_shapes)-5} more")
            else:
                self.success("All referenced shapes exist")

        except json.JSONDecodeError as e:
            self.error(f"JSON parse error: {e}")
        except Exception as e:
            self.error(f"Error during test: {e}")

    def test_textures(self):
        """Test textures."""
        logger.info("\n[5] Testing textures...")

        if not self.textures_dir.exists():
            self.warning(f"Textures directory not found: {self.textures_dir}")
            return

        textures = list(self.textures_dir.glob(f"{TEXTURE_PREFIX_TILE}*.dds"))

        if len(textures) == 0:
            self.warning("No tile textures found")
            return

        self.success(f"{len(textures)} tile textures found")

        # Check file sizes
        empty_textures = []
        large_textures = []

        for tex in textures:
            size = tex.stat().st_size
            if size == 0:
                empty_textures.append(tex.name)
            elif size > TEXTURE_MAX_SIZE_MB * 1024 * 1024:
                large_textures.append(f"{tex.name} ({size/1024/1024:.1f}MB)")

        if empty_textures:
            self.error(f"{len(empty_textures)} empty textures:")
            for tex in empty_textures[:5]:
                self.error(f"  - {tex}")
        else:
            self.success("No empty textures")

        if large_textures:
            self.warning(f"{len(large_textures)} very large textures (>10MB):")
            for tex in large_textures[:5]:
                self.warning(f"  - {tex}")

        # Compute the total size
        total_size = sum(tex.stat().st_size for tex in textures)
        self.success(f"Total texture size: {total_size/1024/1024:.1f} MB")

    def _index_to_coords(self, tile_index_x, tile_index_y):
        """
        Convert tile indices to absolute coordinates.
        Index -2, -1, 0, 1 correspond to coordinates -1000, -500, 0, 500.
        """
        x_coord = tile_index_x * 500
        y_coord = tile_index_y * 500
        return (x_coord, y_coord)

    def test_texture_mapping(self):
        """Test texture mapping: DAE geometry names to texture files."""
        logger.info("\n[5b] Testing texture mapping (DAE <-> textures)...")

        # Load available texture files
        if not self.textures_dir.exists():
            self.warning("Textures directory not found, skipping mapping test")
            return

        texture_files = list(self.textures_dir.glob(f"{TEXTURE_PREFIX_TILE}*.dds"))
        texture_keys = set(f.stem for f in texture_files)

        if not texture_keys:
            self.warning("No tile textures present")
            return

        self.success(f"{len(texture_keys)} texture files present")

        # Test all terrain DAE files
        terrain_daes = list(self.shapes_dir.glob(TERRAIN_DAE_PATTERN))

        if not terrain_daes:
            self.warning(f"No {TERRAIN_DAE_PATTERN} found")
            return

        total_geometries = 0
        unmapped_geometries = []

        for dae_file in terrain_daes:
            try:
                parser = lxml_etree.XMLParser(huge_tree=True)
                tree = lxml_etree.parse(dae_file, parser)
                root = tree.getroot()
                ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

                # Extract all geometry names
                geometries = root.findall(".//collada:geometry", ns)

                for geometry in geometries:
                    geom_name = geometry.get("name", "unknown")
                    total_geometries += 1

                    # Convert geometry name to texture key
                    if geom_name.startswith(TEXTURE_PREFIX_TILE):
                        parts = geom_name.split("_")
                        if len(parts) == 3:  # "tile_X_Y"
                            try:
                                idx_x = int(parts[1])
                                idx_y = int(parts[2])
                                coords = self._index_to_coords(idx_x, idx_y)
                                texture_key = f"{TEXTURE_PREFIX_TILE}{coords[0]}_{coords[1]}"

                                if texture_key not in texture_keys:
                                    unmapped_geometries.append((dae_file.name, geom_name, texture_key))
                            except (ValueError, IndexError):
                                unmapped_geometries.append((dae_file.name, geom_name, "PARSE_ERROR"))

            except Exception as e:
                self.error(f"{dae_file.name}: Could not be parsed - {e}")
                continue

        if unmapped_geometries:
            self.error(f"{len(unmapped_geometries)}/{total_geometries} geometries have no textures:")
            for dae_name, geom_name, tex_key in unmapped_geometries[:10]:
                self.error(f"  {dae_name}: {geom_name} → {tex_key} [MISSING]")
            if len(unmapped_geometries) > 10:
                self.error(f"  ... and {len(unmapped_geometries)-10} more")
        else:
            self.success(f"All {total_geometries} geometries have textures")

    def test_xyz_normalization(self):
        """Test XYZ coordinate normalization of all objects."""
        logger.info("\n[7] Testing XYZ coordinate normalization...")

        import numpy as np
        from tools.dae_loader import load_dae_tile

        # === LOAD TERRAIN DAE ===
        terrain_dae = self.shapes_dir / TERRAIN_HORIZON_DAE.replace("_horizon", "")
        terrain_z_values = []

        if terrain_dae.exists():
            try:
                terrain_data = load_dae_tile(str(terrain_dae))
                terrain_vertices = terrain_data.get("vertices", np.array([]))

                if isinstance(terrain_vertices, np.ndarray) and len(terrain_vertices) > 0:
                    terrain_z_values = terrain_vertices[:, 2].tolist()
                    z_min, z_max = min(terrain_z_values), max(terrain_z_values)
                    z_mean = np.mean(terrain_z_values)

                    self.success(
                        f"Terrain: {len(terrain_z_values)} Vertices, Z=[{z_min:.2f}, {z_max:.2f}], M={z_mean:.2f}"
                    )
            except Exception as e:
                self.warning(f"Terrain DAE analysis: {e}")

        # === LOAD BUILDING DAEs ===
        building_daes = list(self.buildings_dir.glob("*.dae"))
        building_z_values = []
        building_z_ranges = {}

        if building_daes:
            logger.info(f"  [Buildings] Analyzing {len(building_daes)} DAE files...")

            for dae_file in building_daes:
                try:
                    dae_data = load_dae_tile(dae_file)
                    vertices = dae_data.get("vertices", np.array([]))

                    if isinstance(vertices, np.ndarray) and len(vertices) > 0:
                        z_coords = vertices[:, 2].tolist()
                        building_z_values.extend(z_coords)

                        z_min, z_max = min(z_coords), max(z_coords)
                        building_z_ranges[dae_file.name] = {"min": z_min, "max": z_max, "count": len(z_coords)}
                except Exception as e:
                    self.warning(f"{dae_file.name}: {e}")

            if building_z_values:
                z_min, z_max = min(building_z_values), max(building_z_values)
                z_mean = np.mean(building_z_values)

                self.success(
                    f"Buildings: {len(building_z_values)} Vertices, Z=[{z_min:.2f}, {z_max:.2f}], M={z_mean:.2f}"
                )

                # Detailed analysis
                logger.info(f"    Z range per building:")
                sorted_buildings = sorted(building_z_ranges.items(), key=lambda x: x[1]["min"])
                for name, zrange in sorted_buildings[:5]:
                    logger.info(f"      • {name}: [{zrange['min']:.2f}, {zrange['max']:.2f}] ({zrange['count']} Verts)")
                if len(building_z_ranges) > 5:
                    logger.info(f"      ... and {len(building_z_ranges)-5} more")

        # === CONSISTENCY CHECK ===
        logger.info("\n  [Consistency] Comparing Z coordinates between objects:")

        all_z_values = terrain_z_values + building_z_values

        if all_z_values:
            z_min, z_max = min(all_z_values), max(all_z_values)
            z_range = z_max - z_min

            logger.info(f"    • Total Z range: [{z_min:.2f}, {z_max:.2f}] (span: {z_range:.2f}m)")

            # Check whether buildings are positioned on the terrain
            if terrain_z_values and building_z_values:
                terrain_min = min(terrain_z_values)
                building_min = min(building_z_values)
                terrain_max = max(terrain_z_values)
                building_max = max(building_z_values)

                # Check whether the building base is within the terrain range or slightly above it
                # (it is normal for buildings to be somewhat higher than the terrain minimum,
                #  because they stand on variable terrain)
                overlap_min = max(terrain_min, building_min)
                overlap_max = min(terrain_max, building_max)

                if overlap_max > overlap_min:
                    self.success(f"Buildings are positioned on terrain (heights overlap)")
                    logger.info(f"      Terrain Z: [{terrain_min:.2f}, {terrain_max:.2f}]")
                    logger.info(f"      Buildings Z: [{building_min:.2f}, {building_max:.2f}]")
                    logger.info(f"      Overlap: [{overlap_min:.2f}, {overlap_max:.2f}]")
                else:
                    self.warning(f"Building heights and terrain heights do not overlap")

            # Check that the building base is not extremely far below the terrain
            if terrain_z_values and building_z_values:
                terrain_min = min(terrain_z_values)
                building_min = min(building_z_values)
                diff = building_min - terrain_min

                if diff >= -BUILDING_Z_TOLERANCE_M:  # Tolerance for foundations
                    self.success(f"Building base height relative to terrain: {diff:.2f}m")
                else:
                    self.error(f"Buildings too far below terrain: {diff:.2f}m")

            # Check whether the coordinate system is consistent (no wild outliers)
            if len(all_z_values) > 100:
                mean = np.mean(all_z_values)
                std = np.std(all_z_values)

                # Allow large variance (different terrain elevations are normal)
                # Only look for extreme outliers (e.g. 10σ)
                outlier_threshold_high = mean + Z_OUTLIER_SIGMA * std
                outlier_threshold_low = mean - Z_OUTLIER_SIGMA * std

                outliers_high = sum(1 for z in all_z_values if z > outlier_threshold_high)
                outliers_low = sum(1 for z in all_z_values if z < outlier_threshold_low)

                if outliers_high + outliers_low == 0:
                    self.success(f"No extreme Z coordinate outliers found")
                else:
                    self.warning(f"{outliers_high + outliers_low} extreme outliers (>M±{Z_OUTLIER_SIGMA}s) found")
        else:
            self.warning("No Z coordinates available for comparison")

    def test_horizon_dae(self):
        """Test terrain_horizon.dae integrity (only if a horizon item exists in items.json)."""
        logger.info("\n[Horizon] Testing terrain_horizon.dae...")

        # Skip the test if no horizon item is registered
        if not self.has_horizon_item:
            self.warning("No horizon item found in items.json - skipping horizon DAE test")
            return

        horizon_dae = self.shapes_dir / TERRAIN_HORIZON_DAE

        if not horizon_dae.exists():
            self.warning(f"{TERRAIN_HORIZON_DAE} not found - horizon layer not generated")
            return

        self.success(f"{TERRAIN_HORIZON_DAE} found")

        # Parse XML
        try:
            tree = ET.parse(str(horizon_dae))
            root = tree.getroot()

            # Definiere Namespace
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Check the COLLADA root
            if "COLLADA" not in root.tag:
                self.error("No valid COLLADA root element in horizon DAE")
                return

            self.success("Valid COLLADA XML in horizon DAE")

            # Count geometries
            geometries = root.findall(".//collada:geometry", ns)
            if len(geometries) == 0:
                self.error("No geometries found in horizon DAE")
                return
            else:
                self.success(f"{len(geometries)} Geometrie(n) in horizon DAE")

            # Check vertices
            sources = root.findall(".//collada:source", ns)
            if len(sources) < 2:  # At least vertices + UVs
                self.warning(f"Only {len(sources)} source(s) found (expected: vertices + UVs)")

            # Check that UV mapping is present
            uv_sources = [s for s in sources if "uv" in s.get("id", "").lower()]
            if uv_sources:
                self.success("UV mapping found in horizon DAE")
            else:
                self.warning("No UV sources found in horizon DAE")

            # Count faces (triangles/polylist)
            triangles = root.findall(".//collada:triangles", ns)
            polylists = root.findall(".//collada:polylist", ns)

            total_faces = len(triangles) + len(polylists)
            if total_faces == 0:
                self.error("No faces (triangles/polylist) in horizon DAE")
            else:
                self.success(f"{total_faces} face primitives in horizon DAE")

            # Extract vertex count from float_array count
            float_arrays = root.findall(".//collada:float_array", ns)
            for fa in float_arrays:
                count_str = fa.get("count", "")
                if count_str:
                    try:
                        count = int(count_str)
                        if "vertex" in fa.get("id", "").lower():
                            vertex_count = count // 3  # 3 coordinates per vertex
                            self.success(f"{vertex_count} Vertices in horizon DAE")
                        elif "uv" in fa.get("id", "").lower():
                            uv_count = count // 2  # 2 UV coordinates per vertex
                            self.success(f"{uv_count} UV coordinates in horizon DAE")
                    except ValueError:
                        pass

        except ET.ParseError as e:
            self.error(f"Error parsing horizon DAE: {e}")

    def test_horizon_materials(self):
        """Test the horizon material in materials.json (only if a horizon item exists in items.json)."""
        logger.info("\n[Horizon] Testing horizon material...")

        # Skip the test if no horizon item is registered
        if not self.has_horizon_item:
            self.warning("No horizon item found in items.json - skipping horizon material test")
            return

        materials_path = self.beamng_dir / MAIN_DIR / MATERIALS_JSON

        if not materials_path.exists():
            self.warning("materials.json not found")
            return

        try:
            with open(materials_path, "r") as f:
                materials = json.load(f)

            # Check the horizon_terrain material
            if HORIZON_MATERIAL not in materials:
                self.warning(f"{HORIZON_MATERIAL} material not in materials.json")
                return

            horizon_mat = materials[HORIZON_MATERIAL]
            self.success(f"{HORIZON_MATERIAL} material found")

            # Check required fields (except diffuseMap for phase 5)
            required_fields = ["name", "mapTo", "version"]
            for field in required_fields:
                if field in horizon_mat:
                    self.success(f"  - {field}: {horizon_mat[field]}")
                else:
                    self.warning(f"  - {field}: MISSING")

            # Check DDS texture
            diff_map = horizon_mat.get("diffuseMap", "")
            if not diff_map and "Stages" in horizon_mat and horizon_mat["Stages"]:
                diff_map = horizon_mat["Stages"][0].get("colorMap", "")

            if "horizon_sentinel2.dds" in diff_map or "horizon_sentinel2" in diff_map:
                # Convert relative path to absolute Windows path
                dds_path = self._resolve_relative_path(diff_map)
                if dds_path.exists():
                    self.success(f"DDS texture exists: {dds_path.name}")
                else:
                    self.error(f"DDS texture not found: {diff_map} -> {dds_path}")
            else:
                self.warning(f"Unexpected diffuseMap: {diff_map}")

        except json.JSONDecodeError as e:
            self.error(f"Error parsing materials.json: {e}")

    def test_horizon_items(self):
        """Test the horizon item in items.json (only if present)."""
        logger.info("\n[Horizon] Testing horizon item...")

        # Skip the test if no horizon item is registered
        if not self.has_horizon_item:
            self.warning("No horizon item found in items.json - skipping horizon item test")
            return

        try:
            items = self.items  # Check horizon item
            if HORIZON_ITEM_NAME not in items:
                self.warning(f"{HORIZON_ITEM_NAME} item not in items.json")
                return

            horizon_item = items[HORIZON_ITEM_NAME]
            self.success(f"{HORIZON_ITEM_NAME} item found")

            # Check required fields
            required_fields = {
                "__name": HORIZON_ITEM_NAME,
                "className": "TSStatic",
                "datablock": "DefaultStaticShape",
            }

            for field, expected_value in required_fields.items():
                if field in horizon_item:
                    actual = horizon_item[field]
                    if actual == expected_value:
                        self.success(f"  - {field}: {actual} [OK]")
                    else:
                        self.warning(f"  - {field}: {actual} (expected: {expected_value})")
                else:
                    self.warning(f"  - {field}: MISSING")

            # Check position (should be [0, 0, 0] for local)
            position = horizon_item.get("position", [])
            if position == [0, 0, 0]:
                self.success(f"  - position: {position} [OK] (local)")
            else:
                self.warning(f"  - position: {position} (expected: [0, 0, 0] for local coordinates)")

            # The field "rotation" must not be present: BeamNG uses it to tilt objects around the x axis (orient only via rotationMatrix)
            if "rotation" not in horizon_item:
                self.success("  - rotation: not set [OK]")
            else:
                self.warning(f"  - rotation: {horizon_item['rotation']} (remove field: tilts the object in BeamNG)")

            # Check scale (should be [1, 1, 1])
            scale = horizon_item.get("scale", [])
            if scale == [1, 1, 1]:
                self.success(f"  - scale: {scale} [OK]")
            else:
                self.warning(f"  - scale: {scale} (expected: [1, 1, 1])")

            # Check shapeName
            shape_name = horizon_item.get("shapeName", "")
            if TERRAIN_HORIZON_DAE in shape_name or "horizon" in shape_name.lower():
                # Convert relative path to absolute Windows path
                horizon_dae = self._resolve_relative_path(shape_name)
                if horizon_dae.exists():
                    self.success(f"  - shapeName: {shape_name} [OK]")
                else:
                    self.error(f"  - shapeName references non-existent DAE: {shape_name} ({horizon_dae})")
            else:
                self.warning(f"  - shapeName: {shape_name} (expected: {TERRAIN_HORIZON_DAE})")

            # Check meshCulling and originSort
            if horizon_item.get("meshCulling") == 0:
                self.success(f"  - meshCulling: 0 [OK]")
            if horizon_item.get("originSort") == 0:
                self.success(f"  - originSort: 0 [OK]")

        except json.JSONDecodeError as e:
            self.error(f"Error parsing items.json: {e}")

    def test_horizon_uv_mapping(self):
        """Test UV mapping plausibility in the horizon DAE (only if a horizon item exists)."""
        logger.info("\n[Horizon] Testing UV mapping plausibility...")

        # Skip the test if no horizon item is registered
        if not self.has_horizon_item:
            self.warning("No horizon item found in items.json - skipping horizon UV test")
            return

        horizon_dae = self.shapes_dir / TERRAIN_HORIZON_DAE

        if not horizon_dae.exists():
            self.warning("terrain_horizon.dae not found - skipping UV test")
            return

        try:
            tree = ET.parse(str(horizon_dae))
            root = tree.getroot()
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Extract UV coordinates
            float_arrays = root.findall(".//collada:float_array", ns)
            uv_values = []

            for fa in float_arrays:
                if "uv" in fa.get("id", "").lower():
                    text = fa.text.strip()
                    if text:
                        values = [float(v) for v in text.split()]
                        uv_values.extend(values)

            if not uv_values:
                self.warning("No UV values found in horizon DAE")
                return

            # Analyze UV range
            uv_min = min(uv_values)
            uv_max = max(uv_values)

            logger.debug(f"  [i] UV range: [{uv_min:.4f} .. {uv_max:.4f}]")

            # Check whether the UVs are in the expected range (with offset/scaling)
            # Normally they should be between -0.1 and 1.1 (with small offsets)
            if uv_min >= UV_RANGE_MIN and uv_max <= UV_RANGE_MAX:
                self.success(f"UV values in expected range [{uv_min:.4f}..{uv_max:.4f}]")
            else:
                self.warning(f"UV values outside expected range: [{uv_min:.4f}..{uv_max:.4f}]")

            # Check for valid UV density (there should be several distinct values)
            unique_uvs = len(set(round(v, 6) for v in uv_values))
            if unique_uvs > 1:
                self.success(f"{unique_uvs} unique UV values found")
            else:
                self.warning(f"Only {unique_uvs} unique UV value(s) - possible error in UV mapping")

        except ET.ParseError as e:
            self.error(f"Error during UV analysis in horizon DAE: {e}")

    def test_horizon_coordinates(self):
        """Test coordinate plausibility of the horizon mesh (only if a horizon item exists)."""
        logger.info("\n[Horizon] Testing coordinate plausibility...")

        # Skip the test if no horizon item is registered
        if not self.has_horizon_item:
            self.warning("No horizon item found in items.json - skipping coordinate test")
            return

        horizon_dae = self.shapes_dir / TERRAIN_HORIZON_DAE

        if not horizon_dae.exists():
            self.warning(f"{TERRAIN_HORIZON_DAE} not found - skipping coordinate test")
            return

        try:
            tree = ET.parse(str(horizon_dae))
            root = tree.getroot()
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Extract vertices
            float_arrays = root.findall(".//collada:float_array", ns)
            vertices = []

            for fa in float_arrays:
                if "vertices" in fa.get("id", "").lower():
                    text = fa.text.strip() if fa.text else ""
                    if text:
                        values = [float(v) for v in text.split()]
                        # 3 values per vertex (X, Y, Z)
                        for i in range(0, len(values), 3):
                            if i + 2 < len(values):
                                vertices.append((values[i], values[i + 1], values[i + 2]))

            if not vertices:
                self.warning("No vertices found in horizon DAE")
                return

            # Analyze coordinate ranges
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)

            logger.debug(f"  [i] Mesh bounds:")
            logger.info(f"      X: [{x_min:.0f}..{x_max:.0f}] ({x_max - x_min:.0f}m)")
            logger.info(f"      Y: [{y_min:.0f}..{y_max:.0f}] ({y_max - y_min:.0f}m)")
            logger.info(f"      Z: [{z_min:.0f}..{z_max:.0f}] ({z_max - z_min:.0f}m)")

            # Check whether the mesh is in local coordinates (should be around 0,0,0)
            mesh_center_x = (x_min + x_max) / 2
            mesh_center_y = (y_min + y_max) / 2

            if abs(mesh_center_x) < 100000 and abs(mesh_center_y) < 100000:
                self.success(f"Mesh center in local coordinates: ({mesh_center_x:.0f}, {mesh_center_y:.0f})")
            else:
                self.warning(f"Mesh center outside expected range: ({mesh_center_x:.0f}, {mesh_center_y:.0f})")

            # Check whether the height variance is plausible (should not be too small for a horizon)
            if z_max - z_min > 1.0:
                self.success(f"Height variance in mesh: {z_max - z_min:.2f}m")
            else:
                self.warning(f"Small height variance in horizon mesh: {z_max - z_min:.2f}m (flach?)")

        except ET.ParseError as e:
            self.error(f"Error parsing coordinates in horizon DAE: {e}")

    # test_all_face_winding_order() was removed:
    # Winding order is now guaranteed centrally in Mesh.add_face() (optimized, 14x faster)
    # See debug/test_performance_winding_order.py for a performance comparison

    # test_all_face_winding_order() was removed:
    # Winding order is now guaranteed centrally in Mesh.add_face() (optimized, 14x faster)
    # See debug/test_performance_winding_order.py for a performance comparison

    def run_all_tests(self):
        """Run all tests."""
        logger.info("=" * 60)
        logger.info("EXPORT INTEGRITY TEST")
        logger.info("=" * 60)
        logger.info(f"BeamNG directory: {self.beamng_dir}")

        self.test_terrain_dae()
        self.test_terrain_face_materials()
        self.test_building_daes()
        self.test_materials_json()
        self.test_material_chain_integrity()
        self.test_road_materials_debug()  # DEBUG: check road materials
        self.test_dds_metadata_and_pbr_shader()  # DEBUG: check DDS & PBR shader
        self.test_road_material_binding_uv()  # DEBUG: check material binding and UV mapping
        # Winding order is now guaranteed centrally in Mesh.add_face() (optimized, 14x faster)
        self.test_items_json()
        self.test_textures()
        self.test_texture_mapping()
        self.test_xyz_normalization()

        # Horizon-Tests
        self.test_horizon_dae()
        self.test_horizon_materials()
        self.test_horizon_items()
        self.test_horizon_uv_mapping()
        self.test_horizon_coordinates()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        if self.errors:
            logger.info(f"\n❌ {len(self.errors)} ERRORS found:")
            for err in self.errors:
                logger.info(f"  {err}")
        else:
            logger.error("\n✅ No critical errors!")

        if self.warnings:
            logger.info(f"\n⚠️  {len(self.warnings)} WARNINGS:")
            for warn in self.warnings:
                logger.info(f"  {warn}")
        else:
            logger.warning("\n✅ No warnings!")

        if not self.errors and not self.warnings:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 ALL TESTS PASSED! Export is valid.")
            logger.info("=" * 60)
            return 0
        elif not self.errors:
            logger.info("\n" + "=" * 60)
            logger.warning("[OK] Export is functional (with warnings).")
            logger.info("=" * 60)
            return 0
        else:
            logger.info("\n" + "=" * 60)
            logger.error("❌ Export has critical errors!")
            logger.info("=" * 60)
            return 1


def main():
    """Main function."""
    tester = ExportIntegrityTest()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
