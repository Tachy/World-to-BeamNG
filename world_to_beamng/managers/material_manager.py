"""
MaterialManager - central management of all BeamNG materials.

Manages materials for:
- Terrain tiles (with textures)
- Roads (from OSM tags)
- Buildings (LoD2)
- Horizon layer
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


class MaterialManager:
    """
    Central management of all BeamNG materials (singleton).

    Features:
    - Automatic tracking of materials
    - Duplicate detection
    - JSON export/import
    - Material templates (terrain, road, building, horizon)
    - Merge support for multi-tile workflows
    - Singleton: only one instance per export (one materials.json)
    """

    _instance: Optional["MaterialManager"] = None

    def __init__(self, beamng_dir: str):
        """
        Private constructor - use get_instance() instead.

        Args:
            beamng_dir: Path to the BeamNG level directory
        """
        if MaterialManager._instance is not None:
            raise RuntimeError("MaterialManager is a singleton - use get_instance()")

        self.beamng_dir = Path(beamng_dir) # Convert to Path object
        self.materials: Dict[str, Dict[str, Any]] = {}
        self._templates = self._init_templates()
        self._config = self._load_config()  # Whole JSON for buildings, etc.

    def add_terrain_materials(self, entries: Dict[str, Dict]) -> None:
        """
        Registers TerrainMaterial entries (from
        terrain.terrain_materials.build_terrain_material_entries()) for the
        later materials.json export.

        Args:
            entries: {material_name: {...TerrainMaterial JSON...}}
        """
        for mat_name, mat_data in entries.items():
            self.materials[mat_name] = mat_data

    @classmethod
    def get_instance(cls, beamng_dir: Path = None) -> "MaterialManager":
        """
        Gets the singleton instance (creates it if needed).

        Args:
            beamng_dir: Path to the BeamNG level directory (first call only)

        Returns:
            MaterialManager singleton instance
        """
        if cls._instance is None:
            if beamng_dir is None: # Added check for None
                raise ValueError("beamng_dir must be provided for the first call to get_instance")
            cls._instance = cls.__new__(cls)
            cls._instance.beamng_dir = beamng_dir # Already a Path object
            cls._instance.materials = {}
            cls._instance._templates = cls._instance._init_templates()
            cls._instance._config = cls._instance._load_config()  # Load the whole JSON
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance (for a new export run)."""
        cls._instance = None

    def _init_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads material templates from data/material_templates.json (REQUIRED).

        Returns:
            Dict with template names and definitions

        Raises:
            FileNotFoundError: If data/material_templates.json does not exist
        """
        config_path = Path(__file__).parent.parent.parent / "data" / "material_templates.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Material templates not found: {config_path}\n"
                "The file data/material_templates.json is required.\n"
                "Make sure it is part of the repository."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                templates = config.get("templates", {})

                # Filter out: description, note, and other metadata
                cleaned_templates = {}
                for name, template_def in templates.items():
                    # Copy the template, remove meta fields
                    cleaned = {k: v for k, v in template_def.items() if k not in ("description", "note")}
                    cleaned_templates[name] = cleaned

                num_templates = len(cleaned_templates)
                logger.info(f"  [✓] Material templates loaded: {num_templates} from JSON")

                return cleaned_templates

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing {config_path}: {e}\n" "The JSON file is invalid.")
        except Exception as e:
            raise RuntimeError(f"Error loading {config_path}: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the complete material_templates.json configuration.

        This method loads the whole JSON (with buildings, version, description, etc).

        Returns:
            Dict with all configurations
        """
        config_path = Path(__file__).parent.parent.parent / "data" / "material_templates.json"

        # Fallback: minimal config
        default_config = {
            "version": "1.0",
            "description": "Material Templates Configuration",
            "templates": {},
            "buildings": {
                "wall": {
                    "description": "Building wall (fallback)",
                    "template": "building_wall",
                    "tiling_scale": 4.0,
                    "material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"},
                },
                "roof": {
                    "description": "Building roof (fallback)",
                    "template": "building_roof",
                    "tiling_scale": 2.0,
                    "material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"},
                },
            },
        }

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"  [!] Error loading the config: {e}")
                return default_config

        return default_config

    def add_material(self, name: str, template: Optional[str] = None, overwrite: bool = False, **kwargs) -> bool:
        """
        Adds a material.

        Args:
            name: Material name (unique)
            template: Template name ("building_wall", "building_roof", "horizon") or None
            overwrite: Overwrite an existing material
            **kwargs: Additional/override properties

        Returns:
            True if the material was added, False if it already exists and overwrite=False
        """
        if name in self.materials and not overwrite:
            return False

        # Base: template or empty dict
        if template and template in self._templates:
            material = self._templates[template].copy()
            # Deep copy for nested dicts (Stages)
            if "Stages" in material:
                material["Stages"] = [stage.copy() for stage in material["Stages"]]
        else:
            material = {}

        # Set name and mapTo
        material["name"] = name
        material["mapTo"] = name
        material["persistentId"] = str(uuid.uuid4())

        # Merge kwargs (overrides template values)
        for key, value in kwargs.items():
            if key == "Stages" and "Stages" in material:
                # Merge Stages (extend the first stage)
                material["Stages"][0].update(value if isinstance(value, dict) else {})
            else:
                material[key] = value

        self.materials[name] = material
        return True

    def add_building_material(
        self,
        material_name: str,
        color: List[float] = None,
        textures: Dict[str, str] = None,
        tiling_scale: float = 1.0,
        overwrite: bool = False,
        stage_properties: Dict = None,
        **kwargs,
    ) -> str:
        """
        Adds a building material (convenience method).

        Args:
            material_name: Material name (e.g. "lod2_wall_plaster_white", "lod2_roof_red"); "wall" or "roof" in the name
                selects the template
            color: RGBA color [r, g, b, a] (0-1) - optional if textures are given
            textures: Dict with texture paths {baseColorMap, normalMap, roughnessMap} and optionally useAnisotropic
            tiling_scale: 1.0 = no repeat scale (UVs are metric); != 1.0 sets materialFactors
            overwrite: Overwrite an existing material
            stage_properties: Additional properties of the first stage (e.g. roughnessFactor, metallicFactor)
            **kwargs: Additional properties (groundType, materialTag0, etc.)

        Returns:
            Material name
        """
        # Determine the template based on the name
        if "wall" in material_name.lower():
            template = "building_wall"
        elif "roof" in material_name.lower():
            template = "building_roof"
        else:
            template = "building_wall"  # Default

        stages_config = {}

        # ALWAYS use textures if present
        if textures:
            if textures.get("baseColorMap"):
                stages_config["baseColorMap"] = textures["baseColorMap"]
            if textures.get("normalMap"):
                stages_config["normalMap"] = textures["normalMap"]
            if textures.get("roughnessMap"):
                stages_config["roughnessMap"] = textures["roughnessMap"]
            if textures.get("useAnisotropic"):
                stages_config["useAnisotropic"] = True  # Facades/roofs are viewed at a shallow angle

            # Color tint: tinting of the texture (combined with baseColorMap)
            if color:
                stages_config["diffuseColor"] = color
        elif color:
            stages_config["diffuseColor"] = color
        else:
            # Fallback: plain color if neither textures nor a color are given
            # Red for roof, white for wall
            if "roof" in material_name.lower():
                stages_config["diffuseColor"] = [0.6, 0.2, 0.1, 1.0]  # Red
            else:
                stages_config["diffuseColor"] = [0.9, 0.9, 0.9, 1.0]  # White

        if stage_properties:
            stages_config.update(stage_properties)

        # Add the tiling scale (for UV repetition)
        if tiling_scale != 1.0:
            stages_config["materialFactors"] = f"1 1 {tiling_scale} 1"  # e.g. "1 1 4.0 1" for a 4 m repeat

        self.add_material(material_name, template=template, overwrite=overwrite, Stages=stages_config, **kwargs)
        return material_name

    def add_horizon_material(self, texture_path: str, overwrite: bool = False) -> str:
        """
        Adds a horizon material (convenience method).

        Args:
            texture_path: Relative path to the horizon texture
            overwrite: Overwrite an existing material

        Returns:
            Material name
        """
        mat_name = "horizon_terrain"
        self.add_material(mat_name, template="horizon", overwrite=overwrite, Stages={"baseColorMap": texture_path})
        return mat_name

    def exists(self, name: str) -> bool:
        """
        Checks whether a material exists.

        Args:
            name: Material name

        Returns:
            True if the material exists
        """
        return name in self.materials

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Exports materials as a single JSON object (not JSONL).

        Args:
            filepath: Optional custom path, otherwise {beamng_dir}/main/materials.json (from config.MATERIALS_JSON)
        """
        if filepath is None:
            from .. import config

            filepath = self.beamng_dir / config.MATERIALS_JSON

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write as a single JSON object (indented for readability)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.materials, f, ensure_ascii=False, indent=2)

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Loads materials as a single JSON object.

        Args:
            filepath: Optional custom path, otherwise {beamng_dir}/main/materials.json (from config.MATERIALS_JSON)
        """
        if filepath is None:
            from .. import config

            filepath = self.beamng_dir / config.MATERIALS_JSON

        if not filepath.exists():
            return

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                materials_dict = json.load(f)
                # Convert to the internal structure
                self.materials = materials_dict if isinstance(materials_dict, dict) else {}
            except json.JSONDecodeError:
                self.materials = {}

    def clear(self) -> None:
        """Deletes all materials."""
        self.materials.clear()

    def get_templates(self) -> Dict[str, Any]:
        """
        Gets all configurations including material templates and the buildings section.

        Returns:
            Dict with template definitions, buildings config, etc.
        """
        return self._config.copy()

    def __len__(self) -> int:
        """Number of materials."""
        return len(self.materials)

    def __repr__(self) -> str:
        return f"MaterialManager({len(self.materials)} materials, singleton)"
