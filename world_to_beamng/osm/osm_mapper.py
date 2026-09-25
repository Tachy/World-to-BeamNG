import json
import uuid
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


class OSMMapper:
    def __init__(self, config_path="osm_to_beamng.json"):
        """Loads the configuration for the mapping."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Warning: {config_path} not found. Using empty defaults.")
            self.config = {"highway_defaults": {}, "surface_overrides": {}, "surface_types": {}}

        self.defaults = self.config.get("highway_defaults", {})
        self.overrides = self.config.get("surface_overrides", {})
        self.surface_types = self.config.get("surface_types", {})
        self.road_markings = self.config.get("road_markings", {})
        self.forest_types = self.config.get("forest_types", {})
        self.forest_mappings = self.config.get("forest_mappings", {})

    def get_road_properties(self, tags):
        """
        Returns a dictionary with all BeamNG parameters.
        New structure:
        1. highway_defaults[highway type] → internal_name + width
        2. surface_types[internal_name] → complete definition (priority, drivability, textures, groundModelName)
        3. surface_overrides[surface] or [tracktype] → optional: update internal_name

        Tracktype mapping:
        - grade1 → asphalt (best quality)
        """
        if tags is None:
            tags = {}

        # 1. Get the highway type default (with internal_name + width)
        # Exact type first (e.g. 'primary_link' has its own single-lane default), only
        # then the base type before the underscore ('primary_foo' -> 'primary').
        hw_type = tags.get("highway", "unclassified")
        base_type = hw_type.split("_")[0]

        highway_entry = self.defaults.get(hw_type, self.defaults.get(base_type, self.defaults.get("unclassified", {})))

        if not highway_entry:
            # Fallback: use dirt_road
            highway_entry = {"width": 4.0, "internal_name": "dirt_road"}

        # Copy the highway entry
        props = highway_entry.copy()

        # 2. Get the surface type definition (with priority, drivability, textures, groundModelName)
        internal_name = props.get("internal_name", "dirt_road")
        surface_type_def = self.surface_types.get(internal_name, {})

        # Merge: surface type definition (but do not overwrite width)
        for key, value in surface_type_def.items():
            if key != "internal_name":  # internal_name should already be set
                props[key] = value

        # 3. Check the surface override or tracktype override
        # Tracktype mapping (values are keys in surface_overrides):
        # grade1 → asphalt, grade2 → gravel (compacted), grade3-5 → dirt track
        tracktype_mapping = {
            "grade1": "asphalt",  # Best quality → asphalt
            "grade2": "gravel",  # Mostly solid → gravel
            "grade3": "dirt",  # Mixed solid/soft → dirt track
            "grade4": "dirt",  # Mostly soft → dirt track
            "grade5": "dirt",  # Soft (grass/soil) → dirt track
        }

        # Check the surface tag first
        surface = tags.get("surface")
        if surface in self.overrides:
            override_entry = self.overrides[surface]
            self._apply_surface_override(props, override_entry)
        else:
            # If there is no surface tag: check the tracktype tag
            tracktype = tags.get("tracktype")
            if tracktype in tracktype_mapping:
                mapped_surface = tracktype_mapping[tracktype]
                if mapped_surface in self.overrides:
                    override_entry = self.overrides[mapped_surface]
                    self._apply_surface_override(props, override_entry)

        # 4. Compute the width (could be overridden by tags)
        props["width"] = self._calculate_width(tags, props.get("width", 4.0))

        return props

    def _apply_surface_override(self, props, override_entry):
        """
        Applies a surface override (updates internal_name and loads surface_types).

        Args:
            props: Dictionary with current properties (modified in place)
            override_entry: Dict with override data (e.g. {"internal_name": "dirt_road"})
        """
        # Update internal_name (if present)
        if "internal_name" in override_entry:
            props["internal_name"] = override_entry["internal_name"]
            # Get the new surface type definition
            new_surface_def = self.surface_types.get(props["internal_name"], {})
            for key, value in new_surface_def.items():
                if key != "internal_name":
                    props[key] = value

    def get_building_properties(self, building_type="roof"):
        """
        Returns the building material parameters from the config.

        Args:
            building_type: "roof", "roof_edge" or "roof_trim"

        Returns:
            Dict with diffuseColor and, depending on the type, textures or roughnessFactor, metallicFactor.
            The UVs are metric (plaster or roof repeat in meters), there is no tiling scale.
        """
        return self.config.get("buildings", {}).get(building_type, {}).copy()

    def _calculate_width(self, tags, fallback_width):
        """Logic for determining the width."""
        # A. Explicit width tag
        if "width" in tags:
            try:
                # Remove units such as 'm' and convert to float
                return float(str(tags["width"]).lower().replace("m", "").strip())
            except (ValueError, AttributeError):
                pass

        # B. Lanes tag (3.25 m per lane as default)
        if "lanes" in tags:
            try:
                return int(tags["lanes"]) * 3.25
            except (ValueError, TypeError):
                pass

        return fallback_width

    def generate_materials_json_entry(self, mat_name, props):
        """Creates a single entry for main.materials.json."""
        from .. import config

        tex = props.get("textures", {})

        # Fallback for missing textures - use plain colors
        stages_config = {
            "specularPower": 1.0,
            "pixelSpecular": True,
        }

        # Full PBR texture stage for DecalRoad materials (verified against
        # BeamNG's own west_coast_usa/art/road/main.materials.json ->
        # "road_asphalt_2lane": baseColorMap+normalMap+roughnessMap+
        # ambientOcclusionMap+opacityMap is the standard there, not just
        # baseColorMap).
        if tex.get("baseColorMap"):
            stages_config["baseColorMap"] = tex.get("baseColorMap")
        if tex.get("normalMap"):
            stages_config["normalMap"] = tex.get("normalMap")
        if tex.get("roughnessMap"):
            stages_config["roughnessMap"] = tex.get("roughnessMap")
        if tex.get("ambientOcclusionMap"):
            stages_config["ambientOcclusionMap"] = tex.get("ambientOcclusionMap")
        if tex.get("opacityMap"):
            stages_config["opacityMap"] = tex.get("opacityMap")
        # Optional: opacityFactor < 1 lets the terrain underneath show through
        # (BeamNG's "road_gravel" uses 0.721).
        if props.get("opacityFactor") is not None:
            stages_config["opacityFactor"] = props["opacityFactor"]

        # Fallback only if no texture keys are present
        if not any(k in stages_config for k in ["baseColorMap", "normalMap", "roughnessMap"]):
            # Use color from props if present
            color = props.get("color", [0.5, 0.5, 0.5, 1.0])
            if len(color) == 3:
                color.append(1.0)
            stages_config["diffuseColor"] = color

        # groundType belongs at TOP-LEVEL (not in Stages) and MUST be one of the
        # ~32 official, UPPERCASE identifiers from BeamNG's own
        # art/groundmodels.json (e.g. "ASPHALT", "DIRT") - otherwise the
        # ASPHALT fallback for tire physics/sound silently applies. The
        # earlier key "groundModelName" (lowercase) was not evaluated by
        # BeamNG at all.
        ground_type = str(props.get("groundModelName", "asphalt")).upper()
        annotation = "ASPHALT" if ground_type.startswith("ASPHALT") else "NATURE"

        return {
            "__name": mat_name,  # ← IMPORTANT: __name for MaterialManager
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "version": 1.5,
            "groundType": ground_type,  # ← TOP-LEVEL (not in Stages)
            "Stages": [stages_config],
            # materialTag0="RoadAndPath": BeamNG only recognizes that as a road
            # material for traffic AI/navmesh. materialTag1/annotation follow
            # the scheme of BeamNG's own DecalRoad materials
            # (west_coast_usa/art/road/main.materials.json).
            "materialTag0": "RoadAndPath",
            "materialTag1": "beamng",
            "annotation": annotation,
            # translucent/translucentZWrite: REQUIRED for DecalRoad materials
            # (verified against west_coast_usa "road_asphalt_2lane") - without
            # these flags BeamNG cannot correctly blend the decal onto the
            # terrain surface underneath.
            "translucent": True,
            "translucentZWrite": True,
            "persistentId": str(uuid.uuid4()),  # ← CRITICAL: BeamNG needs unique IDs for material persistence!
        }

    def generate_marking_material_entry(self, mat_name, props):
        """
        materials.json entry for a marking line (edge/center line). Schema like BeamNG's own `line_white` /
        `line_dashed_long` (west_coast_usa/art/road/main.materials.json): translucent with opacityMap, no shadows,
        annotation SOLID_LINE or DASHED_LINE. Unlike generate_materials_json_entry(), without "__name".
        """
        tex = props.get("textures", {})
        stage = {key: tex[key] for key in ("baseColorMap", "normalMap", "opacityMap") if tex.get(key)}
        return {
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "version": 1.5,
            "Stages": [stage],
            "annotation": props.get("annotation", "SOLID_LINE"),
            "alphaRef": 255,
            "castShadows": False,
            "materialTag0": "RoadAndPath",
            "materialTag1": "beamng",
            "translucent": True,
            "translucentZWrite": True,
            "persistentId": str(uuid.uuid4()),
        }

    def is_forest(self, tags):
        """
        Checks whether an OSM element is a forest.

        Args:
            tags: Dictionary with OSM tags

        Returns:
            bool: True if the element is classified as a forest
        """
        if tags is None:
            return False

        # Check against all defined forest mappings
        for tag_key, tag_values_dict in self.forest_mappings.items():
            tag_value = tags.get(tag_key)
            if tag_value in tag_values_dict:
                forest_type = tag_values_dict[tag_value]
                return forest_type is not None and forest_type != "open_meadow"

        return False

    def get_forest_properties(self, tags):
        """
        Returns forest generation parameters based on OSM tags.

        Function analogous to get_road_properties(), but for forest areas:
        1. Find the forest type from forest_mappings[tag_key][tag_value]
        2. Get the forest definition from forest_types
        3. Merge with optional tag overrides

        Args:
            tags: Dictionary with OSM tags

        Returns:
            Dict with forest_type, tree_density, tree_species, etc.
            or None if the element is not a forest
        """
        if tags is None:
            tags = {}

        # 1. Find the matching forest type
        forest_type = None

        for tag_key, tag_values_dict in self.forest_mappings.items():
            tag_value = tags.get(tag_key)
            if tag_value in tag_values_dict:
                forest_type = tag_values_dict[tag_value]
                break

        # If no forest type was found or open_meadow: return None
        if forest_type is None or forest_type == "open_meadow":
            return None

        # 2. Get the forest definition
        forest_def = self.forest_types.get(forest_type, {})

        # Copy the forest definition
        props = forest_def.copy()
        props["forest_type"] = forest_type

        # 3. Optional tag overrides
        # e.g. name, description from OSM tags
        if "name" in tags:
            props["name"] = tags["name"]
        if "description" in tags:
            props["description"] = tags["description"]

        return props
