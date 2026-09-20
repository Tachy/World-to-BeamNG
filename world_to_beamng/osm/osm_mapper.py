import json
import uuid
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


class OSMMapper:
    def __init__(self, config_path="osm_to_beamng.json"):
        """Lädt die Konfiguration für das Mapping."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Warnung: {config_path} nicht gefunden. Nutze leere Defaults.")
            self.config = {"highway_defaults": {}, "surface_overrides": {}, "surface_types": {}}

        self.defaults = self.config.get("highway_defaults", {})
        self.overrides = self.config.get("surface_overrides", {})
        self.surface_types = self.config.get("surface_types", {})
        self.forest_types = self.config.get("forest_types", {})
        self.forest_mappings = self.config.get("forest_mappings", {})

    def get_road_properties(self, tags):
        """
        Gibt ein Dictionary mit allen BeamNG-Parametern zurück.
        Neue Struktur:
        1. highway_defaults[highway-typ] → internal_name + width
        2. surface_types[internal_name] → Komplette Definition (priority, drivability, textures, groundModelName)
        3. surface_overrides[surface] oder [tracktype] → Optional: update internal_name

        Tracktype-Mapping:
        - grade1 → asphalt (beste Qualität)
        """
        if tags is None:
            tags = {}

        # 1. Hole Highway-Type Default (mit internal_name + width)
        hw_type = tags.get("highway", "unclassified")
        base_type = hw_type.split("_")[0]  # 'primary_link' -> 'primary'

        highway_entry = self.defaults.get(base_type, self.defaults.get(hw_type, self.defaults.get("unclassified", {})))

        if not highway_entry:
            # Fallback: verwende dirt_road
            highway_entry = {"width": 4.0, "internal_name": "dirt_road"}

        # Kopiere Highway-Entry
        props = highway_entry.copy()

        # 2. Hole Surface-Type Definition (mit priority, drivability, textures, groundModelName)
        internal_name = props.get("internal_name", "dirt_road")
        surface_type_def = self.surface_types.get(internal_name, {})

        # Merge: Surface-Type Definition (aber nicht width überschreiben)
        for key, value in surface_type_def.items():
            if key != "internal_name":  # internal_name sollte schon gesetzt sein
                props[key] = value

        # 3. Prüfe Surface-Override oder Tracktype-Override
        # Tracktype-Mapping (Werte sind Keys in surface_overrides):
        # grade1 → Asphalt, grade2 → Kies (befestigt), grade3-5 → Erdweg
        tracktype_mapping = {
            "grade1": "asphalt",  # Beste Qualität → asphalt
            "grade2": "gravel",  # Überwiegend fest → Kies
            "grade3": "dirt",  # Gemischt fest/weich → Erdweg
            "grade4": "dirt",  # Überwiegend weich → Erdweg
            "grade5": "dirt",  # Weich (Gras/Erde) → Erdweg
        }

        # Prüfe zuerst Surface-Tag
        surface = tags.get("surface")
        if surface in self.overrides:
            override_entry = self.overrides[surface]
            self._apply_surface_override(props, override_entry)
        else:
            # Falls kein Surface-Tag: Prüfe Tracktype-Tag
            tracktype = tags.get("tracktype")
            if tracktype in tracktype_mapping:
                mapped_surface = tracktype_mapping[tracktype]
                if mapped_surface in self.overrides:
                    override_entry = self.overrides[mapped_surface]
                    self._apply_surface_override(props, override_entry)

        # 4. Breite berechnen (könnte von tags überschrieben werden)
        props["width"] = self._calculate_width(tags, props.get("width", 4.0))

        return props

    def _apply_surface_override(self, props, override_entry):
        """
        Wendet einen Surface-Override an (aktualisiert internal_name und lädt surface_types).

        Args:
            props: Dictionary mit aktuellen Eigenschaften (wird in-place modifiziert)
            override_entry: Dict mit Override-Daten (z.B. {"internal_name": "dirt_road"})
        """
        # Update internal_name (wenn vorhanden)
        if "internal_name" in override_entry:
            props["internal_name"] = override_entry["internal_name"]
            # Hole neue Surface-Type Definition
            new_surface_def = self.surface_types.get(props["internal_name"], {})
            for key, value in new_surface_def.items():
                if key != "internal_name":
                    props[key] = value

    def get_building_properties(self, building_type="roof"):
        """
        Gibt die Gebäude-Material-Parameter aus der Config zurück.

        Args:
            building_type: "roof", "roof_edge" oder "roof_trim"

        Returns:
            Dict mit diffuseColor und je nach Typ textures bzw. roughnessFactor, metallicFactor.
            Die UVs sind metrisch (Putz- bzw. Dach-Wiederholung in Metern), es gibt keine Tiling-Skala.
        """
        return self.config.get("buildings", {}).get(building_type, {}).copy()

    def _calculate_width(self, tags, fallback_width):
        """Logik für die Breitenermittlung."""
        # A. Explizites width Tag
        if "width" in tags:
            try:
                # Entferne Einheiten wie 'm' und konvertiere zu float
                return float(str(tags["width"]).lower().replace("m", "").strip())
            except (ValueError, AttributeError):
                pass

        # B. Lanes Tag (3.25m pro Spur als Standard)
        if "lanes" in tags:
            try:
                return int(tags["lanes"]) * 3.25
            except (ValueError, TypeError):
                pass

        return fallback_width

    def generate_materials_json_entry(self, mat_name, props):
        """Erzeugt einen einzelnen Eintrag für die main.materials.json."""
        from .. import config

        tex = props.get("textures", {})

        # Fallback für fehlende Texturen - nutze einfache Farben
        stages_config = {
            "specularPower": 1.0,
            "pixelSpecular": True,
        }

        # Volle PBR-Textur-Stufe für DecalRoad-Materialien (verifiziert gegen
        # BeamNGs eigenes west_coast_usa/art/road/main.materials.json ->
        # "road_asphalt_2lane": baseColorMap+normalMap+roughnessMap+
        # ambientOcclusionMap+opacityMap ist dort der Standard, nicht nur
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
        # Optional: opacityFactor < 1 lässt das Terrain darunter durchscheinen
        # (BeamNGs "road_gravel" nutzt 0.721).
        if props.get("opacityFactor") is not None:
            stages_config["opacityFactor"] = props["opacityFactor"]

        # Fallback nur wenn Texturen-Keys nicht vorhanden sind
        if not any(k in stages_config for k in ["baseColorMap", "normalMap", "roughnessMap"]):
            # Verwende color aus props, falls vorhanden
            color = props.get("color", [0.5, 0.5, 0.5, 1.0])
            if len(color) == 3:
                color.append(1.0)
            stages_config["diffuseColor"] = color

        # groundType gehört auf TOP-LEVEL (nicht in Stages) und MUSS einer der
        # ~32 offiziellen, GROSSGESCHRIEBENEN Bezeichner aus BeamNGs eigener
        # art/groundmodels.json sein (z.B. "ASPHALT", "DIRT") - sonst greift
        # stillschweigend der ASPHALT-Fallback für Reifenphysik/-sound. Der
        # frühere Key "groundModelName" (kleingeschrieben) wurde von BeamNG
        # gar nicht ausgewertet.
        ground_type = str(props.get("groundModelName", "asphalt")).upper()
        annotation = "ASPHALT" if ground_type.startswith("ASPHALT") else "NATURE"

        return {
            "__name": mat_name,  # ← WICHTIG: __name für MaterialManager
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "version": 1.5,
            "groundType": ground_type,  # ← TOP-LEVEL (nicht in Stages)
            "Stages": [stages_config],
            # materialTag0="RoadAndPath": BeamNG erkennt nur das als Straßen-
            # Material für Traffic-KI/Navmesh. materialTag1/annotation nach
            # dem Schema von BeamNGs eigenen DecalRoad-Materialien
            # (west_coast_usa/art/road/main.materials.json).
            "materialTag0": "RoadAndPath",
            "materialTag1": "beamng",
            "annotation": annotation,
            # translucent/translucentZWrite: PFLICHT für DecalRoad-Materialien
            # (verifiziert gegen west_coast_usa "road_asphalt_2lane") - ohne
            # diese Flags kann BeamNG das Decal nicht korrekt auf die
            # Terrain-Oberfläche darunter verblenden.
            "translucent": True,
            "translucentZWrite": True,
            "persistentId": str(uuid.uuid4()),  # ← KRITISCH: BeamNG braucht eindeutige IDs für Material-Persistierung!
        }

    def is_forest(self, tags):
        """
        Prüft, ob ein OSM-Element ein Wald ist.

        Args:
            tags: Dictionary mit OSM-Tags

        Returns:
            bool: True wenn Element als Wald klassifiziert ist
        """
        if tags is None:
            return False

        # Prüfe gegen alle definierten Forest-Mappings
        for tag_key, tag_values_dict in self.forest_mappings.items():
            tag_value = tags.get(tag_key)
            if tag_value in tag_values_dict:
                forest_type = tag_values_dict[tag_value]
                return forest_type is not None and forest_type != "open_meadow"

        return False

    def get_forest_properties(self, tags):
        """
        Gibt Forest-Generierungs-Parameter basierend auf OSM-Tags zurück.

        Funktion analog zu get_road_properties(), aber für Waldgebiete:
        1. Finde Forest-Type aus forest_mappings[tag_key][tag_value]
        2. Hole Forest-Definition aus forest_types
        3. Merge mit optionalen Tag-Overrides

        Args:
            tags: Dictionary mit OSM-Tags

        Returns:
            Dict mit forest_type, tree_density, tree_species, etc.
            oder None wenn Element kein Wald ist
        """
        if tags is None:
            tags = {}

        # 1. Finde passenden Forest-Type
        forest_type = None

        for tag_key, tag_values_dict in self.forest_mappings.items():
            tag_value = tags.get(tag_key)
            if tag_value in tag_values_dict:
                forest_type = tag_values_dict[tag_value]
                break

        # Falls kein Forest-Type gefunden oder open_meadow: return None
        if forest_type is None or forest_type == "open_meadow":
            return None

        # 2. Hole Forest-Definition
        forest_def = self.forest_types.get(forest_type, {})

        # Kopiere Forest-Definition
        props = forest_def.copy()
        props["forest_type"] = forest_type

        # 3. Optionale Tag-Overrides
        # z.B. name, description aus OSM-Tags
        if "name" in tags:
            props["name"] = tags["name"]
        if "description" in tags:
            props["description"] = tags["description"]

        return props
