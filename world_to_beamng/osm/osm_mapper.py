import json
import os


class OSMMapper:
    def __init__(self, config_path="osm_to_beamng.json"):
        """Lädt die Konfiguration für das Mapping."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Warnung: {config_path} nicht gefunden. Nutze leere Defaults.")
            self.config = {"highway_defaults": {}, "surface_overrides": {}}

        self.defaults = self.config.get("highway_defaults", {})
        self.overrides = self.config.get("surface_overrides", {})

    def get_road_properties(self, tags):
        """
        Gibt ein Dictionary mit allen BeamNG-Parametern zurück.
        Kaskade: highway-type -> width-tag -> lanes-tag -> surface-override
        """
        if tags is None:
            tags = {}

        # 1. Basis-Werte über Highway-Typ (z.B. residential)
        hw_type = tags.get("highway", "unclassified")
        base_type = hw_type.split("_")[0]  # 'primary_link' -> 'primary'

        # Hole Default-Werte aus JSON (Fallback auf unclassified)
        props = self.defaults.get(base_type, self.defaults.get(hw_type, self.defaults.get("unclassified", {}))).copy()

        # 2. Breite berechnen
        props["width"] = self._calculate_width(tags, props.get("width", 4.0))

        # 3. Surface-Override (wenn z.B. surface=gravel auf einer residential road steht)
        surface = tags.get("surface")
        if surface in self.overrides:
            # Aktualisiere internal_name, groundModel, drivability und textures
            props.update(self.overrides[surface])

        return props

    def get_building_properties(self, building_type="wall"):
        """
        Gibt ein Dictionary mit allen Gebäude-Material-Parametern zurück.

        Args:
            building_type: "wall" oder "roof"

        Returns:
            Dict mit internal_name, groundModelName, textures, und tiling_scale (Wiederholung in Metern)
        """
        buildings_config = self.config.get("buildings", {})
        building_data = buildings_config.get(building_type, {})

        # Kopiere alle Daten aus Config
        props = building_data.copy()

        # Setze Tiling-Skala basierend auf Typ
        # Wände: 4m Wiederholung, Dächer: 2m Wiederholung
        if building_type == "wall":
            props["tiling_scale"] = 4.0  # Wände: alle 4 Meter wiederholen
        elif building_type == "roof":
            props["tiling_scale"] = 2.0  # Dächer: alle 2 Meter wiederholen
        else:
            props["tiling_scale"] = 1.0  # Default

        return props

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
        tex = props.get("textures", {})

        # Fallback für fehlende Texturen - nutze einfache Farben
        stages_config = {
            "useAnisotropic": True,
            "specularPower": 1.0,
            "pixelSpecular": True,
        }

        # Alle Textur-Typen hinzufügen, die in osm_to_beamng.json definiert sind
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

        # Fallback: Wenn keine Texturen, nutze einfache Farbe
        if not any(k in stages_config for k in ["baseColorMap", "normalMap", "roughnessMap"]):
            stages_config["colorMap"] = "0.5 0.5 0.5 1.0"  # Mittleres Grau

        return {
            "__name": mat_name,  # ← WICHTIG: __name für MaterialManager
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "version": 2,
            "Stages": [stages_config],
            "groundModelName": props.get("groundModelName", "asphalt"),
            "materialTag0": "beamng",
            "materialTag1": "italy",
        }
