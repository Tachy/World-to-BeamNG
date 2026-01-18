import json
import os
import uuid


class OSMMapper:
    def __init__(self, config_path="osm_to_beamng.json"):
        """Lädt die Konfiguration für das Mapping."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Warnung: {config_path} nicht gefunden. Nutze leere Defaults.")
            self.config = {"highway_defaults": {}, "surface_overrides": {}, "surface_types": {}}

        self.defaults = self.config.get("highway_defaults", {})
        self.overrides = self.config.get("surface_overrides", {})
        self.surface_types = self.config.get("surface_types", {})

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
        # Tracktype-Mapping: grade1 (bester Zustand) → asphalt
        tracktype_mapping = {
            "grade1": "asphalt",  # Beste Qualität → asphalt
            "grade2": "gravel",  # Mittlere Qualität → gravel
            "grade3": "gravel",  # Schlechtere Qualität → gravel
            "grade4": "gravel",  # Noch schlechter → gravel
            "grade5": "gravel",  # Schlechteste → gravel
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
        from .. import config

        tex = props.get("textures", {})

        # Fallback für fehlende Texturen - nutze einfache Farben
        stages_config = {
            #            "useAnisotropic": True,
            "specularPower": 1.0,
            "pixelSpecular": True,
        }

        # Texturen IMMER verwenden (wenn vorhanden)
        if tex.get("baseColorMap"):
            stages_config["baseColorMap"] = tex.get("baseColorMap")
        #        if tex.get("normalMap"):
        #            stages_config["normalMap"] = tex.get("normalMap")
        #        if tex.get("roughnessMap"):
        #            stages_config["roughnessMap"] = tex.get("roughnessMap")
        #        if tex.get("ambientOcclusionMap"):
        #            stages_config["ambientOcclusionMap"] = tex.get("ambientOcclusionMap")
        #        if tex.get("opacityMap"):
        #            stages_config["opacityMap"] = tex.get("opacityMap")

        # Fallback nur wenn Texturen-Keys nicht vorhanden sind
        if not any(k in stages_config for k in ["baseColorMap", "normalMap", "roughnessMap"]):
            # Verwende color aus props, falls vorhanden
            color = props.get("color", [0.5, 0.5, 0.5, 1.0])
            if len(color) == 3:
                color.append(1.0)
            stages_config["diffuseColor"] = color

        # groundModelName gehört auf TOP-LEVEL (nicht in Stages)!
        ground_model_name = props.get("groundModelName", "asphalt")

        return {
            "__name": mat_name,  # ← WICHTIG: __name für MaterialManager
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "version": 2,
            "groundModelName": ground_model_name,  # ← TOP-LEVEL (nicht in Stages)
            #            "shader": "PBR",  # ← WICHTIG: PBR-Shader für Textur-Rendering!
            "Stages": [stages_config],
            #            "materialTag0": "RoadAndPath",  # ← KRITISCH: BeamNG erkennt nur "RoadAndPath" als Straßen-Material!
            #            "materialTag1": "custom",  # Custom-Materialien mit BeamNG Standard-Texturen
            "persistentId": str(uuid.uuid4()),  # ← KRITISCH: BeamNG braucht eindeutige IDs für Material-Persistierung!
        }
