"""
MaterialManager - Zentrale Verwaltung aller BeamNG-Materialien.

Verwaltet Materials für:
- Terrain-Tiles (mit Texturen)
- Straßen (aus OSM-Tags)
- Gebäude (LoD2)
- Horizont-Layer
"""

import json
import os
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path


class MaterialManager:
    """
    Zentrale Verwaltung aller BeamNG-Materialien.

    Features:
    - Automatisches Tracking von Materialien
    - Duplikat-Erkennung
    - JSON Export/Import
    - Material-Templates (Terrain, Road, Building, Horizon)
    - Merge-Unterstützung für Multi-Tile-Workflows
    """

    def __init__(self, beamng_dir: str):
        """
        Initialisiere MaterialManager.

        Args:
            beamng_dir: Pfad zum BeamNG Level-Verzeichnis
        """
        self.beamng_dir = beamng_dir
        self.materials: Dict[str, Dict[str, Any]] = {}
        self._templates = self._init_templates()

    def _init_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialisiere Material-Templates.

        Returns:
            Dict mit Template-Namen und Default-Werten
        """
        return {
            "terrain": {
                "class": "Material",
                "version": 2,
                "Stages": [{"specularPower": 1, "pixelSpecular": True}],
                "groundModelName": "grass",
            },
            "road": {
                "class": "Material",
                "version": 2,
                "Stages": [{"specularPower": 1, "pixelSpecular": True}],
            },
            "building_wall": {
                "class": "Material",
                "version": 1.5,
                "Stages": [{"specularPower": 1, "pixelSpecular": True}],
                "groundType": "concrete",
                "materialTag0": "beamng",
                "materialTag1": "Building",
            },
            "building_roof": {
                "class": "Material",
                "version": 1.5,
                "Stages": [{"specularPower": 1, "pixelSpecular": True}],
                "groundType": "concrete",
                "materialTag0": "beamng",
                "materialTag1": "Building",
            },
            "horizon": {
                "class": "Material",
                "version": 1.5,
                "Stages": [{"specularPower": 16, "pixelSpecular": True}],
            },
        }

    def add_material(self, name: str, template: Optional[str] = None, overwrite: bool = False, **kwargs) -> bool:
        """
        Füge Material hinzu.

        Args:
            name: Material-Name (eindeutig)
            template: Template-Name ("terrain", "road", "building_wall", etc.) oder None
            overwrite: Überschreibe existierendes Material
            **kwargs: Zusätzliche/Override Properties

        Returns:
            True wenn Material hinzugefügt wurde, False wenn bereits vorhanden und overwrite=False
        """
        if name in self.materials and not overwrite:
            return False

        # Basis: Template oder leeres Dict
        if template and template in self._templates:
            material = self._templates[template].copy()
            # Deep copy für nested dicts (Stages)
            if "Stages" in material:
                material["Stages"] = [stage.copy() for stage in material["Stages"]]
        else:
            material = {}

        # Setze name und mapTo
        material["name"] = name
        material["mapTo"] = name
        material["persistentId"] = str(uuid.uuid4())

        # Merge kwargs (überschreibt Template-Werte)
        for key, value in kwargs.items():
            if key == "Stages" and "Stages" in material:
                # Merge Stages (erweitere erste Stage)
                material["Stages"][0].update(value if isinstance(value, dict) else {})
            else:
                material[key] = value

        self.materials[name] = material
        return True

    def add_terrain_material(self, tile_x: int, tile_y: int, texture_path: str, overwrite: bool = False) -> str:
        """
        Füge Terrain-Material für Tile hinzu (Convenience-Methode).

        Args:
            tile_x: Tile X-Koordinate (in Metern)
            tile_y: Tile Y-Koordinate (in Metern)
            texture_path: Relativer Pfad zur Textur (z.B. "/levels/.../textures/tile_0_0.dds")
            overwrite: Überschreibe existierendes Material

        Returns:
            Material-Name
        """
        mat_name = f"tile_{tile_x}_{tile_y}"
        self.add_material(mat_name, template="terrain", overwrite=overwrite, Stages={"baseColorMap": texture_path})
        return mat_name

    def add_road_material(self, road_type: str, properties: Dict[str, Any], overwrite: bool = False) -> str:
        """
        Füge Straßen-Material hinzu (Convenience-Methode).

        Args:
            road_type: Road-Typ (z.B. "road_residential", "road_motorway")
            properties: OSM-Properties-Dict (color, friction, textures, etc.)
            overwrite: Überschreibe existierendes Material

        Returns:
            Material-Name
        """
        mat_name = properties.get("internal_name", road_type)

        # Prüfe ob Texturen vorhanden sind
        textures = properties.get("textures", {})
        stages_dict = {}

        if textures:
            # Verwende Texturen (baseColorMap, normalMap, etc.)
            stages_dict.update(textures)
        else:
            # Fallback: Color
            color = properties.get("color", [0.5, 0.5, 0.5, 1.0])
            if len(color) == 3:
                color.append(1.0)  # Alpha hinzufügen
            stages_dict["diffuseColor"] = color

        self.add_material(
            mat_name,
            template="road",
            overwrite=overwrite,
            Stages=stages_dict,
            friction=properties.get("friction", 1.0),
            groundType=properties.get("groundType", "ASPHALT"),
        )
        return mat_name

    def add_building_material(
        self,
        material_name: str,
        color: List[float] = None,
        textures: Dict[str, str] = None,
        tiling_scale: float = 1.0,
        overwrite: bool = False,
        **kwargs,
    ) -> str:
        """
        Füge Gebäude-Material hinzu (Convenience-Methode).

        Args:
            material_name: Material-Name (z.B. "lod2_wall_white", "lod2_roof_red")
            color: RGBA Color [r, g, b, a] (0-1) - Optional wenn Texturen gegeben
            textures: Dict mit Textur-Pfaden {baseColorMap, normalMap, roughnessMap}
            tiling_scale: UV-Wiederholung in Metern (z.B. 4.0 = alle 4m wiederholen)
            overwrite: Überschreibe existierendes Material
            **kwargs: Zusätzliche Properties (groundType, materialTag0, etc.)

        Returns:
            Material-Name
        """
        # Bestimme Template basierend auf Namen
        if "wall" in material_name.lower():
            template = "building_wall"
        elif "roof" in material_name.lower():
            template = "building_roof"
        else:
            template = "building_wall"  # Default

        stages_config = {}

        # Texturen IMMER verwenden wenn vorhanden
        if textures:
            if textures.get("baseColorMap"):
                stages_config["baseColorMap"] = textures["baseColorMap"]
            if textures.get("normalMap"):
                stages_config["normalMap"] = textures["normalMap"]
            if textures.get("roughnessMap"):
                stages_config["roughnessMap"] = textures["roughnessMap"]
        elif color:
            stages_config["diffuseColor"] = color
        else:
            # Fallback: Einfache Farbe wenn keine Texturen und keine Farbe gegeben
            # Rot für Dach, Weiß für Wand
            if "roof" in material_name.lower():
                stages_config["diffuseColor"] = [0.6, 0.2, 0.1, 1.0]  # Rot
            else:
                stages_config["diffuseColor"] = [0.9, 0.9, 0.9, 1.0]  # Weiß

        # Tiling-Skala hinzufügen (für UV-Wiederholung)
        if tiling_scale != 1.0:
            stages_config["materialFactors"] = f"1 1 {tiling_scale} 1"  # z.B. "1 1 4.0 1" für 4m Wiederholung

        self.add_material(material_name, template=template, overwrite=overwrite, Stages=stages_config, **kwargs)
        return material_name

    def add_horizon_material(self, texture_path: str, overwrite: bool = False) -> str:
        """
        Füge Horizont-Material hinzu (Convenience-Methode).

        Args:
            texture_path: Relativer Pfad zur Horizont-Textur
            overwrite: Überschreibe existierendes Material

        Returns:
            Material-Name
        """
        mat_name = "horizon_terrain"
        self.add_material(mat_name, template="horizon", overwrite=overwrite, Stages={"baseColorMap": texture_path})
        return mat_name

    def get_material(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Hole Material.

        Args:
            name: Material-Name

        Returns:
            Material-Dict oder None
        """
        return self.materials.get(name)

    def exists(self, name: str) -> bool:
        """
        Prüfe ob Material existiert.

        Args:
            name: Material-Name

        Returns:
            True wenn Material existiert
        """
        return name in self.materials

    def get_all_names(self) -> List[str]:
        """
        Gebe alle Material-Namen zurück.

        Returns:
            Liste von Material-Namen
        """
        return list(self.materials.keys())

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Exportiere Materials als einzelnes JSON-Objekt (nicht JSONL).

        Args:
            filepath: Optionaler custom Pfad, ansonsten {beamng_dir}/main/materials.json (aus config.MATERIALS_JSON)
        """
        if filepath is None:
            from .. import config

            filepath = os.path.join(self.beamng_dir, config.MATERIALS_JSON)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Schreibe als einzelnes JSON-Objekt (mit Indentation für Lesbarkeit)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.materials, f, ensure_ascii=False, indent=2)

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Lade Materials als einzelnes JSON-Objekt.

        Args:
            filepath: Optionaler custom Pfad, ansonsten {beamng_dir}/main/materials.json (aus config.MATERIALS_JSON)
        """
        if filepath is None:
            from .. import config

            filepath = os.path.join(self.beamng_dir, config.MATERIALS_JSON)

        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                materials_dict = json.load(f)
                # Konvertiere zu interner Struktur
                self.materials = materials_dict if isinstance(materials_dict, dict) else {}
            except json.JSONDecodeError:
                self.materials = {}

    def merge(self, other: "MaterialManager", overwrite: bool = False) -> int:
        """
        Merge Materials von anderem MaterialManager.

        Args:
            other: Anderer MaterialManager
            overwrite: Überschreibe existierende Materials

        Returns:
            Anzahl hinzugefügter Materials
        """
        count = 0
        for name, material in other.materials.items():
            if name not in self.materials or overwrite:
                self.materials[name] = material
                count += 1
        return count

    def clear(self) -> None:
        """Lösche alle Materials."""
        self.materials.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gebe Statistiken zurück.

        Returns:
            Dict mit Statistiken
        """
        stats = {
            "total": len(self.materials),
            "by_template": {},
            "with_textures": 0,
            "with_colors": 0,
        }

        for mat in self.materials.values():
            # Zähle nach Template-Typ (heuristisch)
            if "groundModelName" in mat:
                stats["by_template"]["terrain"] = stats["by_template"].get("terrain", 0) + 1
            elif "Building" in mat.get("materialTag1", ""):
                stats["by_template"]["building"] = stats["by_template"].get("building", 0) + 1
            else:
                stats["by_template"]["other"] = stats["by_template"].get("other", 0) + 1

            # Zähle Texturen
            if mat.get("Stages") and any("colorMap" in stage for stage in mat["Stages"]):
                stats["with_textures"] += 1

            # Zähle Colors
            if mat.get("Stages") and any("diffuseColor" in stage for stage in mat["Stages"]):
                stats["with_colors"] += 1

        return stats

    def __len__(self) -> int:
        """Anzahl der Materials."""
        return len(self.materials)

    def __repr__(self) -> str:
        return f"MaterialManager({len(self.materials)} materials)"
