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
                "version": 2,
                "Stages": [{"specularPower": 1, "pixelSpecular": True}],
                "groundType": "STONE",
                "materialTag0": "beamng",
                "materialTag1": "Building",
            },
            "building_roof": {
                "class": "Material",
                "version": 2,
                "Stages": [{"specularPower": 1, "pixelSpecular": True}],
                "groundType": "STONE",
                "materialTag0": "beamng",
                "materialTag1": "Building",
            },
            "horizon": {
                "class": "Material",
                "version": 2,
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
        self.add_material(mat_name, template="terrain", overwrite=overwrite, Stages={"colorMap": texture_path})
        return mat_name

    def add_road_material(self, road_type: str, properties: Dict[str, Any], overwrite: bool = False) -> str:
        """
        Füge Straßen-Material hinzu (Convenience-Methode).

        Args:
            road_type: Road-Typ (z.B. "road_residential", "road_motorway")
            properties: OSM-Properties-Dict (color, friction, etc.)
            overwrite: Überschreibe existierendes Material

        Returns:
            Material-Name
        """
        mat_name = properties.get("internal_name", road_type)

        # Konvertiere Color zu Stage
        color = properties.get("color", [0.5, 0.5, 0.5, 1.0])
        if len(color) == 3:
            color.append(1.0)  # Alpha hinzufügen

        self.add_material(
            mat_name,
            template="road",
            overwrite=overwrite,
            Stages={"diffuseColor": color},
            friction=properties.get("friction", 1.0),
            groundType=properties.get("groundType", "ASPHALT"),
        )
        return mat_name

    def add_building_material(self, material_type: str, color: List[float], overwrite: bool = False, **kwargs) -> str:
        """
        Füge Gebäude-Material hinzu (Convenience-Methode).

        Args:
            material_type: "wall" oder "roof"
            color: RGBA Color [r, g, b, a] (0-1)
            overwrite: Überschreibe existierendes Material
            **kwargs: Zusätzliche Properties (groundType, materialTag0, etc.)

        Returns:
            Material-Name
        """
        template = f"building_{material_type}"
        mat_name = f"lod2_{material_type}_{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"

        self.add_material(mat_name, template=template, overwrite=overwrite, Stages={"diffuseColor": color}, **kwargs)
        return mat_name

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
        self.add_material(mat_name, template="horizon", overwrite=overwrite, Stages={"colorMap": texture_path})
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
        Exportiere Materials zu main.materials.json.

        Args:
            filepath: Optionaler custom Pfad, ansonsten {beamng_dir}/main.materials.json
        """
        if filepath is None:
            filepath = os.path.join(self.beamng_dir, "main.materials.json")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.materials, f, indent=2, ensure_ascii=False)

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Lade Materials aus main.materials.json.

        Args:
            filepath: Optionaler custom Pfad, ansonsten {beamng_dir}/main.materials.json
        """
        if filepath is None:
            filepath = os.path.join(self.beamng_dir, "main.materials.json")

        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                self.materials = json.load(f)
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
