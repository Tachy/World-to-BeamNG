"""
MaterialManager - Zentrale Verwaltung aller BeamNG-Materialien.

Verwaltet Materials für:
- Terrain-Tiles (mit Texturen)
- Straßen (aus OSM-Tags)
- Gebäude (LoD2)
- Horizont-Layer
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


class MaterialManager:
    """
    Zentrale Verwaltung aller BeamNG-Materialien (Singleton).

    Features:
    - Automatisches Tracking von Materialien
    - Duplikat-Erkennung
    - JSON Export/Import
    - Material-Templates (Terrain, Road, Building, Horizon)
    - Merge-Unterstützung für Multi-Tile-Workflows
    - Singleton: Nur eine Instanz pro Export (eine materials.json)
    """

    _instance: Optional["MaterialManager"] = None

    def __init__(self, beamng_dir: str):
        """
        Private Constructor - verwende get_instance() stattdessen.

        Args:
            beamng_dir: Pfad zum BeamNG Level-Verzeichnis
        """
        if MaterialManager._instance is not None:
            raise RuntimeError("MaterialManager ist ein Singleton - verwende get_instance()")

        self.beamng_dir = Path(beamng_dir) # Convert to Path object
        self.materials: Dict[str, Dict[str, Any]] = {}
        self._templates = self._init_templates()
        self._config = self._load_config()  # Ganze JSON für buildings, etc.

    @classmethod
    def get_instance(cls, beamng_dir: Path = None) -> "MaterialManager":
        """
        Hole die Singleton-Instanz (erstellt sie bei Bedarf).

        Args:
            beamng_dir: Pfad zum BeamNG Level-Verzeichnis (nur beim ersten Aufruf)

        Returns:
            MaterialManager Singleton-Instanz
        """
        if cls._instance is None:
            if beamng_dir is None: # Added check for None
                raise ValueError("beamng_dir must be provided for the first call to get_instance")
            cls._instance = cls.__new__(cls)
            cls._instance.beamng_dir = beamng_dir # Already a Path object
            cls._instance.materials = {}
            cls._instance._templates = cls._instance._init_templates()
            cls._instance._config = cls._instance._load_config()  # Ganze JSON laden
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Setze Singleton-Instanz zurück (für neuen Export-Lauf)."""
        cls._instance = None

    def _init_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Lade Material-Templates aus data/material_templates.json (ZWINGEND erforderlich).

        Returns:
            Dict mit Template-Namen und Definition

        Raises:
            FileNotFoundError: Wenn data/material_templates.json nicht existiert
        """
        config_path = Path(__file__).parent.parent.parent / "data" / "material_templates.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Material-Templates nicht gefunden: {config_path}\n"
                "Die Datei data/material_templates.json ist erforderlich.\n"
                "Stelle sicher, dass sie im Repository enthalten ist."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                templates = config.get("templates", {})

                # Filtere aus: description, note, und andere Metadaten
                cleaned_templates = {}
                for name, template_def in templates.items():
                    # Kopiere Template, entferne Meta-Felder
                    cleaned = {k: v for k, v in template_def.items() if k not in ("description", "note")}
                    cleaned_templates[name] = cleaned

                num_templates = len(cleaned_templates)
                logger.info(f"  [✓] Material-Templates geladen: {num_templates} aus JSON")

                return cleaned_templates

        except json.JSONDecodeError as e:
            raise ValueError(f"Fehler beim Parsen von {config_path}: {e}\n" "Die JSON-Datei ist ungültig.")
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden von {config_path}: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Lade die komplette material_templates.json Konfiguration.

        Diese Methode lädt die ganze JSON (mit buildings, version, description, etc).

        Returns:
            Dict mit allen Konfigurationen
        """
        config_path = Path(__file__).parent.parent.parent / "data" / "material_templates.json"

        # Fallback: Minimale Config
        default_config = {
            "version": "1.0",
            "description": "Material Templates Configuration",
            "templates": {},
            "buildings": {
                "wall": {
                    "description": "Gebäude-Wand (fallback)",
                    "template": "building_wall",
                    "tiling_scale": 4.0,
                    "material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"},
                },
                "roof": {
                    "description": "Gebäude-Dach (fallback)",
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
                logger.error(f"  [!] Fehler beim Laden der Config: {e}")
                return default_config

        return default_config

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

    def remove_material(self, name: str) -> bool:
        """
        Entferne Material aus dem Manager.

        Args:
            name: Material-Name

        Returns:
            True wenn Material entfernt wurde, False wenn nicht vorhanden
        """
        if name in self.materials:
            del self.materials[name]
            return True
        return False

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

            # Color-Tint: Einfärbung der Textur (kombiniert mit baseColorMap)
            if color:
                stages_config["diffuseColor"] = color
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

    def exists(self, name: str) -> bool:
        """
        Prüfe ob Material existiert.

        Args:
            name: Material-Name

        Returns:
            True wenn Material existiert
        """
        return name in self.materials

    def get_all_materials(self) -> Dict[str, Dict[str, Any]]:
        """
        Gebe alle Materials zurück (für DAE-Export).

        Returns:
            Dict {mat_name: mat_data}
        """
        return self.materials.copy()

    def iter_materials(self):
        """
        Iterator über alle Materials.

        Yields:
            Tuple[str, Dict[str, Any]]: (mat_name, mat_data)
        """
        return iter(self.materials.items())

    def extract_textures(self) -> Dict[str, str]:
        """
        Extrahiere {mat_name: texture_path} für alle Materials mit Texturen.

        Returns:
            Dict mapping Material-Namen zu Textur-Pfaden
        """
        textures = {}
        for mat_name, mat_data in self.materials.items():
            stages = mat_data.get("Stages", [])
            if stages and "baseColorMap" in stages[0]:
                textures[mat_name] = stages[0]["baseColorMap"]
        return textures

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Exportiere Materials als einzelnes JSON-Objekt (nicht JSONL).

        Args:
            filepath: Optionaler custom Pfad, ansonsten {beamng_dir}/main/materials.json (aus config.MATERIALS_JSON)
        """
        if filepath is None:
            from .. import config

            filepath = self.beamng_dir / config.MATERIALS_JSON

        filepath.parent.mkdir(parents=True, exist_ok=True)

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

            filepath = self.beamng_dir / config.MATERIALS_JSON

        if not filepath.exists():
            return

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                materials_dict = json.load(f)
                # Konvertiere zu interner Struktur
                self.materials = materials_dict if isinstance(materials_dict, dict) else {}
            except json.JSONDecodeError:
                self.materials = {}

    def clear(self) -> None:
        """Lösche alle Materials."""
        self.materials.clear()

    def get_templates(self) -> Dict[str, Any]:
        """
        Hole alle Konfigurationen inkl. Material-Templates und buildings section.

        Returns:
            Dict mit Template-Definitionen, buildings Config, etc.
        """
        return self._config.copy()

    def __len__(self) -> int:
        """Anzahl der Materials."""
        return len(self.materials)

    def __repr__(self) -> str:
        return f"MaterialManager({len(self.materials)} materials, singleton)"
