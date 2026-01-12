"""
ItemManager - Zentrale Verwaltung aller BeamNG-Items.

Verwaltet Items für:
- Terrain-Tiles (TSStatic)
- Gebäude (TSStatic)
- Horizont-Layer (TSStatic)
- Decals, Prefabs, etc.
"""

import json
import os
import uuid
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class ItemManager:
    """
    Zentrale Verwaltung aller BeamNG-Items.

    Features:
    - Automatisches Tracking von Items
    - Duplikat-Erkennung
    - JSON Export/Import
    - Item-Templates (Terrain, Buildings, etc.)
    - Merge-Unterstützung für Multi-Tile-Workflows
    - Bounds-Berechnung für Terrain-Tiles
    """

    # persistentId der MissionGroup (Hauptelement)
    MISSION_GROUP_ID = "6d21ca3b-3f81-4cd8-aeb9-0e780223c20e"

    # MissionGroup - wird in main.level.json geschrieben
    MISSION_GROUP_LINE = {
        "name": "MissionGroup",
        "class": "SimGroup",
        "persistentId": MISSION_GROUP_ID,
    }

    # Weitere Base-Items - werden in main/items.level.json geschrieben
    OTHER_BASE_LINES = [
        {
            "name": "the_level_info",
            "class": "LevelInfo",
            "persistentId": "64e00688-24f4-417d-a0c8-25e1e7d59cce",
            "gravity": -9.81,
            "parentId": "MissionGroup",
            "levelName": "world_to_beamng",
            "decalsEnabled": True,
            "canSave": True,
            "globalEnvironmentMap": "BNG_Sky_02_cubemap",
        },
        {
            "name": "the_sky",
            "class": "ScatterSky",
            "persistentId": "f0c7b6f6-7e4a-4b2a-8c4f-5c6f0c2a9c55",
            "cloudHeight": 2000,
            "cloudCover": 0.5,
            "cloudSpeed": [0.0005, 0.0],
            "sunScale": 1.2,
            "colorize": [1.0, 0.9, 0.8, 1.0],  # Leichter Gelb/Warmstich
            "ambient": [0.12, 0.12, 0.15, 1.0],
            "brightness": 0.8,  # Etwas dunkler für mehr Atmosphäre
            "skyBrightness": 0.4,
            "fogHeight": 800,
            "fogDensity": 0.0005,
            "rayleighScattering": 0.005,  # Verstärkt den rötlichen Effekt am Horizont
            "mieScattering": 0.001,  # Mehr "Dunst" in der Luft
            "sunSize": 1.5,  # Die Sonne wirkt tiefer stehend größer
            "exposure": 1.1,
            "nightBrightness": 0.05,
            "parentId": "MissionGroup",
        },
        {
            "name": "the_sun",
            "class": "Sun",
            "persistentId": "e75fc72e-4ec9-42ca-b08a-24eca2141534",
            "azimuth": 0,
            "elevation": 60,  # Erhöht von 45 für höhere Sonne
            "brightness": 2.0,  # Erhöht für stärkere Sonne
            "castShadows": True,
            "coronaEnabled": True,
            "parentId": "MissionGroup",
        },
        {
            "name": "PlayerDropPoints",  # SimGroup für Spawn-Punkte (BeamNG-Standard)
            "class": "SimGroup",
            "persistentId": "e8177ef1-0445-4ea5-811a-4eda149ca818",
            "enabled": "1",
            "parentId": "MissionGroup",
        },
        {
            "name": "spawn",  # Spawn-Sphere unter PlayerDropPoints
            "class": "SpawnSphere",
            "dataBlock": "SpawnSphereMarker",
            "persistentId": "3d08e3b2-2514-49f8-8b76-8351a12dea51",
            "position": [0, 0, 400],
            "rotation": [0, 0, 0, 1],
            "spawnClass": "Player",
            "radius": 10,
            "sphereWeight": 100,
            "indoorWeight": 100,
            "parentId": "PlayerDropPoints",  # Child von PlayerDropPoints!
        },
    ]

    # Level-Info für info.json
    LEVEL_INFO = {
        "title": "World to BeamNG",
        "description": "Automatischer Export von OpenStreetmap-Elementen in das BeamNG.drive-Format.",
        "levelName": "world_to_beamng",
        "previews": ["preview.jpg"],
        "size": [2000, 2000],
        "authors": "Tachy AI",
        "supportsTraffic": False,
        "supportsTimeOfDay": False,
        "spawnPointName": "PlayerDropPoints",  # BeamNG sucht nach dieser SimGroup
    }

    def __init__(self, beamng_dir: str):
        """
        Initialisiere ItemManager.

        Args:
            beamng_dir: Pfad zum BeamNG Level-Verzeichnis
        """
        self.beamng_dir = beamng_dir
        self.items: Dict[str, Dict[str, Any]] = {}

    def add_item(
        self,
        name: str,
        item_class: str = "TSStatic",
        shape_name: Optional[str] = None,
        position: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 1, 0),
        scale: Tuple[float, float, float] = (1, 1, 1),
        overwrite: bool = False,
        **kwargs,
    ) -> bool:
        """
        Füge Item hinzu.

        Args:
            name: Item-Name (eindeutig)
            item_class: BeamNG Item-Klasse (z.B. "TSStatic", "DecalRoad")
            shape_name: Pfad zur Shape-Datei (relativ oder absolut)
            position: Position [x, y, z]
            rotation: Rotation [x, y, z, w] (Quaternion)
            scale: Skalierung [x, y, z]
            overwrite: Überschreibe existierendes Item
            **kwargs: Zusätzliche Properties (collisionType, dataBlock, etc.)

        Returns:
            True wenn Item hinzugefügt wurde, False wenn bereits vorhanden und overwrite=False
        """
        if name in self.items and not overwrite:
            return False

        item = {
            "name": name,
            "class": item_class,
            "position": list(position),
            "rotation": list(rotation),
            "scale": list(scale),
        }

        if shape_name:
            item["shapeName"] = shape_name

        # Merge zusätzliche Properties
        item.update(kwargs)

        # Generiere persistentId (UUID v4)
        item["persistentId"] = str(uuid.uuid4())

        # Setze parentId auf MissionGroup
        item["parentId"] = "MissionGroup"

        self.items[name] = item
        return True

    def add_terrain(
        self,
        name: str,
        dae_filename: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        overwrite: bool = False,
    ) -> str:
        """
        Füge Terrain-Item hinzu (Convenience-Methode).

        Args:
            name: Item-Name (z.B. "terrain_0_0")
            dae_filename: DAE-Dateiname (z.B. "terrain_0_0.dae")
            position: Position (normalerweise [0, 0, 0])
            overwrite: Überschreibe existierendes Item

        Returns:
            Item-Name
        """
        # Konstruiere relativen Pfad
        from .. import config

        shape_name = config.RELATIVE_DIR_SHAPES + dae_filename

        self.add_item(
            name,
            item_class="TSStatic",
            shape_name=shape_name,
            position=position,
            overwrite=overwrite,
            collisionType="Visible Mesh Final",
        )
        return name

    def add_building(
        self,
        name: str,
        dae_filename: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float] = (0, 0, 1, 0),
        overwrite: bool = False,
    ) -> str:
        """
        Füge Gebäude-Item hinzu (Convenience-Methode).

        Args:
            name: Item-Name (z.B. "building_tile_0_0")
            dae_filename: DAE-Dateiname (z.B. "buildings_tile_0_0.dae")
            position: Position [x, y, z]
            rotation: Rotation [x, y, z, w]
            overwrite: Überschreibe existierendes Item

        Returns:
            Item-Name
        """
        from .. import config

        shape_name = config.RELATIVE_DIR_BUILDINGS + dae_filename

        self.add_item(
            name,
            item_class="TSStatic",
            shape_name=shape_name,
            position=position,
            rotation=rotation,
            overwrite=overwrite,
            collisionType="Visible Mesh Final",
        )
        return name

    def add_horizon(
        self,
        name: str = "terrain_horizon",
        dae_filename: str = "terrain_horizon.dae",
        position: Tuple[float, float, float] = (0, 0, 0),
        overwrite: bool = False,
    ) -> str:
        """
        Füge Horizont-Item hinzu (Convenience-Methode).

        Args:
            name: Item-Name
            dae_filename: DAE-Dateiname
            position: Position (normalerweise [0, 0, 0])
            overwrite: Überschreibe existierendes Item

        Returns:
            Item-Name
        """
        from .. import config

        shape_name = config.RELATIVE_DIR_SHAPES + dae_filename

        self.add_item(
            name,
            item_class="TSStatic",
            shape_name=shape_name,
            position=position,
            overwrite=overwrite,
            collisionType="None",
            datablock="DefaultStaticShape",
        )
        return name

    def get_item(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Hole Item.

        Args:
            name: Item-Name

        Returns:
            Item-Dict oder None
        """
        return self.items.get(name)

    def exists(self, name: str) -> bool:
        """
        Prüfe ob Item existiert.

        Args:
            name: Item-Name

        Returns:
            True wenn Item existiert
        """
        return name in self.items

    def get_all_names(self) -> List[str]:
        """
        Gebe alle Item-Namen zurück.

        Returns:
            Liste von Item-Namen
        """
        return list(self.items.keys())

    def get_terrain_bounds(self, tile_size: float = 2000.0) -> List[Tuple[float, float, float, float]]:
        """
        Gebe Bounds aller Terrain-Tiles zurück.

        Wichtig für Horizont-Clipping!

        Args:
            tile_size: Größe eines Terrain-Tiles in Metern (Standard: 2000m = 2×2 km)

        Returns:
            Liste von (x_min, y_min, x_max, y_max) Tuples in lokalen Koordinaten
        """
        bounds = []

        for item_name, item_data in self.items.items():
            # Filtere Terrain-Items (Namen wie "terrain_-1000_-1000")
            if not item_name.startswith("terrain_"):
                continue

            # Überspringe terrain_tile_* (das sind Tile-Slices)
            if "tile_" in item_name:
                continue

            # Parse Koordinaten aus Name "terrain_X_Y"
            parts = item_name.split("_")
            if len(parts) < 3:
                continue

            try:
                coord_x = int(parts[-2])
                coord_y = int(parts[-1])

                # Bounds berechnen (Koordinaten sind minimale Ecke)
                x_min = coord_x
                x_max = coord_x + tile_size
                y_min = coord_y
                y_max = coord_y + tile_size

                bounds.append((x_min, y_min, x_max, y_max))
            except ValueError:
                continue

        return bounds

    def _get_spawn_position_with_height(self, height_points, height_elevations, global_offset):
        """
        Berechne Spawn-Position mit Höhendaten.

        Args:
            height_points: Höhendaten-Punkte (XY) - lokal
            height_elevations: Z-Werte - lokal
            global_offset: (origin_x, origin_y) für Transformation WGS84->UTM->Lokal

        Returns:
            Liste [x, y, z] mit automatischer Höhenberechnung
        """
        from .. import config
        from ..geometry.coordinates import transformer_to_utm
        import numpy as np
        from scipy.interpolate import griddata

        if not config.SPAWN_POINT or not global_offset:
            return [0, 0, 400]  # Fallback

        lat, lon = config.SPAWN_POINT
        ox, oy = global_offset

        # Konvertiere WGS84 zu UTM
        x_utm, y_utm = transformer_to_utm.transform(lon, lat)

        # Transformiere zu lokalen Koordinaten
        x_local = x_utm - ox
        y_local = y_utm - oy

        print(
            f"  [i] Berechne Spawn-Punkt: WGS84({lat}, {lon}) -> UTM({x_utm}, {y_utm}) -> Lokal({x_local}, {y_local})"
        )

        # Interpoliere Höhe an diesem Punkt
        if len(height_points) > 0 and len(height_elevations) > 0:
            try:
                height_points_array = np.array(height_points)
                height_elevations_array = np.array(height_elevations)

                # Nutze nearest-neighbor Interpolation
                z_value = griddata(height_points_array, height_elevations_array, (x_local, y_local), method="nearest")

                if z_value is not None and not np.isnan(z_value):
                    z_height = float(z_value)
                else:
                    z_height = 400  # Fallback: 400m über Grund
            except Exception as e:
                print(f"[!] Fehler bei Höheninterpolation: {e}")
                z_height = 400
        else:
            z_height = 400

        final_pos = [x_local, y_local, z_height + 10]  # +10m Sicherheitsabstand über Terrain
        print(f"  [OK] Spawn-Position: {final_pos}")
        return final_pos

    def save(
        self, filepath: Optional[str] = None, height_points=None, height_elevations=None, global_offset=None
    ) -> None:
        """
        Exportiere Items in die richtige BeamNG-Struktur.

        Erzeugt:
        - main/items.level.json: MissionGroup + alle Child-Items (komplette Level-Struktur in JSONL)

        Args:
            filepath: Optionaler custom Pfad, ansonsten aus config.ITEMS_JSON
            height_points: Höhendaten-Punkte für Spawn-Position (optional)
            height_elevations: Z-Werte für Höheninterpolation (optional)
            global_offset: (origin_x, origin_y) für Koordinaten-Transformation (optional)
        """
        if filepath is None:
            from .. import config

            # Nutze ITEMS_JSON (items.level.json - enthält alles)
            filepath = os.path.join(self.beamng_dir, config.ITEMS_JSON)

        # BeamNG erwartet zwei Dateien im JSONL-Format (Line-delimited JSON):
        # 1. main/items.level.json - nur MissionGroup (jede Zeile ein JSON-Objekt)
        # 2. main/MissionGroup/items.level.json - alle anderen Items (jede Zeile ein JSON-Objekt)

        main_items = os.path.join(self.beamng_dir, "main", "items.level.json")
        missiongroup_items = os.path.join(self.beamng_dir, "main", "MissionGroup", "items.level.json")

        # Schreibe main/items.level.json im JSONL-Format (nur MissionGroup)
        os.makedirs(os.path.dirname(main_items), exist_ok=True)
        with open(main_items, "w", encoding="utf-8") as f:
            json.dump(self.MISSION_GROUP_LINE, f, ensure_ascii=False)
            f.write("\n")

        # Schreibe main/MissionGroup/items.level.json im JSONL-Format
        os.makedirs(os.path.dirname(missiongroup_items), exist_ok=True)
        with open(missiongroup_items, "w", encoding="utf-8") as f:
            # Schreibe jedes Item auf eigene Zeile (JSONL-Format)

            # Berechne Spawn-Position mit Höhendaten falls verfügbar
            spawn_position = [0, 0, 400]  # Default
            if height_points is not None and height_elevations is not None and global_offset is not None:
                spawn_position = self._get_spawn_position_with_height(height_points, height_elevations, global_offset)

            # OTHER_BASE_LINES (the_level_info, the_sky, the_sun, spawn)
            for base_line in self.OTHER_BASE_LINES:
                if base_line.get("name") == "spawn":
                    # Überschreibe Position mit berechneter Position
                    base_line = base_line.copy()
                    base_line["position"] = spawn_position
                json.dump(base_line, f, ensure_ascii=False)
                f.write("\n")

            # Alle neu hinzugefügten Items
            for item in self.items.values():
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    def save_info_json(self) -> None:
        """
        Schreibe info.json ins Level-Root-Verzeichnis.

        Diese Datei enthält Metadaten für BeamNG (Titel, Autor, Spawn-Point, etc.).
        """
        info_path = os.path.join(self.beamng_dir, "info.json")

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self.LEVEL_INFO, f, ensure_ascii=False, indent=4)

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Lade Items aus items.json im JSONL-Format (Line-JSON).

        Args:
            filepath: Optionaler custom Pfad, ansonsten aus config.ITEMS_JSON
        """
        if filepath is None:
            from .. import config

            filepath = os.path.join(self.beamng_dir, config.ITEMS_JSON)

        if not os.path.exists(filepath):
            return

        self.items = {}

        # Namen der BASE_LINES die beim Load übersprungen werden sollen
        base_line_names = {line.get("name") for line in self.OTHER_BASE_LINES}
        base_line_names.add("PlayerDropPoint")  # Alter Name falls noch vorhanden
        base_line_names.add("spawn")  # Auch spawn überspringen (wird mit OTHER_BASE_LINES geschrieben)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    item_name = item.get("name", "")

                    # Überspringe BASE_LINES - diese werden beim save() automatisch geschrieben
                    if item_name in base_line_names:
                        continue

                    if item_name:
                        self.items[item_name] = item
                except json.JSONDecodeError:
                    continue

    def merge(self, other: "ItemManager", overwrite: bool = False) -> int:
        """
        Merge Items von anderem ItemManager.

        Args:
            other: Anderer ItemManager
            overwrite: Überschreibe existierende Items

        Returns:
            Anzahl hinzugefügter Items
        """
        count = 0
        for name, item in other.items.items():
            if name not in self.items or overwrite:
                self.items[name] = item
                count += 1
        return count

    def clear(self) -> None:
        """Lösche alle Items."""
        self.items.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gebe Statistiken zurück.

        Returns:
            Dict mit Statistiken
        """
        stats = {"total": len(self.items), "by_class": {}, "by_type": {}}

        for item in self.items.values():
            # Zähle nach Klasse
            item_class = item.get("class", "Unknown")
            stats["by_class"][item_class] = stats["by_class"].get(item_class, 0) + 1

            # Zähle nach Typ (terrain, building, etc.)
            name = item.get("name", "")
            if name.startswith("terrain_"):
                stats["by_type"]["terrain"] = stats["by_type"].get("terrain", 0) + 1
            elif "building" in name.lower():
                stats["by_type"]["building"] = stats["by_type"].get("building", 0) + 1
            else:
                stats["by_type"]["other"] = stats["by_type"].get("other", 0) + 1

        return stats

    def __len__(self) -> int:
        """Anzahl der Items."""
        return len(self.items)

    def __repr__(self) -> str:
        return f"ItemManager({len(self.items)} items)"
