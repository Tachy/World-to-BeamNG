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
            "__name": name,
            "class": item_class,
            "className": item_class,
            "datablock": item_class,
            "position": list(position),
            "rotation": list(rotation),
            "scale": list(scale),
        }

        if shape_name:
            item["shapeName"] = shape_name

        # Merge zusätzliche Properties
        item.update(kwargs)

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
            collisionType="Visible Mesh",
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
            collisionType="Visible Mesh",
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

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Exportiere Items zu items.json im JSONL-Format (Line-JSON).

        Args:
            filepath: Optionaler custom Pfad, ansonsten aus config.ITEMS_JSON
        """
        if filepath is None:
            from .. import config
            filepath = os.path.join(self.beamng_dir, config.ITEMS_JSON)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            for item in self.items.values():
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

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
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    item_name = item.get("__name", "")
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
            name = item.get("__name", "")
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
