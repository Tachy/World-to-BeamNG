"""
ItemManager - Zentrale Verwaltung aller BeamNG-Items.

Verwaltet Items für:
- Terrain-Tiles (TSStatic)
- Gebäude (TSStatic)
- Horizont-Layer (TSStatic)
- Decals, Prefabs, etc.
"""

import copy
import json
import uuid
import shutil
from typing import Dict, Any, Optional, List, Sequence, Tuple
from pathlib import Path
from world_to_beamng import config
from world_to_beamng.managers.environment import build_environment_lines, load_environment_defaults
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


class ItemManager:
    """
    Zentrale Verwaltung aller BeamNG-Items (Singleton).

    Features:
    - Automatisches Tracking von Items
    - Duplikat-Erkennung
    - JSON Export/Import
    - Item-Templates (Terrain, Buildings, etc.)
    - Merge-Unterstützung für Multi-Tile-Workflows
    - Bounds-Berechnung für Terrain-Tiles
    - Singleton: Nur eine Instanz pro Export (eine items.json)
    """

    _instance: Optional["ItemManager"] = None

    # persistentId der MissionGroup (Hauptelement)
    MISSION_GROUP_ID = "6d21ca3b-3f81-4cd8-aeb9-0e780223c20e"

    # MissionGroup - wird in main.level.json geschrieben
    MISSION_GROUP_LINE = {
        "name": "MissionGroup",
        "class": "SimGroup",
        "persistentId": MISSION_GROUP_ID,
    }

    # Weitere Base-Items - werden in main/MissionGroup/items.level.json geschrieben: LevelInfo, ScatterSky (Sonne/Himmel),
    # TimeOfDay, CloudLayer, Precipitation aus BeamNGs eigenen Vorgaben (managers/environment.py) + die PlayerDropPoints-
    # SimGroup. Ein separates Sun-Objekt gibt es bewusst nicht: der ScatterSky liefert die Sonne (wie in den Original-Leveln).
    OTHER_BASE_LINES = build_environment_lines(
        load_environment_defaults(),
        latitude=config.SUN_REFERENCE_LATLON[0],
        longitude=config.SUN_REFERENCE_LATLON[1],
        date=config.ENV_DATE,
        clock=config.ENV_CLOCK_TIME,
        fog_color=config.ENV_FOG_COLOR,
        fog_density=config.LEVEL_FOG_DENSITY,
        visible_distance=config.LEVEL_VISIBLE_DISTANCE,
        environment_map="BNG_Sky_02_cubemap",
    ) + [
        {
            "name": "PlayerDropPoints",  # SimGroup für Spawn-Punkte (BeamNG-Standard)
            "class": "SimGroup",
            "persistentId": "e8177ef1-0445-4ea5-811a-4eda149ca818",
            "enabled": "1",
            "parentId": "MissionGroup",
        },
    ]
    PLAYER_DROPPOINTS_LINE = [
        {
            "name": "spawn",  # Spawn-Sphere unter PlayerDropPoints
            "class": "SpawnSphere",
            "dataBlock": "SpawnSphereMarker",
            "persistentId": "3d08e3b2-2514-49f8-8b76-8351a12dea51",
            "position": [0, 0, 400],
            "spawnClass": "Player",
            "radius": 10,
            "sphereWeight": 100,
            "indoorWeight": 100,
            "parentId": "PlayerDropPoints",  # Child von PlayerDropPoints!
        }
    ]

    # Level-Info für info.json (Fallback für einen ItemManager ohne echten Export-Lauf, z.B. Tests/Tools;
    # ein echter Export überschreibt "size" und "minimap" zur Exportzeit mit der tatsächlichen Terrain-
    # Ausdehnung über set_info_json_fields(), siehe export/beamng_exporter.py und io/aerial.py).
    LEVEL_INFO = {
        "title": "World to BeamNG",
        "description": "Automatischer Export von OpenStreetmap-Elementen in das BeamNG.drive-Format.",
        "levelName": "world_to_beamng",
        "previews": ["preview.jpg"],
        "size": [2000, 2000],
        "authors": "Tachy AI",
        "supportsTraffic": False,
        "supportsTimeOfDay": True,  # TimeOfDay-Objekt vorhanden (managers/environment.py)
        "defaultSpawnPointName": "PlayerDropPoints",  # BeamNG sucht nach dieser SimGroup
    }

    def __init__(self, beamng_dir: Path):
        """
        Private Constructor - verwende get_instance() stattdessen.

        Args:
            beamng_dir: Pfad zum BeamNG Level-Verzeichnis
        """
        if ItemManager._instance is not None:
            raise RuntimeError("ItemManager ist ein Singleton - verwende get_instance()")

        self.beamng_dir = beamng_dir
        self.items: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls, beamng_dir: Path = None) -> "ItemManager":
        """
        Hole die Singleton-Instanz (erstellt sie bei Bedarf).

        Args:
            beamng_dir: Pfad zum BeamNG Level-Verzeichnis (nur beim ersten Aufruf)

        Returns:
            ItemManager Singleton-Instanz
        """
        if cls._instance is None:
            if not beamng_dir:
                raise ValueError("beamng_dir must be provided for the first call to get_instance")
            cls._instance = cls.__new__(cls)
            cls._instance.beamng_dir = Path(beamng_dir)
            cls._instance.items = {}
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Setze Singleton-Instanz zurück (für neuen Export-Lauf)."""
        cls._instance = None

    @property
    def base_lines(self) -> List[Dict[str, Any]]:
        """Basis-Objekte (LevelInfo, ScatterSky, ...) dieser Instanz: eine Kopie, damit Export-Werte die Klasse nicht ändern."""
        if not hasattr(self, "_base_lines"):
            self._base_lines = copy.deepcopy(self.OTHER_BASE_LINES)
        return self._base_lines

    def set_base_line_fields(self, name: str, **fields) -> None:
        """
        Setzt Felder eines Basis-Objekts zur Exportzeit (z.B. fogAtmosphereHeight aus der Terrainhöhe).

        Raises:
            KeyError: wenn es kein Basis-Objekt dieses Namens gibt
        """
        for line in self.base_lines:
            if line.get("name") == name:
                line.update(fields)
                return
        raise KeyError(f"Kein Basis-Objekt '{name}'")

    def add_item(
        self,
        name: str,
        item_class: str = "TSStatic",
        shape_name: Optional[str] = None,
        position: Tuple[float, float, float] = (0, 0, 0),
        rotation_matrix: Optional[Sequence[float]] = None,
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
            rotation_matrix: Ausrichtung als 3x3-Matrix (9 Werte, zeilenweise); ohne Angabe bleibt das Objekt unverdreht.
                Es gibt bewusst kein Feld "rotation": BeamNG liest die Ausrichtung nur aus "rotationMatrix" (offizielle
                Level schreiben nie "rotation"), und ein "rotation": [0, 0, 1, 0] hat jedes DAE-Objekt um ca. 0,04 Grad
                um die x-Achse durch den Ursprung gekippt (Mauern am Hang 1 m zu tief).
            scale: Skalierung [x, y, z]
            overwrite: Überschreibe existierendes Item
            **kwargs: Zusätzliche Properties (collisionType, dataBlock, etc.)

        Returns:
            True wenn Item hinzugefügt wurde, False wenn bereits vorhanden und overwrite=False
        """
        if "rotation" in kwargs:
            raise TypeError('Feld "rotation" nicht verwenden: BeamNG kippt damit das Objekt, Ausrichtung nur über rotation_matrix')
        if name in self.items and not overwrite:
            return False

        item = {
            "name": name,
            "class": item_class,
            "position": list(position),
            "scale": list(scale),
        }

        if rotation_matrix is not None:
            item["rotationMatrix"] = list(rotation_matrix)
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

    def add_terrain_block(
        self,
        name: str,
        terrain_filename: str,
        material_texture_set: str,
        max_height: float,
        z_min: float,
        origin_x: float,
        origin_y: float,
        square_size: float,
        overwrite: bool = False,
    ) -> str:
        """
        Registriert das native BeamNG-Terrain (TerrainBlock, .ter-Datei).

        JSON-Schema verifiziert gegen BeamNGs eigenes template-Level
        (content/levels/template.zip).

        Args:
            name: Item-Name (üblich: "theTerrain")
            terrain_filename: Dateiname der .ter-Datei (z.B. "world_to_beamng.ter"),
                              relativ zum Level-Root abgelegt
            material_texture_set: Name des TerrainMaterialTextureSet
            max_height: Höhenbereich in Metern (config.TERRAIN_MAX_HEIGHT_BUFFER
                       + tatsächliche Elevation-Spanne)
            z_min: absolute Welthöhe (Meter), die Heightmap-Wert 0 entspricht
            origin_x, origin_y: Welt-Koordinaten der Terrain-Ecke [0, 0]
            square_size: Meter pro Rasterzelle (config.TERRAIN_SQUARE_SIZE) -
                        Torque3D TerrainBlock-Feld "squareSize"
            overwrite: Überschreibe existierendes Item

        Returns:
            Item-Name
        """
        from .. import config

        self.add_item(
            name,
            item_class="TerrainBlock",
            position=(origin_x, origin_y, z_min),
            overwrite=overwrite,
            materialTextureSet=material_texture_set,
            maxHeight=max_height,
            terrainFile=f"/levels/{config.LEVEL_NAME}/{terrain_filename}",
            squareSize=square_size,
        )
        return name

    def add_ground_cover(
        self,
        name: str,
        material: str,
        types: List[Dict[str, Any]],
        **fields,
    ) -> str:
        """
        Registriert ein GroundCover-Objekt (Bodenbewuchs: Gras, Blumen, Farn ...).

        Args:
            name: Item-Name (eindeutig, z.B. "gc_mat_grass_grass_short")
            material: Billboard-Material (Textur-Atlas) der Types
            types: Liste von Types (billboardUVs, sizeMin/-Max, Klumpung, layer ...);
                `layer` bindet einen Typ an den Namen eines Terrain-Materials
            **fields: weitere Felder (radius, maxElements, gridSize, Wind ...)

        Returns:
            Item-Name
        """
        self.add_item(
            name,
            item_class="GroundCover",
            overwrite=True,
            material=material,
            Types=types,
            **fields,
        )
        return name

    def add_decal_road(
        self,
        name: str,
        nodes: List[List[float]],
        material: str,
        drivability: float = 1.0,
        overwrite: bool = False,
        **extra,
    ) -> str:
        """
        Registriert eine Straße als BeamNG DecalRoad (Spline-Decal, wird zur
        Laufzeit direkt auf die Terrain-Oberfläche projiziert - siehe
        JSON-Schema verifiziert gegen BeamNGs eigenem gridmap_v2-Level,
        main/MissionGroup/.../decalroads/items.level.json).

        Args:
            name: Item-Name (eindeutig, z.B. "road_<road_id>")
            nodes: Liste von [x, y, z, width]-Knoten entlang der Centerline
            material: Name des Material-Datablocks (siehe
                      OSMMapper.generate_materials_json_entry())
            drivability: AI-Navigations-Gewicht (-1 = nicht nutzbar, 1 = normal)
            overwrite: Überschreibe existierendes Item
            **extra: Zusätzliche DecalRoad-Felder (z.B. autoLanes, autoJunction,
                     improvedSpline, textureLength, renderPriority, distanceFade)

        Returns:
            Item-Name
        """
        position = tuple(nodes[0][:3]) if nodes else (0.0, 0.0, 0.0)

        self.add_item(
            name,
            item_class="DecalRoad",
            position=position,
            overwrite=overwrite,
            nodes=nodes,
            material=material,
            drivability=drivability,
            **extra,
        )
        return name

    def add_building(
        self,
        name: str,
        dae_filename: str,
        position: Tuple[float, float, float],
        overwrite: bool = False,
    ) -> str:
        """
        Füge Gebäude-Item hinzu (Convenience-Methode).

        Args:
            name: Item-Name (z.B. "building_tile_0_0")
            dae_filename: DAE-Dateiname (z.B. "buildings_tile_0_0.dae")
            position: Position [x, y, z]
            overwrite: Überschreibe existierendes Item

        Returns:
            Item-Name
        """
        from .. import config

        shape_name = str(config.RELATIVE_DIR_BUILDINGS / dae_filename)

        self.add_item(
            name,
            item_class="TSStatic",
            shape_name=shape_name,
            position=position,
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

        shape_name = str(config.RELATIVE_DIR_SHAPES / dae_filename)

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

    def _compute_vehicle_spawn(self, road_polygons) -> Tuple[list, list]:
        """
        Platziert das Fahrzeug auf der zur Gebietsmitte (lokal (0, 0), siehe utils.tile_scanner::
        compute_global_center()) nächstgelegenen Straße, ausgerichtet in eine Fahrtrichtung
        entlang dieser Straße (welche der beiden Richtungen ist beliebig).

        Args:
            road_polygons: Liste von Dicts mit "trimmed_centerline" ((N, 3)-Array, lokale
                Koordinaten). Die Z-Werte sind bereits die Straßen-Centerline-Höhe, auf die das
                Terrain später eingebettet wird (siehe terrain/road_embedding.py) - keine eigene
                Höheninterpolation nötig.

        Returns:
            (position [x, y, z], rotationMatrix [9 floats]). Fallback ([0, 0, 400], Identität)
            ohne brauchbare Straßendaten (z. B. Wald-/Horizont-only-Export ohne Straßen).
        """
        import numpy as np

        fallback_position = [0, 0, 400]
        fallback_rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        if not road_polygons:
            return fallback_position, fallback_rotation

        best_point, best_tangent, best_dist = None, None, float("inf")

        for road in road_polygons:
            centerline = road.get("trimmed_centerline")
            if centerline is None or len(centerline) < 2:
                continue
            coords = np.asarray(centerline, dtype=float)

            # Abstand zur Gebietsmitte (lokal (0, 0)) - kein KD-Tree nötig, Centerlines haben nur
            # Dutzende bis wenige Hundert Punkte je Straße, nicht Millionen wie die Höhendaten.
            dist_sq = coords[:, 0] ** 2 + coords[:, 1] ** 2
            idx = int(np.argmin(dist_sq))
            if dist_sq[idx] >= best_dist:
                continue

            # Tangente an diesem Punkt: Richtung zum Nachbarpunkt (am Streckenende der einzig
            # vorhandene Nachbar in die andere Richtung) - welche der beiden Richtungen ist
            # beliebig, siehe Docstring.
            neighbor_idx = idx + 1 if idx + 1 < len(coords) else idx - 1
            tangent = coords[neighbor_idx][:2] - coords[idx][:2]
            norm = float(np.hypot(tangent[0], tangent[1]))
            if norm < 1e-6:
                continue  # zwei identische Punkte - keine brauchbare Richtung

            best_dist = float(dist_sq[idx])
            best_point = coords[idx]
            best_tangent = tangent / norm

        if best_point is None:
            return fallback_position, fallback_rotation

        dx, dy = float(best_tangent[0]), float(best_tangent[1])
        # rotationMatrix (row-major 3x3, siehe test_item_manager_rotation.py): bildet die lokale
        # Fahrzeug-Vorwärtsachse (Torque3D-Konvention: lokal +Y) auf die Fahrtrichtung (dx, dy, 0)
        # im Level ab, lokal +X auf die dazu senkrechte "rechts"-Richtung, lokal +Z bleibt "oben".
        # Identitätsprobe: dx=0, dy=1 (Fahrtrichtung = Welt-Y) ergibt [1,0,0, 0,1,0, 0,0,1]. Die
        # Vorwärtsachsen-Konvention ist nicht offiziell dokumentiert - im Spiel verifizieren.
        rotation_matrix = [dy, dx, 0.0, -dx, dy, 0.0, 0.0, 0.0, 1.0]

        # Kleiner Sicherheitsabstand über der (bereits eingebetteten) Straßenhöhe, damit das
        # Fahrzeug nicht in der Fahrbahn feststeckt.
        position = [float(best_point[0]), float(best_point[1]), float(best_point[2]) + 0.3]

        logger.info(f"  [OK] Fahrzeug-Spawn auf nächster Straße zur Gebietsmitte: {position}")
        return position, rotation_matrix

    def save(self, filepath: Optional[Path] = None, road_polygons=None) -> None:
        """
        Exportiere Items in die richtige BeamNG-Struktur.

        Erzeugt:
        - main/items.level.json: MissionGroup
        - main/MissionGroup/items.level.json: OTHER_BASE_LINES + Terrain/Building Items
        - main/MissionGroup/PlayerDropPoints/items.level.json: Spawn-Points

        Args:
            filepath: Optionaler custom Pfad, ansonsten aus config.ITEMS_JSON
            road_polygons: Straßen-Dicts mit "trimmed_centerline" für die automatische
                Fahrzeug-Spawn-Position (optional) - siehe _compute_vehicle_spawn()
        """
        from .. import config

        # BeamNG erwartet folgende Struktur:
        # 1. main/items.level.json - nur MissionGroup
        # 2. main/MissionGroup/items.level.json - LevelInfo, Sky, Sun + alle Terrain/Building Items
        # 3. main/MissionGroup/PlayerDropPoints/items.level.json - Spawn-Points

        main_items_dir = self.beamng_dir / "main"
        missiongroup_dir = main_items_dir / "MissionGroup"
        playerdroppoints_dir = missiongroup_dir / "PlayerDropPoints"

        main_items = main_items_dir / "items.level.json"
        missiongroup_items = missiongroup_dir / "items.level.json"
        playerdroppoints_items = playerdroppoints_dir / "items.level.json"

        spawn_position, spawn_rotation = self._compute_vehicle_spawn(road_polygons)

        # Schreibe main/items.level.json im JSONL-Format (nur MissionGroup)
        # (json.dumps statt json.dump auf die Datei: der C-Encoder ist ~5x schneller)
        encode = json.JSONEncoder(ensure_ascii=False).encode
        main_items_dir.mkdir(exist_ok=True)
        with open(main_items, "w", encoding="utf-8") as f:
            f.write(encode(self.MISSION_GROUP_LINE) + "\n")

        # Schreibe main/MissionGroup/items.level.json im JSONL-Format
        missiongroup_dir.mkdir(exist_ok=True)
        with open(missiongroup_items, "w", encoding="utf-8") as f:
            # OTHER_BASE_LINES (the_level_info, the_sky, tod, clouds1, rain_coverage, PlayerDropPoints-SimGroup)
            for base_line in self.base_lines:
                f.write(encode(base_line) + "\n")

            # Alle neu hinzugefügten Items (Terrain, Buildings, etc.)
            for item in self.items.values():
                f.write(encode(item) + "\n")

        # Schreibe main/MissionGroup/PlayerDropPoints/items.level.json im JSONL-Format
        playerdroppoints_dir.mkdir(exist_ok=True)
        with open(playerdroppoints_items, "w", encoding="utf-8") as f:
            # PLAYER_DROPPOINTS_LINE mit berechneter Spawn-Position
            for spawn_line in self.PLAYER_DROPPOINTS_LINE:
                if spawn_line.get("name") == "spawn":
                    # Überschreibe Position/Ausrichtung mit der berechneten Fahrzeug-Spawn-Position
                    spawn_line = spawn_line.copy()
                    spawn_line["position"] = spawn_position
                    spawn_line["rotationMatrix"] = spawn_rotation
                f.write(encode(spawn_line) + "\n")

    @property
    def info_json(self) -> Dict[str, Any]:
        """info.json-Inhalt dieser Instanz: eine Kopie, damit Export-Werte (Terrain-Größe, Minimap) die Klasse nicht ändern."""
        if not hasattr(self, "_info_json"):
            self._info_json = copy.deepcopy(self.LEVEL_INFO)
        return self._info_json

    def set_info_json_fields(self, **fields) -> None:
        """Setzt/überschreibt Felder von info.json zur Exportzeit (z.B. "size"/"minimap" aus der echten Terrain-Ausdehnung)."""
        self.info_json.update(fields)

    def save_info_json(self) -> None:
        """
        Schreibe info.json ins Level-Root-Verzeichnis.

        Diese Datei enthält Metadaten für BeamNG (Titel, Autor, Spawn-Point, etc.).
        Kopiert auch data/preview.jpg ins Level-Verzeichnis.
        """
        info_path = self.beamng_dir / "info.json"

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self.info_json, f, ensure_ascii=False, indent=4)

        # Kopiere preview.jpg von data/ nach BEAMNG_DIR
        preview_src = Path("data/preview.jpg")
        preview_dst = self.beamng_dir / "preview.jpg"

        if preview_src.exists():
            try:
                shutil.copy2(preview_src, preview_dst)
                logger.info(f"  [OK] Preview-Bild kopiert: {preview_dst}")
            except Exception as e:
                logger.info(f"  [WARNUNG] Preview-Bild konnte nicht kopiert werden: {e}")
        else:
            logger.info(f"  [INFO] Keine Preview-Datei gefunden: {preview_src}")

    def load(self, filepath: Optional[Path] = None) -> None:
        """
        Lade Items aus items.json im JSONL-Format (Line-JSON).

        Args:
            filepath: Optionaler custom Pfad, ansonsten aus config.ITEMS_JSON
        """
        from .. import config

        load_path = filepath
        if load_path is None:
            load_path = self.beamng_dir / config.ITEMS_JSON

        if not load_path.exists():
            return

        self.items = {}

        # Namen der BASE_LINES die beim Load übersprungen werden sollen
        base_line_names = {line.get("name") for line in self.base_lines}
        base_line_names.add("PlayerDropPoint")  # Alter Name falls noch vorhanden
        base_line_names.add("spawn")  # Auch spawn überspringen (wird mit OTHER_BASE_LINES geschrieben)

        with open(load_path, "r", encoding="utf-8") as f:
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
        return f"ItemManager({len(self.items)} items, singleton)"
