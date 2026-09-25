"""
ItemManager - central management of all BeamNG items.

Manages items for:
- Terrain tiles (TSStatic)
- Buildings (TSStatic)
- Horizon layer (TSStatic)
- Decals, prefabs, etc.
"""

import copy
import json
import re
import uuid
import shutil
from typing import Callable, Dict, Any, Optional, List, Sequence, Tuple
from pathlib import Path
from world_to_beamng import config
from world_to_beamng.managers.environment import build_environment_lines, load_environment_defaults
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


class ItemManager:
    """
    Central management of all BeamNG items (singleton).

    Features:
    - Automatic tracking of items
    - Duplicate detection
    - JSON export/import
    - Item templates (terrain, buildings, etc.)
    - Merge support for multi-tile workflows
    - Bounds calculation for terrain tiles
    - Singleton: only one instance per export (one items.json)
    """

    _instance: Optional["ItemManager"] = None

    # persistentId of the MissionGroup (main element)
    MISSION_GROUP_ID = "6d21ca3b-3f81-4cd8-aeb9-0e780223c20e"

    # MissionGroup - written to main.level.json
    MISSION_GROUP_LINE = {
        "name": "MissionGroup",
        "class": "SimGroup",
        "persistentId": MISSION_GROUP_ID,
    }

    # Further base items - written to main/MissionGroup/items.level.json: LevelInfo, ScatterSky (sun/sky),
    # TimeOfDay, CloudLayer, Precipitation from BeamNG's own defaults (managers/environment.py) + the PlayerDropPoints
    # SimGroup. There is deliberately no separate Sun object: the ScatterSky provides the sun (as in the original levels).
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
            "name": "PlayerDropPoints",  # SimGroup for spawn points (BeamNG default)
            "class": "SimGroup",
            "persistentId": "e8177ef1-0445-4ea5-811a-4eda149ca818",
            "enabled": "1",
            "parentId": "MissionGroup",
        },
    ]
    PLAYER_DROPPOINTS_LINE = [
        {
            "name": "spawn",  # Spawn sphere under PlayerDropPoints
            "class": "SpawnSphere",
            "dataBlock": "SpawnSphereMarker",
            "persistentId": "3d08e3b2-2514-49f8-8b76-8351a12dea51",
            "position": [0, 0, 400],
            "spawnClass": "Player",
            "radius": 10,
            "sphereWeight": 100,
            "indoorWeight": 100,
            "parentId": "PlayerDropPoints",  # Child of PlayerDropPoints!
        }
    ]

    # Level info for info.json (fallback for an ItemManager without a real export run, e.g. tests/tools;
    # a real export overwrites "size" and "minimap" at export time with the actual terrain
    # extent via set_info_json_fields(), see export/beamng_exporter.py and io/aerial.py).
    LEVEL_INFO = {
        "title": "World to BeamNG",
        "description": "Automatic export of OpenStreetMap elements into the BeamNG.drive format.",
        "levelName": "world_to_beamng",
        "previews": ["preview.jpg"],
        "size": [2000, 2000],
        "authors": "Tachy AI",
        "supportsTraffic": False,
        "supportsTimeOfDay": True,  # TimeOfDay object present (managers/environment.py)
        # Name of the SpawnSphere OBJECT (PLAYER_DROPPOINTS_LINE, "spawn"), NOT of the enclosing
        # PlayerDropPoints SimGroup: setSpawnpoint.lua::loadDefaultSpawnpoint() reads this field directly
        # and passes it unchanged to scenetree.findObject() - if it points to the SimGroup, that finds
        # a non-object without getPosition() and core_levels.maybeSpawnDefaultVehicle() crashes fatally
        # on the automatic vehicle spawn (verified against lua/ge/spawn.lua + setSpawnpoint.lua).
        "defaultSpawnPointName": "spawn",
    }

    def __init__(self, beamng_dir: Path):
        """
        Private constructor - use get_instance() instead.

        Args:
            beamng_dir: Path to the BeamNG level directory
        """
        if ItemManager._instance is not None:
            raise RuntimeError("ItemManager is a singleton - use get_instance()")

        self.beamng_dir = beamng_dir
        self.items: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls, beamng_dir: Path = None) -> "ItemManager":
        """
        Gets the singleton instance (creates it if needed).

        Args:
            beamng_dir: Path to the BeamNG level directory (first call only)

        Returns:
            ItemManager singleton instance
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
        """Resets the singleton instance (for a new export run)."""
        cls._instance = None

    @property
    def base_lines(self) -> List[Dict[str, Any]]:
        """Base objects (LevelInfo, ScatterSky, ...) of this instance: a copy so that export values do not change the class."""
        if not hasattr(self, "_base_lines"):
            self._base_lines = copy.deepcopy(self.OTHER_BASE_LINES)
        return self._base_lines

    def set_base_line_fields(self, name: str, **fields) -> None:
        """
        Sets fields of a base object at export time (e.g. fogAtmosphereHeight from the terrain height).

        Raises:
            KeyError: if there is no base object of this name
        """
        for line in self.base_lines:
            if line.get("name") == name:
                line.update(fields)
                return
        raise KeyError(f"No base object '{name}'")

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
        Adds an item.

        Args:
            name: Item name (unique)
            item_class: BeamNG item class (e.g. "TSStatic", "DecalRoad")
            shape_name: Path to the shape file (relative or absolute)
            position: Position [x, y, z]
            rotation_matrix: Orientation as a 3x3 matrix (9 values, row by row); if omitted the object stays unrotated.
                There is deliberately no "rotation" field: BeamNG reads the orientation only from "rotationMatrix" (official
                levels never write "rotation"), and a "rotation": [0, 0, 1, 0] tilted every DAE object by about 0.04 degrees
                around the x-axis through the origin (walls on slopes 1 m too low).
            scale: Scale [x, y, z]
            overwrite: Overwrite an existing item
            **kwargs: Additional properties (collisionType, dataBlock, etc.)

        Returns:
            True if the item was added, False if it already exists and overwrite=False
        """
        if "rotation" in kwargs:
            raise TypeError('Do not use the field "rotation": BeamNG tilts the object with it, orientation only via rotation_matrix')
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

        # Merge additional properties
        item.update(kwargs)

        # Generate persistentId (UUID v4)
        item["persistentId"] = str(uuid.uuid4())

        # Set parentId to MissionGroup
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
        Registers the native BeamNG terrain (TerrainBlock, .ter file).

        JSON schema verified against BeamNG's own template level
        (content/levels/template.zip).

        Args:
            name: Item name (usually: "theTerrain")
            terrain_filename: File name of the .ter file (e.g. "world_to_beamng.ter"),
                              placed relative to the level root
            material_texture_set: Name of the TerrainMaterialTextureSet
            max_height: Height range in meters (config.TERRAIN_MAX_HEIGHT_BUFFER
                       + actual elevation span)
            z_min: absolute world height (meters) that corresponds to heightmap value 0
            origin_x, origin_y: World coordinates of the terrain corner [0, 0]
            square_size: Meters per grid cell (config.TERRAIN_SQUARE_SIZE) -
                        Torque3D TerrainBlock field "squareSize"
            overwrite: Overwrite an existing item

        Returns:
            Item name
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
        Registers a GroundCover object (ground vegetation: grass, flowers, fern ...).

        Args:
            name: Item name (unique, e.g. "gc_mat_grass_grass_short")
            material: Billboard material (texture atlas) of the types
            types: List of types (billboardUVs, sizeMin/-Max, clumping, layer ...);
                `layer` binds a type to the name of a terrain material
            **fields: further fields (radius, maxElements, gridSize, wind ...)

        Returns:
            Item name
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
        Registers a road as a BeamNG DecalRoad (spline decal, projected directly
        onto the terrain surface at runtime - see
        JSON schema verified against BeamNG's own gridmap_v2 level,
        main/MissionGroup/.../decalroads/items.level.json).

        Args:
            name: Item name (unique, e.g. "road_<road_id>")
            nodes: List of [x, y, z, width] nodes along the centerline
            material: Name of the material datablock (see
                      OSMMapper.generate_materials_json_entry())
            drivability: AI navigation weight (-1 = not usable, 1 = normal)
            overwrite: Overwrite an existing item
            **extra: Additional DecalRoad fields (e.g. autoLanes, autoJunction,
                     improvedSpline, textureLength, renderPriority, distanceFade)

        Returns:
            Item name
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
        Adds a building item (convenience method).

        Args:
            name: Item name (e.g. "building_tile_0_0")
            dae_filename: DAE file name (e.g. "buildings_tile_0_0.dae")
            position: Position [x, y, z]
            overwrite: Overwrite an existing item

        Returns:
            Item name
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
        Adds a horizon item (convenience method).

        Args:
            name: Item name
            dae_filename: DAE file name
            position: Position (normally [0, 0, 0])
            overwrite: Overwrite an existing item

        Returns:
            Item name
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
        Places the vehicle on the road nearest to the area center (local (0, 0), see utils.tile_scanner::
        compute_global_center()), oriented in one driving direction
        along this road (which of the two directions is arbitrary).

        Args:
            road_polygons: List of dicts with "trimmed_centerline" ((N, 3) array, local
                coordinates). The Z values are already the road centerline height that the
                terrain is later embedded to (see terrain/road_embedding.py) - no separate
                height interpolation needed.

        Returns:
            (position [x, y, z], rotationMatrix [9 floats]). Fallback ([0, 0, 400], identity)
            without usable road data (e.g. forest/horizon-only export without roads).
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

            # Distance to the area center (local (0, 0)) - no KD-tree needed, centerlines have only
            # dozens to a few hundred points per road, not millions like the elevation data.
            dist_sq = coords[:, 0] ** 2 + coords[:, 1] ** 2
            idx = int(np.argmin(dist_sq))
            if dist_sq[idx] >= best_dist:
                continue

            # Tangent at this point: direction to the neighbor point (at the end of the route the only
            # existing neighbor in the other direction) - which of the two directions is
            # arbitrary, see docstring.
            neighbor_idx = idx + 1 if idx + 1 < len(coords) else idx - 1
            tangent = coords[neighbor_idx][:2] - coords[idx][:2]
            norm = float(np.hypot(tangent[0], tangent[1]))
            if norm < 1e-6:
                continue  # two identical points - no usable direction

            best_dist = float(dist_sq[idx])
            best_point = coords[idx]
            best_tangent = tangent / norm

        if best_point is None:
            return fallback_position, fallback_rotation

        dx, dy = float(best_tangent[0]), float(best_tangent[1])
        # Vehicle looks along the driving direction (dx, dy, 0) - see _heading_rotation_matrix().
        rotation_matrix = self._heading_rotation_matrix(dx, dy)

        # Small safety margin above the (already embedded) road height so that the
        # vehicle does not get stuck in the road surface.
        position = [float(best_point[0]), float(best_point[1]), float(best_point[2]) + 0.3]

        logger.info(f"  [OK] Vehicle spawn on the road nearest to the area center: {position}")
        return position, rotation_matrix

    @staticmethod
    def _spawn_road_segments(road_polygons):
        """
        All centerline segments of drivable roads as arrays (starts, ends), each (M, 3) - built once so that
        _nearest_road_pose() needs only one vectorized query per POI instead of a loop over all roads.
        Tunnels (structure_type) and paths without car traffic (config.POI_SPAWN_EXCLUDED_HIGHWAYS) do not count,
        segments of length 0 are dropped. Order = road order (on equal distance the first one wins).
        """
        import numpy as np

        starts, ends = [], []
        for road in road_polygons or []:
            if road.get("structure_type", "surface") == "tunnel":
                continue
            if (road.get("osm_tags") or {}).get("highway") in config.POI_SPAWN_EXCLUDED_HIGHWAYS:
                continue
            centerline = road.get("trimmed_centerline")
            if centerline is None or len(centerline) < 2:
                continue
            coords = np.asarray(centerline, dtype=float)
            starts.append(coords[:-1])
            ends.append(coords[1:])
        if not starts:
            return np.empty((0, 3)), np.empty((0, 3))
        starts, ends = np.concatenate(starts), np.concatenate(ends)
        seg = ends[:, :2] - starts[:, :2]
        valid = np.einsum("ij,ij->i", seg, seg) > 1e-12
        return starts[valid], ends[valid]

    @staticmethod
    def _nearest_road_pose(road_polygons, target_xy, max_distance: float, segments=None):
        """
        Nearest point on a drivable road centerline to `target_xy` (foot of the perpendicular on the segment, height
        linear along the segment) and the direction of this segment as a unit vector - or None if no
        road lies within max_distance. Tunnels (structure_type) and paths without car traffic
        (config.POI_SPAWN_EXCLUDED_HIGHWAYS) do not count.

        Args:
            segments: optional prebuilt (starts, ends) from _spawn_road_segments(road_polygons) - for
                many queries against the same road network; otherwise it is built here

        Returns:
            ((x, y, z), (dx, dy)) or None
        """
        import numpy as np

        starts, ends = segments if segments is not None else ItemManager._spawn_road_segments(road_polygons)
        if len(starts) == 0:
            return None
        target = np.asarray(target_xy, dtype=float)[:2]
        seg = ends[:, :2] - starts[:, :2]
        seg_len_sq = np.einsum("ij,ij->i", seg, seg)
        t = np.clip(np.einsum("ij,ij->i", target - starts[:, :2], seg) / seg_len_sq, 0.0, 1.0)
        foot = starts + t[:, None] * (ends - starts)
        dist = np.hypot(foot[:, 0] - target[0], foot[:, 1] - target[1])
        i = int(np.argmin(dist))
        if not dist[i] < float(max_distance):
            return None
        return tuple(float(v) for v in foot[i]), tuple(float(v) for v in seg[i] / np.sqrt(seg_len_sq[i]))

    @staticmethod
    def _heading_rotation_matrix(dx: float, dy: float) -> list:
        """rotationMatrix with which a vehicle looks in direction (dx, dy, 0).

        The vehicle front lies on local -Y (jbeam convention; confirmed in game 2026-09-24: with local +Y on the
        heading the tunnel spawns stood parallel to the road but looked away from the tunnel). So local +Y -> -(dx, dy).

        BeamNG stores the images of the local axes in the ROWS (row 0 = local +X, row 1 = local +Y, row 2 =
        local +Z) - derived from vanilla spawn points on diagonal roads (20 of 24 with row 1 parallel to the
        road) and confirmed by the vanilla tunnel in jungle_rock_island (portals exactly on row 1 of the zone). The
        earlier column variant mirrored every orientation on the north-south axis (in game: cars across the road)."""
        return [-dy, dx, 0.0, -dx, -dy, 0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def _slugify_spawn_object_name(display_name: str) -> str:
        """Display name -> valid, readable SpawnSphere object name (e.g. "Hospental" -> "spawn_hospental")."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", display_name).strip("_").lower()
        return f"spawn_{slug}" if slug else "spawn_unnamed"

    def _compute_poi_spawn_points(
        self,
        poi_points,
        max_points: Optional[int] = None,
        preview_builder: Optional[Callable[[str, Tuple[float, float]], Optional[str]]] = None,
        road_polygons=None,
    ) -> List[Dict]:
        """
        One additional spawn point per POI (place or large parking lot, see osm/poi_points.py) selectable in the
        BeamNG vehicle selection - supplements the automatic default spawn
        (_compute_vehicle_spawn()), does not replace it.

        Earlier version (up to and including commit a424cbf) built one spawn per uniquely named
        OSM road instead - but as a label in the vehicle selection that is not very meaningful ("Nuova strada
        del Passo del San Gottardo"). Places/parking lots are easier for players to recognize.

        Ranking when there are more candidates than max_points: first all "place" POIs (places), sorted by
        prominence (osm.poi_points.PLACE_RANK: city before village before hamlet), then "parking" POIs
        sorted by area - a place is a more meaningful spawn landmark than any parking lot.
        The vehicle does not stand on the OSM point of the place (often in the middle of houses or on a meadow),
        but on the nearest drivable road, with heading parallel to its centerline (see
        _nearest_road_pose()). If no road is closer than config.POI_SPAWN_MAX_ROAD_DISTANCE (e.g. an alp with only
        hiking trails), the POI is dropped - before the limit to max_points, the next candidate moves up.
        Without road data all POIs are dropped (only the default spawn remains).

        Args:
            poi_points: List of dicts {"name", "position": [x, y, z], "kind": "place"|"parking",
                "rank": float} - see TerrainWorkflow._collect_poi_points()
            max_points: at most this many points (default: config.MAX_POI_SPAWN_POINTS)
            preview_builder: optional (object_name, (x, y)) -> preview image path (relative to the level
                root) or None - see io/aerial.py::build_poi_preview_image(). Without a preview image
                BeamNG falls back to the level preview image (see levels.lua imageExistsDefault()).
                Centered on the spawn (placed on the road).
            road_polygons: Road dicts with "trimmed_centerline", "osm_tags", "structure_type" (optional)

        Returns:
            List of dicts: {"object_name", "display_name", "position", "rotationMatrix", "preview"}
        """
        if max_points is None:
            max_points = config.MAX_POI_SPAWN_POINTS
        if not poi_points:
            return []
        if not road_polygons:
            logger.info("  [i] No road data - no place/parking spawns")
            return []

        kind_priority = {"place": 0, "parking": 1}
        ranked = sorted(poi_points, key=lambda p: (kind_priority.get(p.get("kind"), 2), -(p.get("rank") or 0.0)))

        # Spawn pose per POI: on the nearest road, or omit the POI if none is within reach
        candidates = []  # (poi, ((x, y, z), (dx, dy)))
        segments = self._spawn_road_segments(road_polygons)
        for poi in ranked:
            if len(candidates) >= max_points:
                break
            x, y, _ = (float(v) for v in poi["position"])
            pose = self._nearest_road_pose(road_polygons, (x, y), config.POI_SPAWN_MAX_ROAD_DISTANCE, segments)
            if pose is None:
                logger.info(f"  [i] Spawn '{poi['name']}' dropped: no road within {config.POI_SPAWN_MAX_ROAD_DISTANCE:.0f} m")
                continue
            candidates.append((poi, pose))

        # Number duplicate display names (mainly unnamed parking lots -> "Parkplatz") so that the
        # vehicle selection lists them distinguishably - the first occurrence stays unnumbered.
        name_occurrence: Dict[str, int] = {}
        display_names = []
        for poi, _ in candidates:
            name = poi["name"]
            name_occurrence[name] = name_occurrence.get(name, 0) + 1
            n = name_occurrence[name]
            display_names.append(name if n == 1 else f"{name} {n}")

        used_object_names = set()
        result = []
        for (poi, ((x, y, z), (dx, dy))), display_name in zip(candidates, display_names):
            rotation = self._heading_rotation_matrix(dx, dy)
            # Small safety margin so that the vehicle does not get stuck in the road/terrain
            position = [x, y, z + 0.3]

            object_name = base_name = self._slugify_spawn_object_name(display_name)
            suffix = 2
            while object_name in used_object_names:
                object_name = f"{base_name}_{suffix}"
                suffix += 1
            used_object_names.add(object_name)

            preview = preview_builder(object_name, (x, y)) if preview_builder else None

            result.append({
                "object_name": object_name,
                "display_name": display_name,
                "position": position,
                "rotationMatrix": rotation,
                "preview": preview,
            })

        return result

    def _fixed_spawn_points(
        self,
        fixed_spawns,
        used_object_names,
        preview_builder: Optional[Callable[[str, Tuple[float, float]], Optional[str]]] = None,
    ) -> List[Dict]:
        """
        Selectable spawn points with a fixed pose (e.g. in front of tunnel entrances, see
        tunnels/entrance_spawns.py) - same output format as _compute_poi_spawn_points().

        Args:
            fixed_spawns: [{"name", "position": (x, y, z) on the road surface, "heading": (dx, dy)}, ...]
            used_object_names: already assigned object names (is extended)
        """
        result = []
        for spawn in fixed_spawns or ():
            x, y, z = (float(v) for v in spawn["position"])
            dx, dy = (float(v) for v in spawn["heading"])
            object_name = base_name = self._slugify_spawn_object_name(spawn["name"])
            suffix = 2
            while object_name in used_object_names:
                object_name = f"{base_name}_{suffix}"
                suffix += 1
            used_object_names.add(object_name)
            result.append({
                "object_name": object_name,
                "display_name": spawn["name"],
                "position": [x, y, z + 0.3],
                "rotationMatrix": self._heading_rotation_matrix(dx, dy),
                "preview": preview_builder(object_name, (x, y)) if preview_builder else None,
            })
        return result

    def save(
        self,
        filepath: Optional[Path] = None,
        road_polygons=None,
        poi_points=None,
        preview_builder: Optional[Callable[[str, Tuple[float, float]], Optional[str]]] = None,
        fixed_spawns=None,
    ) -> None:
        """
        Exports items into the correct BeamNG structure.

        Produces:
        - main/items.level.json: MissionGroup
        - main/MissionGroup/items.level.json: OTHER_BASE_LINES + terrain/building items
        - main/MissionGroup/PlayerDropPoints/items.level.json: spawn points

        Args:
            filepath: Optional custom path, otherwise from config.ITEMS_JSON
            road_polygons: Road dicts with "trimmed_centerline" for the automatic
                vehicle spawn position (optional) - see _compute_vehicle_spawn()
            poi_points: POI dicts (places, large parking lots) for additional, selectable spawn points
                (optional) - see _compute_poi_spawn_points()
            preview_builder: optional (object_name, (x, y)) -> preview image path, passed through to
                _compute_poi_spawn_points()
            fixed_spawns: additional spawn points with a fixed pose (optional) - see _fixed_spawn_points()
        """
        from .. import config

        # BeamNG expects the following structure:
        # 1. main/items.level.json - MissionGroup only
        # 2. main/MissionGroup/items.level.json - LevelInfo, Sky, Sun + all terrain/building items
        # 3. main/MissionGroup/PlayerDropPoints/items.level.json - spawn points

        main_items_dir = self.beamng_dir / "main"
        missiongroup_dir = main_items_dir / "MissionGroup"
        playerdroppoints_dir = missiongroup_dir / "PlayerDropPoints"

        main_items = main_items_dir / "items.level.json"
        missiongroup_items = missiongroup_dir / "items.level.json"
        playerdroppoints_items = playerdroppoints_dir / "items.level.json"

        spawn_position, spawn_rotation = self._compute_vehicle_spawn(road_polygons)
        poi_spawns = self._compute_poi_spawn_points(poi_points, preview_builder=preview_builder, road_polygons=road_polygons)
        poi_spawns += self._fixed_spawn_points(fixed_spawns, {ps["object_name"] for ps in poi_spawns}, preview_builder)

        # Write main/items.level.json in JSONL format (MissionGroup only)
        # (json.dumps instead of json.dump to the file: the C encoder is ~5x faster)
        encode = json.JSONEncoder(ensure_ascii=False).encode
        main_items_dir.mkdir(exist_ok=True)
        with open(main_items, "w", encoding="utf-8") as f:
            f.write(encode(self.MISSION_GROUP_LINE) + "\n")

        # Write main/MissionGroup/items.level.json in JSONL format
        missiongroup_dir.mkdir(exist_ok=True)
        with open(missiongroup_items, "w", encoding="utf-8") as f:
            # OTHER_BASE_LINES (the_level_info, the_sky, tod, clouds1, rain_coverage, PlayerDropPoints-SimGroup)
            for base_line in self.base_lines:
                f.write(encode(base_line) + "\n")

            # All newly added items (terrain, buildings, etc.)
            for item in self.items.values():
                f.write(encode(item) + "\n")

        # Write main/MissionGroup/PlayerDropPoints/items.level.json in JSONL format
        playerdroppoints_dir.mkdir(exist_ok=True)
        with open(playerdroppoints_items, "w", encoding="utf-8") as f:
            # PLAYER_DROPPOINTS_LINE with the computed spawn position
            for spawn_line in self.PLAYER_DROPPOINTS_LINE:
                if spawn_line.get("name") == "spawn":
                    # Overwrite position/orientation with the computed vehicle spawn position
                    spawn_line = spawn_line.copy()
                    spawn_line["position"] = spawn_position
                    spawn_line["rotationMatrix"] = spawn_rotation
                f.write(encode(spawn_line) + "\n")

            # Additional POI spawn points (one SpawnSphere per place/large parking lot)
            for poi_spawn in poi_spawns:
                f.write(
                    encode(
                        {
                            "name": poi_spawn["object_name"],
                            "class": "SpawnSphere",
                            "dataBlock": "SpawnSphereMarker",
                            "persistentId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"world_to_beamng/spawn/{poi_spawn['object_name']}")),
                            "position": poi_spawn["position"],
                            "spawnClass": "Player",
                            "radius": 10,
                            "sphereWeight": 100,
                            "indoorWeight": 100,
                            "parentId": "PlayerDropPoints",
                            "rotationMatrix": poi_spawn["rotationMatrix"],
                        }
                    )
                    + "\n"
                )

        # info.json list for the BeamNG vehicle selection: the default spawn ("spawn") first (thereby gets
        # the 'default' flag, see lua/ge/extensions/core/levels.lua), then the POI spawns.
        if poi_spawns:
            self.set_info_json_fields(
                spawnPoints=[{"objectname": "spawn"}]
                + [
                    {
                        "objectname": ps["object_name"],
                        "name": ps["display_name"],
                        **({"preview": ps["preview"]} if ps.get("preview") else {}),
                    }
                    for ps in poi_spawns
                ]
            )

    @property
    def info_json(self) -> Dict[str, Any]:
        """info.json content of this instance: a copy so that export values (terrain size, minimap) do not change the class."""
        if not hasattr(self, "_info_json"):
            self._info_json = copy.deepcopy(self.LEVEL_INFO)
        return self._info_json

    def set_info_json_fields(self, **fields) -> None:
        """Sets/overwrites fields of info.json at export time (e.g. "size"/"minimap" from the real terrain extent)."""
        self.info_json.update(fields)

    def save_info_json(self) -> None:
        """
        Writes info.json into the level root directory.

        This file contains metadata for BeamNG (title, author, spawn point, etc.).
        Also copies data/preview.jpg into the level directory.
        """
        info_path = self.beamng_dir / "info.json"

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self.info_json, f, ensure_ascii=False, indent=4)

        # Copy preview.jpg from data/ to BEAMNG_DIR
        preview_src = Path("data/preview.jpg")
        preview_dst = self.beamng_dir / "preview.jpg"

        if preview_src.exists():
            try:
                shutil.copy2(preview_src, preview_dst)
                logger.info(f"  [OK] Preview image copied: {preview_dst}")
            except Exception as e:
                logger.info(f"  [WARNING] Preview image could not be copied: {e}")
        else:
            logger.info(f"  [INFO] No preview file found: {preview_src}")

    def load(self, filepath: Optional[Path] = None) -> None:
        """
        Loads items from items.json in JSONL format (line JSON).

        Args:
            filepath: Optional custom path, otherwise from config.ITEMS_JSON
        """
        from .. import config

        load_path = filepath
        if load_path is None:
            load_path = self.beamng_dir / config.ITEMS_JSON

        if not load_path.exists():
            return

        self.items = {}

        # Names of the BASE_LINES that are skipped on load
        base_line_names = {line.get("name") for line in self.base_lines}
        base_line_names.add("PlayerDropPoint")  # Old name, if still present
        base_line_names.add("spawn")  # Also skip spawn (it is written with OTHER_BASE_LINES)

        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    item_name = item.get("name", "")

                    # Skip BASE_LINES - these are written automatically on save()
                    if item_name in base_line_names:
                        continue

                    if item_name:
                        self.items[item_name] = item
                except json.JSONDecodeError:
                    continue

    def clear(self) -> None:
        """Deletes all items."""
        self.items.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics.

        Returns:
            Dict with statistics
        """
        stats = {"total": len(self.items), "by_class": {}, "by_type": {}}

        for item in self.items.values():
            # Count by class
            item_class = item.get("class", "Unknown")
            stats["by_class"][item_class] = stats["by_class"].get(item_class, 0) + 1

            # Count by type (terrain, building, etc.)
            name = item.get("name", "")
            if name.startswith("terrain_"):
                stats["by_type"]["terrain"] = stats["by_type"].get("terrain", 0) + 1
            elif "building" in name.lower():
                stats["by_type"]["building"] = stats["by_type"].get("building", 0) + 1
            else:
                stats["by_type"]["other"] = stats["by_type"].get("other", 0) + 1

        return stats

    def __len__(self) -> int:
        """Number of items."""
        return len(self.items)

    def __repr__(self) -> str:
        return f"ItemManager({len(self.items)} items, singleton)"
