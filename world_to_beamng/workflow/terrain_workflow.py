"""
Terrain export workflow.

Orchestrates the complete terrain export process.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from .tile_processor import TileProcessor
from ..progress import PipelineTask
from .terrain_roads import (
    _attach_width_nodes,
    _guardrail_instances,
    _invisible_road_material,
    _road_marking_lines,
    _road_width_specs,
    _widths_along,
)
from .terrain_structures import (
    _bridge_footprints,
    _bridge_groups,
    _bridge_photo_areas,
    _bridge_stems,
    _dropped_tunnel_road_ids,
    _gallery_embankment_cuts,
    _gallery_embedding,
    _plan_tunnels,
    _roadblock_items,
    _structure_items,
    _tunnel_light_items,
    _tunnel_zone_items,
)

logger = logging.getLogger(__name__)

WATER_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "data" / "water_templates.json"


def make_height_sampler_for_water(heights, origin_x, origin_y):
    """Bilinear elevation lookup on the finished heightmap (as for the vineyard vines)."""
    from ..forest.vineyard_generator import make_height_sampler

    return make_height_sampler(heights, origin_x, origin_y, config.TERRAIN_SQUARE_SIZE)


WATER_BOUNDS_MARGIN = 2.0  # just inside the real data: beyond it the terrain is filled in


def water_bounds(grid_bounds_local):
    """(xmin, ymin, xmax, ymax) of the terrain with real elevation data in which water may be created."""
    x_min, x_max, y_min, y_max = grid_bounds_local
    m = WATER_BOUNDS_MARGIN
    return (x_min + m, y_min + m, x_max - m, y_max - m)


@dataclass
class TileState:
    """Intermediate results of TerrainWorkflow.process_tile(), filled phase by phase (grouped by the _tile_* method
    that sets them; later phases read what earlier ones set)."""

    tiles: List[Dict]
    global_offset: Tuple[float, float]
    # _tile_load_osm()
    local_points: Optional[np.ndarray] = None
    elevations: Optional[np.ndarray] = None
    tile_hash: str = ""
    osm_bbox: Optional[Tuple] = None
    osm_data: Optional[Dict] = None
    road_polygons: List[Dict] = field(default_factory=list)
    # _tile_load_buildings()
    buildings_data: Optional[Dict] = None
    # _tile_road_network()
    grid_bounds_local: Optional[Tuple[float, float, float, float]] = None
    road_slope_polygons_2d: List[Dict] = field(default_factory=list)
    # _tile_heightmap() (heights is changed again by _tile_shape_terrain() and _tile_ponds())
    grid: Optional[Tuple] = None
    nx: int = 0
    ny: int = 0
    heights: Optional[np.ndarray] = None
    terrain_size: int = 0
    terrain_origin_x: float = 0.0
    terrain_origin_y: float = 0.0
    # _tile_shape_terrain()
    surface_road_polygons: List[Dict] = field(default_factory=list)
    structure_road_polygons: List[Dict] = field(default_factory=list)
    gallery_roads: List[Dict] = field(default_factory=list)
    bridge_roads: List[Dict] = field(default_factory=list)
    tunnel_plans: List[Dict] = field(default_factory=list)
    tunnel_holes: Optional[np.ndarray] = None
    dropped_tunnel_road_ids: frozenset = frozenset()
    # _tile_ponds()
    landuse_polygons: List[Dict] = field(default_factory=list)
    natural_heights: Optional[np.ndarray] = None
    # _tile_layer_map()
    layer_map: Optional[np.ndarray] = None
    terrain_material_names: List[str] = field(default_factory=list)
    photo_tile_names: List[str] = field(default_factory=list)
    photo_tiles: Optional[Dict] = None
    road_surface_union: Optional[object] = None
    road_shapes: List = field(default_factory=list)
    building_shapes: List = field(default_factory=list)
    tree_exclusion: Optional[object] = None
    # _tile_scene_objects()
    vineyard_instances: List[Dict] = field(default_factory=list)
    water: Dict = field(default_factory=dict)
    wall_meshes: List[Dict] = field(default_factory=list)
    tunnel_meshes: List[Dict] = field(default_factory=list)
    roadblocks: List[Dict] = field(default_factory=list)
    tunnel_zones: List[Dict] = field(default_factory=list)
    tunnel_lights: List[Dict] = field(default_factory=list)
    poi_points: List[Dict] = field(default_factory=list)
    tunnel_spawns: List[Dict] = field(default_factory=list)


class TerrainWorkflow:
    """
    Orchestrates the terrain export workflow.

    Responsible for:
    - Mesh generation
    - Road integration
    - DAE export
    - Material/item management
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = MaterialManager.get_instance()  # Singleton
        self.items = ItemManager.get_instance()  # Singleton
        self.dae = dae_exporter
        self.tile_processor = TileProcessor(cache_manager)

    def process_tile(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float],
        task: PipelineTask,
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Process all given tiles as ONE contiguous area.

        The elevation data of all tiles is combined into a single point cloud
        (assumes that they form a gapless, rectangular
        area - user responsibility, see utils.tile_scanner).
        From here on, all remaining processing (BBox, OSM query,
        grid, road mesh, junction detection, embankment, heightmap) runs ONCE
        over the whole area instead of once per tile - clipping only happens
        at the outer edge of the whole area, no longer at the former
        tile borders. A single tile is simply the special case
        len(tiles) == 1 of the same code path.

        The work is split into phases (the _tile_* methods) that run in this order and pass their results on in a
        TileState - the order matters, e.g. the embankment needs the still natural heightmap.

        Args:
            tiles: List of tile metadata (typically all DGM1 tiles
                of an export)
            global_offset: Global offset (origin_x, origin_y)
            task: PipelineTask for the progress display of the subtasks
            bbox_margin: BBox expansion in meters
            buildings_data: Optional - LoD2 building data

        Returns:
            Dict with processing results
        """
        s = TileState(tiles=tiles, global_offset=global_offset)

        sub = task.begin_subtask("Load OSM data")
        failure = self._tile_load_osm(s, bbox_margin, sub)
        if failure:
            return failure
        sub.finish()  # covers the OSM query AND road extraction - both part of "Load OSM data"

        # 6a. Aerial photos: are NO longer processed per tile here - with
        # several tiles the file existence check ("is there already
        # any .dds?") would skip the export for all tiles except the first.
        # Instead, BeamNGExporter.export_complete_level() calls
        # process_aerial_images() once for the total BBox of all tiles,
        # before the tile loop begins.

        sub = task.begin_subtask("Normalize buildings")
        self._tile_load_buildings(s, buildings_data)
        sub.finish(f"{len(s.buildings_data)} buildings" if s.buildings_data else "no LoD2 buildings")

        sub = task.begin_subtask("Road network + infrastructure")
        self._tile_road_network(s)
        self._tile_heightmap(s)
        self._tile_shape_terrain(s)
        self._tile_ponds(s)
        self._tile_building_ground(s)
        self._tile_layer_map(s)
        self._tile_scene_objects(s)

        heights = s.heights
        z_min = float(heights.min())
        z_max = float(heights.max())
        max_height = (z_max - z_min) + config.TERRAIN_MAX_HEIGHT_BUFFER

        sub.finish(f"{len(s.road_slope_polygons_2d)} road segments")

        photo_tiles = s.photo_tiles
        return {
            "status": "success",
            "heightmap": heights,
            "terrain_size": s.terrain_size,
            "terrain_origin_x": s.terrain_origin_x,
            "terrain_origin_y": s.terrain_origin_y,
            "z_min": z_min,
            "max_height": max_height,
            "layer_map": s.layer_map,
            "terrain_material_names": s.terrain_material_names,
            "photo_tile_names": s.photo_tile_names,
            # Four-photo mode (otherwise None): tile variants of the layers, their photo and the photo size per tile
            "layer_variants": photo_tiles["layer_variants"] if photo_tiles else None,
            "variant_parents": photo_tiles["variant_parents"] if photo_tiles else None,
            "photo_extents": photo_tiles["photo_extents"] if photo_tiles else None,
            "poi_points": s.poi_points,  # Villages/towns and large parking lots for ItemManager._compute_poi_spawn_points()
            "vineyard_instances": s.vineyard_instances,  # Forest-Items (grape_vine)
            "water": s.water,  # {"rivers": [...], "ponds": [...]} for export_water()
            "wall_meshes": s.wall_meshes,  # Mesh dicts of the rubble stone walls for export_walls()
            "tunnel_meshes": s.tunnel_meshes,  # Tunnel/gallery mesh dicts for export_tunnels()
            "tunnel_spawns": s.tunnel_spawns,  # Spawn points in front of tunnel entrances for ItemManager.save(fixed_spawns=...)
            "roadblocks": s.roadblocks,  # Roadblocks in front of entrances of tunnels beyond the map border, for export_roadblocks()
            "tunnel_zones": s.tunnel_zones,  # Zone boxes for dark tunnel tubes, for export_tunnel_zones()
            "tunnel_lights": s.tunnel_lights,  # SpotLight fixtures inside the tubes, for export_tunnel_lights()
            "dropped_tunnel_road_ids": s.dropped_tunnel_road_ids,  # pieces of pass-through tunnels: no AI DecalRoad
            "grid": s.grid,
            "road_polygons": s.road_polygons,
            "road_slope_polygons_2d": s.road_slope_polygons_2d,  # For DecalRoad export
            "structure_road_polygons": s.structure_road_polygons,  # Bridges/tunnels/galleries - for export_bridges()/export_tunnels()
            "bridge_photo_areas": _bridge_photo_areas(s.structure_road_polygons),  # retouched out of the aerial photo
            "road_surface_union": s.road_surface_union,  # unioned road surface for exclusion zones (or None)
            "tree_exclusion": s.tree_exclusion,  # road surfaces plus the areas under bridges: no trees there (or None)
            "grid_bounds_local": s.grid_bounds_local,
            "global_offset": s.global_offset,
            "buildings_data": s.buildings_data,  # Pass on the building data
            "height_points": s.local_points,  # For spawn point calculation
            "height_elevations": s.elevations,  # For spawn point calculation
            "height_hash": s.tile_hash,  # For cache consistency in the forest workflow
        }

    def _tile_load_osm(self, s: "TileState", bbox_margin: float, sub) -> Optional[Dict]:
        """Elevation point cloud, OSM data and road polygons of the whole area. Returns a failure result dict (and
        fails `sub`) when there is no height or OSM data, otherwise None."""
        from ..osm.parser import calculate_bbox_from_height_data, extract_roads_from_osm
        from ..osm.downloader import get_osm_data
        from ..geometry.polygon import get_road_polygons
        from ..io.cache import calculate_global_tiles_hash

        tiles, global_offset = s.tiles, s.global_offset

        # 1. Combine the elevation data of all tiles into one point cloud
        height_points, height_elevations = self.tile_processor.load_height_data_multi(tiles)
        if height_points is None:
            sub.fail("no height data")
            return {"status": "failed", "reason": "no_height_data"}

        # Combined hash over all tiles - cache identity for OSM/
        # elevation/grid (replaces the former per-tile file hash)
        tile_hash = calculate_global_tiles_hash(tiles) if tiles else "unknown"

        # 2. Compute the BBox (BEFORE the local transformation!)
        # Compute the BBox with margin directly in UTM (meters), then transform to WGS84
        osm_bbox = calculate_bbox_from_height_data(height_points, margin=bbox_margin)

        # 3. Transform to local coordinates
        local_points, elevations = self.tile_processor.ensure_local_offset(
            global_offset, height_points, height_elevations
        )

        # 4. Load OSM data (with tile_hash for the tile-specific cache)
        osm_data = get_osm_data(osm_bbox, height_hash=tile_hash)

        if not osm_data:
            logger.warning("  [!] No OSM data")
            sub.fail("no OSM data")
            return {"status": "failed", "reason": "no_osm_data"}

        # 5. Extract roads
        roads = extract_roads_from_osm(osm_data)

        # 6. Road polygons (converts OSM data to coords)
        # IMPORTANT: Pass LOCAL coordinates! All internal calculations in local!

        road_polygons = get_road_polygons(roads, osm_bbox, local_points, elevations, global_offset, tile_hash=tile_hash)

        s.local_points, s.elevations, s.tile_hash = local_points, elevations, tile_hash
        s.osm_bbox, s.osm_data, s.road_polygons = osm_bbox, osm_data, road_polygons
        return None

    def _tile_load_buildings(self, s: "TileState", buildings_data: Optional[Dict]) -> None:
        """LoD2 buildings (if enabled and not passed in), with church towers marked for a tower clock."""
        osm_data, global_offset = s.osm_data, s.global_offset

        # 6b. Load LoD2 buildings (if enabled and not yet passed in)
        if buildings_data is None and config.LOD2_ENABLED:
            from ..io.lod2 import cache_lod2_buildings, load_buildings_from_cache

            # Compute Z-min from the elevation data for full 3D normalization
            # IMPORTANT: Do NOT normalize building Z coordinates!
            # The terrain itself has absolute elevations (263-580m), not normalized.
            # The buildings in CityGML also have absolute elevations above sea level.
            # Therefore: z_offset = 0 (no Z normalization for buildings!)
            z_offset = 0.0
            # Extend global_offset to a 3D offset
            local_offset_3d = (global_offset[0], global_offset[1], z_offset)

            # Try to load normalized buildings directly from the cache
            # (they were already normalized by cache_lod2_buildings)
            buildings_cache_path = cache_lod2_buildings(
                lod2_dir=config.LOD2_DATA_DIR,
                bbox=s.osm_bbox,  # WGS84-BBox
                local_offset=local_offset_3d,  # 3D offset with Z-min!
                cache_dir=config.CACHE_DIR,
                height_hash=s.tile_hash,
            )
            if buildings_cache_path:
                buildings_data = load_buildings_from_cache(buildings_cache_path)
                if buildings_data:
                    logger.info(f"  [OK] {len(buildings_data)} normalized buildings loaded from cache")

            if not buildings_data:
                logger.info("  [i] No LoD2 buildings found")

        # Church towers: no windows, a tower clock instead (church from OSM, tower walls from the geometry)
        if buildings_data:
            from ..facade.church_towers import ChurchTowerFinder
            from ..osm.landuse_polygons import make_local_transform

            towers = ChurchTowerFinder.from_osm(osm_data, make_local_transform(global_offset)).mark(buildings_data)
            logger.info(f"  [OK] {towers} churches with a tower detected (tower clock instead of windows)")

        s.buildings_data = buildings_data

    def _tile_road_network(self, s: "TileState") -> None:
        """Clipped road network with junctions, underpass heights and lane splits, converted into the road dicts
        (road_slope_polygons_2d) that the terrain and the DecalRoad export work on."""
        from ..geometry.polygon import clip_road_polygons
        from ..geometry.junctions import build_junction_network

        local_points, road_polygons = s.local_points, s.road_polygons

        # Compute grid bounds from local points for clipping
        grid_bounds_local = (
            float(local_points[:, 0].min()),
            float(local_points[:, 0].max()),
            float(local_points[:, 1].min()),
            float(local_points[:, 1].max()),
        )

        # Use ROAD_CLIP_MARGIN from the config (negative = expand!)
        road_polygons = clip_road_polygons(road_polygons, grid_bounds_local, margin=config.ROAD_CLIP_MARGIN)

        # 7. Junction detection (needs road_polygons with coords): detect → split → mark, tunnels excluded
        # (they lie on a different level, never form junctions - see build_junction_network())
        road_polygons, junctions = build_junction_network(road_polygons)

        # Roads that pass under a bridge: the terrain model shows the deck there, so their sampled height climbs to it -
        # interpolate from before to behind the bridge (the embedding below then cuts them in with slopes on both sides)
        from ..geometry.road_structures import fix_underpass_elevations

        underpasses = fix_underpass_elevations(
            road_polygons,
            lambda road: config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"] / 2.0,
            max_search=config.UNDERPASS_MAX_SEARCH,
            stable_length=config.UNDERPASS_STABLE_LENGTH,
            max_grade=config.UNDERPASS_STABLE_GRADE,
            min_rise=config.UNDERPASS_MIN_RISE,
        )
        if underpasses:
            logger.debug(f"  [OK] {underpasses} road(s) under bridges: height interpolated")

        # Lane splits (motorway exits/entrances, turn lanes): OSM draws all ways into one node - every branch is moved
        # into its lanes of the trunk's cross-section there and fades into its own course (bridges and ground alike)
        from ..geometry.lane_splits import directional_lanes, find_lane_splits, shift_branches_into_slots

        def _mapper_width(road):
            return config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"]

        lane_splits = find_lane_splits(
            road_polygons, _mapper_width,
            lanes_of=lambda road: directional_lanes(
                road.get("osm_tags", {}), _mapper_width(road), config.ROAD_MARKING_MIN_TWO_LANE_WIDTH
            ),
        )
        shift_branches_into_slots(
            lane_splits, road_polygons, max_connector=config.ROAD_LANE_SPLIT_MAX_CONNECTOR, length=config.ROAD_LANE_SPLIT_LENGTH,
            leave_gap=2.0 * config.BRIDGE_CURB_WIDTH,  # a branch leaves the bridge deck where both have room for a curb
        )
        if lane_splits:
            logger.info(f"  [OK] {len(lane_splits)} lane split(s): branches moved into the trunk's lanes")

        # Convert road_polygons into road_slope_polygons_2d (for classification)
        # IMPORTANT: AFTER junction detection, so that the split roads are used!
        # IMPORTANT: Create actual road polygons (buffer around the centerline)
        from shapely.geometry import LineString
        from ..config import OSM_MAPPER
        from ..geometry.road_structures import classify_structure
        from ..utils.debug_exporter import DebugNetworkExporter

        road_slope_polygons_2d = []
        debug_exporter = DebugNetworkExporter.get_instance()

        for road in road_polygons:
            coords = np.asarray(road.get("coords", []), dtype=float)
            if len(coords) < 2:
                continue

            # Compute the road width from OSM tags
            osm_tags = road.get("osm_tags", {})
            road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]

            # Create the polygon by buffering the centerline
            centerline_2d = coords[:, :2]
            try:
                line = LineString(centerline_2d)
                road_poly = line.buffer(road_width / 2.0, cap_style=2)  # cap_style=2 = flat
                road_polygon_2d = np.array(road_poly.exterior.coords[:-1])  # without duplicate
            except Exception:
                # Fallback: use the centerline directly
                road_polygon_2d = centerline_2d

            road_id = road.get("id")
            road_slope_polygons_2d.append(
                {
                    "road_id": road_id,  # Important for material mapping
                    "road_polygon": road_polygon_2d,
                    "trimmed_centerline": coords,
                    "osm_tags": osm_tags,
                    "structure_type": classify_structure(osm_tags),
                    "daylight_slopes": bool(road.get("underpass")),  # road under a bridge: slopes up to the terrain
                    **{key: road[key] for key in ("lane_split_trunk", "lane_split_trunk_nodes", "lane_split_branch") if key in road},
                }
            )

            # Export the road for debug visualization
            debug_exporter.add_line(
                coords,
                color=[0.0, 0.0, 1.0],
                width=2.0,
                label=f"Road_{road_id}",
            )

        # The terrain follows the widths that the DecalRoads get along the width transitions (lane changes, structures)
        widened = _attach_width_nodes(road_slope_polygons_2d)
        if widened:
            logger.debug(f"  [OK] {widened} road(s) with a width transition: embankment follows the blended width")

        s.grid_bounds_local, s.road_polygons, s.road_slope_polygons_2d = grid_bounds_local, road_polygons, road_slope_polygons_2d

    def _tile_heightmap(self, s: "TileState") -> None:
        """Regular grid and the still natural terrain heightmap built from it."""
        # 8. Create the grid (with builder)
        from ..builders import GridBuilder
        from ..terrain.heightmap import build_heightmap

        grid_builder = GridBuilder()
        grid = (
            grid_builder.with_points(s.local_points)
            .with_elevations(s.elevations)
            .with_spacing(config.GRID_SPACING)
            .with_cache_key(f"grid_{s.tile_hash}")
            .build()
        )

        # 9. Extract grid dimensions (vertex classification no longer applies -
        # the terrain is no longer triangulated, see task 9)
        grid_points, grid_elevations, nx, ny = grid

        # 10. Terrain heightmap instead of mesh triangulation.
        # Since the switch to DecalRoad, roads are no longer built as a mesh
        # (no RoadMeshBuilder/junction fan material majority vote
        # needed anymore) - see export_decal_roads().
        heightmap_result = build_heightmap(
            grid_points, grid_elevations, nx, ny, config.TERRAIN_SQUARE_SIZE
        )
        s.grid, s.nx, s.ny = grid, nx, ny
        s.heights = heightmap_result["heights"]
        s.terrain_size = heightmap_result["size"]
        s.terrain_origin_x = heightmap_result["origin_x"]
        s.terrain_origin_y = heightmap_result["origin_y"]

    def _tile_shape_terrain(self, s: "TileState") -> None:
        """Embankments and embedding of the surface roads and galleries, bridge abutments capped to the deck, tunnel
        cover and portals - all on the heightmap, in this order."""
        from ..config import OSM_MAPPER
        from ..geometry.road_structures import split_by_structure_type
        from ..terrain.road_embedding import (
            embed_roads_into_heightmap,
            build_road_embankment_profiles,
            apply_embankment_blend,
            sample_heightmap_bilinear,
        )

        heights, terrain_origin_x, terrain_origin_y = s.heights, s.terrain_origin_x, s.terrain_origin_y

        # Embankment: create the transition from the road edge to the natural surroundings directly
        # in the heightmap (the mesh no longer generates embankment geometry - see spec section 4b).
        # IMPORTANT: must run on the still UNMODIFIED heights, so that
        # "natural height" is really natural (before embed_roads_into_heightmap).
        # Bridges/tunnels are NOT embedded into the terrain and get no embankment
        # (the terrain below/beside them stays completely natural). Galleries,
        # on the other hand, ARE embedded like normal roads (see below) - no separate terrain hole
        # needed anymore, since floor/wall/roof are solid boxes (tunnels/gallery_mesh.py).
        surface_road_polygons, structure_road_polygons = split_by_structure_type(s.road_slope_polygons_2d)

        # Embed galleries like normal roads (same embankment/embedding parameters), but with
        # fixed instead of computed embankment widths on both sides (slope_width_override, see
        # build_road_embankment_profiles() docstring):
        # - Mountain side: GALLERY_MOUNTAIN_EMBED_MARGIN (1 m) beyond the inner edge of the (solid) wall,
        #   FLAT at road surface height (flat_shoulder_sides, no interpolation to the terrain - the wall reaches
        #   into the slope anyway, this narrow fringe only ensures a clean wall-floor transition).
        # - Valley side: GALLERY_VALLEY_SLOPE_WIDTH, real downward interpolation to the natural terrain (the
        #   DGM shows the real valley-side structure right at the road edge instead of natural terrain, see
        #   build_road_embankment_profiles() docstring - a computed width would be noisy/faceted).
        # Without an avalanche_protector:left/right tag the valley side is determined from the (still natural) terrain -
        # the gallery mesh later uses the same side (see _gallery_embedding()).
        def _natural_ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        def _gallery_road(r):
            override, flat_sides = _gallery_embedding(r, _natural_ground_at)
            return {**r, "slope_width_override": override, "flat_shoulder_sides": flat_sides}

        gallery_roads = [_gallery_road(r) for r in structure_road_polygons if r.get("structure_type") == "gallery"]
        # Gallery embankments end flush with the gallery, the approach roads' flush with the transition
        _gallery_embankment_cuts(surface_road_polygons, gallery_roads, config.ROAD_CONTINUATION_ENDPOINT_TOL)
        embeddable_roads = surface_road_polygons + gallery_roads

        embankment_profiles = build_road_embankment_profiles(
            embeddable_roads,
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            OSM_MAPPER,
            config.SLOPE_ANGLE,
            config.MIN_SLOPE_WIDTH,
            max_slope_width=config.MAX_SLOPE_WIDTH,
        )
        heights = apply_embankment_blend(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, embankment_profiles)

        # Road embedding: set the terrain on the pure road surface exactly to
        # centerline height (the embankment is already covered by
        # apply_embankment_blend) - see the road_embedding.py
        # module docstring for the DecalRoad rationale.
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            embeddable_roads,
        )

        # Bridges: cap terrain that lies HIGHER than the deck within the bridge width to deck level
        # (not necessarily set it as above) - this practically only affects the bridge ends (abutments), where the
        # road surface merges into the natural terrain and is not necessarily flat across the driving direction;
        # without capping, the terrain could poke through the (flat) deck in places. The valley floor that
        # the bridge spans stays visible unchanged (well below deck level, clamp_to_max
        # does not apply there).
        bridge_roads = [r for r in structure_road_polygons if r.get("structure_type") == "bridge"]
        if bridge_roads:
            heights = embed_roads_into_heightmap(
                heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, bridge_roads,
                clamp_to_max=True,
            )

        # Tunnel: join pieces into chains, define portals and adapt the terrain to them - cover
        # above the tube, portal zone at floor height, hole cells behind the portal (see terrain/tunnel_terrain.py).
        # After the road embedding; the surface roads themselves are left untouched.
        from ..geometry.road_surfaces import union_road_surfaces

        tunnel_plans = []
        tunnel_holes = None
        dropped_tunnel_road_ids = frozenset()
        if config.TUNNELS_ENABLED:
            from ..terrain.tunnel_terrain import shape_terrain_for_tunnels

            tunnel_plans = _plan_tunnels(structure_road_polygons)
            if tunnel_plans:
                import shapely

                all_tunnel_piece_ids = {pid for plan in tunnel_plans for pid in plan.get("piece_ids", [plan["id"]])}

                protected = union_road_surfaces(surface_road_polygons + gallery_roads)
                if protected is not None:
                    shapely.prepare(protected)
                heights, tunnel_holes = shape_terrain_for_tunnels(
                    heights,
                    terrain_origin_x,
                    terrain_origin_y,
                    config.TERRAIN_SQUARE_SIZE,
                    tunnel_plans,
                    cover=config.TUNNEL_COVER,
                    protected=protected,
                    bounds=s.grid_bounds_local,
                    edge_margin=config.MAP_EDGE_TUNNEL_MARGIN,
                )
                # tunnel_plans was filtered in place: pass-through chains without a reachable portal are gone
                dropped_tunnel_road_ids = _dropped_tunnel_road_ids(all_tunnel_piece_ids, tunnel_plans)

        s.heights = heights
        s.surface_road_polygons, s.structure_road_polygons = surface_road_polygons, structure_road_polygons
        s.gallery_roads, s.bridge_roads = gallery_roads, bridge_roads
        s.tunnel_plans, s.tunnel_holes, s.dropped_tunnel_road_ids = tunnel_plans, tunnel_holes, dropped_tunnel_road_ids

    def _tile_building_ground(self, s: "TileState") -> None:
        """Ground level per wall on the finished terrain for buildings whose body reaches into the ground
        (swissBUILDINGS3D, see io/swissbuildings_dxf.py::attach_wall_ground())."""
        if not s.buildings_data:
            return
        from ..io.swissbuildings_dxf import attach_wall_ground
        from ..terrain.road_embedding import sample_heightmap_bilinear

        heights, origin_x, origin_y = s.heights, s.terrain_origin_x, s.terrain_origin_y
        count = attach_wall_ground(
            s.buildings_data,
            lambda xy: sample_heightmap_bilinear(heights, origin_x, origin_y, config.TERRAIN_SQUARE_SIZE, np.asarray(xy)),
        )
        if count:
            logger.debug(f"  [OK] {count} building(s): ground level per wall from the terrain")

    def _tile_ponds(self, s: "TileState") -> None:
        """OSM land use polygons and the pond basins lowered into the terrain; keeps the heights before the basins as
        natural_heights (the water level comes from the natural edge)."""
        from ..osm.landuse_polygons import build_landuse_polygons, make_local_transform

        heights = s.heights

        # At this point osm_data still contains RAW Overpass geometry (lat/lon).
        # Land use polygons are built from ways AND multipolygon relations
        # (large forest/vineyard/residential areas are usually relations in OSM).
        landuse_polygons = build_landuse_polygons(s.osm_data, make_local_transform(s.global_offset))

        # Pond basins: lower the terrain within the water areas (before everything that reuses the elevations:
        # streams, trees, vines). The water level comes from the natural edge, see _build_water().
        natural_heights = heights
        if config.WATER_ENABLED:
            from ..terrain.water import carve_pond_basins, select_pond_areas

            pond_areas = select_pond_areas(landuse_polygons, water_bounds(s.grid_bounds_local))
            if pond_areas:
                heights = carve_pond_basins(
                    heights,
                    s.terrain_origin_x,
                    s.terrain_origin_y,
                    config.TERRAIN_SQUARE_SIZE,
                    pond_areas,
                    depth=config.WATER_POND_BANK_DEPTH,
                    slope_deg=config.WATER_POND_BANK_SLOPE_DEG,
                )
                logger.info(
                    f"  [OK] Pond basins: {len(pond_areas)} water area(s), terrain {config.WATER_POND_BANK_DEPTH * 100:.0f} cm lower "
                    f"(bank {config.WATER_POND_BANK_SLOPE_DEG:.0f} degrees)"
                )

        s.landuse_polygons, s.natural_heights, s.heights = landuse_polygons, natural_heights, heights

    def _tile_layer_map(self, s: "TileState") -> None:
        """Terrain layer map: aerial photo base, OSM land use painted on top, ground cover masked off roads, buildings
        and near bridge decks, holes (padding, tunnel portals), split per photo tile in four-photo mode. Also the
        road/building shapes and the tree exclusion zone that the vegetation reuses."""
        from ..geometry.road_surfaces import union_road_surfaces
        from ..osm.landuse_polygons import build_landuse_polygons, make_local_transform
        from ..terrain.terrain_materials import (
            DEFAULT_LANDUSE_CATEGORY,
            build_photo_fallback_layer,
            mark_padding_as_holes,
            mask_layer_map_with_photo,
            paint_landuse_materials,
        )

        heights, terrain_size = s.heights, s.terrain_size
        terrain_origin_x, terrain_origin_y = s.terrain_origin_x, s.terrain_origin_y
        osm_data, global_offset = s.osm_data, s.global_offset
        tunnel_plans, bridge_roads = s.tunnel_plans, s.bridge_roads

        # Layer map: ONE aerial photo material for the whole area, then OSM
        # land use on top (see build_photo_fallback_layer()).
        layer_map, photo_tile_names = build_photo_fallback_layer(terrain_size)

        # background_category: areas without any land use polygon (no OSM element covers them)
        # still get meadow this way instead of staying on the photo fallback forever - closes the gap that
        # get_landuse_category()'s DEFAULT_LANDUSE_CATEGORY fallback leaves open (it only applies to an
        # EXISTING but unknown landuse tag value, see paint_landuse_materials() docstring).
        layer_map, terrain_material_names = paint_landuse_materials(
            layer_map,
            photo_tile_names,
            terrain_size,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            s.landuse_polygons,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            background_category=DEFAULT_LANDUSE_CATEGORY,
        )

        # Road and building areas: (1) ground cover grows on the layer - there it goes
        # back to the aerial photo, otherwise grass grows through decals and houses; (2) exclusion zone
        # for the vineyard vines.
        # All road surfaces unioned ONCE (simplified): serves the mask, vine exclusion and the forest.
        # Surface roads AND galleries (now embedded into the terrain like normal roads, see above) -
        # only bridges/tunnels are left out, they should not block vegetation, otherwise e.g. a tunnel
        # would leave a bare strip over the whole ridge. The tunnel portal blocks, however,
        # do count (otherwise trees/grass would grow through the block).
        from ..tunnels.tunnel_portal import portal_footprint

        portal_footprints = [
            {"road_polygon": np.array(portal_footprint(portal))} for plan in tunnel_plans for portal in plan["portals"] if portal["open"]
        ]
        road_surface_union = union_road_surfaces(s.surface_road_polygons + s.gallery_roads + portal_footprints)
        road_shapes = [road_surface_union] if road_surface_union is not None else []
        building_shapes = [
            p["geometry"]
            for p in build_landuse_polygons(osm_data, make_local_transform(global_offset), tag_keys=("building",))
        ]

        if config.GROUND_COVER_ENABLED:
            layer_map = mask_layer_map_with_photo(
                layer_map,
                terrain_size,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
                road_shapes,
                buffer=config.GROUND_COVER_ROAD_MARGIN,
            )
            layer_map = mask_layer_map_with_photo(
                layer_map,
                terrain_size,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
                building_shapes,
                buffer=config.GROUND_COVER_BUILDING_MARGIN,
            )

        # Bridges: no grass where the terrain lies so close below the deck that it would grow through it (hillside
        # bridges); under bridges spanning a valley the grass stays. No trees anywhere under a bridge.
        bridge_footprints = _bridge_footprints(bridge_roads, config.BRIDGE_CURB_WIDTH + config.BRIDGE_UNDERGROWTH_MARGIN)
        if config.GROUND_COVER_ENABLED and bridge_footprints:
            from ..terrain.road_embedding import near_deck_mask

            near_deck = near_deck_mask(
                heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, bridge_footprints,
                clearance=config.BRIDGE_UNDERGROWTH_CLEARANCE,
            )
            layer_map = layer_map.copy()
            layer_map[near_deck] = 0  # aerial photo layer: nothing grows on it
        tree_exclusion = road_surface_union
        bridge_tree_areas = _bridge_footprints(bridge_roads, config.BRIDGE_CURB_WIDTH + config.BRIDGE_TREE_MARGIN)
        if bridge_tree_areas:
            from shapely import union_all
            from shapely.geometry import Polygon

            parts = [Polygon(b["road_polygon"]) for b in bridge_tree_areas]
            tree_exclusion = union_all(parts + ([road_surface_union] if road_surface_union is not None else []))

        # Excess border of the power-of-two heightmap (extrapolation only) as a hole: the visible
        # terrain ends exactly at the data edge, the horizon covers the strip behind it.
        # Last, so that painting/masking above run unchanged on the full layer map.
        if config.TERRAIN_PADDING_AS_HOLES:
            layer_map = mark_padding_as_holes(layer_map, data_cols=s.nx, data_rows=s.ny)
        # Tunnel portals: hole cells at the portal level (hidden by the tube shell or collar)
        tunnel_holes = s.tunnel_holes
        if tunnel_holes is not None and np.any(tunnel_holes):
            from ..terrain.ter_writer import EMPTY_LAYER_VALUE

            layer_map = layer_map.copy()
            layer_map[tunnel_holes] = EMPTY_LAYER_VALUE

        # Four-photo mode: only now (painting, masks and holes are done) is the layer map split per tile into
        # physical materials - each tile gets its own photo. The photo tiling is a
        # FIXED grid over the whole area (config.PHOTO_TILE_SIZE_M), independent of the size/number of the
        # raw elevation data tiles (see terrain/photo_tiles.py module docstring) - export/beamng_exporter.py
        # builds the same grid from the same inputs (deterministic, without having to share data).
        from ..terrain.photo_tiles import build_processing_tile_grid
        from ..utils.tile_scanner import compute_global_bbox

        processing_tiles = build_processing_tile_grid(compute_global_bbox(s.tiles), config.PHOTO_TILE_SIZE_M)

        photo_tiles = None
        if config.AERIAL_PHOTO_PER_TILE and len(processing_tiles) > 1:
            from ..terrain.photo_tiles import split_layers_by_tile

            photo_tiles = split_layers_by_tile(
                layer_map,
                terrain_material_names,
                processing_tiles,
                global_offset,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
            )
            layer_map = photo_tiles["layer_map"]
            terrain_material_names = photo_tiles["material_names"]
            photo_tile_names = photo_tiles["photo_tile_names"]
            logger.info(
                f"  [OK] Four-photo mode: {len(photo_tile_names)} aerial photos, {len(terrain_material_names)} terrain materials"
            )

        s.layer_map, s.terrain_material_names, s.photo_tile_names, s.photo_tiles = (
            layer_map, terrain_material_names, photo_tile_names, photo_tiles
        )
        s.road_surface_union, s.road_shapes, s.building_shapes, s.tree_exclusion = (
            road_surface_union, road_shapes, building_shapes, tree_exclusion
        )

    def _tile_scene_objects(self, s: "TileState") -> None:
        """Everything placed on the finished heightmap: vineyard vines, water, stone walls, tunnel/gallery meshes with
        zones, lights and roadblocks, POI and tunnel entrance spawn points."""
        heights, terrain_origin_x, terrain_origin_y = s.heights, s.terrain_origin_x, s.terrain_origin_y
        osm_data, global_offset, grid_bounds_local = s.osm_data, s.global_offset, s.grid_bounds_local
        landuse_polygons, tunnel_plans = s.landuse_polygons, s.tunnel_plans

        # Vineyard vines (forest items) along the fall line, on the finished heightmap
        vineyard_instances = []
        if config.VINEYARDS_ENABLED and config.FORESTS_ENABLED:
            from ..forest.vineyard_generator import build_exclusion_geometry, generate_vineyards, make_height_sampler
            from shapely.geometry import box

            vineyard_instances = generate_vineyards(
                landuse_polygons,
                config.OSM_MAPPER.config.get("landuse_mappings", {}),
                make_height_sampler(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE),
                exclusion=build_exclusion_geometry(s.road_shapes + s.building_shapes, config.VINEYARD_EXCLUSION_MARGIN),
                # Only over real elevation data: the OSM query extends beyond the terrain,
                # and the terrain edge is filled in (there are no real elevations there)
                bounds=box(
                    grid_bounds_local[0] + config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[2] + config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[1] - config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[3] - config.VINEYARD_EXCLUSION_MARGIN,
                ),
            )
            logger.debug(f"  [OK] {len(vineyard_instances)} vine row segments generated")

        # Real water: streams as River splines, water areas as WaterBlocks (on the finished heightmap)
        water = {"rivers": [], "ponds": []}
        if config.WATER_ENABLED:
            water = self._build_water(
                osm_data,
                landuse_polygons,
                global_offset,
                make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y),
                grid_bounds_local,
                rim_height_at=make_height_sampler_for_water(s.natural_heights, terrain_origin_x, terrain_origin_y),
            )

        # Rubble stone walls (OSM barrier=wall with height) on the finished heightmap
        wall_meshes = []
        if config.WALLS_ENABLED:
            wall_meshes, _ = self._build_wall_meshes(
                osm_data,
                global_offset,
                make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y),
                s.road_slope_polygons_2d,
            )

        # Bridges (deck + piers) are built in export_bridges(): their width follows the blended widths of the transitions,
        # which export_decal_roads() computes first - see bridges/bridge_mesh.py

        # Tunnels (tube + portals) and galleries (roof + supports) on the finished heightmap - see tunnels/
        tunnel_meshes = []
        roadblocks = []
        tunnel_zones = []
        tunnel_lights = []
        if config.TUNNELS_ENABLED:
            from ..tunnels.tunnel_lights import build_lamp_mesh

            tunnel_zones = _tunnel_zone_items(tunnel_plans)
            tunnel_lights = _tunnel_light_items(tunnel_plans)
            tunnel_meshes = self._build_tunnels(s.structure_road_polygons, tunnel_plans, heights, terrain_origin_x, terrain_origin_y)
            lamps = build_lamp_mesh(
                tunnel_lights, config.TUNNEL_LAMP_MATERIAL_NAME, config.TUNNEL_LAMP_LENGTH, config.TUNNEL_LAMP_WIDTH,
                config.TUNNEL_LAMP_HEIGHT, config.TUNNEL_LIGHT_CEILING_MARGIN,
            )
            if lamps is not None:
                tunnel_meshes.append({"id": "tunnel_lamps", **lamps})
            roadblocks = _roadblock_items(
                tunnel_plans, heights, terrain_origin_x, terrain_origin_y, grid_bounds_local, s.surface_road_polygons
            )

        # POI candidates (villages/towns, large parking lots) for additional spawn points selectable in the
        # vehicle selection - see osm/poi_points.py and ItemManager._compute_poi_spawn_points(). Height sampled on
        # the FINISHED heightmap (the positions come as pure XY points from OSM, not from
        # a road centerline).
        poi_points = self._collect_poi_points(osm_data, global_offset, heights, terrain_origin_x, terrain_origin_y, grid_bounds_local)

        # Selectable spawn points in front of tunnel chain entrances, facing into the tunnel (tunnels/entrance_spawns.py)
        from ..tunnels.entrance_spawns import plan_entrance_spawns

        tunnel_spawns = plan_entrance_spawns(s.road_slope_polygons_2d, config.TUNNEL_SPAWN_DISTANCE, config.POI_SPAWN_EXCLUDED_HIGHWAYS)

        s.vineyard_instances, s.water, s.wall_meshes = vineyard_instances, water, wall_meshes
        s.tunnel_meshes, s.roadblocks, s.tunnel_zones, s.tunnel_lights = tunnel_meshes, roadblocks, tunnel_zones, tunnel_lights
        s.poi_points, s.tunnel_spawns = poi_points, tunnel_spawns

    def _build_water(self, osm_data, landuse_polygons, global_offset, height_at, grid_bounds_local, rim_height_at=None) -> Dict:
        """
        Computes stream nodes and pond blocks (see terrain/water.py). The water heights are derived from the finished
        heightmap; the pond water level from the edge of the NATURAL heightmap (`rim_height_at`, without the
        pond basin - otherwise it would be too low by the bank depth), otherwise from `height_at`.

        Returns:
            {"rivers": [{"name", "waterway", "nodes"}], "ponds": [{"name", "blocks"}]}
        """
        from shapely.ops import unary_union

        from ..osm.landuse_polygons import make_local_transform
        from ..terrain.water import (
            build_pond_blocks,
            build_river_nodes,
            clip_line_to_bounds,
            cut_line_by_area,
            select_pond_areas,
            select_waterways,
            split_nodes,
        )

        bounds = water_bounds(grid_bounds_local)

        # Ponds first: the streams end at their bank
        ponds = []
        pond_areas = select_pond_areas(landuse_polygons, bounds)
        for geometry in pond_areas:
            blocks = build_pond_blocks(
                geometry,
                rim_height_at or height_at,
                depth=config.WATER_POND_DEPTH,
                cell=config.WATER_POND_CELL,
                margin=config.WATER_POND_MARGIN,
            )
            if blocks:
                ponds.append({"name": f"pond_{len(ponds)}", "blocks": blocks})
        pond_area = unary_union(pond_areas) if pond_areas else None

        rivers = []
        for way in select_waterways(osm_data, make_local_transform(global_offset), config.WATERWAY_WIDTHS):
            for clipped in clip_line_to_bounds(way["coords"], bounds):
                for part in cut_line_by_area(clipped, pond_area):
                    if len(part) < 2:
                        continue
                    nodes = build_river_nodes(
                        part,
                        height_at,
                        width=way["width"],
                        depth=config.WATER_RIVER_DEPTH,
                        spacing=config.WATER_NODE_SPACING,
                        lift=config.WATER_STREAM_LIFT,
                    )
                    for chunk in split_nodes(nodes, config.WATER_MAX_RIVER_NODES):
                        if len(chunk) >= 2:
                            rivers.append({"name": f"river_{len(rivers)}", "waterway": way["waterway"], "nodes": chunk})

        length = sum(sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(r["nodes"], r["nodes"][1:])) for r in rivers)
        logger.debug(
            f"  [OK] Water: {len(rivers)} river object(s) ({length:.0f} m of stream), "
            f"{len(ponds)} water area(s) with {sum(len(p['blocks']) for p in ponds)} WaterBlocks"
        )
        return {"rivers": rivers, "ponds": ponds}

    def _build_wall_meshes(self, osm_data, global_offset, height_at, road_polygons=None):
        """
        Rubble stone walls from OSM (barrier=wall/retaining_wall, only with a height tag), following the terrain (see
        walls/wall_mesh.py); at most config.WALL_ROAD_SNAP_M beside a road centerline at its height (road_base.py).

        Args:
            road_polygons: Road dicts with "trimmed_centerline" (see road_slope_polygons_2d)

        Returns:
            (mesh dicts for the DAE export, statistics {"built", "length", "without_height"})
        """
        from ..osm.landuse_polygons import make_local_transform
        from ..textures import library
        from ..walls.road_base import RoadBaseHeight, centerlines_from_roads
        from ..walls.wall_mesh import build_walls

        meshes, stats = build_walls(
            osm_data,
            make_local_transform(global_offset),
            height_at,
            config.WALL_MATERIAL_NAME,
            thickness=config.WALL_THICKNESS,
            sink=config.WALL_SINK,
            max_step=config.WALL_MAX_SEGMENT,
            tile_m=library.texture_tile_m(config.WALL_TEXTURE_NAME, config.WALL_TEXTURE_TILE_M),
            road_base_at=RoadBaseHeight(centerlines_from_roads(road_polygons or []), config.WALL_ROAD_SNAP_M),
            cap_thickness=config.WALL_CAP_THICKNESS,
            cap_overhang=config.WALL_CAP_OVERHANG,
            cap_plate_length=config.WALL_CAP_PLATE_LENGTH,
            cap_joint=config.WALL_CAP_JOINT,
        )
        logger.debug(
            f"  [OK] Walls: {stats['built']} rubble wall(s) with height ({stats['length']:.0f} m), "
            f"{stats['without_height']} without height skipped"
        )
        return meshes, stats

    def _build_bridges(
        self, structure_road_polygons: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float,
        bridge_widths: Optional[Dict] = None,
    ) -> List[Dict]:
        """Bridge meshes (deck + piers) for all roads with structure_type == "bridge" (see bridges/bridge_mesh.py).
        `bridge_widths`: road id -> DecalRoad nodes [x, y, z, width] with the blended widths (export_decal_roads()); the
        deck follows them."""
        from ..bridges.bridge_mesh import build_bridges
        from ..terrain.road_embedding import sample_heightmap_bilinear

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        bridge_roads = [road for road in structure_road_polygons if road.get("structure_type") == "bridge"]
        groups = _bridge_groups(
            bridge_roads,
            reach=config.ROAD_LANE_SPLIT_MAX_CONNECTOR + config.ROAD_LANE_SPLIT_LENGTH + config.BRIDGE_GROUP_EXTRA_REACH,
        )
        stems = _bridge_stems(bridge_roads)
        bridges = [
            {
                "id": road["road_id"],
                "group": groups.get(road["road_id"]),
                "stem": stems.get(road["road_id"]),
                "coords": road["trimmed_centerline"],
                "width": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"],
                "deck_material": f"{config.OSM_MAPPER.get_road_properties(road.get('osm_tags', {})).get('internal_name', 'road_default')}_structure",
                "widths": _widths_along(road["trimmed_centerline"], (bridge_widths or {}).get(road["road_id"])),
            }
            for road in bridge_roads
        ]
        return build_bridges(
            bridges,
            ground_at,
            pier_material=config.BRIDGE_MATERIAL_NAME,
            railing_material=config.BRIDGE_RAILING_MATERIAL_NAME,
            deck_thickness=config.BRIDGE_DECK_THICKNESS,
            pier_spacing=config.BRIDGE_PIER_SPACING,
            pier_width_fraction=config.BRIDGE_PIER_WIDTH_FRACTION,
            pier_depth_fraction=config.BRIDGE_PIER_DEPTH_FRACTION,
            pier_burial=config.BRIDGE_PIER_BURIAL,
            min_pier_clearance=config.BRIDGE_MIN_PIER_CLEARANCE,
            curb_width=config.BRIDGE_CURB_WIDTH,
            curb_height=config.BRIDGE_CURB_HEIGHT,
            railing_height=config.BRIDGE_RAILING_HEIGHT,
            railing_post_spacing=config.BRIDGE_RAILING_POST_SPACING,
            railing_post_size=config.BRIDGE_RAILING_POST_SIZE,
            road_texture_length=config.ROAD_DECAL_TEXTURE_LENGTH,
        )

    def export_bridges(self, mesh_data: Dict) -> int:
        """
        Exports bridges as ONE DAE (deck + piers per bridge) with ONE TSStatic and registers the road surface
        and concrete material. Without bridges, leftovers of a previous export are removed.

        Returns:
            Number of exported bridges
        """
        bridges_dir = config.BEAMNG_DIR_SHAPES / "bridges"
        meshes = mesh_data.get("bridge_meshes")  # ready-made mesh dicts (tests, callers that build them themselves)
        if meshes is None:
            # Built once, with the widths of the transitions from export_decal_roads() (the deck follows them)
            meshes = []
            if config.BRIDGES_ENABLED and mesh_data.get("heightmap") is not None:
                meshes = self._build_bridges(
                    mesh_data["structure_road_polygons"], mesh_data["heightmap"], mesh_data["terrain_origin_x"],
                    mesh_data["terrain_origin_y"], mesh_data.get("bridge_widths"),
                )
        if not config.BRIDGES_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (bridges_dir / f"bridges{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )
        railing = registry.prepared_textures()[config.RAILING_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_RAILING_MATERIAL_NAME,
            textures={**railing, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        unique_deck_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") != "bridge":
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            mat_name = f"{props.get('internal_name', 'road_default')}_structure"
            unique_deck_materials[mat_name] = props

        for mat_name, props in unique_deck_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=bridges_dir / "bridges.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "bridges",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "bridges" / "bridges.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.debug(f"  [OK] {len(meshes)} bridges exported (bridges.dae)")
        return len(meshes)

    def _build_tunnels(
        self, structure_road_polygons: List[Dict], tunnel_plans: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float
    ) -> List[Dict]:
        """Tunnel meshes (tube + portal blocks, from the tunnel plans, see tunnels/tunnel_portal.py) and gallery meshes
        (roof + supports) for all roads with structure_type "gallery" - see tunnels/gallery_mesh.py."""
        from ..terrain.road_embedding import sample_heightmap_bilinear
        from ..tunnels.gallery_mesh import build_galleries
        from ..tunnels.tunnel_mesh import build_tunnels

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        tunnel_meshes = build_tunnels(
            tunnel_plans,
            wall_material=config.TUNNEL_MATERIAL_NAME,
            portal_material=config.TUNNEL_MATERIAL_NAME,
            arc_segments=config.TUNNEL_ARC_SEGMENTS,
            transition_cover=config.TUNNEL_TRANSITION_COVER_THICKNESS,
            road_texture_length=config.ROAD_DECAL_TEXTURE_LENGTH,
        )
        gallery_meshes = build_galleries(
            _structure_items(structure_road_polygons, "gallery"),
            ground_at,
            roof_material=config.TUNNEL_MATERIAL_NAME,
            height=config.GALLERY_HEIGHT,
            column_spacing=config.GALLERY_COLUMN_SPACING,
            roof_thickness=config.GALLERY_ROOF_THICKNESS,
            floor_thickness=config.GALLERY_FLOOR_THICKNESS,
            wall_thickness=config.GALLERY_WALL_THICKNESS,
            column_size=config.GALLERY_COLUMN_SIZE,
            curb_height=config.GALLERY_CURB_HEIGHT,
            curb_width=config.GALLERY_CURB_WIDTH,
            road_texture_length=config.ROAD_DECAL_TEXTURE_LENGTH,
        )
        return tunnel_meshes + gallery_meshes

    def _collect_poi_points(
        self,
        osm_data: List[Dict],
        global_offset: Tuple[float, float],
        heights: np.ndarray,
        terrain_origin_x: float,
        terrain_origin_y: float,
        grid_bounds_local: Tuple[float, float, float, float],
    ) -> List[Dict]:
        """POI candidates (villages/towns, large parking lots) with their height on the finished heightmap - see
        osm/poi_points.py. Only within the real terrain area (the OSM query extends beyond the
        terrain, see the generate_vineyards() caller)."""
        if not config.POI_SPAWN_POINTS_ENABLED:
            return []

        from ..osm.landuse_polygons import make_local_transform
        from ..osm.poi_points import extract_parking_points, extract_place_points

        to_local = make_local_transform(global_offset)
        candidates = extract_place_points(osm_data, to_local) + extract_parking_points(
            osm_data, to_local, min_area_m2=config.POI_MIN_PARKING_AREA_M2
        )
        if not candidates:
            return []

        margin = config.POI_SPAWN_BOUNDS_MARGIN
        x_min, x_max, y_min, y_max = grid_bounds_local
        in_bounds = [
            c for c in candidates
            if x_min + margin <= c["position_xy"][0] <= x_max - margin
            and y_min + margin <= c["position_xy"][1] <= y_max - margin
        ]
        if not in_bounds:
            return []

        height_at = make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y)
        xs = np.array([c["position_xy"][0] for c in in_bounds])
        ys = np.array([c["position_xy"][1] for c in in_bounds])
        zs = height_at(xs, ys)
        return [
            {**candidate, "position": [float(x), float(y), float(z)]}
            for candidate, x, y, z in zip(in_bounds, xs, ys, zs)
        ]

    def export_tunnel_zones(self, mesh_data: Dict) -> int:
        """Zone and portal objects that darken the tunnel tubes (see _tunnel_zone_items()).

        Returns:
            Number of zones
        """
        zones = mesh_data.get("tunnel_zones") or []
        for zone in zones:
            self.items.add_item(
                zone["name"],
                item_class=zone["class"],
                position=zone["position"],
                rotation_matrix=zone["rotation_matrix"],
                scale=zone["scale"],
                overwrite=True,
                **zone["fields"],
            )
        return len(zones)

    def export_tunnel_lights(self, mesh_data: Dict) -> int:
        """SpotLight fixtures inside tunnel tubes (see _tunnel_light_items()).

        Returns:
            Number of lights
        """
        lights = mesh_data.get("tunnel_lights") or []
        for light in lights:
            self.items.add_item(
                light["name"],
                item_class=light["class"],
                position=light["position"],
                rotation_matrix=light["rotation_matrix"],
                overwrite=True,
                **light["fields"],
            )
        return len(lights)

    def export_roadblocks(self, mesh_data: Dict) -> int:
        """Roadblocks (see _roadblock_items()) as TSStatic with the BeamNG default asset config.ROADBLOCK_SHAPE.

        Returns:
            Number of barrier elements
        """
        blocks = mesh_data.get("roadblocks") or []
        for block in blocks:
            self.items.add_item(
                block["name"],
                item_class="TSStatic",
                shape_name=config.ROADBLOCK_SHAPE,
                position=block["position"],
                rotation_matrix=block["rotation_matrix"],
                overwrite=True,
            )
        return len(blocks)

    def export_tunnels(self, mesh_data: Dict) -> int:
        """
        Exports tunnels (tube + 2 portal frames per tunnel) and galleries (roof + supports) as ONE DAE with
        ONE TSStatic and registers the road surface and concrete material. Without tunnels/galleries, leftovers of a
        previous export are removed.

        Returns:
            Number of exported tunnel/portal/gallery meshes
        """
        tunnels_dir = config.BEAMNG_DIR_SHAPES / "tunnels"
        meshes = mesh_data.get("tunnel_meshes") or []
        if not config.TUNNELS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (tunnels_dir / f"tunnels{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.TUNNEL_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        # Lamp bodies: glowing without a texture (PBR stage like vanilla's emissive materials)
        self.materials.materials[config.TUNNEL_LAMP_MATERIAL_NAME] = {
            "name": config.TUNNEL_LAMP_MATERIAL_NAME,
            "mapTo": config.TUNNEL_LAMP_MATERIAL_NAME,
            "class": "Material",
            "version": 1.5,
            "Stages": [{
                "baseColorFactor": [*config.TUNNEL_LAMP_COLOR, 1.0],
                "emissive": True,
                "emissiveFactor": list(config.TUNNEL_LAMP_COLOR),
                "emissiveIntensityNits": config.TUNNEL_LAMP_EMISSIVE_NITS,
                "roughnessFactor": 0.4,
            }, {}, {}, {}],
            "castShadows": False,
        }

        unique_floor_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") not in ("tunnel", "gallery"):
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            mat_name = f"{props.get('internal_name', 'road_default')}_structure"
            unique_floor_materials[mat_name] = props

        for mat_name, props in unique_floor_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=tunnels_dir / "tunnels.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "tunnels",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "tunnels" / "tunnels.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.debug(f"  [OK] {len(meshes)} tunnel/gallery mesh(es) exported (tunnels.dae)")
        return len(meshes)

    def _set_fog_height(self, heights: np.ndarray) -> None:
        """
        fogAtmosphereHeight (height above which the height fog thins out) = highest terrain point + margin. All original
        levels set a value on the order of their terrain height; a fixed value would be wrong for our terrain (here
        236-689 m absolute).
        """
        height = float(np.max(heights)) + float(config.ENV_FOG_HEIGHT_MARGIN)
        self.items.set_base_line_fields("theLevelInfo", fogAtmosphereHeight=round(height, 1))

    def export_water(self, mesh_data: Dict) -> int:
        """
        Registers streams (`River`) and ponds/lakes (`WaterBlock`) as BeamNG objects. The render
        parameters come from BeamNG's own east_coast_usa level (data/water_templates.json,
        core textures only), see tools/extract_water_templates.py.

        Returns:
            Number of created water objects
        """
        water = mesh_data.get("water") or {}
        if not config.WATER_ENABLED or not (water.get("rivers") or water.get("ponds")):
            return 0

        import copy

        templates = json.loads(WATER_TEMPLATES_PATH.read_text(encoding="utf-8"))
        count = 0

        for river in water.get("rivers", []):
            fields = copy.deepcopy(templates["stream"]["fields"])
            fields.pop("class", None)
            nodes = river["nodes"]
            self.items.add_item(
                river["name"],
                item_class="River",
                position=tuple(nodes[0][:3]),
                overwrite=True,
                nodes=nodes,
                **fields,
            )
            count += 1

        for pond in water.get("ponds", []):
            for index, block in enumerate(pond["blocks"]):
                fields = copy.deepcopy(templates["pond"]["fields"])
                fields.pop("class", None)
                fields["cubemap"] = config.WATER_POND_CUBEMAP
                # The water grid must not be larger than the block (otherwise BeamNG warns and shortens it itself)
                fields["gridElementSize"] = float(min(fields.get("gridElementSize", 5.0), block["scale"][0], block["scale"][1]))
                self.items.add_item(
                    f"{pond['name']}_{index}",
                    item_class="WaterBlock",
                    position=tuple(block["position"]),
                    scale=tuple(block["scale"]),
                    overwrite=True,
                    **fields,
                )
                count += 1

        logger.debug(f"  [OK] {count} water object(s) exported")
        return count

    def export_walls(self, mesh_data: Dict) -> int:
        """
        Exports the rubble stone walls as ONE DAE (each wall a node) with ONE TSStatic and registers the
        stone material. Without walls, leftovers of a previous export are removed.

        Returns:
            Number of exported walls
        """
        walls_dir = config.BEAMNG_DIR_SHAPES / "walls"
        meshes = mesh_data.get("wall_meshes") or []
        if not config.WALLS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (walls_dir / f"walls{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        stone = registry.prepared_textures()[config.WALL_TEXTURE_NAME]  # if the photo is missing, the export was aborted long ago
        self.materials.add_building_material(
            config.WALL_MATERIAL_NAME,
            textures={**stone, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )
        self.dae.export_multi_mesh(output_path=walls_dir / "walls.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "walls",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "walls" / "walls.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.debug(f"  [OK] {len(meshes)} wall(s) exported (walls.dae)")
        return len(meshes)

    def _export_structure_road_assets(self, marking_lines: List[Dict]) -> None:
        """
        Files for the roads on structures (see config.STRUCTURE_AI_ROADS): the fully transparent texture of the invisible
        AI DecalRoad and the marking lines as mesh strips (geometry/marking_mesh.py) in ONE DAE with ONE TSStatic
        without collision (the strips must not make bumps). Without lines, leftovers of a previous export are removed.
        """
        from PIL import Image

        from ..geometry.marking_mesh import build_marking_meshes

        if config.STRUCTURE_AI_ROADS:
            config.BEAMNG_DIR_TEXTURES.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(config.BEAMNG_DIR_TEXTURES / f"{config.STRUCTURE_AI_ROAD_MATERIAL}.png")

        markings_dir = config.BEAMNG_DIR_SHAPES / "structure_markings"
        texture_lengths = {name: entry["textureLength"] for name, entry in config.OSM_MAPPER.road_markings.items()}
        meshes = build_marking_meshes(marking_lines, texture_lengths, config.STRUCTURE_MARKING_LIFT)
        if not meshes:
            for suffix in (".dae", ".cdae"):
                (markings_dir / f"structure_markings{suffix}").unlink(missing_ok=True)
            return
        self.dae.export_multi_mesh(output_path=markings_dir / "structure_markings.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "structure_markings",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "structure_markings" / "structure_markings.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="None",
        )
        logger.debug(f"  [OK] {len(meshes)} marking strip(s) on structures exported (structure_markings.dae)")

    def export_decal_roads(self, mesh_data: Dict) -> int:
        """
        Exports each road as its own BeamNG `DecalRoad` item - a
        spline decal that is projected directly onto the terrain surface
        at runtime (see the road_embedding.py module docstring for the
        rationale). Completely replaces the former mesh construction (prepare_road_export()/
        export_merged_roads()/RoadMeshBuilder/DAE export) - no
        road mesh, no junction fan geometry, no
        face material majority vote needed anymore: each road gets
        its own DecalRoad item with its own (OSM-derived)
        material, BeamNG's `autoJunction` connects adjacent roads
        automatically.

        Args:
            mesh_data: Result of process_tile() (needs
                "road_slope_polygons_2d")

        Returns:
            Number of created DecalRoad items
        """
        from ..config import OSM_MAPPER
        from ..geometry.decal_chunks import split_decal_nodes
        from ..geometry.road_width_transitions import close_continuation_gaps

        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        unique_materials: Dict[str, Dict] = {}
        specs, node_lists = _road_width_specs(road_slope_polygons_2d, mesh_data.get("dropped_tunnel_road_ids") or frozenset())

        if config.GUARDRAILS_ENABLED and mesh_data.get("heightmap") is not None:
            mesh_data["guardrail_instances"] = _guardrail_instances(specs, node_lists, mesh_data)

        # Extend the carriageway decals at kinked straight-through joints past the joint point (otherwise a wedge gap
        # on the outside, see close_continuation_gaps()). Only for the carriageway - the markings keep using node_lists.
        decal_node_lists = close_continuation_gaps(
            node_lists, config.ROAD_CONTINUATION_ENDPOINT_TOL, config.ROAD_CONTINUATION_MAX_ANGLE_DEG
        )

        # Bridges follow the blended widths of the transitions: export_bridges() rebuilds the deck from them
        mesh_data["bridge_widths"] = {
            poly["road_id"]: nodes
            for (poly, _, _), nodes in zip(specs, node_lists)
            if poly.get("structure_type") == "bridge"
        }

        count = 0
        for (poly, props, _), nodes in zip(specs, decal_node_lists):
            if poly.get("structure_type", "surface") != "surface":
                # Structure: only the AI road network - the visible carriageway is part of the structure mesh
                mat_name = config.STRUCTURE_AI_ROAD_MATERIAL
                self.materials.materials[mat_name] = _invisible_road_material(mat_name)
            else:
                mat_name = props.get("internal_name", "road_default")
                unique_materials[mat_name] = props

            # Derive renderPriority from the existing "priority" field
            # (surface_types in data/osm_to_beamng.json): at junctions the
            # ends (which always have full width) of several DecalRoad
            # objects overlap - without an explicit, consistent
            # drawing order BeamNG sorts them arbitrarily, which looks like a
            # "patchwork" at junctions. BeamNG draws
            # in DESCENDING renderPriority (smallest value on top, see
            # config.ROAD_RENDER_PRIORITY_BASE) - higher-grade roads
            # (asphalt) therefore get the smaller value and lie above
            # lower-grade ones (dirt/concrete).
            render_priority = config.ROAD_RENDER_PRIORITY_BASE - int(props.get("priority", 0))

            # Split long carriageways into pieces - BeamNG only draws a limited amount of geometry per DecalRoad (see
            # geometry/decal_chunks.py). The marking lines stay unsplit: narrow, far below the budget.
            chunks = split_decal_nodes(nodes, config.ROAD_DECAL_MAX_AREA, config.ROAD_DECAL_MIN_TAIL_LENGTH)
            for chunk_idx, chunk in enumerate(chunks):
                name = f"road_{poly.get('road_id')}" if len(chunks) == 1 else f"road_{poly.get('road_id')}_{chunk_idx}"
                self.items.add_decal_road(
                    name=name,
                    nodes=chunk,
                    material=mat_name,
                    drivability=props.get("drivability", 1.0),
                    overwrite=True,
                    autoLanes=True,
                    autoJunction=True,
                    improvedSpline=True,
                    renderPriority=render_priority,
                )
                count += 1

        road_material_entries = [
            OSM_MAPPER.generate_materials_json_entry(mat_name, props) for mat_name, props in unique_materials.items()
        ]
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        # Road markings as their own narrow DecalRoads on top (geometry/road_markings.py). drivability=-1:
        # BeamNG's AI road network (lua/ge/map.lua) only picks up DecalRoads with drivability > 0.
        marking_count = 0
        structure_lines = []
        if config.ROAD_MARKINGS_ENABLED:
            used_markings = set()
            for line in _road_marking_lines(specs, node_lists):
                used_markings.add(line["material"])
                if line["structure"]:
                    structure_lines.append(line)  # mesh strips on the structure floor, see _export_structure_road_assets()
                    continue
                marking = OSM_MAPPER.road_markings[line["material"]]
                self.items.add_decal_road(
                    name=line["name"],
                    nodes=line["nodes"],
                    material=line["material"],
                    drivability=-1,
                    overwrite=True,
                    improvedSpline=True,
                    textureLength=marking["textureLength"],
                    renderPriority=config.ROAD_MARKING_RENDER_PRIORITY,
                )
                marking_count += 1
            for mat_name in sorted(used_markings):
                self.materials.materials[mat_name] = OSM_MAPPER.generate_marking_material_entry(
                    mat_name, OSM_MAPPER.road_markings[mat_name]
                )
        self._export_structure_road_assets(structure_lines)

        logger.debug(
            f"  [OK] {count} DecalRoad item(s) exported ({len(unique_materials)} materials), "
            f"{marking_count} marking line(s)"
        )
        return count

    def export_ground_cover(
        self, layer_map: np.ndarray, terrain_material_names: List[str], layer_variants: Optional[Dict] = None
    ) -> int:
        """
        Registers ground cover (grass, flowers, fern, weeds) as GroundCover
        objects for each terrain layer that actually occurs in the layer map,
        together with the billboard materials (shared BeamNG assets).

        Args:
            layer_map: finished global layer map (index into terrain_material_names)
            terrain_material_names: Layer names in index order
            layer_variants: Four-photo mode: layer -> tile variants (names in terrain_material_names)

        Returns:
            Number of created GroundCover objects
        """
        if not config.GROUND_COVER_ENABLED:
            return 0

        from ..terrain.ground_cover import (
            build_billboard_material_entries,
            build_ground_cover_items,
            load_ground_cover_templates,
        )

        used_physical = [terrain_material_names[i] for i in np.unique(layer_map) if i < len(terrain_material_names)]
        # In four-photo mode the layer map only contains variants (mat_grass_t0 ...): map back to the layer
        logical_of = {variant: layer for layer, variants in (layer_variants or {}).items() for variant in variants}
        used_layers = list(dict.fromkeys(logical_of.get(name, name) for name in used_physical))
        templates_data = load_ground_cover_templates()
        items = build_ground_cover_items(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            used_layers,
            templates_data,
            max_elements=config.GROUND_COVER_MAX_ELEMENTS,
            max_radius=config.GROUND_COVER_MAX_RADIUS,
            layer_variants=layer_variants,
        )
        for item in items:
            fields = dict(item)
            self.items.add_ground_cover(fields.pop("name"), fields.pop("material"), fields.pop("Types"), **fields)

        self.materials.materials.update(build_billboard_material_entries(items, templates_data))
        logger.info(f"  [OK] {len(items)} GroundCover object(s) for {len(used_layers) - 1} land use layers")
        return len(items)

    def export_merged_terrain(
        self,
        heights: np.ndarray,
        layer_map: np.ndarray,
        terrain_material_names: List[str],
        terrain_origin_x: float,
        terrain_origin_y: float,
        terrain_size: int,
        z_min: float,
        max_height: float,
        photo_tile_names: List[str],
        layer_variants: Optional[Dict] = None,
        variant_parents: Optional[Dict] = None,
        photo_extents: Optional[Dict] = None,
    ) -> None:
        """
        Writes ONE .ter and registers TerrainBlock + TerrainMaterials.

        Args:
            heights, layer_map: finished (already padded) global arrays,
                shape (terrain_size, terrain_size)
            terrain_material_names: global, deduplicated material list
                (index corresponds to layer_map values)
            terrain_origin_x, terrain_origin_y: World coordinates of cell [0, 0]
            z_min, max_height: see ter_writer.encode_heights_to_u16()
            photo_tile_names: Subset of terrain_material_names that refer to
                the composed aerial photo (see
                terrain_materials.build_terrain_material_entries)
        """
        from ..terrain.ter_writer import write_ter, encode_heights_to_u16
        from ..terrain.terrain_materials import (
            build_terrain_material_entries,
            build_terrain_material_texture_set,
            ensure_flat_pbr_placeholders,
            ensure_landuse_detail_textures_sized,
            DETAIL_TEX_SIZE,
        )

        self._set_fog_height(heights)
        heightmap_u16 = encode_heights_to_u16(heights, z_min, max_height)
        ter_filename = f"{config.LEVEL_NAME}.ter"
        ter_path = config.BEAMNG_DIR / ter_filename
        write_ter(ter_path, heightmap_u16, layer_map.astype("uint8"), terrain_material_names)
        logger.debug(f"  [OK] Terrain exported: {ter_filename} ({terrain_size}x{terrain_size})")

        placeholders = ensure_flat_pbr_placeholders(
            config.BEAMNG_DIR_TEXTURES, config.LEVEL_NAME, config.TERRAIN_BASE_TEX_PIXEL_SIZE
        )
        sized_landuse_mappings = ensure_landuse_detail_textures_sized(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            DETAIL_TEX_SIZE,
            config.BEAMNG_DIR,
            config.BEAMNG_DIR_TEXTURES,
            config.LEVEL_NAME,
        )
        # photo_extent_size: the value we report to BeamNG as baseColorBaseTexSize
        # for the aerial photo material.
        #
        # DIAGNOSIS 2026-09-18 (three measurement points with a real user test, see
        # the [[project_road_embed_slope_margin]] neighbor memory for the context
        # of this session):
        #   - Grid @ 2m/cell: terrain_size=1024, TERRAIN_SQUARE_SIZE=2.0,
        #     physical size 2048m -> declared value 1024 was correct.
        #   - Grid @ 1m/cell: terrain_size=2048, TERRAIN_SQUARE_SIZE=1.0,
        #     physical size UNCHANGED 2048m -> the same value 1024 (from the
        #     old "physical_size * 0.5" formula) was now WRONG, the correct one
        #     is 2048.
        # The physical map size stayed identical in both cases (2048m),
        # only the grid resolution changed - nevertheless the
        # correct value had to double with the grid resolution. That means
        # BeamNG apparently interprets baseColorBaseTexSize in heightmap
        # grid cells, NOT in world meters (contrary to the
        # documented formula "world_size = size * squareSize"): the correct value
        # is simply terrain_size, the plain .ter grid point count, completely
        # independent of TERRAIN_SQUARE_SIZE.
        photo_extent_size = float(terrain_size)
        terrain_material_entries = build_terrain_material_entries(
            terrain_material_names,
            photo_tile_names,
            sized_landuse_mappings,
            config.LEVEL_NAME,
            photo_extent_size,
            placeholders,
            variant_parents=variant_parents,
            photo_extents=photo_extents,
        )
        texture_set_name = f"{config.LEVEL_NAME}TerrainMaterialTextureSet"
        terrain_material_entries.update(
            build_terrain_material_texture_set(
                texture_set_name, base_tex_size=config.TERRAIN_BASE_TEX_PIXEL_SIZE
            )
        )
        self.materials.add_terrain_materials(terrain_material_entries)

        self.export_ground_cover(layer_map, terrain_material_names, layer_variants=layer_variants)

        self.items.add_terrain_block(
            name="theTerrain",
            terrain_filename=ter_filename,
            material_texture_set=texture_set_name,
            max_height=max_height,
            z_min=z_min,
            origin_x=terrain_origin_x,
            origin_y=terrain_origin_y,
            square_size=config.TERRAIN_SQUARE_SIZE,
            overwrite=True,
        )

    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict, task: PipelineTask) -> int:
        """
        Export ONE single tile completely (DecalRoad roads + its own .ter).

        Convenience wrapper around export_decal_roads()/export_merged_terrain()
        for callers that only export a single tile
        (export_single_tile()/export_terrain_only()).

        Args:
            tile_x, tile_y: unused, only for caller compatibility
            mesh_data: Mesh data from process_tile()
            task: PipelineTask for the progress display of the subtasks

        Returns:
            Number of created DecalRoad items
        """
        with task.subtask("DecalRoads") as sub:
            road_count = self.export_decal_roads(mesh_data)
            sub.finish(f"{road_count} roads" if road_count else "no roads")

        with task.subtask("Water") as sub:
            count = self.export_water(mesh_data)
            sub.finish(f"{count} objects" if count else "no water areas")

        with task.subtask("Walls") as sub:
            count = self.export_walls(mesh_data)
            sub.finish(f"{count} walls" if count else "no walls")

        with task.subtask("Bridges") as sub:
            count = self.export_bridges(mesh_data)
            sub.finish(f"{count} bridges" if count else "no bridges")

        with task.subtask("Tunnels/galleries") as sub:
            count = self.export_tunnels(mesh_data)
            blocked = self.export_roadblocks(mesh_data)
            zones = self.export_tunnel_zones(mesh_data)
            lights = self.export_tunnel_lights(mesh_data)
            sub.finish(
                (f"{count} mesh(es)" if count else "no tunnels/galleries")
                + (f", {zones} darkness zones" if zones else "")
                + (f", {lights} lights" if lights else "")
                + (f", {blocked} barrier elements" if blocked else "")
            )

        with task.subtask("Terrain export") as sub:
            self.export_merged_terrain(
                heights=mesh_data["heightmap"],
                layer_map=mesh_data["layer_map"],
                terrain_material_names=list(mesh_data["terrain_material_names"]),
                terrain_origin_x=mesh_data["terrain_origin_x"],
                terrain_origin_y=mesh_data["terrain_origin_y"],
                terrain_size=mesh_data["terrain_size"],
                z_min=mesh_data["z_min"],
                max_height=mesh_data["max_height"],
                photo_tile_names=mesh_data["photo_tile_names"],
                layer_variants=mesh_data.get("layer_variants"),
                variant_parents=mesh_data.get("variant_parents"),
                photo_extents=mesh_data.get("photo_extents"),
            )
            sub.finish()

        return road_count
