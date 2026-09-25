"""
Central configuration for World-to-BeamNG.
"""

import logging
import os
from pathlib import Path, PurePosixPath

from .logging_config import LoggerConfig

# === LOGGING ===
# Must happen BEFORE any other module import in this file: modules like osm/osm_mapper.py
# already call logger = LoggerConfig.get_logger() on their own import. LoggerConfig is a
# singleton with an "only create once" guard (see logging_config.py::get_instance()) - whoever calls
# get_instance()/get_logger() first fixes the level for good; every later call (even
# with different values, as here) then silently becomes a no-op. Before this fix
# `from .osm.osm_mapper import OSMMapper` ran first and froze the level to the default values (INFO) -
# DEBUG_VERBOSE/LOGGING_LEVEL therefore never had any effect.
#
# Controlled exclusively via LOG_LEVEL (can be overridden via an environment variable, no need to edit
# config.py): DEBUG | INFO | WARNING | ERROR | CRITICAL.
#   PowerShell:  $env:LOG_LEVEL = "INFO"; python world_to_beamng.py
#   Bash:        LOG_LEVEL=INFO python world_to_beamng.py
LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "WARNING").upper()  # empty/not set -> WARNING
LOGGING_FILE = None  # Path("logs/world_to_beamng.log")  # Optional; None = stdout only
LoggerConfig.get_instance(log_file=LOGGING_FILE, level=LOG_LEVEL)

from .osm.osm_mapper import OSMMapper

LEVEL_NAME = "world_to_beamng"  # Name of the BeamNG level (must match BEAMNG_DIR)

# OSM mapper singleton (loads data/osm_to_beamng.json)
OSM_MAPPER = OSMMapper(config_path=Path("data/osm_to_beamng.json"))

# Approximate reference point ONLY for the sun position calculation (managers/environment.py, see
# ENV_DATE/ENV_CLOCK_TIME below) - NOT the vehicle spawn point. That one is computed automatically
# (area center, snapped to the nearest road, see managers/item_manager.py::
# _compute_vehicle_spawn()) and no longer needs a setting. The deviation of this reference
# point from the actual area makes practically no difference for the sun angle.
SUN_REFERENCE_LATLON = (47.842840, 7.684767)

# BeamNG user folder of the current version (BeamNG creates it on first launch):
# %LOCALAPPDATA%\BeamNG\BeamNG.drive\current. The level is created in it under levels/<LEVEL_NAME>.
BEAMNG_USER_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "BeamNG" / "BeamNG.drive" / "current"
BEAMNG_DIR = BEAMNG_USER_DIR / "levels" / LEVEL_NAME
BEAMNG_DIR_SHAPES = BEAMNG_DIR / "art" / "shapes"
BEAMNG_DIR_TEXTURES = BEAMNG_DIR_SHAPES / "textures"
BEAMNG_DIR_BUILDINGS = BEAMNG_DIR_SHAPES / "buildings"

# Checked-in, generated-once or photo-based textures (one folder per texture + manifest.json), see textures/library.py
TEXTURE_LIBRARY_DIR = Path("data/textures")

# In-Game relative paths (MUST use forward slashes for BeamNG)
RELATIVE_DIR = PurePosixPath("levels") / LEVEL_NAME
RELATIVE_DIR_SHAPES = RELATIVE_DIR / "art" / "shapes"
RELATIVE_DIR_TEXTURES = RELATIVE_DIR_SHAPES / "textures"
RELATIVE_DIR_BUILDINGS = RELATIVE_DIR_SHAPES / "buildings"


# === BEAMNG LEVEL STRUCTURE ===
ITEMS_JSON = Path("main") / "MissionGroup" / "items.level.json"  # items in the MissionGroup directory
MATERIALS_JSON = Path("main") / "materials.json"  # contains material definitions

# Flow control
LOD2_ENABLED = False  # process LoD2 buildings
PHASE5_ENABLED = True  # enable the horizon layer (requires DGM30 + DOP300 data)
FORESTS_ENABLED = True  # enable/disable forest export globally

# In addition to the automatic default spawn (nearest road to the area center), every POI found
# (place or large parking lot, see osm/poi_points.py) gets its own spawn point selectable in the vehicle
# selection - see ItemManager._compute_poi_spawn_points(). Road names ("Nuova strada del Passo del
# San Gottardo") were not very meaningful as labels, hence the switch to places/parking lots. If there are
# more POI candidates than the limit, the larger/better-known places win first, then the largest
# parking lots (see osm.poi_points.PLACE_RANK).
POI_SPAWN_POINTS_ENABLED = True
MAX_POI_SPAWN_POINTS = 20
POI_MIN_PARKING_AREA_M2 = 500.0  # smaller parking lots (residential street, single garage) are no sensible spawn location
POI_SPAWN_BOUNDS_MARGIN = 20.0  # margin to the terrain edge, in meters (the OSM query extends beyond the terrain)
# The spawn of a POI is placed on the nearest drivable road (heading parallel to the centerline), provided one is
# at most this far away (in meters) - see ItemManager._nearest_road_pose().
POI_SPAWN_MAX_ROAD_DISTANCE = 300.0
# Never a spawn road: ways without car traffic (tunnels are also excluded via structure_type - their
# centerline often runs in 2D directly under a place, e.g. the Gotthard road tunnel under Airolo)
POI_SPAWN_EXCLUDED_HIGHWAYS = frozenset({"path", "footway", "steps", "cycleway", "bridleway", "pedestrian"})

# Selectable spawn points in front of tunnel entrances (tunnels/entrance_spawns.py): this far in front of the portal
# on the approach, in meters
TUNNEL_SPAWN_DISTANCE = 20.0

# POI preview image ("preview" in info.json spawnPoints[], see lua/ge/extensions/core/levels.lua): top-down
# crop of the already built aerial photo, place centered - see io/aerial.py::build_poi_preview_image().
POI_PREVIEW_CROP_SIZE_M = 300.0  # edge length (meters) of the aerial photo crop
POI_PREVIEW_PIXEL_SIZE = 512  # edge length (pixels) of the saved preview image


# === MATERIAL SETTINGS ===
# Materials ALWAYS use textures (no color fallbacks)

HORIZON_GRID_SPACING = 200  # horizon grid resolution in meters (200 m)
HORIZON_HALF_SIZE_M = 50000  # the horizon extends this far from the area center in each direction (i.e. 100 x 100 km)
HORIZON_IMAGE_SIZE_PX = 8192  # edge length of the horizon texture; the satellite image is scaled to it
# Seam terrain <-> horizon (see terrain/horizon_seam.py): the horizon has an exactly fitting hole
# for the terrain block, a fine border ring with the terrain edge heights and a smooth height transition.
TERRAIN_PADDING_AS_HOLES = True  # treat the padded heightmap border (beyond the data) as a hole - the horizon covers it
HORIZON_SEAM_STEP = 1.0  # point spacing of the border ring at the terrain hole, m (= TERRAIN_SQUARE_SIZE: exact seam, DGM1 unchanged)
HORIZON_BLEND_DISTANCE = 1000.0  # length of the height transition terrain -> DGM30 in meters
HORIZON_FLANGE_INSET = 5.0  # width of the flange under the terrain (hides residual cracks), 0 = off
HORIZON_FLANGE_SINK = 15.0  # depth of the flange below the terrain height in meters

# === VISIBILITY / FOG (LevelInfo) ===
# Without visibleDistance BeamNG uses its default (~1 km) - everything beyond is clipped.
# Original levels: utah 5000, east_coast 12000, west_coast 15667, italy 25000. The horizon extends
# +-50 km; the fog hides the cut-off (at 0.0002 only ~0.7 % is still visible after 25 km).
# Larger values show more of the horizon, but cost depth precision (not tested above 25000).
LEVEL_VISIBLE_DISTANCE = 25000  # view distance in meters (LevelInfo.visibleDistance)
LEVEL_FOG_DENSITY = 0.0002  # fog density (LevelInfo.fogDensity), smaller = clearer distant view
# Light/weather: sun position, sky and clouds come from BeamNG's own defaults (data/environment_defaults.json,
# managers/environment.py). Initial state = "sunny"; time of day, clouds and weather can be adjusted in game.
ENV_FOG_COLOR = [0.741176, 0.815686, 0.92549, 1.0]  # haze color (light blue like most original levels; otherwise the far horizon is gray)
ENV_FOG_HEIGHT_MARGIN = 50.0  # fogAtmosphereHeight = highest terrain point + this margin in m (originals: ~terrain height)
ENV_DATE = (2026, 6, 21)  # date for the sun position (summer: the DOP20 aerial photos are summer shots)
ENV_CLOCK_TIME = "11:00"  # start time "HH:MM" (local time; 12:00 = highest sun position)

# === WATER (real BeamNG objects: River for streams, WaterBlock for ponds/lakes) ===
WATER_ENABLED = True
WATERWAY_WIDTHS = {"stream": 2.5, "river": 8.0, "canal": 5.0}  # default width per type in m (tag "width" wins); no ditches
WATER_RIVER_DEPTH = 1.0  # depth of the River volume in m
WATER_NODE_SPACING = 10.0  # spacing of the River nodes in m
WATER_STREAM_LIFT = 0.2  # water level above the DGM1 channel bed in m (banks are 0.4 m higher at the median)
WATER_MAX_RIVER_NODES = 40  # longer streams are split into several River objects
WATER_POND_MARGIN = 2.0  # WaterBlocks extend this far beyond the edge of the OSM polygon (fill the whole hole), in m
WATER_POND_DEPTH = 3.0  # depth of the WaterBlocks in m
WATER_POND_BANK_DEPTH = 0.5  # inside the OSM water polygon the terrain is lowered by this much, in m
WATER_POND_BANK_SLOPE_DEG = 45.0  # embankment angle of the depression inward (45 degrees = 1 m deeper per meter inward)
WATER_POND_CELL = 6.0  # maximum edge length of the tiles used to fill ponds, in m
WATER_POND_CUBEMAP = "DefaultSkyCubemap"  # engine's own cubemap (the template's one is level-specific and was missing)

# === FOREST GENERATION PARAMETERS ===
FOREST_ROAD_MARGIN = 2.0  # buffer around the OSM centerline (fallback; the carriageway edge below is decisive), in meters
FOREST_BUILDING_MARGIN = 2.5  # buffer around buildings: no trees/bushes stand there (in meters)
FOREST_ROAD_SURFACE_MARGIN = 2.0  # distance to the edge of the embedded carriageway (applies to every trunk, incl. group trees)
FOREST_ROW_SURFACE_MARGIN = 1.0  # the same for tree rows: trunks not on the carriageway
FOREST_ROW_ROAD_MARGIN = 3.0  # tree rows (avenues) stand closer to roads than forest: smaller buffer (in meters)
# Group assets have several trunks up to ~9 m next to the origin. The distances above also apply to every trunk
# (forest/tree_footprints.py), not only to the origin.
FOREST_TRUNK_MAX_FLOAT = 0.5  # a trunk base may stand this far above the ground after lowering (in meters)
FOREST_TRUNK_MAX_SINK = 1.0  # a tree may be lowered at most this far, otherwise the type is switched (in meters)

# The embankment is NOT created in the mesh: roads are DecalRoads (workflow/terrain_workflow.py::export_decal_roads()),
# the transition to the surroundings is created directly in the terrain heightmap
# (terrain/road_embedding.py::apply_embankment_blend()).
# Minimum embankment width (meters) regardless of height differences
MIN_SLOPE_WIDTH = 2
# Upper limit of the embankment width (meters), regardless of how large the
# height difference between road edge and natural terrain is
# (prevents unrealistically wide embankment corridors in extreme cases).
MAX_SLOPE_WIDTH = 30.0
SLOPE_ANGLE = 45.0  # inclination angle of the embankment in degrees (45° = 1:1 gradient)
# Pre-reduction via a coarser grid (strategy 2). For finer terrain set e.g. 1.0.
GRID_SPACING = 1.0  # spacing between grid points in meters (native DGM1 resolution; 10.0 = coarse)

# Fallback CRS (EPSG code) for elevation/aerial photo data without its own embedded CRS (e.g. pure
# ASCII XYZ point clouds as from LGL Baden-Wuerttemberg). For GeoTIFF sources the embedded
# CRS is detected automatically (geometry.coordinates.set_source_crs()) and takes precedence over this value.
SOURCE_CRS_EPSG = 25832

# === NATIVE TERRAIN (.terrain heightmap) ===
# Meters per heightmap grid cell. = GRID_SPACING for resolution parity with the
# previous mesh approach (see spec section 2, requirement 2).
TERRAIN_SQUARE_SIZE = GRID_SPACING
# No ROAD_EMBED_MARGIN/gradient compensation needed anymore (former, now
# removed constants): since the switch to BeamNG `DecalRoad` (see
# terrain/road_embedding.py module docstring) there is no second, separately
# encoded road surface anymore that would have to be hit - the
# terrain is set directly to exactly the road centerline height.
# Buffer (meters) above the actual height max/min when computing
# maxHeight for the .terrain file (see spec section 8).
TERRAIN_MAX_HEIGHT_BUFFER = 50.0

# DEBUG / EXPORTS
DEBUG_EXPORTS = True  # enable debug dumps (network, grid) only when needed

# === LAND USE / GROUND COVER ===
# Ground vegetation (grass, flowers, fern ...) as BeamNG GroundCover objects per terrain layer.
GROUND_COVER_ENABLED = True
GROUND_COVER_MAX_ELEMENTS = 100000  # upper limit of simultaneously drawn elements per object (performance!)
GROUND_COVER_MAX_RADIUS = 100.0  # upper limit of the view distance in meters per object (BeamNG originals: 50-120)
# Under roads (plus shoulder) and buildings the layer is reset to the aerial photo,
# so that no grass grows through decals/houses there.
GROUND_COVER_ROAD_MARGIN = 1.0
GROUND_COVER_BUILDING_MARGIN = 0.5
# Vineyard vines (forest items) - require FORESTS_ENABLED.
VINEYARDS_ENABLED = True
# Distance of the vines to the edge of roads/buildings in meters. Otherwise the rows run right up to the
# polygon boundary of the vineyard and end here to the centimeter at this exclusion zone.
VINEYARD_EXCLUSION_MARGIN = 2.0

# === ROAD SMOOTHING / OPTIONS ===
ENABLE_ROAD_SMOOTHING = True  # False = spline smoothing completely off
SAMPLE_SPACING_FACTOR = 0.5  # factor for segment spacing: road_width * SAMPLE_SPACING_FACTOR
ROAD_SMOOTH_ITERATIONS = 1  # number of smoothing iterations (1-3; higher = smoother)
ROAD_SMOOTH_WEIGHT = 0.6  # Chaikin filter weight (0.5-0.9; higher = less smoothing, 0.75 = mild)

# Minimum spacing between two DecalRoad nodes in meters. BeamNG does not draw a
# DecalRoad at all if it has a segment that is too short (0.10 m -> invisible,
# 0.32 m -> ok; verified on Eichgasse). Typical spacing after
# resampling is ~0.8 m.
DECAL_ROAD_MIN_NODE_SPACING = 0.5
# BeamNG draws only a limited amount of geometry per DecalRoad (the decal is clipped to the terrain triangles under its
# area; whatever exceeds the budget is missing without an error message - in game: cut-off after ~510 m^2 at 6.5 m width).
# Carriageway decals are therefore split into pieces of at most this much area (geometry/decal_chunks.py), in m^2;
# remainders shorter than ROAD_DECAL_MIN_TAIL_LENGTH (m) are attached to the previous piece.
ROAD_DECAL_MAX_AREA = 250.0
ROAD_DECAL_MIN_TAIL_LENGTH = 5.0

# === WIDTH TRANSITIONS / ROAD MARKINGS ===
# See docs/superpowers/plans/2026-09-24-road-markings-width-transitions.md.
# Width transition at straight-through joints of two DecalRoads (geometry/road_width_transitions.py): over 10 m,
# 5 m before and after the joint point, cubic Hermite spline; in the zone one node per ROAD_WIDTH_TRANSITION_STEP meters.
ROAD_WIDTH_TRANSITION_LENGTH = 10.0
ROAD_WIDTH_TRANSITION_STEP = 1.0
ROAD_WIDTH_TRANSITION_MIN_DELTA = 0.05  # smaller width differences stay unchanged, in meters
ROAD_CONTINUATION_ENDPOINT_TOL = 0.5  # two road ends must be this close together, in meters
ROAD_CONTINUATION_MAX_ANGLE_DEG = 30.0  # largest bend that still counts as "continuing straight"

# Markings (geometry/road_markings.py): separate narrow DecalRoads above the carriageway as in BeamNG's vanilla levels.
ROAD_MARKINGS_ENABLED = True
ROAD_MARKING_HIGHWAYS = frozenset(
    {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    }
)
ROAD_MARKING_SURFACE = "asphalt_road_standard"  # asphalt only - no paving, gravel, dirt track
ROAD_MARKING_MIN_TWO_LANE_WIDTH = 5.5  # without lanes tag: narrower roads are single-lane (edge lines only)
ROAD_MARKING_LINE_WIDTH = 0.15  # line width in meters (vanilla: 0.15-0.2 m)
ROAD_MARKING_EDGE_INSET = 0.25  # distance of the edge line center from the carriageway edge, in meters
ROAD_MARKING_EDGE_MATERIAL = "line_edge_white"  # entries in data/osm_to_beamng.json -> road_markings
ROAD_MARKING_DIVIDER_MATERIAL = "line_divider_dashed"
# renderPriority of DecalRoads: BeamNG/Torque3D draws them in DESCENDING order - the smallest value comes
# last and lies on top (confirmed in game: lines with 20 lay under the asphalt with 8; vanilla: lines 1-2,
# roads up to 12; Road Architect gives the carriageway decal the highest value). Carriageways get
# ROAD_RENDER_PRIORITY_BASE - surface_types[*].priority (asphalt 12 above gravel 16 above dirt track 18), markings the
# smallest value, right on top.
ROAD_RENDER_PRIORITY_BASE = 20
ROAD_MARKING_RENDER_PRIORITY = 1
ROAD_MARKING_JUNCTION_CLEARANCE = 0.5  # the gap in the edge line extends this far beyond the joining carriageway
ROAD_MARKING_MIN_PIECE_LENGTH = 2.0  # shorter line remnants after the junction cut are dropped, in meters
# Ways whose T-junction does NOT cut a gap into the edge line (farm track, footway ...)
ROAD_MARKING_NO_GAP_HIGHWAYS = frozenset({"track", "path", "footway", "cycleway", "bridleway", "steps"})

# === CLIPPING ===
ENABLE_ROAD_CLIPPING = True  # True = clip + segment subdivision at the grid edge, False = skip (test mode)
ROAD_CLIP_MARGIN = -20.0  # clipping distance from the grid edge in meters (faces < 3 m from the edge are removed)

# === TILE-EXPORT (DAE) ===
BUILDINGS_AS_ONE_OBJECT = True  # True: ALL buildings in ONE DAE/ONE object over the whole area (like the roads)
MAX_BUILDINGS_PER_SHAPE = 1500  # BeamNG discards everything beyond 2048 nodes per shape (1 node per building) -> split
TILE_SIZE = 500  # size per DAE tile in meters (only with BUILDINGS_AS_ONE_OBJECT = False)

# === WALLS (OSM barrier=wall / retaining_wall, only with height tag) ===
WALLS_ENABLED = True
WALL_THICKNESS = 0.5  # wall thickness in meters (rubble stone wall)
WALL_ROAD_SNAP_M = 5.0  # walls at most this far from a road centerline take its height as base (else terrain), in meters
WALL_CAP_THICKNESS = 0.05  # cap slabs on top of the wall: thickness in m (0 = no slabs); the total height stays the OSM height
WALL_CAP_OVERHANG = 0.04  # overhang of the slabs beyond the wall body (per side and at open ends), in meters
WALL_CAP_PLATE_LENGTH = 0.8  # average slab length in meters (length varies by +-25 %)
WALL_CAP_JOINT = 0.01  # joint between two slabs in meters
WALL_SINK = 0.3  # the bottom edge reaches this far below the ground (no gap at the base), in meters
WALL_MAX_SEGMENT = 1.0  # longest sub-segment so that the wall follows the terrain, in meters
WALL_TEXTURE_TILE_M = 1.2  # fallback tile size of the rubble stone texture in m; tile_m in the manifest of data/textures is decisive
WALL_TEXTURE_NAME = "rubble_stone_wall"  # name of the rubble stone texture in data/textures (photo, tools/make_seamless_texture.py)
WALL_MATERIAL_NAME = "rubble_stone_wall"  # "wall" in the name picks the wall template; textures: WALL_TEXTURE_NAME (textures/registry.py)

# Bridges (OSM highway=* with bridge=*): generic concrete deck with the road's carriageway material on top and
# support piers down to the natural terrain below (see design spec docs/superpowers/specs/
# 2026-09-22-bridges-tunnels-design.md section 4). For these roads it replaces the normal terrain embedding
# and the DecalRoad export.
BRIDGES_ENABLED = True
BRIDGE_DECK_THICKNESS = 0.6  # deck thickness in meters
BRIDGE_PIER_SPACING = 25.0  # spacing of the support piers along the bridge, in meters
BRIDGE_PIER_SIZE = 1.5  # cross-section of the (square) support piers, in meters
BRIDGE_MIN_PIER_CLEARANCE = 1.0  # no pier if the distance between deck underside and terrain is smaller, in meters
BRIDGE_MATERIAL_NAME = "bridge_concrete"  # pier/curb material (texture: CONCRETE_TEXTURE_NAME, see Task 10)

# Real road bridge cross-section profile: concrete edge (curb) on both sides of the carriageway, on it a
# simple railing (posts + continuous handrail), instead of an edgeless deck slab.
BRIDGE_CURB_WIDTH = 0.25  # width of the curb per side, in meters (the carriageway becomes correspondingly narrower)
BRIDGE_CURB_HEIGHT = 0.15  # height of the curb above the carriageway, in meters
BRIDGE_RAILING_HEIGHT = 0.9  # handrail height above the curb top edge, in meters
BRIDGE_RAILING_POST_SPACING = 2.0  # post spacing along the bridge, in meters
BRIDGE_RAILING_POST_SIZE = 0.08  # cross-section of the (square) posts and the handrail, in meters
BRIDGE_RAILING_MATERIAL_NAME = "bridge_railing"  # texture: RAILING_TEXTURE_NAME

# Some OSM bridges are dimensioned too tightly and already begin in the middle of the hillside instead of at
# road level (the linear height interpolation between the way end points then yields an unrealistically
# steep ramp). Fix: the bridge is extended into its neighboring road until normal gradient prevails there again
# (see geometry.polygon.extend_short_bridges_to_natural_grade).
BRIDGE_APPROACH_SLOPE_THRESHOLD = 0.10  # gradient at which the extension stops (10 % ≈ 5.7°)
BRIDGE_APPROACH_MAX_EXTENSION = 40.0  # the bridge is extended into the neighboring road at most this far, in meters

# Tunnels (OSM highway=* with tunnel=yes/culvert/building_passage) and galleries (tunnel=avalanche_protector):
# tube or valley-side open gallery along the linearly interpolated height profile (see design spec
# section 5/6). For these roads it replaces the normal terrain embedding and the DecalRoad export.
TUNNELS_ENABLED = True  # also covers galleries (tunnel=avalanche_protector)
TUNNEL_WIDTH_MARGIN = 1.5  # additional width beyond the carriageway width, in meters
TUNNEL_ARC_SEGMENTS = 32  # discretization of the 240° arc, 7.5° per segment (radius/crown height follow from the width)
TUNNEL_SEGMENT_STEP = 10.0  # extrusion step size along the axis, in meters (coarse, since straight)
# The OSM tunnel start often already lies in the hillside (DGM = portal embankment instead of road level): portal
# height and the last meters of the approach are brought to the stable gradient of the approach (see
# geometry.polygon.settle_tunnel_portals_to_approach_grade).
TUNNEL_APPROACH_SLOPE_THRESHOLD = 0.10  # from this gradient on the approach counts as "still hillside"
TUNNEL_APPROACH_STABLE_LENGTH = 6.0  # the gradient must stay below it for this long, in meters
TUNNEL_APPROACH_MAX_DISTANCE = 40.0  # the search goes at most this far away from the portal, in meters
# Tube, portal + terrain (see tunnels/tunnel_mesh.py, tunnels/tunnel_portal.py, terrain/tunnel_terrain.py): the
# tube is a cylinder with an outer shell and may stand free; the terrain is only adapted directly at tube and portal
# - where it protrudes into the tube, there is TUNNEL_COVER of soil above the shell (no embankments, no fills).
# At the open portal the tube shell or collar covers the terrain holes at the portal plane.
# No tunnel construction for ways without road/bicycle traffic (path "tunnels" in the mountains, construction sites):
# only roads and cycleways
TUNNEL_EXCLUDED_HIGHWAYS = frozenset({
    "path", "footway", "steps", "bridleway", "pedestrian", "corridor", "via_ferrata", "elevator", "construction", "proposed",
})
TUNNEL_SHELL_RATIO = 1.0 / 15.0  # wall thickness of the tube shell : tube diameter (1:15) - small tunnels, thin walls
TUNNEL_COVER = 0.5  # at least this much soil above the tube shell where the terrain protrudes into the tube, in meters
# Portal collar at entrance and transition to the gallery: rectangular on the outside, at the thinnest point (left,
# right, top) this wall thickness relative to the tube diameter (1:10); the terrain in the portal area is cut down to
# its top edge.
TUNNEL_PORTAL_COLLAR_RATIO = 0.1
# Collar sides left/right at least this thick: the hole cells at the portal step extend up to one grid diagonal
# (1.41 m for a 1 m grid) sideways beyond the tube radius and must stay hidden in the collar
TUNNEL_PORTAL_COLLAR_MIN_SIDE = 1.5  # in meters
# Front face of the tunnel entrances (not the transitions into a gallery) tilted by this many degrees toward the
# mountain side; the flat portal zone then extends to behind the front face at the outer crown - the hole edge lies
# hidden there.
TUNNEL_PORTAL_TILT_DEG = 20.0
TUNNEL_PORTAL_FLAT_DEPTH = 1.5  # the terrain still lies at ground level this far behind the portal plane, in meters
TUNNEL_PORTAL_LENGTH = 3.5  # length of the portal collar into the mountain, in meters (> FLAT_DEPTH + 1 hole cell)
# Transition tunnel -> gallery (docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md): a
# tunnel portal that is at most this far from a gallery end point is a transition (round portal plus faces
# between the tube arc and the gallery cross-section).
TUNNEL_TRANSITION_ENDPOINT_TOL = 0.5  # in meters
TUNNEL_TRANSITION_COVER_THICKNESS = 0.2  # solid cover slabs between tube arc and gallery cross-section, in meters
# Height profile of tunnel/gallery chains (geometry/polygon.py::apply_structure_elevation_profiles): at a gallery
# the DGM shows the roof - support points (DGM - GALLERY_HEIGHT - GALLERY_ROOF_THICKNESS) only this far from the chain
# ends and from each other, median over +- GALLERY_ROOF_SAMPLE_WINDOW meters.
GALLERY_ROOF_SAMPLE_END_DISTANCE = 100.0
GALLERY_ROOF_SAMPLE_SPACING = 200.0
GALLERY_ROOF_SAMPLE_WINDOW = 10.0
# Tunnel across the map boundary: a chain end at most this far before the terrain edge (or beyond it) counts as
# "leaves the map" - the tube then lies flat at entrance height, a roadblock stands in front of the entrance.
MAP_EDGE_TUNNEL_MARGIN = 25.0
ROADBLOCK_SHAPE = "/art/shapes/garage_and_dealership/Clutter/hr_plasticbarrier_red.dae"  # BeamNG default asset
ROADBLOCK_DISTANCE = 5.0  # the roadblock stands this far in front of the portal plane, in meters
ROADBLOCK_SIDE_MARGIN = 0.5  # the roadblock extends this far beyond the carriageway on each side, in meters
ROADBLOCK_SPACING = 1.5  # spacing of the barrier elements (vanilla median 1.48 m), in meters
# Darkness in tunnel tubes (tunnels/tunnel_zones.py): zone boxes along the tube as in BeamNG's own levels
TUNNEL_ZONE_MAX_LENGTH = 50.0  # at most this long per zone, in meters
TUNNEL_ZONE_MAX_DEVIATION = 0.5  # tube axis at most this far from the zone axis (curves), in meters
TUNNEL_ZONE_END_OVERLAP = 0.5  # overlap of adjacent zones per side, in meters
TUNNEL_ZONE_WIDTH_MARGIN = 2.0  # zone this much wider than the tube, in meters
TUNNEL_ZONE_HEIGHT_MARGIN = 2.0  # zone this much taller than the crown (half each at the bottom/top), in meters
TUNNEL_ZONE_PORTAL_INSET = 1.0  # zones begin this far behind the portal plane (portal stays bright), in meters
TUNNEL_ZONE_PORTAL_DEPTH = 3.0  # depth of the portal objects at the zone ends (vanilla: 3.6-5.9 m), in meters
GALLERY_HEIGHT = 5.0  # clear height of the (rectangular, not circular) gallery, in meters
GALLERY_COLUMN_SPACING = 6.0  # column spacing on the open valley side, in meters
# Floor/roof/mountain-side wall are real boxes (not just thin faces), so that the gallery also looks like a solid
# structure from outside/at the ends instead of a hollow shell - see
# tunnels/gallery_mesh.py::build_gallery_mesh(). The ends lie exactly at the original
# OSM way boundary points (no artificial extension anymore).
GALLERY_ROOF_THICKNESS = 0.5  # roof thickness upward, in meters
GALLERY_FLOOR_THICKNESS = 5.0  # floor thickness downward, in meters
GALLERY_WALL_THICKNESS = 5.0  # thickness of the mountain-side wall into the hillside, m (flush with the roof top edge)
GALLERY_COLUMN_SIZE = 0.4  # cross-section of the (square) columns, in meters
GALLERY_CURB_HEIGHT = 0.5  # height of the plinth on the column side, in meters
GALLERY_CURB_WIDTH = 0.4  # width of the plinth (offset inward from the carriageway edge), in meters - matches
# GALLERY_COLUMN_SIZE, so that the column sits exactly on the plinth footprint (see
# gallery_mesh.py::build_gallery_mesh(), column loop) and its outer edge stays flush with the roof edge.

# Galleries are (unlike bridges/tunnels) embedded into the terrain like normal roads (see
# terrain_workflow.py::process_tile(), embankment + embed_roads_into_heightmap) instead of getting a separate terrain
# hole: floor/wall/roof are solid boxes (GALLERY_FLOOR_THICKNESS/GALLERY_WALL_THICKNESS/
# GALLERY_ROOF_THICKNESS) that meet the terrain cleanly on their own. Both embankment sides get a fixed
# value via build_road_embankment_profiles()'s "slope_width_override" instead of the usual one computed from the
# height difference (see there):
# - Mountain side: GALLERY_MOUNTAIN_EMBED_MARGIN - not an embankment angle, but a narrow FLAT rim at
#   carriageway height (flat_shoulder_sides) directly at the inner edge of the (solid) wall, for a clean
#   wall-floor transition. Beyond it the terrain stays at natural height - the wall
#   extends into the hillside anyway.
# - Valley side: GALLERY_VALLEY_SLOPE_WIDTH instead of the computed width - at a gallery the DGM does not show
#   the original terrain there, but the real valley-side structure (parapet/roof overhang); the
#   embankment width derived from it would be noisy and would give a visibly faceted embankment instead of
#   a smooth, short blend into the terrain.
GALLERY_VALLEY_SLOPE_WIDTH = 5.0  # minimum embankment width on the valley side, in meters
# Search the valley-side reference behind the gallery roof in the DGM
# (terrain_workflow._gallery_valley_slope_widths()): downvalley to the first point with lower terrain (at most
# STRUCTURE_HEIGHT above the carriageway) - that is the reference.
# 0: only terrain below the carriageway counts; with +2 m the reference was often still at the steep roof edge in the
# DGM.
GALLERY_VALLEY_STRUCTURE_HEIGHT = 0.0  # DGM higher than carriageway + this = still gallery roof, in meters
GALLERY_VALLEY_SEARCH_STEP = 0.5  # search step downvalley, in meters
GALLERY_VALLEY_SEARCH_MAX = 20.0  # the embankment on the valley side becomes at most this wide, in meters
GALLERY_MOUNTAIN_EMBED_MARGIN = 1.3  # flat rim on the mountain side beyond the wall inner edge, in meters
TUNNEL_MATERIAL_NAME = "tunnel_concrete"  # wall/ceiling/frame/roof material (texture: CONCRETE_TEXTURE_NAME)

# === LOD2 ROOF TEXTURE ===
# t_roof_slates_rounded_b.color.dds (256 px) shows 5 beaver-tail tiles side by side per repeat (6 rows
# on top of each other). The UVs are metric: one repeat = ROOF_REPEAT_M meters in the roof plane, so a tile
# is always ROOF_TILE_WIDTH_M wide, no matter how steep the roof is.
ROOF_TILE_WIDTH_M = 0.20
ROOF_TILES_PER_REPEAT = 5
ROOF_REPEAT_M = ROOF_TILE_WIDTH_M * ROOF_TILES_PER_REPEAT

# Sloped roofs project beyond walls (the LOD2 data ends exactly at the wall): ROOF_EAVE_OVERHANG_M at the eave,
# ROOF_VERGE_OVERHANG_M at the gable (verge). The overhang is modeled ROOF_OVERHANG_THICKNESS_M thick (fascia board
# and soffit).
ROOF_EAVE_OVERHANG_M = 0.6
ROOF_VERGE_OVERHANG_M = 0.3
ROOF_OVERHANG_THICKNESS_M = 0.10

# Flat roofs (slope up to FLAT_ROOF_MAX_SLOPE_DEG): gravel surface (tileable texture, FLAT_ROOF_GRAVEL_REPEAT_M meters
# per repeat) with a surrounding sheet-metal rim as real geometry.
FLAT_ROOF_MAX_SLOPE_DEG = 5.0
FLAT_ROOF_GRAVEL_REPEAT_M = 2.0  # tile size of the gravel texture; generate_gravel_texture.py writes it into the manifest
FLAT_ROOF_GRAVEL_TEXTURE = "roof_gravel"  # name in data/textures
FLAT_ROOF_GRAVEL_TEXTURE_PX = 1024
FLAT_ROOF_EDGE_HEIGHT_M = 0.25
FLAT_ROOF_EDGE_THICKNESS_M = 0.05

# Procedural concrete texture for bridge piers/curbs, tunnel walls/ceiling/portals and gallery roof/columns.
CONCRETE_TEXTURE_NAME = "tunnel_concrete"
CONCRETE_TEXTURE_TILE_M = 2.0
CONCRETE_TEXTURE_PX = 1024

# Procedural steel texture for bridge railings (posts + handrail).
RAILING_TEXTURE_NAME = "bridge_railing"
RAILING_TEXTURE_TILE_M = 0.3
RAILING_TEXTURE_PX = 512

# === LOD2 WALLS (plastered) ===
# Wall = ONE uncut polygon with a seamless, metrically tiled plaster texture (FACADE_PLASTER_REPEAT_M meters per
# repeat); there are no cells or seams in the plaster. The color is fixed per building (weighted choice, see
# facade/facade_styles.py), each color is its own material on the same normal/roughness texture.
# Windows and doors are separate, small faces (sprites from an atlas) that sit FACADE_WINDOW_OFFSET_M in front of the
# wall.
FACADE_PLASTER_TEXTURE_PX = 1024
FACADE_PLASTER_REPEAT_M = 3.0
FACADE_STOREY_HEIGHT_M = 3.0  # storey height; the storeys are counted DOWNWARD from the eave
FACADE_BAY_WIDTH_M = 2.5  # target bay width; in practice the wall width is distributed evenly over whole bays
FACADE_NARROW_WALL_M = 2.5  # narrower walls (projections, end faces) get no windows
FACADE_WINDOW_OFFSET_M = 0.03  # windows sit this far in front of the wall (against z-fighting)
FACADE_WINDOW_SILL_M = 0.9  # sill height: window lower edge above the storey floor
FACADE_WINDOW_ATLAS_PX_PER_M = 200  # resolution of the window sprites
FACADE_GUTTER_PX = 8  # border around each sprite into which the edges are replicated (against bleeding when filtering)
FACADE_MAX_MIP_LEVELS = 6  # longer mip chains let neighboring sprites bleed into each other
# Raised basement: if a remainder of at least FACADE_BASEMENT_MIN_REMAINDER_M is left below the full storeys,
# it is a basement storey; basement windows appear where it is visible above the ground by at least
# FACADE_BASEMENT_MIN_EXPOSED_M, with lower edge FACADE_BASEMENT_SILL_M above the ground.
FACADE_BASEMENT_MIN_REMAINDER_M = 0.6
FACADE_BASEMENT_MIN_EXPOSED_M = 1.0
FACADE_BASEMENT_SILL_M = 0.3
FACADE_STOREY_ROUNDING = 0.2  # fraction of a storey height by which the lowest storey may be too short
FACADE_DOOR_MAX_HEIGHT_ABOVE_BASE_M = 0.5  # doors only where the storey floor is at most this high above the ground
# Church towers: no windows, but a tower clock on the front side. A church is an OSM polygon (building=church/
# cathedral/chapel, amenity=place_of_worship) that covers at least CHURCH_OVERLAP_MIN of a LOD2 building. In the
# LOD2 data nave and tower are ONE building; tower walls are the walls whose top edge reaches at least
# CHURCH_TOWER_HEIGHT_FRACTION of the way from the median of the wall top edges to the highest wall (and the highest
# wall lies at least CHURCH_TOWER_MIN_RISE_M above the median), plus walls in OSM bell tower polygons.
CHURCH_OVERLAP_MIN = 0.5
CHURCH_TOWER_HEIGHT_FRACTION = 0.6
CHURCH_TOWER_MIN_RISE_M = 4.0
CHURCH_CLOCK_MIN_WALL_M = 3.2  # narrower tower walls get no clock
CHURCH_CLOCK_BELOW_TOP_M = 3.5  # clock center at most this far below the wall top edge (else lower, where the wall is wide enough)
CHURCH_CLOCK_MIN_HEIGHT_M = 6.0  # clock center at least this high above the wall base
# Green channel of the normal maps: True = green points up (OpenGL). Measured on t_roof_slates_rounded_nm.normal.dds
# (correlation of G with the vertical AO gradient +0.61, of R with the horizontal -0.80): BeamNG uses
# green-up here.
TEXTURE_NORMAL_GREEN_UP = True

# Pixel edge length of the ONE combined aerial photo for the whole
# area (io/aerial.py::process_aerial_images() - since 2026-09-18 no
# photo material per 500 m tile anymore, see the docstring there). MUST match the
# baseTexSize of the TerrainMaterialTextureSet (terrain_workflow.py),
# otherwise BeamNG packs a wrongly sized texture into the
# terrain material atlas (see terrain material crash fix of 2026-09-17).
#
# 8192 instead of BeamNG's "typical" upper limit of 4096, because our official
# documentation research (2026-09-18, https://documentation.beamng.com/modding/
# levels/level_formats/terrain/) explicitly allows higher values "if the
# base is a unique overall terrain map" - exactly our case (ONE
# aerial photo for the whole area instead of many tiles). 8192 instead of 16384
# (as of 2026-09-18) at the user's request because of file size (every base texture
# must have exactly the baseTexSize according to BeamNG; the land use layers
# use the same aerial photo as base, see build_terrain_material_entries()).
# 2048 m tile at 8192 px ≈ 0.25 m/pixel, close to the native DOP20 resolution
# (0.2 m/pixel).
TERRAIN_BASE_TEX_PIXEL_SIZE = 8192


# Four-image mode: the area is divided into PHOTO_TILE_SIZE_M tiles, each gets its OWN photo
# (TERRAIN_BASE_TEX_PIXEL_SIZE px for 2 km = 0.244 m/px instead of ONE overall photo with 0.5 m/px at 4x4 km). Cost:
# the land use layers and the GroundCover types are kept per tile (see terrain/photo_tiles.py).
# False = one overall photo (with a single tile it is always one photo anyway).
AERIAL_PHOTO_PER_TILE = True

# Tile size (meters) of the four-image mode - independent of the size/number of the raw data tiles (which, depending on
# the source, can be e.g. 1 km instead of 2 km, see terrain/photo_tiles.py::build_processing_tile_grid()).
# Default 2000.0 corresponds to the previous implicit behavior for LGL Baden-Wuerttemberg (one photo per 2x2 km ZIP).
PHOTO_TILE_SIZE_M = 2000.0

# BigMap preview image (info.json field "minimap") from the already built aerial photo PNGs
# (io/aerial.py::build_minimap_image()) - without this field BigMap stays usable, but shows only an
# empty background instead of the terrain.
MINIMAP_ENABLED = True
MINIMAP_PIXEL_SIZE = 2048  # edge length of the minimap in pixels, independent of the actual terrain size

# === DIRECTORIES ===
CACHE_DIR = Path("cache")  # directory for cache files
HEIGHT_DATA_DIR = Path("data/height")  # directory with elevation data (DGM1)
AERIAL_DATA_DIR = Path("data/satellite")  # directory with aerial photos (DOP20)
LOD2_DATA_DIR = Path("data/buildings")  # directory with 3D building models (LoD2/CityGML)
# 30 m elevation data for the horizon: ALWAYS downloaded fully automatically from Copernicus (dgm30_fetch.py),
# never placed manually - therefore belongs under cache/ (always safe to delete/reload),
# not under data/ (which stays for self-supplied raw data such as elevation/aerial photo/buildings).
DGM30_CACHE_DIR = CACHE_DIR / "dgm30"

# === AUTOMATIC DOWNLOAD: HORIZON SOURCE DATA (DGM30 + SENTINEL-2) ===

DGM30_AUTO_DOWNLOAD = True
DGM30_S3_BUCKET = "copernicus-dem-30m"          # verified in the README (working link)
DGM30_S3_REGION = "eu-central-1"
DGM30_FETCH_MAX_RETRIES = 3
DGM30_FETCH_TIMEOUT_S = 60
DGM30_NOT_FOUND_CACHE_TTL_DAYS = 30

EOX_AUTO_DOWNLOAD = True
EOX_WMS_URL = "https://tiles.maps.eox.at/wms"   # verified via GetCapabilities on 2026-09-22
EOX_WMS_LAYER = "s2cloudless-2025_3857"         # verified: layer list contains s2cloudless-<year>_3857,
                                                 # currently up to 2025; when implemented in future, adopt the newest
                                                 # available year from GetCapabilities
EOX_WMS_VERSION = "1.1.1"                       # verified: server supports only 1.1.1, NOT 1.3.0
                                                 # (axis order in EPSG:3857 identical for 1.1.1 vs. 1.3.0,
                                                 # hence uncritical for the BBOX calculation)
EOX_WMS_FORMAT = "image/jpeg"                   # verified available
EOX_MAX_REQUEST_PX = 2048                       # GetCapabilities lists no MaxWidth/MaxHeight - left
                                                 # conservative, lower on 400/429 in a real run
EOX_TARGET_RESOLUTION_M = 10.0
EOX_MOSAIC_MAX_PX = 12000
EOX_FETCH_MARGIN_FACTOR = 1.02
EOX_FETCH_MAX_RETRIES = 3
EOX_FETCH_TIMEOUT_S = 60
EOX_USER_AGENT = "World-to-BeamNG/1.0 (private OSM-to-BeamNG conversion tool)"
EOX_KEEP_RAW_MOSAIC = True
EOX_MOSAIC_CACHE_DIR = CACHE_DIR / "horizon_source"    # raw mosaic (before cropping), hashed per area+layer
EOX_TEXTURE_CACHE_DIR = CACHE_DIR / "horizon_texture"  # finished, cropped texture, hashed per area+size+
                                                        # resampling+layer - NOT data/DOP300/, which
                                                        # remains the manual override slot (see
                                                        # sentinel2_fetch.ensure_horizon_texture())
EOX_ATTRIBUTION_NOTICE = (
    "Horizon background image: EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH "
    "(Contains modified Copernicus Sentinel data 2025). License: CC BY-NC-SA 4.0 "
    "(non-commercial use, attribution + ShareAlike required) - see "
    "https://cloudless.eox.at/documentation/license. Commercial use requires a separate "
    "EOX Commercial Attribution-RestrictedUse license."
)


# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Several Overpass servers reject requests with the generic "python-requests/x.x"
# user agent (406 Not Acceptable) or are more likely to throttle them (429 Too Many
# Requests). Community recommendation of all three servers above: state project name + a
# way to contact you. If needed, extend here with your own contact details
# (e-mail/project URL) - sent as a header to the Overpass servers.
OVERPASS_USER_AGENT = "World-to-BeamNG/1.0 (private OSM-to-BeamNG conversion tool)"
