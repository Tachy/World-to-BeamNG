"""
DAE Viewer - visualizes the exported tile DAEs

NEW ARCHITECTURE: Loads separate DAE files per tile (tile_X_Y.dae)
Each DAE contains only ONE geometry → prevents overlapping textures!

Controls:
    X = Toggle textures (on/off) - triggers a rebuild (rendering ↔ grid)
        Rendering mode: shows textures with material assignments and UV coordinates
        Grid mode: shows wireframe with colors for a faster preview

    In grid view (X=off) show/hide individual layers (WITHOUT rebuild):
        T = Toggle terrain layer
        S = Toggle road layer
        H = Toggle building layer
        D = Toggle debug layer (junctions, centerlines, boundaries)

    General:
        K = Load camera | Shift+K = Save camera
        L = Reload DAE
        Up/Down = Change zoom

    Mouse:
        Double left click = Set camera to the clicked point (40 m distance)
        Right-click drag = Rotate camera
        Scroll = Zoom

Texture Debug Features:
    • On startup the available textures and materials are listed
    • In rendering mode the texture assignments are shown for each layer (✓/○/✗)
    • Console shows: which materials were mapped to textures, which fallbacks
    • UVs are loaded from the DAE automatically and used for texture mapping

Texture System:
    • Tile textures: art/shapes/textures/tile_*.dds (terrain per 500×500 m tile)
    • Material textures: main.materials.json → art/shapes/materials/... (for roads/buildings)
    • UV coordinates per layer:
        - Terrain: 0..1 per tile (500×500 m), normalized in the DAE
        - Roads: 0..unbounded for length, 0..1 for width (tiling)
        - Buildings: scaled for 4 m (walls) / 2 m (roofs) texture repetition
"""

import pyvista as pv
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()
import numpy as np
import sys
import json
import atexit
import time
from pathlib import Path, PurePosixPath
from PIL import Image

# Import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from world_to_beamng import config
from tools.dae_loader import load_all_viewer_data


# Resolve BeamNG-relative paths ("/levels/<Level>/...") to absolute paths
def _resolve_beamng_path(path_str: str) -> str | None:
    if not path_str:
        return None

    p_posix = PurePosixPath(path_str)  # Treat input as a Posix path

    # 1) Prefix /levels/<LEVEL_NAME>/...
    level_prefix_posix = PurePosixPath("/levels") / config.LEVEL_NAME
    if p_posix.is_relative_to(level_prefix_posix):
        return str(config.BEAMNG_DIR / p_posix.relative_to(level_prefix_posix))

    # 2) Prefix from config.RELATIVE_DIR (identical, but provided)
    # config.RELATIVE_DIR is already a PurePosixPath 'levels/world_to_beamng'
    if p_posix.is_relative_to(config.RELATIVE_DIR):
        return str(config.BEAMNG_DIR / p_posix.relative_to(config.RELATIVE_DIR))

    # 3) art/… prefix relative to the level root (Path(path_str) will handle "art/...")
    if p_posix.parts[0] == "art":  # check if first part is 'art'
        return str(config.BEAMNG_DIR / p_posix)

    # 4) Fallback: treat as a relative shape specification
    return str(config.BEAMNG_DIR_SHAPES / p_posix)


class DAETileViewer:
    def __init__(self):
        # Load items and materials from JSON
        items_path = config.BEAMNG_DIR / config.ITEMS_JSON

        # Look for materials.json in config.BEAMNG_DIR/main/
        materials_path = config.BEAMNG_DIR / "main" / "materials.json"

        logger.info(f"Loading items from: {items_path}")

        logger.info(f"Loading materials from: {materials_path}")
        if materials_path.exists():
            with open(materials_path, "r", encoding="utf-8") as f:
                self.materials = json.load(f)
                logger.info(f"  [✓] {len(self.materials)} materials loaded")
        else:
            # Generate materials from the osm_to_beamng.json config
            logger.error(f"  [!] {materials_path} not found, generating from config...")
            try:
                from world_to_beamng.io.lod2 import create_materials_json

                self.materials = create_materials_json()
                logger.info(f"  [✓] Materials generated from config")
            except Exception as e:
                logger.error(f"  [!] Error generating materials: {e}")
                self.materials = {}

        # Load ALL viewer data centrally (DAE + forest)
        loader_result = load_all_viewer_data(config.BEAMNG_DIR, items_path, _resolve_beamng_path)

        self.dae_files = loader_result["dae_files"]
        self.tile_data = loader_result["tile_data"]
        self.forest_data = loader_result["forest_data"]

        if not self.dae_files:
            logger.info("No DAE files found in items.level.json!")
            return

        if not self.tile_data:
            logger.info("No geometry found in DAE files!")
            return

        # Initialize config_path EARLY (needed for _load_layers_state)
        self.config_path = Path(__file__).parent / "dae_viewer.cfg"

        # Visibility flags (load saved values)
        saved_layers = self._load_layers_state()
        self.show_terrain = saved_layers.get("terrain", True)
        self.show_roads = saved_layers.get("roads", True)
        self.show_buildings = saved_layers.get("buildings", True)  # Buildings toggle
        self.show_horizon = saved_layers.get("horizon", True)  # Horizon toggle
        self.show_forest = saved_layers.get("forest", True)  # Trees toggle
        self.use_textures = saved_layers.get("textures", True)  # Textures on by default
        self.show_debug = saved_layers.get("debug", False)  # Debug layers (junctions, centerlines)

        # Store actor references for the visibility toggles
        self.terrain_actors = []  # List of terrain mesh actors
        self.road_actors = []  # List of road mesh actors
        self.building_actors = []  # List of building mesh actors
        self.horizon_actors = []  # List of horizon mesh actors
        self.forest_actors = []  # List of forest point actors (trees)
        self.debug_actors = []  # List of debug actors (junctions, centerlines)
        self.debug_loaded = False  # Flag: debug layer already loaded?
        self.forest_loaded = False  # Flag: forest layer already loaded?
        self._first_update_view = True  # Flag: update_view() called for the first time?

        # Load grid colors from debug_network.json (for the grid view)
        self.grid_colors = self._load_grid_colors()

        # Load textures
        self.textures_dir = config.BEAMNG_DIR_SHAPES / "textures"
        self.textures = self._load_textures()

        # Load material textures from main.materials.json
        self.material_textures = self._load_material_textures()

        if self.textures:
            logger.info(f"  -> {len(self.textures)} tile textures loaded")
        if self.material_textures:
            logger.info(f"  -> {len(self.material_textures)} material textures loaded")

        # Debug: show available textures and material assignments
        self._print_texture_debug_info()

        # forest_data was already loaded in load_all_viewer_data()

        # Status actors
        self._reload_actor = None
        self._camera_status_actor = None
        self._active_layers_actor = None
        self._render_update_counter = 0  # For RenderEvent throttling
        self._last_click_ts = 0.0  # For manual double-click detection

        # Global material properties (central definition)
        self.material_ambient = 0.6
        self.material_diffuse = 0.8
        self.material_specular = 0.0

        # PyVista Setup
        self.plotter = pv.Plotter()
        self.plotter.set_background("skyblue")  # Sky blue
        # self.plotter.enable_shadows()  # TEMP DISABLED - could block rendering!

        self._reinit_lights()

        try:
            # RTX 4090 tuning: maximum shadow map resolution
            try:
                for renderer in self.plotter.renderers:
                    pv.set_new_attribute(renderer, "shadow_map_size", 8192)  # Maximum sharpness for a high-end GPU!
            except Exception as shadow_e:
                logger.error(f"  [i] Shadow map tuning failed: {shadow_e}")
        except Exception as e:
            logger.debug(f"  [i] Light setup: {e}")

        # Restore window position/size
        self._apply_saved_window_state()

        # Observer
        self.plotter.iren.add_observer("KeyPressEvent", self._on_key_press)
        self.plotter.iren.add_observer("ExitEvent", self._on_close_save_window_state)
        self.plotter.iren.add_observer("ScrollEvent", self._on_camera_change)
        self.plotter.iren.add_observer("EndInteractionEvent", self._on_camera_change)
        self.plotter.iren.add_observer("InteractionEvent", self._on_camera_change)
        self.plotter.iren.add_observer("RenderEvent", self._on_render_event)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_mouse_click)

        # Register atexit handler as a fallback (for safe saving on exit)
        atexit.register(self._on_close_save_window_state)

        logger.info(f"\nDAE loaded with tile geometries")
        logger.info("\nControls:")
        logger.info("  X = Toggle textures (Rendering ↔ Grid) - triggers rebuild")
        logger.info("\nShow/hide layers WITHOUT rebuild (available in BOTH views):")
        logger.info("  T = Toggle Terrain")
        logger.info("  S = Toggle roads")
        logger.info("  H = Toggle buildings")
        logger.info("  C = Toggle forests (trees)")
        logger.debug("  D = Toggle Debug (Junctions, Centerlines, Boundaries)")
        logger.info("\nGeneral:")
        logger.info("  K = Load camera | Shift+K = Save camera")
        logger.info("  L = Reload DAE")
        logger.info("  Up/Down = Change zoom")
        logger.info("  Double left click = Set camera to point (40 m distance)")

        self.update_view()
        # NOTE: _apply_saved_camera_state() is NOT called at startup
        # to make sure the camera fits the geometry!
        # It is only called when the viewer is started with show()

        # Load the debug layer at startup if enabled
        if self.show_debug:
            self._update_debug_visibility()

        # Load the forest layer at startup if enabled
        if self.show_forest:
            self._update_forest_visibility()

    def _on_key_press(self, obj, event):
        """KeyPress Event Handler."""
        key = obj.GetKeySym()
        key_lower = key.lower()

        if key_lower == "s":
            # Toggle roads only (no rebuild)
            self.show_roads = not self.show_roads
            logger.info(f"\n[Roads] {'ON' if self.show_roads else 'OFF'}")
            self._update_visibility()

        elif key_lower == "t":
            # Toggle terrain only (no rebuild)
            self.show_terrain = not self.show_terrain
            logger.info(f"\n[Terrain] {'ON' if self.show_terrain else 'OFF'}")
            self._update_visibility()

        elif key_lower == "h":
            # Toggle buildings only (no rebuild)
            self.show_buildings = not self.show_buildings
            logger.info(f"\n[Buildings] {'ON' if self.show_buildings else 'OFF'}")
            self._update_visibility()

        elif key_lower == "c":
            # Toggle forests/trees only (with lazy loading on the first toggle)
            self.show_forest = not self.show_forest
            logger.info(f"\n[Forests] {'ON' if self.show_forest else 'OFF'}")
            self._update_forest_visibility()

        if key == "o":
            self.show_horizon = not self.show_horizon
            logger.info(f"\n[Horizon] {'ON' if self.show_horizon else 'OFF'}")
            self._update_visibility()

        elif key_lower == "x":
            # Toggle textures (with rebuild!)
            self.use_textures = not self.use_textures
            logger.info(f"\n[{'Rendering' if self.use_textures else 'Grid'}-view]")
            self.update_view()
            # Debug layers stay loaded permanently and their visibility is kept

        elif key_lower == "d":
            # Toggle debug in BOTH views (rendering and grid)
            self.show_debug = not self.show_debug
            logger.debug(f"\n[Debug] {'ON' if self.show_debug else 'OFF'}")
            self._update_debug_visibility()

        elif key == "K":  # Shift+K (MUST come BEFORE "k"!)
            self.save_camera_state()
        elif key_lower == "k":
            self.load_camera_state()

        elif key == "Up":
            self._adjust_zoom(-5.0)  # Zoom in
        elif key == "Down":
            self._adjust_zoom(5.0)  # Zoom out

        elif key_lower == "l":
            self.reload_dae_file()

    def _update_visibility(self):
        """Update visibility of the terrain/road/building/forest actors without reload."""
        for actor in self.terrain_actors:
            actor.SetVisibility(self.show_terrain)
        for actor in self.road_actors:
            actor.SetVisibility(self.show_roads)
        for actor in self.building_actors:
            actor.SetVisibility(self.show_buildings)
        for actor in self.horizon_actors:
            actor.SetVisibility(self.show_horizon)
        for actor in self.forest_actors:
            actor.SetVisibility(self.show_forest)

        self._update_active_layers_text()
        self.plotter.render()

    def _update_debug_visibility(self):
        """Update visibility of the debug actors without reload."""
        if not self.debug_loaded:
            # Load the debug layer for the first time
            self._load_debug_layer()
            self.debug_loaded = True

        # Toggle visibility
        for actor in self.debug_actors:
            actor.SetVisibility(self.show_debug)

        self._update_active_layers_text()
        self.plotter.render()

    def _update_forest_visibility(self):
        """Update visibility of the forest actors without reload."""
        if not self.forest_loaded:
            # Load the forest layer for the first time
            self._load_forest_layer()
            self.forest_loaded = True

        # Toggle visibility
        for actor in self.forest_actors:
            actor.SetVisibility(self.show_forest)

        self._update_active_layers_text()
        self.plotter.render()

    def _reinit_lights(self):
        """Re-initialize lights after clear()."""
        try:
            self.plotter.remove_all_lights()
        except:
            pass

        try:
            # Main light: sun with parallel rays (directional light)
            sun_light = pv.Light(
                position=[300, -300, 600],
                focal_point=[0, 0, 0],
                positional=False,  # Parallel rays instead of a point light
                cone_angle=80,
                intensity=1.0,
                shadow_attenuation=0.95,
            )
            self.plotter.add_light(sun_light)

            # Diffuse fill light: even illumination
            fill_light = pv.Light(
                light_type="headlight",  # Diffuse light
                intensity=0.0,
            )
            self.plotter.add_light(fill_light)
        except Exception as e:
            logger.error(f"[!] Error setting up lights: {e}")

    def _print_texture_debug_info(self):
        """Print debug information about available textures and material assignments."""
        logger.debug("\n[TEXTURE DEBUG INFO]")
        logger.info("=" * 80)

        # Tile textures
        if self.textures:
            logger.info(f"\n[Tile textures] {len(self.textures)} available:")
            for key in sorted(self.textures.keys())[:10]:  # Show the first 10
                logger.info(f"  • {key}")
            if len(self.textures) > 10:
                logger.info(f"  ... and {len(self.textures) - 10} more")
        else:
            logger.info("\n[Tile textures] NONE found (textures/ directory empty?)")

        # Material textures
        if self.material_textures:
            logger.info(f"\n[Material textures] {len(self.material_textures)} found:")
            for mat_name in sorted(self.material_textures.keys()):
                logger.info(f"  • {mat_name}")
        else:
            logger.warning("\n[Material textures] NONE found (main.materials.json has no textures?)")

        # Material structure
        if self.materials:
            logger.info(f"\n[Materials JSON] {len(self.materials)} materials defined:")
            roads = [m for m in self.materials.keys() if "road" in m.lower()]
            buildings = [
                m for m in self.materials.keys() if "build" in m.lower() or "wall" in m.lower() or "roof" in m.lower()
            ]
            other = [m for m in self.materials.keys() if m not in roads and m not in buildings]

            if roads:
                logger.info(f"  Roads ({len(roads)}): {', '.join(roads[:3])}")
            if buildings:
                logger.info(f"  Buildings ({len(buildings)}): {', '.join(buildings[:3])}")
            if other:
                logger.info(f"  Other ({len(other)}): {', '.join(other[:3])}")

        logger.info("=" * 80 + "\n")

    def update_view(self):
        """Update the 3D view."""
        # Save the camera before clear() - BUT: on the FIRST call ignore the default camera (1,1,1)!
        camera_pos = None
        camera_focal = None
        camera_up = None

        # Only save if NOT the first call (update_view from __init__)
        if not self._first_update_view:
            try:
                cam = self.plotter.camera
                if cam is not None:
                    camera_pos = list(cam.position)
                    camera_focal = list(cam.focal_point)
                    camera_up = list(cam.up)
            except Exception as e:
                logger.error(f"[!] Error saving camera before update_view: {e}")
        else:
            # First call - remember that we are now in "reload" mode
            self._first_update_view = False

        # Save the debug actors BEFORE clear() - to preserve them
        saved_debug_actors = self.debug_actors.copy() if self.debug_actors else []
        saved_debug_visibility = self.show_debug

        # Save the forest actors BEFORE clear() - to preserve them
        saved_forest_actors = self.forest_actors.copy() if self.forest_actors else []
        saved_forest_visibility = self.show_forest

        self.plotter.clear()
        self._reinit_lights()

        # Clear ONLY the terrain/road/building/horizon/forest actor lists
        self.terrain_actors = []
        self.road_actors = []
        self.building_actors = []
        self.horizon_actors = []
        self.forest_actors = []
        # Debug actors and forest actors were deleted by clear(), but we reload them afterwards

        # IMPORTANT: reset forest_loaded so that forest.forest4.json is reloaded on the next access
        self.forest_loaded = False

        # Iterate over all loaded DAE files
        for item_name, tile_data in self.tile_data:
            logger.info(f"Rendering {item_name}...")
            self._render_single_dae(item_name, tile_data)

        # Status lines
        # Top left: usage instructions
        bedienung = "S: Roads | T: Terrain | H: Buildings | C: Forests | O: Horizon | D: Debug | X: Textures | K: Cam | L: Reload | 2xLMB: Jump"
        self.plotter.add_text(
            bedienung,
            position="upper_left",
            font_size=10,
        )

        # Top right: active layers
        self._update_active_layers_text()

        # Fit camera to all actors (only at startup, not on reload)
        if camera_pos is None:
            # First initialization - use view_isometric for automatic framing
            try:
                self.plotter.view_isometric()
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()

                logger.debug(f"  [i] Camera positioned with view_isometric()")
                logger.info(f"      Position: {self.plotter.camera.position}")

                # Try to load the saved camera (overrides view_isometric)
                saved_camera = self._load_camera_state()
                if saved_camera:
                    try:
                        pos = saved_camera.get("position")
                        focal = saved_camera.get("focal_point")
                        up = saved_camera.get("up_vector")
                        if pos and focal and up:
                            self.plotter.camera.position = pos
                            self.plotter.camera.focal_point = focal
                            self.plotter.camera.up = up
                            self.plotter.reset_camera_clipping_range()
                            self.plotter.render()
                            logger.info(f"  [✓] Saved camera loaded")
                            logger.info(f"      Position: {pos}")
                    except Exception as e:
                        logger.error(f"  [!] Error loading saved camera: {e}")

            except Exception as e:
                logger.error(f"  [!] Error initializing camera: {e}")
        else:
            # Reload - restore the old camera
            try:
                cam = self.plotter.camera
                cam.position = camera_pos
                cam.focal_point = camera_focal
                cam.up = camera_up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
            except Exception as e:
                logger.error(f"  [!] Error restoring camera: {e}")

        self._update_camera_status()

        # Re-add the debug actors to the plotter (if they existed)
        if saved_debug_actors:
            self.debug_actors = []
            for actor in saved_debug_actors:
                try:
                    self.plotter.add_actor(actor)
                    self.debug_actors.append(actor)
                    # Restore visibility
                    actor.SetVisibility(saved_debug_visibility)
                except Exception as e:
                    logger.error(f"[!] Error restoring debug actor: {e}")

        # Re-add the forest actors to the plotter (if they existed)
        if saved_forest_actors:
            self.forest_actors = []
            for actor in saved_forest_actors:
                try:
                    self.plotter.add_actor(actor)
                    self.forest_actors.append(actor)
                    # Restore visibility
                    actor.SetVisibility(saved_forest_visibility)
                except Exception as e:
                    logger.error(f"[!] Error restoring forest actor: {e}")

    def _index_to_coords(self, item_name, tile_index_x, tile_index_y):
        """
        Convert tile indices (e.g. tile_-2_-2) to absolute coordinates.

        The indices are grid positions with 500 m spacing.
        Index -2, -1, 0, 1 correspond to coordinates -1000, -500, 0, 500.

        Returns: (x_coord, y_coord)
        """
        x_coord = tile_index_x * 500
        y_coord = tile_index_y * 500
        return (x_coord, y_coord)

    def _get_actor_list_for_item(self, item_name):
        """Determine which actor list an item belongs to, based on item_name."""
        is_terrain = item_name.startswith("terrain_") or item_name.startswith("tile_")
        is_horizon = "horizon" in item_name.lower()
        is_building = item_name.startswith("buildings_")

        if is_horizon:
            return self.horizon_actors
        elif is_building:
            return self.building_actors
        elif is_terrain:
            return self.terrain_actors
        else:
            return self.terrain_actors  # Default: terrain_actors

    def _render_single_dae(self, item_name, tile_data):
        """Render a single DAE file (terrain or building)."""
        vertices = tile_data.get("vertices", [])
        faces = tile_data.get("faces", [])
        materials = tile_data.get("materials", [])
        tiles_info = tile_data.get("tiles", {})

        if len(vertices) == 0:
            logger.error(f"  [!] {item_name}: No vertices")
            return

        # Determine whether terrain or building
        is_terrain = item_name.startswith("terrain_") or item_name.startswith("tile_")
        is_horizon = "horizon" in item_name.lower()
        is_building = item_name.startswith("buildings_")

        # Colors from grid_colors
        face_colors = {
            "terrain": self.grid_colors.get("terrain", {}).get("face", [0.8, 0.95, 0.8]),
            "road": self.grid_colors.get("road", {}).get("face", [1.0, 1.0, 1.0]),
            "building_wall": self.grid_colors.get("building_wall", {}).get("face", [0.95, 0.95, 0.95]),
            "building_roof": self.grid_colors.get("building_roof", {}).get("face", [0.6, 0.2, 0.1]),
        }
        edge_colors = {
            "terrain": self.grid_colors.get("terrain", {}).get("edge", [0.2, 0.5, 0.2]),
            "road": self.grid_colors.get("road", {}).get("edge", [1.0, 0.0, 0.0]),
            "building_wall": self.grid_colors.get("building_wall", {}).get("edge", [0.3, 0.3, 0.3]),
            "building_roof": self.grid_colors.get("building_roof", {}).get("edge", [0.3, 0.1, 0.05]),
        }

        # Categorize faces by material (for both rendering modes)
        terrain_faces = []
        road_faces_by_material = {}  # {material_name: [faces]}
        wall_faces = []
        roof_faces = []

        for face_idx, material in enumerate(materials):
            mat_lower = material.lower()
            # Categorization is based on material names and item context
            # Priority: wall > roof > road > terrain > building_default > fallback
            if "wall" in mat_lower:
                wall_faces.append(faces[face_idx])
            elif "roof" in mat_lower:
                roof_faces.append(faces[face_idx])
            elif "terrain" in mat_lower or "tile" in mat_lower or material == "terrain":
                # Explicitly "terrain" or with "terrain"/"tile" in the name
                terrain_faces.append(faces[face_idx])
            elif "road" in mat_lower or (
                not is_building and not "terrain" in mat_lower and not "tile" in mat_lower and material != "unknown"
            ):
                # Road: has "road" in the name OR (is not a building and not an unknown material)
                # This also catches faces inserted by stitch_gaps that are exported as "terrain" material
                # but may show up under other names in the DAE
                if material not in road_faces_by_material:
                    road_faces_by_material[material] = []
                road_faces_by_material[material].append(faces[face_idx])
            elif is_building:
                # In buildings: everything else is a wall
                wall_faces.append(faces[face_idx])
            else:
                # In terrain: everything else is terrain
                terrain_faces.append(faces[face_idx])

        # Rendering with textures (terrain only)
        if self.use_textures and tiles_info and (is_terrain or is_horizon):
            terrain_texture_log = []
            for tile_name, tile_info in tiles_info.items():
                tile_vertices_local = tile_info.get("vertices", [])
                tile_faces_local = tile_info.get("faces_local", [])
                tile_uvs = tile_info.get("uvs", [])

                if len(tile_faces_local) == 0 or len(tile_vertices_local) == 0:
                    continue

                if not isinstance(tile_vertices_local, np.ndarray):
                    tile_vertices_local = np.array(tile_vertices_local)

                mesh = self._create_mesh_with_uvs(tile_vertices_local, tile_faces_local, tile_uvs)

                # tile_name is already in coordinate format (e.g. "tile_-1000_-1000")
                # NO conversion needed anymore, since the DAE export now uses world coordinates!
                lookup_key = tile_name.lower()
                texture = self.textures.get(lookup_key)

                if texture is None and is_horizon:
                    # Fallback: use the known horizon texture if the tile name does not match
                    texture = self.textures.get("horizon_sentinel2") or next(
                        (tex for key, tex in self.textures.items() if "horizon" in key),
                        None,
                    )

                if texture is not None and len(tile_uvs) > 0:
                    try:
                        actor = self.plotter.add_mesh(
                            mesh,
                            texture=texture,
                            opacity=1.0,
                            label=f"{item_name}_{tile_name}",
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        self._get_actor_list_for_item(item_name).append(actor)
                        visibility = self.show_horizon if "horizon" in item_name.lower() else self.show_terrain
                        actor.SetVisibility(visibility)
                        terrain_texture_log.append(f"✓ {tile_name} → {lookup_key}")
                    except Exception as e:
                        logger.error(f"  [!] Texture error for {tile_name}: {e}")
                        terrain_texture_log.append(f"✗ {tile_name} → ERROR: {str(e)[:40]}")
                        # Fall back to color
                        actor = self.plotter.add_mesh(
                            mesh, color=[0.6, 0.5, 0.4], opacity=0.5, label=f"{item_name}_{tile_name}"
                        )
                        self._get_actor_list_for_item(item_name).append(actor)
                        visibility = self.show_horizon if "horizon" in item_name.lower() else self.show_terrain
                        actor.SetVisibility(visibility)
                else:
                    # No texture or no UVs
                    reason = "NO UVs" if len(tile_uvs) == 0 else f"Texture not found: {lookup_key}"
                    terrain_texture_log.append(f"○ {tile_name} → {reason}")
                    actor = self.plotter.add_mesh(
                        mesh, color=[0.6, 0.5, 0.4], opacity=0.5, label=f"{item_name}_{tile_name}"
                    )
                    self._get_actor_list_for_item(item_name).append(actor)
                    visibility = self.show_horizon if "horizon" in item_name.lower() else self.show_terrain
                    actor.SetVisibility(visibility)

            # Debug output
            if terrain_texture_log:
                logger.info(f"\n[{item_name}] Terrain texture assignment:")
                for entry in terrain_texture_log[:5]:  # Show the first 5
                    logger.info(f"  {entry}")
                if len(terrain_texture_log) > 5:
                    logger.info(f"  ... and {len(terrain_texture_log) - 5} more")
        else:
            # Grid view: render terrain with colors
            if terrain_faces:
                terrain_mesh = self._create_mesh(vertices, terrain_faces)
                actor = self.plotter.add_mesh(
                    terrain_mesh,
                    color=face_colors["terrain"],
                    label=f"{item_name}_terrain",
                    opacity=0.5,
                    show_edges=True,
                    edge_color=edge_colors["terrain"],
                    line_width=1.0,
                    lighting=True,
                    ambient=self.material_ambient,
                    diffuse=self.material_diffuse,
                    specular=self.material_specular,
                )
                self._get_actor_list_for_item(item_name).append(actor)
                visibility = self.show_horizon if "horizon" in item_name.lower() else self.show_terrain
                actor.SetVisibility(visibility)

        # Render roads per material (always, regardless of texture or grid)
        if road_faces_by_material and tiles_info:
            road_opacity = self.grid_colors.get("road", {}).get("face_opacity", 0.5)

            # Extract UVs from tiles_info (for all vertices)
            global_uvs = self._extract_global_uvs(tiles_info, len(vertices))

            for road_material, road_faces in road_faces_by_material.items():
                # Create mesh with UVs (if available)
                if global_uvs is not None and len(global_uvs) == len(vertices):
                    road_mesh = self._create_mesh_with_uvs(vertices, road_faces, global_uvs)
                    has_uvs = True
                else:
                    road_mesh = self._create_mesh(vertices, road_faces)
                    has_uvs = False

                # In texture view: try to use the material texture (only if UVs are present)
                if self.use_textures and has_uvs:
                    if road_material in self.material_textures:
                        texture = self.material_textures[road_material]
                        try:
                            actor = self.plotter.add_mesh(
                                road_mesh,
                                texture=texture,
                                label=f"{item_name}_road_{road_material}",
                                opacity=1.0,
                                show_edges=False,
                                lighting=True,
                                ambient=self.material_ambient,
                                diffuse=self.material_diffuse,
                                specular=self.material_specular,
                            )
                            self.road_actors.append(actor)
                            actor.SetVisibility(self.show_roads)
                            logger.info(f"  [✓ Road] {road_material}: Texture applied ({len(road_faces)} faces)")
                        except Exception as e:
                            logger.error(f"  [! Road] {road_material}: Texture error: {e}. Falling back to color.")
                            actor = self.plotter.add_mesh(
                                road_mesh,
                                color=face_colors["road"],
                                label=f"{item_name}_road_{road_material}",
                                opacity=road_opacity,
                                show_edges=True,
                                edge_color=edge_colors["road"],
                                line_width=2.0,
                                lighting=True,
                                ambient=self.material_ambient,
                                diffuse=self.material_diffuse,
                                specular=self.material_specular,
                            )
                            self.road_actors.append(actor)
                            actor.SetVisibility(self.show_roads)
                    else:
                        # Fallback: color
                        logger.info(
                            f"  [○ Road] {road_material}: Texture not found. Color fallback ({len(road_faces)} faces)."
                        )
                        actor = self.plotter.add_mesh(
                            road_mesh,
                            color=face_colors["road"],
                            label=f"{item_name}_road_{road_material}",
                            opacity=road_opacity,
                            show_edges=True,
                            edge_color=edge_colors["road"],
                            line_width=2.0,
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        self.road_actors.append(actor)
                        actor.SetVisibility(self.show_roads)
                else:
                    # Grid view or no UVs: color with edges
                    reason = "Grid view" if not self.use_textures else "No UVs"
                    actor = self.plotter.add_mesh(
                        road_mesh,
                        color=face_colors["road"],
                        label=f"{item_name}_road_{road_material}",
                        opacity=road_opacity,
                        show_edges=True,
                        edge_color=edge_colors["road"],
                        line_width=2.0,
                        lighting=True,
                        ambient=self.material_ambient,
                        diffuse=self.material_diffuse,
                        specular=self.material_specular,
                    )
                    self.road_actors.append(actor)
                    actor.SetVisibility(self.show_roads)

        # Render buildings (walls + roofs) - unified with terrain rendering
        if is_building and (wall_faces or roof_faces):
            # Collect UVs from ALL tiles (each building is a separate geometry/tile in the DAE)
            building_uvs = None
            if tiles_info:
                all_uvs = []
                for tile_name, tile_data in sorted(tiles_info.items()):
                    tile_uvs = tile_data.get("uvs")
                    if tile_uvs is not None and len(tile_uvs) > 0:
                        all_uvs.append(tile_uvs)

                if all_uvs:
                    building_uvs = np.vstack(all_uvs)
                    if len(building_uvs) != len(vertices):
                        logger.error(f"  [!] UV/Vertex Mismatch: {len(building_uvs)} UVs vs {len(vertices)} Vertices")
                        building_uvs = None

            # Render walls
            if wall_faces:
                wall_mesh = self._create_mesh_with_uvs(vertices, wall_faces, building_uvs)

                # Texture view: try the material texture
                if self.use_textures:
                    wall_material = next((mat for mat in materials if "wall" in mat.lower()), "lod2_wall_white")
                    if wall_material in self.material_textures:
                        texture = self.material_textures[wall_material]
                        actor = self.plotter.add_mesh(
                            wall_mesh,
                            texture=texture,
                            opacity=1.0,
                            label=f"{item_name}_walls",
                            show_edges=False,
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        logger.info(f"  [✓ Walls] {wall_material} with texture")
                    else:
                        # Fallback: white color
                        actor = self.plotter.add_mesh(
                            wall_mesh,
                            color=face_colors["building_wall"],
                            opacity=1.0,
                            label=f"{item_name}_walls",
                            show_edges=False,
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        logger.error(f"  [○ Walls] Color fallback (material {wall_material} not found)")
                else:
                    # Grid view: color with edges
                    actor = self.plotter.add_mesh(
                        wall_mesh,
                        color=face_colors["building_wall"],
                        opacity=0.8,
                        label=f"{item_name}_walls",
                        show_edges=True,
                        edge_color=edge_colors["building_wall"],
                        line_width=1.0,
                        lighting=True,
                        ambient=self.material_ambient,
                        diffuse=self.material_diffuse,
                        specular=self.material_specular,
                    )
                self.building_actors.append(actor)
                actor.SetVisibility(self.show_buildings)

            # Render roofs
            if roof_faces:
                roof_mesh = self._create_mesh_with_uvs(vertices, roof_faces, building_uvs)

                # Texture view: try the material texture
                if self.use_textures:
                    roof_material = next((mat for mat in materials if "roof" in mat.lower()), "lod2_roof_red")
                    if roof_material in self.material_textures:
                        texture = self.material_textures[roof_material]
                        actor = self.plotter.add_mesh(
                            roof_mesh,
                            texture=texture,
                            opacity=1.0,
                            label=f"{item_name}_roofs",
                            show_edges=False,
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        logger.info(f"  [✓ Roofs] {roof_material} with texture")
                    else:
                        # Fallback: red color
                        actor = self.plotter.add_mesh(
                            roof_mesh,
                            color=face_colors["building_roof"],
                            opacity=1.0,
                            label=f"{item_name}_roofs",
                            show_edges=False,
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        logger.error(f"  [○ Roofs] Color fallback (material {roof_material} not found)")
                else:
                    # Grid view: color with edges
                    actor = self.plotter.add_mesh(
                        roof_mesh,
                        color=face_colors["building_roof"],
                        opacity=0.8,
                        label=f"{item_name}_roofs",
                        show_edges=True,
                        edge_color=edge_colors["building_roof"],
                        line_width=1.0,
                        lighting=True,
                        ambient=self.material_ambient,
                        diffuse=self.material_diffuse,
                        specular=self.material_specular,
                    )
                self.building_actors.append(actor)
                actor.SetVisibility(self.show_buildings)

    def _create_mesh(self, vertices, faces):
        """Create a PyVista PolyData mesh from vertices and faces."""
        # PyVista expects: [num_points_in_face, pt0, pt1, pt2, ...]
        pyvista_faces = []
        for face in faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(vertices, pyvista_faces)
        # CRUCIAL: split_sharp_edges=True produces hard edges for buildings!
        # Without it: walls look "round" or completely unshaded
        try:
            mesh = mesh.compute_normals(
                cell_normals=True, point_normals=True, split_sharp_edges=True  # ESSENTIAL for sharp shading
            )
        except TypeError:
            # Fallback: older PyVista without split_sharp_edges
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True)
        return mesh

    def _create_mesh_with_uvs(self, vertices, faces, uvs):
        """
        Create a PyVista PolyData mesh with texture coordinates.

        IMPORTANT: This builds a REMAPPED mesh in which only the vertices needed by the
        faces are used. This produces correct UV indexing!
        """
        # Collect unique vertices used by faces
        unique_vertex_indices = set()
        for face in faces:
            unique_vertex_indices.update(face)

        unique_vertex_indices = sorted(unique_vertex_indices)

        # Create remapping: old_index → new_index
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

        # Remapped vertices and UVs
        remapped_vertices = vertices[unique_vertex_indices]
        remapped_uvs = uvs[unique_vertex_indices] if uvs is not None else None

        # Remapped faces (with new indices)
        remapped_faces = []
        for face in faces:
            remapped_face = [vertex_map[v_idx] for v_idx in face]
            remapped_faces.append(remapped_face)

        # Create PyVista faces
        pyvista_faces = []
        for face in remapped_faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(remapped_vertices, pyvista_faces)

        # Set texture coordinates if present
        if remapped_uvs is not None:
            mesh.active_texture_coordinates = remapped_uvs % 1.0

        # Compute normals
        try:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True, split_sharp_edges=True)
        except TypeError:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True)

        return mesh

    def _extract_building_uvs(self, tiles_info, vertices):
        """
        Extract UV coordinates for a building from tiles_info.

        Args:
            tiles_info: Dict with tile information (contains UVs)
            vertices: NumPy array of vertices

        Returns:
            NumPy array of UV coordinates (n, 2), or an empty array
        """
        if not tiles_info:
            logger.debug(f"  [DEBUG] _extract_building_uvs: tiles_info is empty")
            return np.array([])

        logger.debug(f"  [DEBUG] _extract_building_uvs: tiles_info keys = {list(tiles_info.keys())}")

        # Collect UVs from all building tiles (they are already in the right order)
        all_uvs = []
        for tile_name, tile_data in tiles_info.items():
            logger.info(f"    Checking tile: {tile_name}")
            uvs = tile_data.get("uvs", np.array([]))
            if len(uvs) > 0:
                logger.info(f"      → Has UVs: shape={uvs.shape}")
                all_uvs.append(uvs)
            else:
                logger.info(f"      → No UVs")

        # Combine all UVs
        if all_uvs:
            combined_uvs = np.vstack(all_uvs)
            logger.debug(f"  [DEBUG] Combined UVs: shape={combined_uvs.shape}, vertices shape={vertices.shape}")
            return combined_uvs

        logger.debug(f"  [DEBUG] No UVs found!")
        return np.array([])

    def _extract_global_uvs(self, tiles_info, num_vertices):
        """
        Extract global UV coordinates from tiles_info.

        The DAE stores vertices and UVs per tile. This function
        combines the UVs of all tiles into one global UV array.

        IMPORTANT: The order of the tiles must match the vertex order!
        Therefore do NOT use sorted()!

        Args:
            tiles_info: Dict with tile information
            num_vertices: Number of global vertices

        Returns:
            NumPy array (num_vertices, 2) of UV coordinates, or None
        """
        if not tiles_info:
            return None

        # Collect UVs from all tiles (WITHOUT sorted, so the order is preserved!)
        all_uvs = []
        for tile_name, tile_data in tiles_info.items():
            uvs = tile_data.get("uvs", np.array([]))
            if len(uvs) > 0:
                all_uvs.append(uvs)

        # Combine all tile UVs
        if all_uvs:
            combined_uvs = np.vstack(all_uvs)
            if len(combined_uvs) == num_vertices:
                return combined_uvs
            else:
                logger.error(f"  [!] UV array size mismatch: {len(combined_uvs)} UVs vs {num_vertices} Vertices")
                return None

        return None

    def _load_grid_colors(self):
        """Load grid colors from debug_network.json."""
        debug_network_path = Path(__file__).parent.parent / "cache" / "debug_network.json"

        # Default grid colors
        default_colors = {
            "terrain": {
                "face": [0.8, 0.95, 0.8],
                "edge": [0.2, 0.5, 0.2],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "road": {
                "face": [1.0, 1.0, 1.0],
                "edge": [1.0, 0.0, 0.0],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "building_wall": {
                "face": [0.95, 0.95, 0.95],
                "edge": [0.3, 0.3, 0.3],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "building_roof": {
                "face": [0.6, 0.2, 0.1],
                "edge": [0.3, 0.1, 0.05],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "junction": {
                "color": [0.0, 0.0, 1.0],
                "opacity": 0.5,
            },
            "centerline": {
                "color": [0.0, 0.0, 1.0],
                "line_width": 2.0,
                "opacity": 1.0,
            },
            "boundary": {
                "color": [1.0, 0.0, 1.0],
                "line_width": 2.0,
                "opacity": 1.0,
            },
        }

        if not debug_network_path.exists():
            return default_colors

        try:
            with open(debug_network_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)
            return debug_data.get("grid_colors", default_colors)
        except Exception as e:
            logger.error(f"  [!] Error loading grid colors: {e}")
            return default_colors

    def _load_textures(self):
        """Load all tile textures from the textures directory."""
        textures = {}

        if not self.textures_dir.exists():
            logger.error(f"  [!] Textures directory not found: {self.textures_dir}")
            return textures

        patterns = ["*.jpg", "*.jpeg", "*.png", "*.dds"]
        texture_files = []

        for pattern in patterns:
            texture_files.extend(Path(self.textures_dir).glob(pattern))

        for texture_path in texture_files:
            texture_key = texture_path.stem.lower()  # e.g. "tile_0_0" or "horizon_sentinel2"

            try:
                if texture_path.suffix.lower() == ".dds":
                    try:
                        import importlib

                        imageio = importlib.import_module("imageio.v2")
                        img_array = imageio.imread(str(texture_path))
                    except ImportError:
                        logger.error(f"  [!] imageio not available, skipping DDS texture {texture_path.name}")
                        continue
                else:
                    img = Image.open(texture_path)
                    img_array = np.array(img.convert("RGB"))

                if img_array.ndim == 2:  # Grayscale -> duplicate to RGB
                    img_array = np.stack([img_array] * 3, axis=-1)

                textures[texture_key] = pv.Texture(img_array)

            except Exception as e:
                logger.error(f"  [!] Error loading {texture_path.name}: {e}")

        return textures

    def _load_material_textures(self):
        """
        Load textures from main.materials.json for roads and buildings.

        Returns:
            Dict {material_name: pv.Texture}
        """
        material_textures = {}

        if not self.materials:
            return material_textures

        for mat_name, mat_data in self.materials.items():
            stages = mat_data.get("Stages", [])
            if not stages or not isinstance(stages, list) or len(stages) == 0:
                continue

            stage = stages[0]  # Use the first stage

            if not isinstance(stage, dict):
                continue

            # Look for baseColorMap (primary texture)
            texture_path = stage.get("baseColorMap")
            diffuse_color = stage.get("diffuseColor")  # Optional tint or plain color

            if not texture_path and diffuse_color:
                # No image, but a color is present -> 1x1 color patch as texture
                try:
                    color_rgb = self._normalize_diffuse_color(diffuse_color)
                    img_array = np.array([[color_rgb]], dtype=np.uint8)
                    texture = pv.Texture(img_array)
                    texture.mipmap = True
                    texture.interpolate = True
                    material_textures[mat_name] = texture
                    logger.info(f"  [✓] Material color texture generated: {mat_name} (diffuseColor)")
                except Exception as e:
                    logger.error(f"  [!] Could not generate diffuseColor for {mat_name}: {e}")
                continue

            if not texture_path:
                continue

            # Convert BeamNG path to absolute path
            abs_texture_path = self._resolve_asset_path(texture_path)

            if not abs_texture_path:
                # Show the path that was actually searched
                if texture_path.startswith("/assets/"):
                    rel_path_posix = PurePosixPath(texture_path[1:])
                    data_dir = Path(__file__).parent.parent / "data"
                    abs_path = (data_dir / rel_path_posix).resolve()
                    logger.error(f"  [!] Material texture for {mat_name} not found: {abs_path}")
                elif texture_path.startswith("/levels/") or texture_path.startswith(str(config.RELATIVE_DIR)):
                    attempted_path = _resolve_beamng_path(texture_path)
                    logger.error(
                        f"  [!] Material texture for {mat_name} not found: {attempted_path or texture_path}"
                    )
                else:
                    logger.error(
                        f"  [!] Material texture for {mat_name} not resolvable: {Path(texture_path).as_posix()}"
                    )
                continue

            if not Path(abs_texture_path).exists():
                logger.error(f"  [!] Material texture for {mat_name} not found: {abs_texture_path}")
                continue

            try:
                # Load texture
                if Path(abs_texture_path).suffix.lower() == ".dds":
                    try:
                        import importlib

                        imageio = importlib.import_module("imageio.v2")
                        img_array = imageio.imread(str(abs_texture_path))
                        # Convert RGBA to RGB (remove alpha channel for full opacity)
                        if img_array.ndim == 3 and img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                    except ImportError:
                        logger.error(f"  [!] imageio not available, skipping {mat_name} DDS texture")
                        continue
                else:
                    img = Image.open(abs_texture_path)
                    img_array = np.array(img.convert("RGB"))
                    logger.info(f"  [✓] Material texture loaded: {mat_name} -> {abs_texture_path.replace('/', os.sep)}")

                if img_array.ndim == 2:  # Grayscale -> RGB
                    img_array = np.stack([img_array] * 3, axis=-1)

                # Apply optional tint
                if diffuse_color:
                    try:
                        img_array = self._apply_diffuse_tint(img_array, diffuse_color)
                        logger.info(f"  [✓] diffuseColor applied: {mat_name}")
                    except Exception as e:
                        logger.error(f"  [!] Could not apply diffuseColor for {mat_name}: {e}")

                texture = pv.Texture(img_array)
                # Enable mipmap and interpolation for better quality
                texture.mipmap = True
                texture.interpolate = True
                material_textures[mat_name] = texture

            except Exception as e:
                logger.error(f"  [!] Error loading material texture {mat_name}: {e}")

        return material_textures

    def _normalize_diffuse_color(self, color):
        """Normalize diffuseColor (0-1 floats) to uint8 RGB."""
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            raise ValueError("diffuseColor must have at least 3 components")
        # Use only RGB, alpha is ignored for the texture
        rgb = [max(0.0, min(1.0, float(c))) for c in color[:3]]
        return [int(round(c * 255)) for c in rgb]

    def _apply_diffuse_tint(self, img_array, color):
        """Apply diffuseColor as a multiplier to the texture."""
        rgb = np.array(self._normalize_diffuse_color(color), dtype=np.float32) / 255.0
        # Make sure the image has 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        tinted = np.clip(img_array.astype(np.float32) * rgb, 0, 255).astype(np.uint8)
        return tinted

    def _resolve_asset_path(self, texture_path: str) -> str:
        """
        Convert a BeamNG asset path to an absolute file system path.

        Textures for roads and buildings live in the local data/ directory.

        Args:
            texture_path: BeamNG asset path (e.g. "/assets/materials/...")

        Returns:
            Absolute path or None
        """
        if not texture_path:
            return None

        # 1. Level-specific paths (/levels/World_to_BeamNG/...) -> use _resolve_beamng_path
        if texture_path.startswith("/levels/") or texture_path.startswith(
            str(config.RELATIVE_DIR)
        ):  # config.RELATIVE_DIR is PurePosixPath
            return _resolve_beamng_path(texture_path)

        # 2. Asset paths (/assets/materials/...) -> search in data/assets/
        if texture_path.startswith("/assets/"):
            rel_path = Path(texture_path[1:])  # Convert to Path to use / operator
            # Search relative to the current directory
            data_dir = Path(__file__).parent.parent / "data"
            abs_path = data_dir / rel_path
            abs_path = abs_path.resolve()  # Normalize path
            return str(abs_path) if abs_path.exists() else None

        return None

    def _update_active_layers_text(self):
        """Update the active-layer text at the top right."""
        active_items = []
        if self.show_terrain:
            active_items.append("T")
        if self.show_roads:
            active_items.append("S")  # S for roads
        if self.show_buildings:
            active_items.append("H")  # H for buildings
        if self.show_forest:
            active_items.append("C")  # C for forest/trees
        if self.show_horizon:
            active_items.append("O")  # O for horizon
        if self.use_textures:
            active_items.append("X")
        if self.show_debug:
            active_items.append("D")

        active_text = " ".join(active_items) if active_items else "-"

        try:
            self.plotter.remove_actor("active_layers_text")
        except Exception as e:
            logger.error(f"[!] Error removing the active layer text: {e}")

        try:
            self._active_layers_actor = self.plotter.add_text(
                active_text,
                position="upper_right",
                font_size=10,
                name="active_layers_text",
            )
        except Exception as e:
            logger.error(f"[!] Error creating the active layer text: {e}")
            self._active_layers_actor = None

    def _update_camera_status(self):
        """Show camera status at the bottom left and correct roll=0°, zoom=30°."""
        cam = self.plotter.camera
        if cam is None:
            return

        try:
            # Set roll to 0° and zoom to 30° automatically
            try:
                cam.up = [0.0, 0.0, 1.0]
                cam.view_angle = 30.0
            except Exception as e:
                logger.error(f"[!] Error setting camera properties: {e}")

            pos = np.array(cam.position, dtype=float)
            focal = np.array(cam.focal_point, dtype=float)

            # Read the up vector correctly
            try:
                up = np.array(cam.up, dtype=float)
            except Exception as e:
                logger.error(f"[!] Error reading the up vector: {e}")
                up = np.array([0.0, 0.0, 1.0], dtype=float)

            forward = focal - pos
            f_norm = np.linalg.norm(forward)
            if f_norm > 1e-9:
                forward = forward / f_norm
            else:
                forward = np.array([0.0, 0.0, 1.0])

            yaw = np.degrees(np.arctan2(forward[1], forward[0]))
            tilt = np.degrees(np.arctan2(forward[2], np.linalg.norm(forward[:2]) + 1e-9))

            up_proj = up - np.dot(up, forward) * forward
            u_norm = np.linalg.norm(up_proj)
            if u_norm > 1e-9:
                up_proj /= u_norm
            else:
                up_proj = np.array([0.0, 0.0, 1.0])
            roll = np.degrees(
                np.arctan2(
                    np.dot(np.cross(up_proj, [0, 0, 1]), forward),
                    np.dot(up_proj, [0, 0, 1]) + 1e-9,
                )
            )

            # Read zoom from view_angle
            try:
                zoom = cam.view_angle
            except Exception as e:
                logger.error(f"[!] Error reading the zoom value: {e}")
                zoom = 30.0

            text = (
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                f"Tilt: {tilt:.1f}° | Roll: {roll:.1f}° | Yaw: {yaw:.1f}° | Zoom: {zoom:.1f}°"
            )

            # Remove the old text actor
            try:
                self.plotter.remove_actor("camera_status_text")
            except Exception as e:
                logger.error(f"[!] Error removing the camera status text: {e}")

            try:
                self._camera_status_actor = self.plotter.add_text(
                    text,
                    position="lower_left",
                    font_size=10,
                    color="black",
                    shadow=True,
                    name="camera_status_text",
                )
            except Exception as e:
                logger.error(f"[!] Error creating the camera status text: {e}")
                self._camera_status_actor = None
        except Exception as e:
            logger.error(f"[!] Error in _update_camera_status: {e}")

    def _on_camera_change(self, obj, event):
        """Update the status line after camera changes."""
        try:
            self._update_camera_status()
        except Exception as e:
            logger.error(f"[!] Error in _on_camera_change: {e}")

    def _on_render_event(self, obj, event):
        """Update the status line on RenderEvent with throttling."""
        try:
            self._render_update_counter += 1
            if self._render_update_counter >= 5:
                self._render_update_counter = 0
                self._update_camera_status()
        except Exception as e:
            logger.error(f"[!] Error in _on_render_event: {e}")

    def _adjust_zoom(self, delta):
        """Change zoom (view_angle) by delta degrees."""
        cam = self.plotter.camera
        if cam is None:
            return
        try:
            current = cam.view_angle
            new_angle = max(5.0, min(120.0, current + delta))
            cam.view_angle = new_angle
            self._update_camera_status()
            self.plotter.render()
        except Exception as e:
            logger.error(f"[!] Error changing zoom: {e}")

    def _load_config(self):
        """Load the config file."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_config(self, data):
        """Save the config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_camera_state(self):
        """Load the camera state from the config."""
        cfg = self._load_config()
        return cfg.get("camera")

    def _load_layers_state(self):
        """Load the layer settings from the config."""
        cfg = self._load_config()
        return cfg.get("layers", {})

    def load_camera_state(self):
        """Load the saved camera position (K key)."""
        state = self._load_camera_state()
        if not state:
            logger.info("[Camera] No saved camera found")
            return
        cam = self.plotter.camera
        if cam is None:
            logger.info("[Camera] Camera not available")
            return
        try:
            pos = state.get("position")
            focal = state.get("focal_point")
            up = state.get("up_vector")
            if pos and focal and up:
                cam.position = pos
                cam.focal_point = focal
                cam.up = up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
                logger.info("[Camera] Loaded")
            else:
                logger.info("[Camera] Invalid camera state")
        except Exception as e:
            logger.error(f"[Camera] Error loading: {e}")

    def _apply_saved_camera_state(self):
        """Apply the saved camera at startup."""
        state = self._load_camera_state()
        if not state:
            return
        cam = self.plotter.camera
        if cam is None:
            return
        try:
            pos = state.get("position")
            focal = state.get("focal_point")
            up = state.get("up_vector")
            if pos and focal and up:
                cam.position = pos
                cam.focal_point = focal
                cam.up = up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
        except Exception as e:
            logger.error(f"[!] Error applying camera state: {e}")

    def save_camera_state(self):
        """Save the camera position (Shift+K)."""
        cam = self.plotter.camera
        if cam is None:
            logger.info("[Camera] Camera not available")
            return
        try:
            state = {
                "position": list(cam.position),
                "focal_point": list(cam.focal_point),
                "up_vector": list(cam.up),
            }
            cfg = self._load_config()
            cfg["camera"] = state
            self._save_config(cfg)
            logger.info(f"[Camera] Saved to {self.config_path}")
        except Exception as e:
            logger.error(f"[Camera] Error saving: {e}")
        self._save_window_state()

    def _load_window_state(self):
        """Load the window state from the config."""
        cfg = self._load_config()
        return cfg.get("window")

    def _apply_saved_window_state(self):
        """Apply the saved window position/size at startup."""
        state = self._load_window_state()
        if not state:
            return
        try:
            x = int(state.get("x", 0))
            y = int(state.get("y", 0))
            w = int(state.get("w", 0))
            h = int(state.get("h", 0))
        except Exception as e:
            logger.error(f"[!] Error converting window state values: {e}")
            return

        if w < 200 or h < 150:
            return
        if x < -5000 or y < -5000:
            return

        try:
            win = self.plotter.render_window
            win.SetSize(w, h)
            win.SetPosition(x, y)
        except Exception as e:
            logger.error(f"[!] Error applying window position/size: {e}")

    def _save_window_state(self):
        """Save the window position/size."""
        try:
            win = self.plotter.render_window
            if win is None:
                logger.error("[!] render_window is None, cannot save window state")
                return

            pos = win.GetPosition()
            size = win.GetSize()
        except Exception as e:
            logger.error(f"[!] Error reading window state: {e}")
            return

        state = {
            "x": int(pos[0]),
            "y": int(pos[1]),
            "w": int(size[0]),
            "h": int(size[1]),
        }
        cfg = self._load_config()
        cfg["window"] = state
        # Also save the current camera
        cam = self.plotter.camera
        if cam is not None:
            cfg["camera"] = {
                "position": list(cam.position),
                "focal_point": list(cam.focal_point),
                "up_vector": list(cam.up),
            }

        # Also save the layer settings
        cfg["layers"] = {
            "terrain": self.show_terrain,
            "roads": self.show_roads,
            "textures": self.use_textures,
            "debug": self.show_debug,
        }
        self._save_config(cfg)

    def _on_close_save_window_state(self, *args, **kwargs):
        """Save the window state on close (ExitEvent + atexit)."""
        try:
            # Check whether the plotter is still valid
            if self.plotter is None or self.plotter.render_window is None:
                return

            self._save_window_state()
            logger.info(f"\n[Config] Window state and camera position saved")
        except Exception as e:
            logger.error(f"[!] Error saving config: {e}")

    def reload_dae_file(self):
        """Reload all DAE files (L key)."""
        self._show_reload_overlay()
        try:
            logger.info(f"\n[Reload] Loading all DAE files from items.level.json...")

            # Save camera AND debug layer status
            camera_pos = None
            camera_focal = None
            camera_up = None
            debug_was_visible = self.show_debug

            try:
                camera_pos = self.plotter.camera.position
                camera_focal = self.plotter.camera.focal_point
                camera_up = self.plotter.camera.up
            except Exception as e:
                logger.error(f"[!] Error saving camera position: {e}")

            # Reload items with the CENTRAL function from dae_loader
            items_path = config.BEAMNG_DIR / config.ITEMS_JSON

            try:
                loader_result = load_all_viewer_data(config.BEAMNG_DIR, items_path, _resolve_beamng_path)
                self.dae_files = loader_result["dae_files"]
                self.tile_data = loader_result["tile_data"]
                self.forest_data = loader_result["forest_data"]
            except Exception as e:
                logger.error(f"  [!] Error loading viewer data: {e}")
                import traceback

                traceback.print_exc()
                self.dae_files = []
                self.tile_data = []
                self.forest_data = None

            # Reload textures
            self.textures = self._load_textures()

            logger.info(f"  ✓ {len(self.tile_data)} DAE files reloaded")
            if self.forest_data:
                logger.info(f"  ✓ forest.forest4.json reloaded ({len(self.forest_data.get('instances', []))} instances)")

            # Reset the debug layer status (reloaded AFTER update_view)
            self.debug_loaded = False
            self.debug_actors = []
            # Reset the forest layer status (reloaded AFTER update_view)
            self.forest_loaded = False
            self.forest_actors = []
            self.update_view()

            # AFTER update_view: reload the debug layer (so it is not deleted by plotter.clear())
            if debug_was_visible:
                self._load_debug_layer()
                self.debug_loaded = True
                self.show_debug = True
                # Set visibility
                for actor in self.debug_actors:
                    actor.SetVisibility(True)
                self.plotter.render()

            # Restore the camera
            if camera_pos is not None:
                try:
                    self.plotter.camera.position = camera_pos
                    self.plotter.camera.focal_point = camera_focal
                    self.plotter.camera.up = camera_up
                    logger.info("  ✓ Camera position kept")
                except Exception as e:
                    logger.error(f"[!] Error restoring camera position: {e}")

            return True
        except Exception as e:
            logger.error(f"  ✗ Error during reload: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self._hide_reload_overlay()

    def _show_reload_overlay(self):
        """Show the reload overlay."""
        if self._reload_actor is not None:
            self._hide_reload_overlay()
        try:
            self._reload_actor = self.plotter.add_text(
                "Reload...",
                position=(0.45, 0.5),
                viewport=True,
                font_size=18,
                color="white",
                shadow=True,
                name="reload_overlay",
            )
            self.plotter.render()
        except Exception as e:
            logger.error(f"[!] Error showing reload overlay: {e}")
            self._reload_actor = None

    def _hide_reload_overlay(self):
        """Hide the reload overlay."""
        if self._reload_actor is not None:
            try:
                self.plotter.remove_actor(self._reload_actor)
                self.plotter.render()
            except Exception as e:
                logger.error(f"[!] Error hiding reload overlay: {e}")
            self._reload_actor = None

    def _load_debug_layer(self):
        """Load the debug layer from primitives (new format of DebugNetworkExporter)."""
        logger.debug("  [Debug] Loading debug layer...")

        # Load primitive data from cache/debug_network.json (local project directory)
        debug_network_path = Path(__file__).parent.parent / "cache" / "debug_network.json"

        if not debug_network_path.exists():
            logger.debug(f"  [Debug] No debug data found: {debug_network_path}")
            return

        try:
            with open(debug_network_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)
        except Exception as e:
            logger.error(f"  [!] Error loading debug data: {e}")
            return

        primitives = debug_data.get("primitives", [])

        if not primitives:
            logger.debug(f"  [Debug] No primitives found in debug data")
            return

        logger.debug(f"  [Debug] Loading {len(primitives)} primitives")

        # Collect primitives by type
        lines = []
        points = []
        polygons = []
        circles = []
        labels = []

        for prim in primitives:
            prim_type = prim.get("type", "line")
            coords = prim.get("coords", [])
            color = prim.get("color", [0.0, 0.0, 1.0])

            if prim_type == "line" and len(coords) >= 2:
                lines.append((coords, color))
            elif prim_type == "point" and len(coords) >= 1:
                points.append((coords[0], color))
            elif prim_type == "polygon" and len(coords) >= 3:
                polygons.append((coords, color))
            elif prim_type == "circle" and len(coords) >= 1:
                circles.append((coords[0], prim.get("radius", 1.0), color))
            elif prim_type == "label":
                # Text label with position
                text = prim.get("text", "Label")
                position = prim.get("position", [0, 0, 0])
                size = prim.get("size", 12.0)
                labels.append((text, position, color, size))

        actor_count = 0

        # Render lines (e.g. centerlines)
        if lines:
            all_points = []
            all_lines = []
            point_offset = 0

            for line_coords, color in lines:
                coords_array = np.array(line_coords)
                n = len(coords_array)
                all_points.extend(coords_array)

                for i in range(n - 1):
                    all_lines.append([2, point_offset + i, point_offset + i + 1])
                point_offset += n

            if all_points:
                all_points_array = np.array(all_points)
                all_lines_array = np.array(all_lines)
                centerlines_mesh = pv.PolyData(all_points_array, lines=all_lines_array)
                actor = self.plotter.add_mesh(
                    centerlines_mesh,
                    color=lines[0][1],  # Use the color of the first line
                    line_width=2.0,
                    opacity=0.8,
                    label="Centerlines",
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Render points (e.g. junctions)
        if points:
            point_coords = np.array([p[0] for p in points])
            point_colors = [p[1] for p in points]

            # Create spheres for junctions
            junction_blocks = pv.MultiBlock()
            for coord, color in points:
                sphere = pv.Sphere(radius=2.0, center=coord)
                junction_blocks.append(sphere)

            if len(junction_blocks) > 0:
                actor = self.plotter.add_mesh(
                    junction_blocks,
                    color=points[0][1],  # Use the color of the first point
                    opacity=0.5,
                    label="Junctions",
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Render circles (combined into one actor)
        if circles:
            circles_blocks = pv.MultiBlock()
            for center, radius, color in circles:
                circle = pv.Sphere(radius=radius, center=center)
                circles_blocks.append(circle)

            if len(circles_blocks) > 0:
                actor = self.plotter.add_mesh(
                    circles_blocks,
                    color=circles[0][2],  # Use the color of the first circle
                    opacity=0.3,
                    label="Circles",
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Render polygons (combined into one actor - as line outlines)
        if polygons:
            all_poly_points = []
            all_poly_lines = []
            point_offset = 0

            for poly_coords, color in polygons:
                coords_array = np.array(poly_coords)
                if len(coords_array) >= 3:
                    n = len(coords_array)
                    all_poly_points.extend(coords_array)

                    # Create closed polygon as lines (not as faces)
                    for i in range(n):
                        next_i = (i + 1) % n  # Close polygon
                        all_poly_lines.append([2, point_offset + i, point_offset + next_i])
                    point_offset += n

            if all_poly_points:
                all_poly_points_array = np.array(all_poly_points)
                all_poly_lines_array = np.array(all_poly_lines)
                polygons_mesh = pv.PolyData(all_poly_points_array, lines=all_poly_lines_array)
                actor = self.plotter.add_mesh(
                    polygons_mesh,
                    color=polygons[0][1],  # Use the color of the first polygon
                    line_width=2.0,
                    opacity=1.0,
                    label="Polygons",
                    render_lines_as_tubes=False,
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Render labels (text at positions) - in batches for performance
        if labels:
            try:
                positions = []
                texts = []
                for text, position, color, size in labels:
                    pos = np.array(position) if not isinstance(position, np.ndarray) else position
                    positions.append(pos)
                    texts.append(str(text))

                if positions:
                    positions_array = np.array(positions)
                    # Batch rendering with add_point_labels (much faster!)
                    label_actors = self.plotter.add_point_labels(
                        positions_array,
                        texts,
                        font_size=10,
                        text_color="white",
                        render=False,  # render=False to save performance
                    )
                    # Add label actors to debug_actors (for the D toggle)
                    if label_actors is not None:
                        if isinstance(label_actors, list):
                            self.debug_actors.extend(label_actors)
                        else:
                            self.debug_actors.append(label_actors)
                        actor_count += 1
            except Exception as e:
                logger.error(f"  [!] Error rendering labels: {e}")

        logger.info(
            f"  [Debug] {actor_count} debug actors rendered ({len(points)} Junctions, {len(lines)} Centerlines, {len(labels)} Labels)"
        )

    def _load_forest_layer(self):
        """Load the forest layer from previously loaded forest_data."""
        logger.info("  [Forest] Loading forest layer...")

        # Use already loaded forest_data (loaded in __init__ or reload)
        if not self.forest_data:
            logger.error(f"  [!] No forest data available (forest_data is None)")
            return

        # Use the central forest_loader function to create actors
        from tools.forest_loader import load_forest_layer

        # Create a temporary path variable for forest_loader (not used there,
        # but the signature requires it). Alternatively forest_loader could be refactored.
        try:
            actor = load_forest_layer(self, config.BEAMNG_DIR / "forest" / "forest.forest4.json")

            if actor is not None:
                logger.info(f"  [✓] Forest layer actors created")
                return

        except Exception as e:
            logger.error(f"  [!] Error creating forest layer: {e}")
            import traceback

            traceback.print_exc()

    def _on_left_mouse_click(self, obj, event):
        """Handler for left double click: set the camera pivot to the clicked point."""
        try:
            now_ts = time.perf_counter()
            if self._last_click_ts and (now_ts - self._last_click_ts) <= 0.2:
                # Double click detected
                self._last_click_ts = 0.0
            else:
                # First click: remember the time and abort
                self._last_click_ts = now_ts
                return
            # Get the mouse position in the window
            try:
                click_pos = obj.GetEventPosition()
            except AttributeError:
                click_pos = obj.get_event_position()

            # Perform ray casting
            hit_point = self._raycast_to_mesh(click_pos)

            if hit_point is not None:
                self._set_camera_to_point(hit_point)
            else:
                logger.info("[Raycast] No mesh hit at this position")

        except Exception as e:
            logger.error(f"[!] Error during mouse click raycasting: {e}")
            import traceback

            traceback.print_exc()

    def _raycast_to_mesh(self, screen_pos):
        """Perform ray casting from the mouse position to the mesh.

        Args:
            screen_pos: (x, y) tuple of the mouse position in the window

        Returns:
            hit_point: (x, y, z) NumPy array of the intersection point, or None
        """
        try:
            # Get renderer and camera
            renderer = self.plotter.renderer
            camera = self.plotter.camera

            # Convert screen coordinates to display coordinates (normalized 0..1)
            win_size = self.plotter.window_size
            x_norm = screen_pos[0] / win_size[0]
            y_norm = screen_pos[1] / win_size[1]

            # PyVista's pick_mouse_position uses a cell picker (more performant than OBBTree)
            # BUT: we need the exact point, not just the cell!

            # Alternatively: use VTK's picker directly for a precise point
            try:
                picker = self.plotter.iren.GetPicker()
            except AttributeError:
                try:
                    picker = self.plotter.iren.get_picker()
                except Exception:
                    picker = None

            if picker is None:
                # Create cell picker if not present
                import vtk

                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.005)  # 0.5% tolerance
                try:
                    # Attach the picker to the interactor so future calls use it
                    self.plotter.iren.SetPicker(picker)
                except Exception:
                    pass

            # Perform pick (x, y in display coordinates, z=0)
            result = picker.Pick(screen_pos[0], screen_pos[1], 0, renderer)

            if result:
                # Successful hit - get intersection point
                hit_point = np.array(picker.GetPickPosition())
                logger.info(f"[Raycast] Hit at: ({hit_point[0]:.1f}, {hit_point[1]:.1f}, {hit_point[2]:.1f})")
                return hit_point
            else:
                return None

        except Exception as e:
            logger.error(f"[!] Error during raycasting: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _set_camera_to_point(self, target_point):
        """Set the camera pivot to a point and move the camera 40 m in front of it.

        Args:
            target_point: (x, y, z) NumPy array of the target point
        """
        try:
            camera = self.plotter.camera

            # Get the current viewing direction (normalized)
            current_pos = np.array(camera.position)
            current_focal = np.array(camera.focal_point)
            view_direction = current_focal - current_pos
            view_dist = np.linalg.norm(view_direction)

            if view_dist > 1e-6:
                view_direction = view_direction / view_dist
            else:
                # Fallback: look from south to north
                view_direction = np.array([0.0, 1.0, 0.0])

            # The new focal point is the clicked point
            new_focal = np.array(target_point)

            # New camera position: 40 m in the opposite viewing direction
            camera_distance = 40.0
            new_position = new_focal - view_direction * camera_distance

            # Set camera
            camera.focal_point = new_focal
            camera.position = new_position
            camera.up = [0.0, 0.0, 1.0]  # Z axis is up

            # Update clipping range and render
            self.plotter.reset_camera_clipping_range()
            self.plotter.render()

            # Update status display
            self._update_camera_status()

            logger.info(f"[Camera] Pivot: ({new_focal[0]:.1f}, {new_focal[1]:.1f}, {new_focal[2]:.1f})")
            logger.info(f"[Camera] Position: ({new_position[0]:.1f}, {new_position[1]:.1f}, {new_position[2]:.1f})")
            logger.info(f"[Camera] Distance: {camera_distance:.1f}m")

        except Exception as e:
            logger.error(f"[!] Error setting camera: {e}")
            import traceback

            traceback.print_exc()

    def show(self):
        """Show the viewer window."""
        self.plotter.show()


if __name__ == "__main__":
    viewer = DAETileViewer()
    if hasattr(viewer, "plotter") and viewer.plotter is not None:
        viewer.show()
    else:
        logger.error("[!] No plotter initialized (probably no DAE files loaded).")
