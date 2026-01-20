"""
DAE Viewer - Visualisiere die exportierten Tile-DAEs

NEUE ARCHITEKTUR: Lädt separate DAE-Dateien pro Tile (tile_X_Y.dae)
Jede DAE enthält nur EINE Geometrie → verhindert überlappende Texturen!

Steuerung:
    X = Toggle Texturen (An/Aus) - triggert Neuaufbau (Rendering ↔ Grid)
        Rendering-Modus: Zeigt Texturen mit Material-Zuordnungen und UV-Koordinaten
        Grid-Modus: Zeigt Drahtgitter mit Farben für schnellere Vorschau

    In Grid-Ansicht (X=aus) individuelle Layer ein-/ausblenden (OHNE Neuaufbau):
        T = Toggle Terrainebene
        S = Toggle Straßenebene
        H = Toggle Häuserebene
        D = Toggle Debugebene (Junctions, Centerlines, Boundaries)

    Allgemein:
        K = Kamera laden | Shift+K = Kamera speichern
        L = DAE neu laden
        Up/Down = Zoom ändern

    Maus:
        Doppel-Links-Klick = Kamera auf angeklickten Punkt setzen (40m Entfernung)
        Rechtsklick-Drag = Kamera drehen
        Scroll = Zoom

Texture Debug Features:
    • Beim Start werden verfügbare Texturen und Materialien aufgelistet
    • Im Rendering-Modus werden Texture-Zuordnungen für jeden Layer gezeigt (✓/○/✗)
    • Console zeigt an: Welche Materialien zu Texturen gemappt wurden, welche Fallbacks
    • UVs werden automatisch aus DAE geladen und für Textur-Mapping verwendet

Texture System:
    • Tile-Texturen: art/shapes/textures/tile_*.dds (Terrain pro 500×500m Tile)
    • Material-Texturen: main.materials.json → art/shapes/materials/... (für Roads/Buildings)
    • UV-Koordinaten pro Layer:
        - Terrain: 0..1 pro Tile (500×500m), normalisiert in DAE
        - Roads: 0..unbounded für Länge, 0..1 für Breite (Tiling)
        - Buildings: Skaliert für 4m (Walls) / 2m (Roofs) Textur-Wiederholung
"""

import pyvista as pv
import numpy as np
import sys
import json
import atexit
import time
from pathlib import Path, PurePosixPath
from PIL import Image

# Importiere config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from world_to_beamng import config
from tools.dae_loader import load_dae_tile, load_all_dae_files as load_dae_tile_all_from_items


# BeamNG-Relative Pfade ("/levels/<Level>/...") nach absoluten Pfaden auflösen
def _resolve_beamng_path(path_str: str) -> str | None:
    if not path_str:
        return None

    p_posix = PurePosixPath(path_str)  # Treat input as a Posix path

    # 1) Präfix /levels/<LEVEL_NAME>/...
    level_prefix_posix = PurePosixPath("/levels") / config.LEVEL_NAME
    if p_posix.is_relative_to(level_prefix_posix):
        return str(config.BEAMNG_DIR / p_posix.relative_to(level_prefix_posix))

    # 2) Präfix aus config.RELATIVE_DIR (identisch, aber bereitgestellt)
    # config.RELATIVE_DIR is already a PurePosixPath 'levels/world_to_beamng'
    if p_posix.is_relative_to(config.RELATIVE_DIR):
        return str(config.BEAMNG_DIR / p_posix.relative_to(config.RELATIVE_DIR))

    # 3) art/… Präfix relativ zum Level-Root (Path(path_str) will handle "art/...")
    if p_posix.parts[0] == "art":  # check if first part is 'art'
        return str(config.BEAMNG_DIR / p_posix)

    # 4) Fallback: behandle als relative Shape-Angabe
    return str(config.BEAMNG_DIR_SHAPES / p_posix)


class DAETileViewer:
    def __init__(self):
        # Lade Items und Materialien aus JSON
        items_path = config.BEAMNG_DIR / config.ITEMS_JSON

        # Suche materials.json in config.BEAMNG_DIR/main/
        materials_path = config.BEAMNG_DIR / "main" / "materials.json"

        print(f"Lade Items aus: {items_path}")

        print(f"Lade Materialien aus: {materials_path}")
        if materials_path.exists():
            with open(materials_path, "r", encoding="utf-8") as f:
                self.materials = json.load(f)
                print(f"  [✓] {len(self.materials)} Materialien geladen")
        else:
            # Generiere materials aus osm_to_beamng.json Config
            print(f"  [!] {materials_path} nicht gefunden, generiere aus Config...")
            try:
                from world_to_beamng.io.lod2 import create_materials_json

                self.materials = create_materials_json()
                print(f"  [✓] Materials generiert aus Config")
            except Exception as e:
                print(f"  [!] Fehler beim Generieren von Materials: {e}")
                self.materials = {}

        # Lade alle DAE-Dateien aus gemeinsamer Funktion
        self.dae_files, self.tile_data = load_dae_tile_all_from_items(
            config.BEAMNG_DIR, items_path, _resolve_beamng_path
        )

        if not self.dae_files:
            print("Keine DAE-Dateien in items.level.json gefunden!")
            return

        if not self.tile_data:
            print("Keine Geometrie in DAE-Dateien gefunden!")
            return

        # Initialisiere config_path FRÜH (wird für _load_layers_state benötigt)
        self.config_path = Path(__file__).parent / "dae_viewer.cfg"

        # Sichtbarkeits-Flags (lade gespeicherte Werte)
        saved_layers = self._load_layers_state()
        self.show_terrain = saved_layers.get("terrain", True)
        self.show_roads = saved_layers.get("roads", True)
        self.show_buildings = saved_layers.get("buildings", True)  # Häuser Toggle
        self.show_horizon = saved_layers.get("horizon", True)  # Horizont Toggle
        self.show_forest = saved_layers.get("forest", True)  # Bäume Toggle
        self.use_textures = saved_layers.get("textures", True)  # Texturen standardmäßig an
        self.show_debug = saved_layers.get("debug", False)  # Debug-Layer (Junctions, Centerlines)

        # Speichere Actor-Referenzen für Sichtbarkeits-Toggles
        self.terrain_actors = []  # Liste von Terrain-Mesh-Actors
        self.road_actors = []  # Liste von Road-Mesh-Actors
        self.building_actors = []  # Liste von Building-Mesh-Actors
        self.horizon_actors = []  # Liste von Horizon-Mesh-Actors
        self.forest_actors = []  # Liste von Forest-Punkt-Actors (Bäume)
        self.debug_actors = []  # Liste von Debug-Actors (Junctions, Centerlines)
        self.debug_loaded = False  # Flag: Debug-Layer bereits geladen?
        self.forest_loaded = False  # Flag: Forest-Layer bereits geladen?
        self._first_update_view = True  # Flag: Erstes Mal update_view() aufgerufen?

        # Lade Grid-Farben aus debug_network.json (für Grid-Ansicht)
        self.grid_colors = self._load_grid_colors()

        # Lade Texturen
        self.textures_dir = config.BEAMNG_DIR_SHAPES / "textures"
        self.textures = self._load_textures()

        # Lade Material-Texturen aus main.materials.json
        self.material_textures = self._load_material_textures()

        if self.textures:
            print(f"  -> {len(self.textures)} Tile-Texturen geladen")
        if self.material_textures:
            print(f"  -> {len(self.material_textures)} Material-Texturen geladen")

        # Debug: Zeige verfügbare Texturen und Material-Zuordnungen
        self._print_texture_debug_info()

        # Status-Actors
        self._reload_actor = None
        self._camera_status_actor = None
        self._active_layers_actor = None
        self._render_update_counter = 0  # Für RenderEvent Drosselung
        self._last_click_ts = 0.0  # Für manuelle Doppelklick-Erkennung

        # Global Material Properties (zentrale Definition)
        self.material_ambient = 0.6
        self.material_diffuse = 0.8
        self.material_specular = 0.0

        # PyVista Setup
        self.plotter = pv.Plotter()
        self.plotter.set_background("skyblue")  # Himmelblau
        # self.plotter.enable_shadows()  # TEMP DISABLED - könnte das Rendering blocken!

        self._reinit_lights()

        try:
            # RTX 4090 Tuning: Maximale Shadow-Map Auflösung
            try:
                for renderer in self.plotter.renderers:
                    pv.set_new_attribute(renderer, "shadow_map_size", 8192)  # Maximale Schärfe für High-End GPU!
            except Exception as shadow_e:
                print(f"  [i] Shadow-Map Tuning fehlgeschlagen: {shadow_e}")
        except Exception as e:
            print(f"  [i] Lichter-Setup: {e}")

        # Stelle Fensterposition/-größe wieder her
        self._apply_saved_window_state()

        # Observer
        self.plotter.iren.add_observer("KeyPressEvent", self._on_key_press)
        self.plotter.iren.add_observer("ExitEvent", self._on_close_save_window_state)
        self.plotter.iren.add_observer("ScrollEvent", self._on_camera_change)
        self.plotter.iren.add_observer("EndInteractionEvent", self._on_camera_change)
        self.plotter.iren.add_observer("InteractionEvent", self._on_camera_change)
        self.plotter.iren.add_observer("RenderEvent", self._on_render_event)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_mouse_click)

        # Registriere atexit-Handler als Fallback (für sicheres Speichern beim Exit)
        atexit.register(self._on_close_save_window_state)

        print(f"\nDAE geladen mit Tile-Geometrien")
        print("\nSteuerung:")
        print("  X = Toggle Texturen (Rendering ↔ Grid) - triggert Neuaufbau")
        print("\nLayer ein-/ausblenden OHNE Neuaufbau (in BEIDEN Ansichten verfügbar):")
        print("  T = Toggle Terrain")
        print("  S = Toggle Straßen")
        print("  H = Toggle Häuser")
        print("  C = Toggle Wälder (Bäume)")
        print("  D = Toggle Debug (Junctions, Centerlines, Boundaries)")
        print("\nAllgemein:")
        print("  K = Kamera laden | Shift+K = Kamera speichern")
        print("  L = DAE neu laden")
        print("  Up/Down = Zoom ändern")
        print("  Doppel-Links-Klick = Kamera auf Punkt setzen (40m Entfernung)")

        self.update_view()
        # Hinweis: _apply_saved_camera_state() wird NICHT beim Start aufgerufen
        # um sicherzustellen dass die Kamera auf die Geometrie passt!
        # Sie wird nur aufgerufen wenn der Viewer mit show() gestartet wird

        # Lade Debug-Layer am Start wenn aktiviert
        if self.show_debug:
            self._update_debug_visibility()

        # Lade Forest-Layer am Start wenn aktiviert
        if self.show_forest:
            self._update_forest_visibility()

    def _on_key_press(self, obj, event):
        """KeyPress Event Handler."""
        key = obj.GetKeySym()
        key_lower = key.lower()

        if key_lower == "s":
            # Toggle nur Roads (kein Neuaufbau)
            self.show_roads = not self.show_roads
            print(f"\n[Straßen] {'AN' if self.show_roads else 'AUS'}")
            self._update_visibility()

        elif key_lower == "t":
            # Toggle nur Terrain (kein Neuaufbau)
            self.show_terrain = not self.show_terrain
            print(f"\n[Terrain] {'AN' if self.show_terrain else 'AUS'}")
            self._update_visibility()

        elif key_lower == "h":
            # Toggle nur Häuser (kein Neuaufbau)
            self.show_buildings = not self.show_buildings
            print(f"\n[Häuser] {'AN' if self.show_buildings else 'AUS'}")
            self._update_visibility()

        elif key_lower == "c":
            # Toggle nur Wälder/Bäume (mit Lazy-Loading beim ersten Toggle)
            self.show_forest = not self.show_forest
            print(f"\n[Wälder] {'AN' if self.show_forest else 'AUS'}")
            self._update_forest_visibility()

        if key == "o":
            self.show_horizon = not self.show_horizon
            print(f"\n[Horizont] {'AN' if self.show_horizon else 'AUS'}")
            self._update_visibility()

        elif key_lower == "x":
            # Toggle Texturen (mit Neuaufbau!)
            self.use_textures = not self.use_textures
            print(f"\n[{'Rendering' if self.use_textures else 'Grid'}-Ansicht]")
            self.update_view()
            # Debug-Layer bleiben dauerhaft geladen und ihre Sichtbarkeit wird beibehalten

        elif key_lower == "d":
            # Toggle Debug in BEIDEN Ansichten (Rendering und Grid)
            self.show_debug = not self.show_debug
            print(f"\n[Debug] {'AN' if self.show_debug else 'AUS'}")
            self._update_debug_visibility()

        elif key == "K":  # Shift+K (MUSS VOR "k" kommen!)
            self.save_camera_state()
        elif key_lower == "k":
            self.load_camera_state()

        elif key == "Up":
            self._adjust_zoom(-5.0)  # Zoom rein
        elif key == "Down":
            self._adjust_zoom(5.0)  # Zoom raus

        elif key_lower == "l":
            self.reload_dae_file()

    def _update_visibility(self):
        """Aktualisiere Sichtbarkeit der Terrain/Road/Building/Forest Actors ohne Reload."""
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
        """Aktualisiere Sichtbarkeit der Debug-Actors ohne Reload."""
        if not self.debug_loaded:
            # Debug-Layer erstmalig laden
            self._load_debug_layer()
            self.debug_loaded = True

        # Toggle Visibility
        for actor in self.debug_actors:
            actor.SetVisibility(self.show_debug)

        self._update_active_layers_text()
        self.plotter.render()

    def _update_forest_visibility(self):
        """Aktualisiere Sichtbarkeit der Forest-Actors ohne Reload."""
        if not self.forest_loaded:
            # Forest-Layer erstmalig laden
            self._load_forest_layer()
            self.forest_loaded = True

        # Toggle Visibility
        for actor in self.forest_actors:
            actor.SetVisibility(self.show_forest)

        self._update_active_layers_text()
        self.plotter.render()

    def _reinit_lights(self):
        """Lichter neu initialisieren nach clear()."""
        try:
            self.plotter.remove_all_lights()
        except:
            pass

        try:
            # Hauptlicht: Sonne mit parallelen Strahlen (Richtungslicht)
            sun_light = pv.Light(
                position=[300, -300, 600],
                focal_point=[0, 0, 0],
                positional=False,  # Parallele Strahlen statt Punktlicht
                cone_angle=80,
                intensity=1.0,
                shadow_attenuation=0.95,
            )
            self.plotter.add_light(sun_light)

            # Diffuses Fill-Light: Gleichmäßige Ausleuchtung
            fill_light = pv.Light(
                light_type="headlight",  # Diffuses Licht
                intensity=0.0,
            )
            self.plotter.add_light(fill_light)
        except Exception as e:
            print(f"[!] Fehler beim Lichter-Setup: {e}")

    def _print_texture_debug_info(self):
        """Drucke Debug-Informationen über verfügbare Texturen und Material-Zuordnungen."""
        print("\n[TEXTURE DEBUG INFO]")
        print("=" * 80)

        # Tile-Texturen
        if self.textures:
            print(f"\n[Tile-Texturen] {len(self.textures)} verfügbar:")
            for key in sorted(self.textures.keys())[:10]:  # Zeige erste 10
                print(f"  • {key}")
            if len(self.textures) > 10:
                print(f"  ... und {len(self.textures) - 10} weitere")
        else:
            print("\n[Tile-Texturen] KEINE gefunden (textures/ Verzeichnis leer?)")

        # Material-Texturen
        if self.material_textures:
            print(f"\n[Material-Texturen] {len(self.material_textures)} gefunden:")
            for mat_name in sorted(self.material_textures.keys()):
                print(f"  • {mat_name}")
        else:
            print("\n[Material-Texturen] KEINE gefunden (main.materials.json hat keine Texturen?)")

        # Material-Struktur
        if self.materials:
            print(f"\n[Materials JSON] {len(self.materials)} Materialien definiert:")
            roads = [m for m in self.materials.keys() if "road" in m.lower()]
            buildings = [
                m for m in self.materials.keys() if "build" in m.lower() or "wall" in m.lower() or "roof" in m.lower()
            ]
            other = [m for m in self.materials.keys() if m not in roads and m not in buildings]

            if roads:
                print(f"  Roads ({len(roads)}): {', '.join(roads[:3])}")
            if buildings:
                print(f"  Buildings ({len(buildings)}): {', '.join(buildings[:3])}")
            if other:
                print(f"  Sonstige ({len(other)}): {', '.join(other[:3])}")

        print("=" * 80 + "\n")

    def update_view(self):
        """Aktualisiere 3D-View."""
        # Speichere Kamera vor clear() - ABER: Beim ERSTEN Aufruf ignoriere die Default-Kamera (1,1,1)!
        camera_pos = None
        camera_focal = None
        camera_up = None

        # Nur speichern wenn NICHT der erste Aufruf (update_view von __init__)
        if not self._first_update_view:
            try:
                cam = self.plotter.camera
                if cam is not None:
                    camera_pos = list(cam.position)
                    camera_focal = list(cam.focal_point)
                    camera_up = list(cam.up)
            except Exception as e:
                print(f"[!] Fehler beim Speichern der Kamera vor update_view: {e}")
        else:
            # Erster Aufruf - merke dass wir jetzt in den "Reload"-Modus sind
            self._first_update_view = False

        # Speichere Debug-Actors VOR clear() - um sie zu bewahren
        saved_debug_actors = self.debug_actors.copy() if self.debug_actors else []
        saved_debug_visibility = self.show_debug

        # Speichere Forest-Actors VOR clear() - um sie zu bewahren
        saved_forest_actors = self.forest_actors.copy() if self.forest_actors else []
        saved_forest_visibility = self.show_forest

        self.plotter.clear()
        self._reinit_lights()

        # Leere NUR Terrain/Road/Building/Horizon/Forest Actor-Listen
        self.terrain_actors = []
        self.road_actors = []
        self.building_actors = []
        self.horizon_actors = []
        self.forest_actors = []
        # Debug-Actors und Forest-Actors wurden durch clear() gelöscht, aber wir laden sie danach wieder

        # Iteriere über alle geladenen DAE-Dateien
        for item_name, tile_data in self.tile_data:
            print(f"Rendere {item_name}...")
            self._render_single_dae(item_name, tile_data)

        # Statuszeilen
        # Oben links: Bedienungsanleitung
        bedienung = "S: Straßen | T: Terrain | H: Häuser | C: Wälder | O: Horizont | D: Debug | X: Texturen | K: Cam | L: Reload | 2xLMB: Jump"
        self.plotter.add_text(
            bedienung,
            position="upper_left",
            font_size=10,
        )

        # Oben rechts: Aktive Layer
        self._update_active_layers_text()

        # Fit Kamera zu allen Actors (aber nur beim Start, nicht bei reload)
        if camera_pos is None:
            # Erste Initialisierung - nutze view_isometric für automatisches Framing
            try:
                self.plotter.view_isometric()
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()

                print(f"  [i] Kamera mit view_isometric() positioniert")
                print(f"      Position: {self.plotter.camera.position}")

                # Versuche gespeicherte Kamera zu laden (überschreibt view_isometric)
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
                            print(f"  [✓] Gespeicherte Kamera geladen")
                            print(f"      Position: {pos}")
                    except Exception as e:
                        print(f"  [!] Fehler beim Laden der gespeicherten Kamera: {e}")

            except Exception as e:
                print(f"  [!] Fehler beim Initialisieren der Kamera: {e}")
        else:
            # Reload - stelle alte Kamera wieder her
            try:
                cam = self.plotter.camera
                cam.position = camera_pos
                cam.focal_point = camera_focal
                cam.up = camera_up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
            except Exception as e:
                print(f"  [!] Fehler beim Wiederherstellen der Kamera: {e}")

        self._update_camera_status()

        # Füge Debug-Actors wieder zum Plotter hinzu (falls sie existierten)
        if saved_debug_actors:
            self.debug_actors = []
            for actor in saved_debug_actors:
                try:
                    self.plotter.add_actor(actor)
                    self.debug_actors.append(actor)
                    # Stelle Sichtbarkeit wieder her
                    actor.SetVisibility(saved_debug_visibility)
                except Exception as e:
                    print(f"[!] Fehler beim Wiederherstellen des Debug-Actors: {e}")

        # Füge Forest-Actors wieder zum Plotter hinzu (falls sie existierten)
        if saved_forest_actors:
            self.forest_actors = []
            for actor in saved_forest_actors:
                try:
                    self.plotter.add_actor(actor)
                    self.forest_actors.append(actor)
                    # Stelle Sichtbarkeit wieder her
                    actor.SetVisibility(saved_forest_visibility)
                except Exception as e:
                    print(f"[!] Fehler beim Wiederherstellen des Forest-Actors: {e}")

    def _index_to_coords(self, item_name, tile_index_x, tile_index_y):
        """
        Konvertiere Tile-Indizes (z.B. tile_-2_-2) zu absoluten Koordinaten.

        Die Indizes sind Grid-Positionen mit 500m Abstände.
        Index -2, -1, 0, 1 correspond zu Koordinaten -1000, -500, 0, 500.

        Returns: (x_coord, y_coord)
        """
        x_coord = tile_index_x * 500
        y_coord = tile_index_y * 500
        return (x_coord, y_coord)

    def _get_actor_list_for_item(self, item_name):
        """Bestimme, zu welcher Actor-Liste ein Item gehört basierend auf item_name."""
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
        """Rendere ein einzelnes DAE-File (terrain oder building)."""
        vertices = tile_data.get("vertices", [])
        faces = tile_data.get("faces", [])
        materials = tile_data.get("materials", [])
        tiles_info = tile_data.get("tiles", {})

        if len(vertices) == 0:
            print(f"  [!] {item_name}: Keine Vertices")
            return

        # Bestimme ob Terrain oder Building
        is_terrain = item_name.startswith("terrain_") or item_name.startswith("tile_")
        is_horizon = "horizon" in item_name.lower()
        is_building = item_name.startswith("buildings_")

        # Farben aus grid_colors
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

        # Kategorisiere Faces nach Material (für beide Rendering-Modi)
        terrain_faces = []
        road_faces_by_material = {}  # {material_name: [faces]}
        wall_faces = []
        roof_faces = []

        for face_idx, material in enumerate(materials):
            mat_lower = material.lower()
            # Kategorisierung basiert auf Material-Namen und Item-Kontext
            # Priorität: wall > roof > road > terrain > building_default > fallback
            if "wall" in mat_lower:
                wall_faces.append(faces[face_idx])
            elif "roof" in mat_lower:
                roof_faces.append(faces[face_idx])
            elif "terrain" in mat_lower or "tile" in mat_lower or material == "terrain":
                # Explizit "terrain" oder mit "terrain"/"tile" im Namen
                terrain_faces.append(faces[face_idx])
            elif "road" in mat_lower or (
                not is_building and not "terrain" in mat_lower and not "tile" in mat_lower and material != "unknown"
            ):
                # Road: Hat "road" im Namen ODER (ist nicht Building und kein unbekanntes Material)
                # Dies fängt auch stitch_gaps-eingefügte Faces, die als "terrain" Material exportiert aber in DAE als andere Namen auftauchen können
                if material not in road_faces_by_material:
                    road_faces_by_material[material] = []
                road_faces_by_material[material].append(faces[face_idx])
            elif is_building:
                # In Buildings: Alles andere ist Wall
                wall_faces.append(faces[face_idx])
            else:
                # In Terrain: Alles andere ist Terrain
                terrain_faces.append(faces[face_idx])

        # Rendering mit Texturen (nur für Terrain)
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

                # tile_name ist bereits im Koordinaten-Format (z.B. "tile_-1000_-1000")
                # KEINE Konvertierung mehr nötig, da DAE-Export jetzt Weltkoordinaten verwendet!
                lookup_key = tile_name.lower()
                texture = self.textures.get(lookup_key)

                if texture is None and is_horizon:
                    # Fallback: nutze die bekannte Horizont-Textur, falls der Tile-Name nicht passt
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
                        print(f"  [!] Textur-Fehler für {tile_name}: {e}")
                        terrain_texture_log.append(f"✗ {tile_name} → FEHLER: {str(e)[:40]}")
                        # Fallback zu Farbe
                        actor = self.plotter.add_mesh(
                            mesh, color=[0.6, 0.5, 0.4], opacity=0.5, label=f"{item_name}_{tile_name}"
                        )
                        self._get_actor_list_for_item(item_name).append(actor)
                        visibility = self.show_horizon if "horizon" in item_name.lower() else self.show_terrain
                        actor.SetVisibility(visibility)
                else:
                    # Keine Textur oder keine UVs
                    reason = "KEINE UVs" if len(tile_uvs) == 0 else f"Textur nicht gefunden: {lookup_key}"
                    terrain_texture_log.append(f"○ {tile_name} → {reason}")
                    actor = self.plotter.add_mesh(
                        mesh, color=[0.6, 0.5, 0.4], opacity=0.5, label=f"{item_name}_{tile_name}"
                    )
                    self._get_actor_list_for_item(item_name).append(actor)
                    visibility = self.show_horizon if "horizon" in item_name.lower() else self.show_terrain
                    actor.SetVisibility(visibility)

            # Debug-Output
            if terrain_texture_log:
                print(f"\n[{item_name}] Terrain-Textur-Zuordnung:")
                for entry in terrain_texture_log[:5]:  # Zeige erste 5
                    print(f"  {entry}")
                if len(terrain_texture_log) > 5:
                    print(f"  ... und {len(terrain_texture_log) - 5} weitere")
        else:
            # Grid-Ansicht: Rendere Terrain mit Farben
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

        # Rendere Roads pro Material (immer, egal ob Texture oder Grid)
        if road_faces_by_material and tiles_info:
            road_opacity = self.grid_colors.get("road", {}).get("face_opacity", 0.5)

            # Extrahiere UVs aus tiles_info (für alle Vertices)
            global_uvs = self._extract_global_uvs(tiles_info, len(vertices))

            for road_material, road_faces in road_faces_by_material.items():
                # Erstelle Mesh mit UVs (wenn verfügbar)
                if global_uvs is not None and len(global_uvs) == len(vertices):
                    road_mesh = self._create_mesh_with_uvs(vertices, road_faces, global_uvs)
                    has_uvs = True
                else:
                    road_mesh = self._create_mesh(vertices, road_faces)
                    has_uvs = False

                # In Textur-Ansicht: Versuche Material-Textur zu verwenden (nur wenn UVs vorhanden)
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
                            print(f"  [✓ Road] {road_material}: Textur angewendet ({len(road_faces)} faces)")
                        except Exception as e:
                            print(f"  [! Road] {road_material}: Textur-Fehler: {e}. Fallback zu Farbe.")
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
                        # Fallback: Farbe
                        print(
                            f"  [○ Road] {road_material}: Textur nicht gefunden. Farbe-Fallback ({len(road_faces)} faces)."
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
                    # Grid-Ansicht oder keine UVs: Farbe mit Kanten
                    reason = "Grid-Ansicht" if not self.use_textures else "Keine UVs"
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

        # Rendere Buildings (Walls + Roofs) - Vereinheitlicht mit Terrain-Rendering
        if is_building and (wall_faces or roof_faces):
            # Sammle UVs aus ALLEN Tiles (jedes Building ist ein separates Geometry/Tile im DAE)
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
                        print(f"  [!] UV/Vertex Mismatch: {len(building_uvs)} UVs vs {len(vertices)} Vertices")
                        building_uvs = None

            # Rendere Walls
            if wall_faces:
                wall_mesh = self._create_mesh_with_uvs(vertices, wall_faces, building_uvs)

                # Textur-Ansicht: Versuche Material-Textur
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
                        print(f"  [✓ Walls] {wall_material} mit Textur")
                    else:
                        # Fallback: Weiße Farbe
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
                        print(f"  [○ Walls] Farbe-Fallback (Material {wall_material} nicht gefunden)")
                else:
                    # Grid-Ansicht: Farbe mit Kanten
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

            # Rendere Roofs
            if roof_faces:
                roof_mesh = self._create_mesh_with_uvs(vertices, roof_faces, building_uvs)

                # Textur-Ansicht: Versuche Material-Textur
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
                        print(f"  [✓ Roofs] {roof_material} mit Textur")
                    else:
                        # Fallback: Rote Farbe
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
                        print(f"  [○ Roofs] Farbe-Fallback (Material {roof_material} nicht gefunden)")
                else:
                    # Grid-Ansicht: Farbe mit Kanten
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
        """Erstelle ein PyVista PolyData Mesh aus Vertices und Faces."""
        # PyVista erwartet: [num_points_in_face, pt0, pt1, pt2, ...]
        pyvista_faces = []
        for face in faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(vertices, pyvista_faces)
        # ÜBERLEBENSWICHTIG: split_sharp_edges=True erzeugt harte Kanten für Häuser!
        # Ohne das: Wände wirken "rund" oder völlig unschattiert
        try:
            mesh = mesh.compute_normals(
                cell_normals=True, point_normals=True, split_sharp_edges=True  # ESSENTIAL für scharfe Schattierung
            )
        except TypeError:
            # Fallback: älter PyVista ohne split_sharp_edges
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True)
        return mesh

    def _create_mesh_with_uvs(self, vertices, faces, uvs):
        """
        Erstelle ein PyVista PolyData Mesh mit Texture-Koordinaten.

        WICHTIG: Dieses macht ein REMAPPED mesh, wo nur die Vertices verwendet werden,
        die von den Faces benötigt werden. Das erzeugt korrekte UV-Indizierung!
        """
        # Sammle unique Vertices, die von Faces benutzt werden
        unique_vertex_indices = set()
        for face in faces:
            unique_vertex_indices.update(face)

        unique_vertex_indices = sorted(unique_vertex_indices)

        # Erstelle Remapping: old_index → new_index
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

        # Remapped Vertices und UVs
        remapped_vertices = vertices[unique_vertex_indices]
        remapped_uvs = uvs[unique_vertex_indices] if uvs is not None else None

        # Remapped Faces (mit neuen Indizes)
        remapped_faces = []
        for face in faces:
            remapped_face = [vertex_map[v_idx] for v_idx in face]
            remapped_faces.append(remapped_face)

        # Erstelle PyVista Faces
        pyvista_faces = []
        for face in remapped_faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(remapped_vertices, pyvista_faces)

        # Setze Texture-Koordinaten wenn vorhanden
        if remapped_uvs is not None:
            mesh.active_texture_coordinates = remapped_uvs % 1.0

        # Compute Normals
        try:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True, split_sharp_edges=True)
        except TypeError:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True)

        return mesh

    def _extract_building_uvs(self, tiles_info, vertices):
        """
        Extrahiere UV-Koordinaten für Building aus tiles_info.

        Args:
            tiles_info: Dict mit Tile-Informationen (enthält UVs)
            vertices: NumPy Array mit Vertices

        Returns:
            NumPy Array mit UV-Koordinaten (n, 2) oder leeres Array
        """
        if not tiles_info:
            print(f"  [DEBUG] _extract_building_uvs: tiles_info ist leer")
            return np.array([])

        print(f"  [DEBUG] _extract_building_uvs: tiles_info keys = {list(tiles_info.keys())}")

        # Sammle UVs von allen Building-Tiles (sie sind bereits in der richtigen Reihenfolge)
        all_uvs = []
        for tile_name, tile_data in tiles_info.items():
            print(f"    Checking tile: {tile_name}")
            uvs = tile_data.get("uvs", np.array([]))
            if len(uvs) > 0:
                print(f"      → Hat UVs: shape={uvs.shape}")
                all_uvs.append(uvs)
            else:
                print(f"      → Keine UVs")

        # Kombiniere alle UVs
        if all_uvs:
            combined_uvs = np.vstack(all_uvs)
            print(f"  [DEBUG] Combined UVs: shape={combined_uvs.shape}, vertices shape={vertices.shape}")
            return combined_uvs

        print(f"  [DEBUG] Keine UVs gefunden!")
        return np.array([])

    def _extract_global_uvs(self, tiles_info, num_vertices):
        """
        Extrahiere globale UV-Koordinaten aus tiles_info.

        Das DAE hat Vertices und UVs pro Tile gespeichert. Diese Funktion
        kombiniert die UVs aller Tiles zu einem globalen UV-Array.

        WICHTIG: Die Reihenfolge der Tiles muss mit der Vertex-Reihenfolge
        übereinstimmen! Daher KEIN sorted() verwenden!

        Args:
            tiles_info: Dict mit Tile-Informationen
            num_vertices: Anzahl der globalen Vertices

        Returns:
            NumPy Array (num_vertices, 2) mit UV-Koordinaten oder None
        """
        if not tiles_info:
            return None

        # Sammle UVs von allen Tiles (OHNE sorted, damit Reihenfolge erhalten bleibt!)
        all_uvs = []
        for tile_name, tile_data in tiles_info.items():
            uvs = tile_data.get("uvs", np.array([]))
            if len(uvs) > 0:
                all_uvs.append(uvs)

        # Kombiniere alle Tile-UVs
        if all_uvs:
            combined_uvs = np.vstack(all_uvs)
            if len(combined_uvs) == num_vertices:
                return combined_uvs
            else:
                print(f"  [!] UV-Array Größe stimmt nicht: {len(combined_uvs)} UVs vs {num_vertices} Vertices")
                return None

        return None

    def _create_mesh_with_uvs(self, vertices, faces, uvs):
        """Erstelle ein PyVista PolyData Mesh mit UV-Koordinaten."""
        pyvista_faces = []
        for face in faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(vertices, pyvista_faces)

        # ÜBERLEBENSWICHTIG: split_sharp_edges auch hier für korrekte Terrain-Schattierung!
        try:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=True, split_sharp_edges=True)
        except TypeError:
            # Fallback für ältere Versionen
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=True)

        # Füge UV-Koordinaten hinzu (als texture coordinates)
        if len(uvs) > 0 and len(uvs) == len(vertices):
            uv_array = np.array(uvs) if not isinstance(uvs, np.ndarray) else uvs
            mesh.active_texture_coordinates = uv_array

        return mesh

    def _load_grid_colors(self):
        """Lade Grid-Farben aus debug_network.json."""
        debug_network_path = Path(__file__).parent.parent / "cache" / "debug_network.json"

        # Default Grid-Farben
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
            print(f"  [!] Fehler beim Laden der Grid-Farben: {e}")
            return default_colors

    def _load_textures(self):
        """Lade alle Tile-Texturen aus dem textures-Verzeichnis."""
        textures = {}

        if not self.textures_dir.exists():
            print(f"  [!] Textures-Verzeichnis nicht gefunden: {self.textures_dir}")
            return textures

        patterns = ["*.jpg", "*.jpeg", "*.png", "*.dds"]
        texture_files = []

        for pattern in patterns:
            texture_files.extend(Path(self.textures_dir).glob(pattern))

        for texture_path in texture_files:
            texture_key = texture_path.stem.lower()  # z.B. "tile_0_0" oder "horizon_sentinel2"

            try:
                if texture_path.suffix.lower() == ".dds":
                    try:
                        import importlib

                        imageio = importlib.import_module("imageio.v2")
                        img_array = imageio.imread(str(texture_path))
                    except ImportError:
                        print(f"  [!] imageio nicht verfügbar, überspringe DDS Textur {texture_path.name}")
                        continue
                else:
                    img = Image.open(texture_path)
                    img_array = np.array(img.convert("RGB"))

                if img_array.ndim == 2:  # Grauwerte -> RGB duplizieren
                    img_array = np.stack([img_array] * 3, axis=-1)

                textures[texture_key] = pv.Texture(img_array)

            except Exception as e:
                print(f"  [!] Fehler beim Laden von {texture_path.name}: {e}")

        return textures

    def _load_material_textures(self):
        """
        Lade Texturen aus main.materials.json für Straßen und Gebäude.

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

            stage = stages[0]  # Erste Stage nutzen

            if not isinstance(stage, dict):
                continue

            # Suche nach baseColorMap (primäre Textur)
            texture_path = stage.get("baseColorMap")
            diffuse_color = stage.get("diffuseColor")  # Optionaler Tint oder reine Farbe

            if not texture_path and diffuse_color:
                # Kein Bild, aber Farbe vorhanden -> 1x1 Farbfeld als Textur
                try:
                    color_rgb = self._normalize_diffuse_color(diffuse_color)
                    img_array = np.array([[color_rgb]], dtype=np.uint8)
                    texture = pv.Texture(img_array)
                    texture.mipmap = True
                    texture.interpolate = True
                    material_textures[mat_name] = texture
                    print(f"  [✓] Material-Farbtextur generiert: {mat_name} (diffuseColor)")
                except Exception as e:
                    print(f"  [!] diffuseColor für {mat_name} konnte nicht erzeugt werden: {e}")
                continue

            if not texture_path:
                continue

            # Konvertiere BeamNG-Pfad zu absolutem Pfad
            abs_texture_path = self._resolve_asset_path(texture_path)

            if not abs_texture_path:
                # Zeige den Pfad, der tatsächlich gesucht wurde
                if texture_path.startswith("/assets/"):
                    rel_path_posix = PurePosixPath(texture_path[1:])
                    data_dir = Path(__file__).parent.parent / "data"
                    abs_path = (data_dir / rel_path_posix).resolve()
                    print(f"  [!] Material-Textur für {mat_name} nicht gefunden: {abs_path}")
                elif texture_path.startswith("/levels/") or texture_path.startswith(str(config.RELATIVE_DIR)):
                    attempted_path = _resolve_beamng_path(texture_path)
                    print(f"  [!] Material-Textur für {mat_name} nicht gefunden: {attempted_path or texture_path}")
                else:
                    print(f"  [!] Material-Textur für {mat_name} nicht auflösbar: {Path(texture_path).as_posix()}")
                continue

            if not Path(abs_texture_path).exists():
                print(f"  [!] Material-Textur für {mat_name} nicht gefunden: {abs_texture_path}")
                continue

            try:
                # Lade Textur
                if Path(abs_texture_path).suffix.lower() == ".dds":
                    try:
                        import importlib

                        imageio = importlib.import_module("imageio.v2")
                        img_array = imageio.imread(str(abs_texture_path))
                        # Konvertiere RGBA zu RGB (entferne Alpha-Kanal für volle Opazität)
                        if img_array.ndim == 3 and img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                    except ImportError:
                        print(f"  [!] imageio nicht verfügbar, überspringe {mat_name} DDS Textur")
                        continue
                else:
                    img = Image.open(abs_texture_path)
                    img_array = np.array(img.convert("RGB"))
                    print(f"  [✓] Material-Textur geladen: {mat_name} -> {abs_texture_path.replace('/', os.sep)}")

                if img_array.ndim == 2:  # Grauwerte -> RGB
                    img_array = np.stack([img_array] * 3, axis=-1)

                # Wende optionalen Tint an
                if diffuse_color:
                    try:
                        img_array = self._apply_diffuse_tint(img_array, diffuse_color)
                        print(f"  [✓] diffuseColor angewendet: {mat_name}")
                    except Exception as e:
                        print(f"  [!] diffuseColor für {mat_name} konnte nicht angewendet werden: {e}")

                texture = pv.Texture(img_array)
                # Aktiviere Mipmap und Interpolation für bessere Qualität
                texture.mipmap = True
                texture.interpolate = True
                material_textures[mat_name] = texture

            except Exception as e:
                print(f"  [!] Fehler beim Laden der Material-Textur {mat_name}: {e}")

        return material_textures

    def _normalize_diffuse_color(self, color):
        """Normiere diffuseColor (0-1 floats) zu uint8 RGB."""
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            raise ValueError("diffuseColor muss mindestens 3 Komponenten haben")
        # Nutze nur RGB, Alpha wird ignoriert für die Textur
        rgb = [max(0.0, min(1.0, float(c))) for c in color[:3]]
        return [int(round(c * 255)) for c in rgb]

    def _apply_diffuse_tint(self, img_array, color):
        """Wende diffuseColor als Multiplikator auf die Textur an."""
        rgb = np.array(self._normalize_diffuse_color(color), dtype=np.float32) / 255.0
        # Stelle sicher, dass Bild 3 Kanäle hat
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        tinted = np.clip(img_array.astype(np.float32) * rgb, 0, 255).astype(np.uint8)
        return tinted

    def _resolve_asset_path(self, texture_path: str) -> str:
        """
        Konvertiere BeamNG Asset-Pfad zu absolutem Dateisystempfad.

        Texturen für Straßen und Gebäude liegen im lokalen data/ Verzeichnis.

        Args:
            texture_path: BeamNG Asset-Pfad (z.B. "/assets/materials/...")

        Returns:
            Absoluter Pfad oder None
        """
        if not texture_path:
            return None

        # 1. Level-spezifische Pfade (/levels/World_to_BeamNG/...) -> nutze _resolve_beamng_path
        if texture_path.startswith("/levels/") or texture_path.startswith(
            str(config.RELATIVE_DIR)
        ):  # config.RELATIVE_DIR is PurePosixPath
            return _resolve_beamng_path(texture_path)

        # 2. Asset-Pfade (/assets/materials/...) -> suche in data/assets/
        if texture_path.startswith("/assets/"):
            rel_path = Path(texture_path[1:])  # Convert to Path to use / operator
            # Suche relativ zum aktuellen Verzeichnis
            data_dir = Path(__file__).parent.parent / "data"
            abs_path = data_dir / rel_path
            abs_path = abs_path.resolve()  # Normalisiere Pfad
            return str(abs_path) if abs_path.exists() else None

        return None

    def _update_active_layers_text(self):
        """Aktualisiere Aktive-Layer-Text oben rechts."""
        active_items = []
        if self.show_terrain:
            active_items.append("T")
        if self.show_roads:
            active_items.append("S")  # S für Straßen
        if self.show_buildings:
            active_items.append("H")  # H für Häuser
        if self.show_forest:
            active_items.append("C")  # C für Forst/Bäume
        if self.show_horizon:
            active_items.append("O")  # O für Horizont
        if self.use_textures:
            active_items.append("X")
        if self.show_debug:
            active_items.append("D")

        active_text = " ".join(active_items) if active_items else "-"

        try:
            self.plotter.remove_actor("active_layers_text")
        except Exception as e:
            print(f"[!] Fehler beim Entfernen des aktiven Layer-Textes: {e}")

        try:
            self._active_layers_actor = self.plotter.add_text(
                active_text,
                position="upper_right",
                font_size=10,
                name="active_layers_text",
            )
        except Exception as e:
            print(f"[!] Fehler beim Erstellen des aktiven Layer-Textes: {e}")
            self._active_layers_actor = None

    def _update_camera_status(self):
        """Zeige Kamera-Status unten links und korrigiere Roll=0°, Zoom=30°."""
        cam = self.plotter.camera
        if cam is None:
            return

        try:
            # Setze Roll auf 0° und Zoom auf 30° automatisch
            try:
                cam.up = [0.0, 0.0, 1.0]
                cam.view_angle = 30.0
            except Exception as e:
                print(f"[!] Fehler beim Setzen der Kamera-Eigenschaften: {e}")

            pos = np.array(cam.position, dtype=float)
            focal = np.array(cam.focal_point, dtype=float)

            # up Vector korrekt auslesen
            try:
                up = np.array(cam.up, dtype=float)
            except Exception as e:
                print(f"[!] Fehler beim Lesen des up-Vektors: {e}")
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

            # Zoom aus view_angle lesen
            try:
                zoom = cam.view_angle
            except Exception as e:
                print(f"[!] Fehler beim Lesen des Zoom-Werts: {e}")
                zoom = 30.0

            text = (
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                f"Tilt: {tilt:.1f}° | Roll: {roll:.1f}° | Yaw: {yaw:.1f}° | Zoom: {zoom:.1f}°"
            )

            # Entferne alten Text-Actor
            try:
                self.plotter.remove_actor("camera_status_text")
            except Exception as e:
                print(f"[!] Fehler beim Entfernen des Kamera-Status-Textes: {e}")

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
                print(f"[!] Fehler beim Erstellen des Kamera-Status-Textes: {e}")
                self._camera_status_actor = None
        except Exception as e:
            print(f"[!] Fehler in _update_camera_status: {e}")

    def _on_camera_change(self, obj, event):
        """Update Statuszeile nach Kamera-Änderungen."""
        try:
            self._update_camera_status()
        except Exception as e:
            print(f"[!] Fehler in _on_camera_change: {e}")

    def _on_render_event(self, obj, event):
        """Update Statuszeile bei RenderEvent mit Drosselung."""
        try:
            self._render_update_counter += 1
            if self._render_update_counter >= 5:
                self._render_update_counter = 0
                self._update_camera_status()
        except Exception as e:
            print(f"[!] Fehler in _on_render_event: {e}")

    def _adjust_zoom(self, delta):
        """Ändere Zoom (view_angle) um delta Grad."""
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
            print(f"[!] Fehler beim Ändern des Zoom: {e}")

    def _load_config(self):
        """Lade Config-Datei."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_config(self, data):
        """Speichere Config-Datei."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_camera_state(self):
        """Lade Kamera-State aus Config."""
        cfg = self._load_config()
        return cfg.get("camera")

    def _load_layers_state(self):
        """Lade Ebenen-Einstellungen aus Config."""
        cfg = self._load_config()
        return cfg.get("layers", {})

    def load_camera_state(self):
        """Lade gespeicherte Kamera-Position (K-Taste)."""
        state = self._load_camera_state()
        if not state:
            print("[Kamera] Keine gespeicherte Kamera gefunden")
            return
        cam = self.plotter.camera
        if cam is None:
            print("[Kamera] Kamera nicht verfügbar")
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
                print("[Kamera] Geladen")
            else:
                print("[Kamera] Ungültiger Kamera-State")
        except Exception as e:
            print(f"[Kamera] Fehler beim Laden: {e}")

    def _apply_saved_camera_state(self):
        """Wende gespeicherte Kamera beim Start an."""
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
            print(f"[!] Fehler beim Anwenden der Kamera-State: {e}")

    def save_camera_state(self):
        """Speichere Kamera-Position (Shift+K)."""
        cam = self.plotter.camera
        if cam is None:
            print("[Kamera] Kamera nicht verfügbar")
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
            print(f"[Kamera] Gespeichert nach {self.config_path}")
        except Exception as e:
            print(f"[Kamera] Fehler beim Speichern: {e}")
        self._save_window_state()

    def _load_window_state(self):
        """Lade Fenster-State aus Config."""
        cfg = self._load_config()
        return cfg.get("window")

    def _apply_saved_window_state(self):
        """Wende gespeicherte Fensterposition/-größe beim Start an."""
        state = self._load_window_state()
        if not state:
            return
        try:
            x = int(state.get("x", 0))
            y = int(state.get("y", 0))
            w = int(state.get("w", 0))
            h = int(state.get("h", 0))
        except Exception as e:
            print(f"[!] Fehler beim Konvertieren der Fenster-State-Werte: {e}")
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
            print(f"[!] Fehler beim Anwenden der Fenster-Position/-Größe: {e}")

    def _save_window_state(self):
        """Speichere Fensterposition/-größe."""
        try:
            win = self.plotter.render_window
            if win is None:
                print("[!] render_window ist None, kann Fenster-State nicht speichern")
                return

            pos = win.GetPosition()
            size = win.GetSize()
        except Exception as e:
            print(f"[!] Fehler beim Auslesen der Fenster-State: {e}")
            return

        state = {
            "x": int(pos[0]),
            "y": int(pos[1]),
            "w": int(size[0]),
            "h": int(size[1]),
        }
        cfg = self._load_config()
        cfg["window"] = state
        # Speichere auch aktuelle Kamera
        cam = self.plotter.camera
        if cam is not None:
            cfg["camera"] = {
                "position": list(cam.position),
                "focal_point": list(cam.focal_point),
                "up_vector": list(cam.up),
            }

        # Speichere auch Ebenen-Einstellungen
        cfg["layers"] = {
            "terrain": self.show_terrain,
            "roads": self.show_roads,
            "textures": self.use_textures,
            "debug": self.show_debug,
        }
        self._save_config(cfg)

    def _on_close_save_window_state(self, *args, **kwargs):
        """Speichere Fenster-State beim Schließen (ExitEvent + atexit)."""
        try:
            # Prüfe ob Plotter noch gültig ist
            if self.plotter is None or self.plotter.render_window is None:
                return

            self._save_window_state()
            print(f"\n[Config] Fenster-State und Kamera-Position gespeichert")
        except Exception as e:
            print(f"[!] Fehler beim Speichern der Config: {e}")

    def reload_dae_file(self):
        """Lade alle DAE-Dateien neu (L-Taste)."""
        self._show_reload_overlay()
        try:
            print(f"\n[Reload] Lade alle DAE-Dateien aus items.level.json...")

            # Speichere Kamera UND Debug-Layer-Status
            camera_pos = None
            camera_focal = None
            camera_up = None
            debug_was_visible = self.show_debug

            try:
                camera_pos = self.plotter.camera.position
                camera_focal = self.plotter.camera.focal_point
                camera_up = self.plotter.camera.up
            except Exception as e:
                print(f"[!] Fehler beim Speichern der Kamera-Position: {e}")

            # Lade Items neu mit gemeinsamer Funktion aus dae_loader
            items_path = config.BEAMNG_DIR / config.ITEMS_JSON

            try:
                self.dae_files, self.tile_data = load_dae_tile_all_from_items(
                    config.BEAMNG_DIR, items_path, _resolve_beamng_path
                )
            except Exception as e:
                print(f"  [!] Fehler beim Laden von DAE-Dateien: {e}")
                import traceback

                traceback.print_exc()
                self.dae_files = []
                self.tile_data = []

            # Lade Texturen neu
            self.textures = self._load_textures()

            print(f"  ✓ {len(self.tile_data)} DAE-Dateien neu geladen")

            # Setze Debug-Layer-Status zurück (wird NACH update_view neu geladen)
            self.debug_loaded = False
            self.debug_actors = []

            # update_view() lädt Terrain/Road/Building actors
            self.update_view()

            # NACH update_view: Lade Debug-Layer neu (damit sie nicht von plotter.clear() gelöscht werden)
            if debug_was_visible:
                self._load_debug_layer()
                self.debug_loaded = True
                self.show_debug = True
                # Setze Sichtbarkeit
                for actor in self.debug_actors:
                    actor.SetVisibility(True)
                self.plotter.render()

            # Stelle Kamera wieder her
            if camera_pos is not None:
                try:
                    self.plotter.camera.position = camera_pos
                    self.plotter.camera.focal_point = camera_focal
                    self.plotter.camera.up = camera_up
                    print("  ✓ Kamera-Position beibehalten")
                except Exception as e:
                    print(f"[!] Fehler beim Wiederherstellen der Kamera-Position: {e}")

            return True
        except Exception as e:
            print(f"  ✗ Fehler beim Reload: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self._hide_reload_overlay()

    def _show_reload_overlay(self):
        """Zeige Reload-Overlay."""
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
            print(f"[!] Fehler beim Anzeigen des Reload-Overlays: {e}")
            self._reload_actor = None

    def _hide_reload_overlay(self):
        """Verstecke Reload-Overlay."""
        if self._reload_actor is not None:
            try:
                self.plotter.remove_actor(self._reload_actor)
                self.plotter.render()
            except Exception as e:
                print(f"[!] Fehler beim Verstecken des Reload-Overlays: {e}")
            self._reload_actor = None

    def _load_debug_layer(self):
        """Lade Debug-Layer aus Primitives (neues Format von DebugNetworkExporter)."""
        print("  [Debug] Lade Debug-Layer...")

        # Lade Primitive-Daten aus cache/debug_network.json (lokales Project-Verzeichnis)
        debug_network_path = Path(__file__).parent.parent / "cache" / "debug_network.json"

        if not debug_network_path.exists():
            print(f"  [Debug] Keine Debug-Daten gefunden: {debug_network_path}")
            return

        try:
            with open(debug_network_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)
        except Exception as e:
            print(f"  [!] Fehler beim Laden der Debug-Daten: {e}")
            return

        primitives = debug_data.get("primitives", [])

        if not primitives:
            print(f"  [Debug] Keine Primitives in Debug-Daten gefunden")
            return

        print(f"  [Debug] Lade {len(primitives)} Primitives")

        # Sammle Primitives nach Typ
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
                # Text-Label mit Position
                text = prim.get("text", "Label")
                position = prim.get("position", [0, 0, 0])
                size = prim.get("size", 12.0)
                labels.append((text, position, color, size))

        actor_count = 0

        # Rendere Lines (z.B. Centerlines)
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
                    color=lines[0][1],  # Nutze Farbe der ersten Line
                    line_width=2.0,
                    opacity=0.8,
                    label="Centerlines",
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Rendere Points (z.B. Junctions)
        if points:
            point_coords = np.array([p[0] for p in points])
            point_colors = [p[1] for p in points]

            # Erstelle Spheres für Junctions
            junction_blocks = pv.MultiBlock()
            for coord, color in points:
                sphere = pv.Sphere(radius=2.0, center=coord)
                junction_blocks.append(sphere)

            if len(junction_blocks) > 0:
                actor = self.plotter.add_mesh(
                    junction_blocks,
                    color=points[0][1],  # Nutze Farbe des ersten Point
                    opacity=0.5,
                    label="Junctions",
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Rendere Circles (kombiniert in einen Actor)
        if circles:
            circles_blocks = pv.MultiBlock()
            for center, radius, color in circles:
                circle = pv.Sphere(radius=radius, center=center)
                circles_blocks.append(circle)

            if len(circles_blocks) > 0:
                actor = self.plotter.add_mesh(
                    circles_blocks,
                    color=circles[0][2],  # Nutze Farbe des ersten Circle
                    opacity=0.3,
                    label="Circles",
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Rendere Polygons (kombiniert in einen Actor - als Linien-Outline)
        if polygons:
            all_poly_points = []
            all_poly_lines = []
            point_offset = 0

            for poly_coords, color in polygons:
                coords_array = np.array(poly_coords)
                if len(coords_array) >= 3:
                    n = len(coords_array)
                    all_poly_points.extend(coords_array)

                    # Erstelle geschlossenes Polygon als Linien (nicht als Faces)
                    for i in range(n):
                        next_i = (i + 1) % n  # Schließe Polygon
                        all_poly_lines.append([2, point_offset + i, point_offset + next_i])
                    point_offset += n

            if all_poly_points:
                all_poly_points_array = np.array(all_poly_points)
                all_poly_lines_array = np.array(all_poly_lines)
                polygons_mesh = pv.PolyData(all_poly_points_array, lines=all_poly_lines_array)
                actor = self.plotter.add_mesh(
                    polygons_mesh,
                    color=polygons[0][1],  # Nutze Farbe des ersten Polygon
                    line_width=2.0,
                    opacity=1.0,
                    label="Polygons",
                    render_lines_as_tubes=False,
                )
                self.debug_actors.append(actor)
                actor_count += 1

        # Rendere Labels (Text an Positionen) - Batch-weise für Performance
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
                    # Batch-Rendering mit add_point_labels (viel schneller!)
                    label_actors = self.plotter.add_point_labels(
                        positions_array,
                        texts,
                        font_size=10,
                        text_color="white",
                        render=False,  # render=False um Performance zu sparen
                    )
                    # Füge Label-Actors zu debug_actors hinzu (für D-Toggle)
                    if label_actors is not None:
                        if isinstance(label_actors, list):
                            self.debug_actors.extend(label_actors)
                        else:
                            self.debug_actors.append(label_actors)
                        actor_count += 1
            except Exception as e:
                print(f"  [!] Fehler beim Rendern von Labels: {e}")

        print(
            f"  [Debug] {actor_count} Debug-Actors gerendert ({len(points)} Junctions, {len(lines)} Centerlines, {len(labels)} Labels)"
        )

    def _load_forest_layer(self):
        """Lade Forest-Layer aus forest.json."""
        print("  [Forest] Lade Forest-Layer...")

        # Suche forest.json im BEAMNG_DIR/main/ Verzeichnis
        from forest_loader import load_forest_layer

        forest_json_path = config.BEAMNG_DIR / "main" / "forest.json"

        if not forest_json_path.exists():
            print(f"  [!] Forest-JSON nicht gefunden: {forest_json_path}")
            return

        try:
            # Lade forest.json und erstelle Punkt-Cloud
            actor = load_forest_layer(self, forest_json_path)

            if actor is not None:
                print(f"  [✓] Forest-Layer geladen")
                return

        except Exception as e:
            print(f"  [!] Fehler beim Laden des Forest-Layers: {e}")
            import traceback

            traceback.print_exc()

    def _on_left_mouse_click(self, obj, event):
        """Handler für linken Doppel-Klick: Setze Kamera-Pivot auf angeklickten Punkt."""
        try:
            now_ts = time.perf_counter()
            if self._last_click_ts and (now_ts - self._last_click_ts) <= 0.2:
                # Doppelklick erkannt
                self._last_click_ts = 0.0
            else:
                # Erster Klick: Zeit merken und abbrechen
                self._last_click_ts = now_ts
                return
            # Hole Mausposition im Fenster
            try:
                click_pos = obj.GetEventPosition()
            except AttributeError:
                click_pos = obj.get_event_position()

            # Führe Raycasting durch
            hit_point = self._raycast_to_mesh(click_pos)

            if hit_point is not None:
                self._set_camera_to_point(hit_point)
            else:
                print("[Raycast] Kein Mesh an dieser Position getroffen")

        except Exception as e:
            print(f"[!] Fehler beim Mausklick-Raycasting: {e}")
            import traceback

            traceback.print_exc()

    def _raycast_to_mesh(self, screen_pos):
        """Führe Raycasting von Mausposition zum Mesh durch.

        Args:
            screen_pos: (x, y) Tupel der Mausposition im Fenster

        Returns:
            hit_point: (x, y, z) NumPy-Array des Schnittpunkts oder None
        """
        try:
            # Hole Renderer und Kamera
            renderer = self.plotter.renderer
            camera = self.plotter.camera

            # Konvertiere Screen-Koordinaten zu Display-Koordinaten (normalisiert 0..1)
            win_size = self.plotter.window_size
            x_norm = screen_pos[0] / win_size[0]
            y_norm = screen_pos[1] / win_size[1]

            # PyVista's pick_mouse_position nutzt Cell-Picker (performanter als OBBTree)
            # ABER: Wir brauchen den genauen Punkt, nicht nur die Zelle!

            # Alternativ: Nutze VTK's Picker direkt für präzisen Punkt
            try:
                picker = self.plotter.iren.GetPicker()
            except AttributeError:
                try:
                    picker = self.plotter.iren.get_picker()
                except Exception:
                    picker = None

            if picker is None:
                # Erstelle Cell-Picker falls nicht vorhanden
                import vtk

                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.005)  # 0.5% Toleranz
                try:
                    # Hänge Picker an Interactor, damit zukünftige Calls ihn nutzen
                    self.plotter.iren.SetPicker(picker)
                except Exception:
                    pass

            # Führe Pick durch (x, y in Display-Koordinaten, z=0)
            result = picker.Pick(screen_pos[0], screen_pos[1], 0, renderer)

            if result:
                # Erfolgreicher Hit - hole Schnittpunkt
                hit_point = np.array(picker.GetPickPosition())
                print(f"[Raycast] Hit at: ({hit_point[0]:.1f}, {hit_point[1]:.1f}, {hit_point[2]:.1f})")
                return hit_point
            else:
                return None

        except Exception as e:
            print(f"[!] Fehler beim Raycasting: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _set_camera_to_point(self, target_point):
        """Setze Kamera-Pivot auf Punkt und bewege Kamera 40m davor.

        Args:
            target_point: (x, y, z) NumPy-Array des Zielpunkts
        """
        try:
            camera = self.plotter.camera

            # Hole aktuelle Blickrichtung (normalisiert)
            current_pos = np.array(camera.position)
            current_focal = np.array(camera.focal_point)
            view_direction = current_focal - current_pos
            view_dist = np.linalg.norm(view_direction)

            if view_dist > 1e-6:
                view_direction = view_direction / view_dist
            else:
                # Fallback: Blicke von Süden nach Norden
                view_direction = np.array([0.0, 1.0, 0.0])

            # Neue Focal-Point ist der angeklickte Punkt
            new_focal = np.array(target_point)

            # Neue Kamera-Position: 40m in entgegengesetzter Blickrichtung
            camera_distance = 40.0
            new_position = new_focal - view_direction * camera_distance

            # Setze Kamera
            camera.focal_point = new_focal
            camera.position = new_position
            camera.up = [0.0, 0.0, 1.0]  # Z-Achse ist oben

            # Aktualisiere Clipping-Range und rendere
            self.plotter.reset_camera_clipping_range()
            self.plotter.render()

            # Aktualisiere Status-Anzeige
            self._update_camera_status()

            print(f"[Kamera] Pivot: ({new_focal[0]:.1f}, {new_focal[1]:.1f}, {new_focal[2]:.1f})")
            print(f"[Kamera] Position: ({new_position[0]:.1f}, {new_position[1]:.1f}, {new_position[2]:.1f})")
            print(f"[Kamera] Distanz: {camera_distance:.1f}m")

        except Exception as e:
            print(f"[!] Fehler beim Setzen der Kamera: {e}")
            import traceback

            traceback.print_exc()

    def show(self):
        """Zeige das Viewer-Fenster."""
        self.plotter.show()


if __name__ == "__main__":
    viewer = DAETileViewer()
    if hasattr(viewer, "plotter") and viewer.plotter is not None:
        viewer.show()
    else:
        print("[!] Kein Plotter initialisiert (vermutlich keine DAE-Dateien geladen).")
