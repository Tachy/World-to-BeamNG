"""
DAE Viewer - Visualisiere die exportierte terrain.dae mit allen Tiles

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
import os
import sys
import json
import atexit
from pathlib import Path
from PIL import Image

# Importiere config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from world_to_beamng import config
from tools.dae_loader import load_dae_tile


# BeamNG-Relative Pfade ("/levels/<Level>/...") nach absoluten Pfaden auflösen
def _resolve_beamng_path(path_str: str) -> str | None:
    if not path_str:
        return None

    p = path_str.replace("\\", "/")

    # 1) Präfix /levels/<LEVEL_NAME>/...
    level_prefix = f"/levels/{config.LEVEL_NAME}/"
    if p.startswith(level_prefix):
        rel = p[len(level_prefix) :]
        return os.path.join(config.BEAMNG_DIR, rel.replace("/", os.sep))

    # 2) Präfix aus config.RELATIVE_DIR (identisch, aber bereitgestellt)
    if p.startswith(config.RELATIVE_DIR):
        rel = p[len(config.RELATIVE_DIR) :]
        return os.path.join(config.BEAMNG_DIR, rel.replace("/", os.sep))

    # 3) art/… Präfix relativ zum Level-Root
    if p.startswith("art/"):
        return os.path.join(config.BEAMNG_DIR, p.replace("/", os.sep))

    # 4) Fallback: behandle als relative Shape-Angabe
    return os.path.join(config.BEAMNG_DIR_SHAPES, p)


class DAETileViewer:
    def __init__(self):
        # Lade Items und Materialien aus JSON
        items_path = os.path.join(config.BEAMNG_DIR, config.ITEMS_JSON)
        
        # Suche materials.json in config.BEAMNG_DIR/main/
        materials_path = os.path.join(config.BEAMNG_DIR, "main", "materials.json")

        print(f"Lade Items aus: {items_path}")
        
        # Versuche items.json zu laden (kann fehlschlagen, wenn nicht in BeamNG Format)
        self.items = {}
        try:
            with open(items_path, "r", encoding="utf-8") as f:
                items_data = json.load(f)
                # items.json könnte verschiedene Strukturen haben - prüfe verschiedene Patterns
                if isinstance(items_data, dict):
                    # Pattern 1: Direct dict mit shapeName
                    if any(isinstance(v, dict) and 'shapeName' in v for v in items_data.values()):
                        self.items = items_data
                    # Pattern 2: Nested 'Instances'
                    elif 'Instances' in items_data and isinstance(items_data['Instances'], dict):
                        self.items = items_data['Instances']
        except Exception as e:
            print(f"  [!] items.json konnte nicht geladen werden: {e}")
            print(f"  [i] Suche stattdessen direkt nach DAE-Dateien im Niveau...")

        print(f"Lade Materialien aus: {materials_path}")
        if os.path.exists(materials_path):
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

        # Extrahiere alle DAE-Dateien aus Items oder suche direkt
        self.dae_files = []
        
        # Method 1: Aus items.json
        if self.items:
            for item_name, item_data in self.items.items():
                if isinstance(item_data, dict):
                    shape_name = item_data.get("shapeName")
                    if shape_name and shape_name.endswith(".dae"):
                        dae_path = _resolve_beamng_path(shape_name)
                        if dae_path and os.path.exists(dae_path):
                            self.dae_files.append((item_name, dae_path))
                        else:
                            print(f"  [!] DAE nicht gefunden: {shape_name}")
        
        # Method 2: Direkt im Verzeichnis suchen (Fallback oder Ergänzung)
        if not self.dae_files:
            print(f"  [i] Suche nach DAE-Dateien im Level-Verzeichnis...")
            shapes_dir = os.path.join(config.BEAMNG_DIR, "art", "shapes")
            if os.path.exists(shapes_dir):
                for dae_file in Path(shapes_dir).glob("*.dae"):
                    item_name = dae_file.stem  # Verwende Dateinamen als item_name
                    self.dae_files.append((item_name, str(dae_file)))
                    print(f"  [✓] Gefunden: {item_name}")

        if not self.dae_files:
            print("Keine DAE-Dateien in items.json oder art/shapes gefunden!")
            return

        print(f"  -> {len(self.dae_files)} DAE-Dateien gefunden")

        # Lade alle DAE-Dateien
        self.tile_data = []
        for item_name, dae_path in self.dae_files:
            print(f"  Lade {item_name}: {os.path.basename(dae_path)}")
            data = load_dae_tile(dae_path)
            if data:
                self.tile_data.append((item_name, data))

        if not self.tile_data:
            print("Keine Geometrie in DAE-Dateien gefunden!")
            return

        print(f"  -> {len(self.tile_data)} DAE-Dateien geladen")

        # Initialisiere config_path FRÜH (wird für _load_layers_state benötigt)
        self.config_path = os.path.join(os.path.dirname(__file__), "dae_viewer.cfg")

        # Sichtbarkeits-Flags (lade gespeicherte Werte)
        saved_layers = self._load_layers_state()
        self.show_terrain = saved_layers.get("terrain", True)
        self.show_roads = saved_layers.get("roads", True)
        self.show_buildings = saved_layers.get("buildings", True)  # Häuser Toggle
        self.use_textures = saved_layers.get("textures", True)  # Texturen standardmäßig an
        self.show_debug = saved_layers.get("debug", False)  # Debug-Layer (Junctions, Centerlines)

        # Speichere Actor-Referenzen für Sichtbarkeits-Toggles
        self.terrain_actors = []  # Liste von Terrain-Mesh-Actors
        self.road_actors = []  # Liste von Road-Mesh-Actors
        self.building_actors = []  # Liste von Building-Mesh-Actors
        self.debug_actors = []  # Liste von Debug-Actors (Junctions, Centerlines)
        self.debug_loaded = False  # Flag: Debug-Layer bereits geladen?
        self._first_update_view = True  # Flag: Erstes Mal update_view() aufgerufen?

        # Lade Grid-Farben aus debug_network.json (für Grid-Ansicht)
        self.grid_colors = self._load_grid_colors()

        # Lade Texturen
        self.textures_dir = os.path.join(config.BEAMNG_DIR_SHAPES, "textures")
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
                    renderer.shadow_map_size = 8192  # Maximale Schärfe für High-End GPU!
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

        # Registriere atexit-Handler als Fallback (für sicheres Speichern beim Exit)
        atexit.register(self._on_close_save_window_state)

        print(f"\nDAE geladen mit Tile-Geometrien")
        print("\nSteuerung:")
        print("  X = Toggle Texturen (Rendering ↔ Grid) - triggert Neuaufbau")
        print("\nIn Grid-Ansicht (X=aus) - Layer ein-/ausblenden OHNE Neuaufbau:")
        print("  T = Toggle Terrain")
        print("  S = Toggle Straßen")
        print("  H = Toggle Häuser")
        print("  D = Toggle Debug (Junctions, Centerlines, Boundaries)")
        print("\nAllgemein:")
        print("  K = Kamera laden | Shift+K = Kamera speichern")
        print("  L = DAE neu laden")
        print("  Up/Down = Zoom ändern")

        self.update_view()
        # Hinweis: _apply_saved_camera_state() wird NICHT beim Start aufgerufen
        # um sicherzustellen dass die Kamera auf die Geometrie passt!
        # Sie wird nur aufgerufen wenn der Viewer mit show() gestartet wird

        # Wenn Debug-Layer aktiviert sein soll, lade ihn nach update_view()
        if self.show_debug:
            self._update_debug_visibility()

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

        elif key_lower == "x":
            # Toggle Texturen (mit Neuaufbau!)
            self.use_textures = not self.use_textures
            print(f"\n[{'Rendering' if self.use_textures else 'Grid'}-Ansicht]")
            self.update_view()
            # Debug-Layer bleiben dauerhaft geladen und ihre Sichtbarkeit wird beibehalten

        elif key_lower == "d":
            # Toggle Debug nur in Grid-Ansicht
            if not self.use_textures:
                self.show_debug = not self.show_debug
                print(f"\n[Debug] {'AN' if self.show_debug else 'AUS'}")
                self._update_debug_visibility()
            else:
                print("\n[Debug] Nur in Grid-Ansicht verfügbar (X drücken)")

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
        """Aktualisiere Sichtbarkeit der Terrain/Road/Building Actors ohne Reload."""
        for actor in self.terrain_actors:
            actor.SetVisibility(self.show_terrain)
        for actor in self.road_actors:
            actor.SetVisibility(self.show_roads)
        for actor in self.building_actors:
            actor.SetVisibility(self.show_buildings)

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
            buildings = [m for m in self.materials.keys() if "build" in m.lower() or "wall" in m.lower() or "roof" in m.lower()]
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

        self.plotter.clear()
        self._reinit_lights()

        # Leere NUR Terrain/Road/Building Actor-Listen
        self.terrain_actors = []
        self.road_actors = []
        self.building_actors = []
        # Debug-Actors wurden durch clear() gelöscht, aber wir laden sie danach wieder

        # Iteriere über alle geladenen DAE-Dateien
        for item_name, tile_data in self.tile_data:
            print(f"\nRendere {item_name}...")
            self._render_single_dae(item_name, tile_data)

        # Statuszeilen
        # Oben links: Bedienungsanleitung
        bedienung = "S: Straßen | T: Terrain | D: Debug | X: Texturen | K: Cam | L: Reload | Up/Down: Zoom"
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
        is_terrain = item_name.startswith("terrain_")
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
            if "road" in mat_lower:
                # Sammle Road-Faces gruppiert nach Material
                if material not in road_faces_by_material:
                    road_faces_by_material[material] = []
                road_faces_by_material[material].append(faces[face_idx])
            elif "wall" in mat_lower:
                wall_faces.append(faces[face_idx])
            elif "roof" in mat_lower:
                roof_faces.append(faces[face_idx])
            elif "terrain" in mat_lower or "tile" in mat_lower:
                terrain_faces.append(faces[face_idx])
            else:
                # Fallback basierend auf item_name
                if is_building:
                    wall_faces.append(faces[face_idx])
                else:
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

                # Konvertiere tile_name von Index (z.B. "tile_-2_-2") zu Koordinaten
                texture_key = tile_name
                if tile_name.startswith("tile_"):
                    parts = tile_name.split("_")
                    if len(parts) == 3:  # "tile_X_Y"
                        try:
                            tile_idx_x = int(parts[1])
                            tile_idx_y = int(parts[2])
                            coords = self._index_to_coords(item_name, tile_idx_x, tile_idx_y)
                            if coords:
                                texture_key = f"tile_{coords[0]}_{coords[1]}"
                        except:
                            pass  # Fallback zu Original tile_name

                lookup_key = texture_key.lower()
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
                        self.terrain_actors.append(actor)
                        actor.SetVisibility(self.show_terrain)
                        terrain_texture_log.append(f"✓ {tile_name} → {lookup_key}")
                    except Exception as e:
                        print(f"  [!] Textur-Fehler für {tile_name}: {e}")
                        terrain_texture_log.append(f"✗ {tile_name} → FEHLER: {str(e)[:40]}")
                        # Fallback zu Farbe
                        actor = self.plotter.add_mesh(
                            mesh, color=[0.6, 0.5, 0.4], opacity=0.5, label=f"{item_name}_{tile_name}"
                        )
                        self.terrain_actors.append(actor)
                        actor.SetVisibility(self.show_terrain)
                else:
                    # Keine Textur oder keine UVs
                    reason = "KEINE UVs" if len(tile_uvs) == 0 else f"Textur nicht gefunden: {lookup_key}"
                    terrain_texture_log.append(f"○ {tile_name} → {reason}")
                    actor = self.plotter.add_mesh(
                        mesh, color=[0.6, 0.5, 0.4], opacity=0.5, label=f"{item_name}_{tile_name}"
                    )
                    self.terrain_actors.append(actor)
                    actor.SetVisibility(self.show_terrain)
            
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
                self.terrain_actors.append(actor)
                actor.SetVisibility(self.show_terrain)
        
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
                        print(f"  [○ Road] {road_material}: Textur nicht gefunden. Farbe-Fallback ({len(road_faces)} faces).")
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
                    print(f"  [○ Road] {road_material}: {reason}. Farbe-Rendering ({len(road_faces)} faces).")
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

            # Rendere Buildings (Walls + Roofs)
            if wall_faces:
                # Extrahiere UVs für Building aus tiles_info
                building_uvs = self._extract_building_uvs(tiles_info, vertices)

                if len(building_uvs) > 0 and len(building_uvs) == len(vertices):
                    wall_mesh = self._create_mesh_with_uvs(vertices, wall_faces, building_uvs)
                    has_uvs = True
                else:
                    wall_mesh = self._create_mesh(vertices, wall_faces)
                    has_uvs = False

                # In Textur-Ansicht: Versuche Material-Textur zu verwenden
                if self.use_textures and materials:
                    # Finde das Wall-Material
                    wall_material = next((mat for mat in materials if "wall" in mat.lower()), None)
                    if wall_material and wall_material in self.material_textures:
                        texture = self.material_textures[wall_material]
                        try:
                            actor = self.plotter.add_mesh(
                                wall_mesh,
                                texture=texture,
                                label=f"{item_name}_walls",
                                opacity=1.0,
                                show_edges=False,
                                lighting=True,
                                ambient=0.9,
                                diffuse=0.9,
                                specular=0.1,
                            )
                            self.building_actors.append(actor)
                            actor.SetVisibility(self.show_buildings)
                            print(f"  [✓ Walls] Textur angewendet: {wall_material} (UVs: {'ja' if has_uvs else 'nein'})")
                        except Exception as e:
                            print(f"  [! Walls] Textur-Fehler: {e}. Fallback zu Farbe.")
                            actor = self.plotter.add_mesh(
                                wall_mesh,
                                color=face_colors["building_wall"],
                                label=f"{item_name}_walls",
                                opacity=1.0,
                                show_edges=False,
                                lighting=True,
                                ambient=1.0,
                                diffuse=1.0,
                                specular=0.1,
                            )
                            self.building_actors.append(actor)
                            actor.SetVisibility(self.show_buildings)
                    else:
                        # Fallback: Farbe
                        print(f"  [○ Walls] Material '{wall_material}' nicht in material_textures. Farbe-Fallback.")
                        actor = self.plotter.add_mesh(
                            wall_mesh,
                            color=face_colors["building_wall"],
                            label=f"{item_name}_walls",
                            opacity=1.0,
                            show_edges=False,
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        self.building_actors.append(actor)
                        actor.SetVisibility(self.show_buildings)
                else:
                    # Grid-Ansicht oder keine UVs: Farbe (kein Drahtgitter für bessere Sichtbarkeit)
                    reason = "Grid-Ansicht" if not self.use_textures else "Keine Materials"
                    print(f"  [○ Walls] {reason}. Farbe-Rendering.")
                    actor = self.plotter.add_mesh(
                        wall_mesh,
                        color=face_colors["building_wall"],
                        label=f"{item_name}_walls",
                        opacity=1.0,
                        show_edges=False,
                        lighting=True,
                        ambient=self.material_ambient,
                        diffuse=self.material_diffuse,
                        specular=self.material_specular,
                    )
                    self.building_actors.append(actor)
                    actor.SetVisibility(self.show_buildings)

            if roof_faces:
                # Extrahiere UVs für Building aus tiles_info
                building_uvs = self._extract_building_uvs(tiles_info, vertices)

                if len(building_uvs) > 0 and len(building_uvs) == len(vertices):
                    roof_mesh = self._create_mesh_with_uvs(vertices, roof_faces, building_uvs)
                    has_uvs = True
                else:
                    roof_mesh = self._create_mesh(vertices, roof_faces)
                    has_uvs = False

                # In Textur-Ansicht: Versuche Material-Textur zu verwenden
                if self.use_textures and materials:
                    # Finde das Roof-Material
                    roof_material = next((mat for mat in materials if "roof" in mat.lower()), None)
                    if roof_material and roof_material in self.material_textures:
                        texture = self.material_textures[roof_material]
                        try:
                            actor = self.plotter.add_mesh(
                                roof_mesh,
                                texture=texture,
                                label=f"{item_name}_roofs",
                                opacity=1.0,
                                show_edges=False,
                                lighting=True,
                                ambient=0.9,
                                diffuse=0.9,
                                specular=0.1,
                            )
                            self.building_actors.append(actor)
                            actor.SetVisibility(self.show_buildings)
                            print(f"  [✓ Roofs] Textur angewendet: {roof_material} (UVs: {'ja' if has_uvs else 'nein'})")
                        except Exception as e:
                            print(f"  [! Roofs] Textur-Fehler: {e}. Fallback zu Farbe.")
                            actor = self.plotter.add_mesh(
                                roof_mesh,
                                color=face_colors["building_roof"],
                                label=f"{item_name}_roofs",
                                opacity=1.0,
                                show_edges=False,
                                lighting=True,
                                ambient=1.0,
                                diffuse=1.0,
                                specular=0.1,
                            )
                            self.building_actors.append(actor)
                            actor.SetVisibility(self.show_buildings)
                    else:
                        # Fallback: Farbe
                        print(f"  [○ Roofs] Material '{roof_material}' nicht in material_textures. Farbe-Fallback.")
                        actor = self.plotter.add_mesh(
                            roof_mesh,
                            color=face_colors["building_roof"],
                            label=f"{item_name}_roofs",
                            opacity=1.0,
                            show_edges=False,
                            lighting=True,
                            ambient=1.0,
                            diffuse=1.0,
                            specular=0.1,
                        )
                        self.building_actors.append(actor)
                        actor.SetVisibility(self.show_buildings)
                else:
                    # Grid-Ansicht oder keine UVs: Farbe (kein Drahtgitter für bessere Sichtbarkeit)
                    reason = "Grid-Ansicht" if not self.use_textures else "Keine Materials"
                    print(f"  [○ Roofs] {reason}. Farbe-Rendering.")
                    actor = self.plotter.add_mesh(
                        roof_mesh,
                        color=face_colors["building_roof"],
                        label=f"{item_name}_roofs",
                        opacity=1.0,
                        show_edges=False,
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
        
        Args:
            tiles_info: Dict mit Tile-Informationen
            num_vertices: Anzahl der globalen Vertices
            
        Returns:
            NumPy Array (num_vertices, 2) mit UV-Koordinaten oder None
        """
        if not tiles_info:
            return None
        
        # Sammle UVs von allen Tiles
        all_uvs = []
        for tile_name, tile_data in sorted(tiles_info.items()):
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
        debug_network_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "debug_network.json")

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

        if not os.path.exists(debug_network_path):
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

        if not os.path.exists(self.textures_dir):
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
                        img_array = imageio.imread(texture_path)
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

            if not texture_path:
                continue

            # Konvertiere BeamNG-Pfad zu absolutem Pfad
            abs_texture_path = self._resolve_asset_path(texture_path)

            if not abs_texture_path:
                # Zeige den Pfad, der tatsächlich gesucht wurde
                if texture_path.startswith("/assets/"):
                    rel_path = texture_path[1:].replace("/", os.sep)
                    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
                    attempted_path = os.path.abspath(os.path.join(data_dir, rel_path))
                    print(f"  [!] Material-Textur für {mat_name} nicht gefunden: {attempted_path}")
                elif texture_path.startswith("/levels/") or texture_path.startswith(config.RELATIVE_DIR):
                    attempted_path = _resolve_beamng_path(texture_path)
                    print(f"  [!] Material-Textur für {mat_name} nicht gefunden: {attempted_path or texture_path}")
                else:
                    print(f"  [!] Material-Textur für {mat_name} nicht auflösbar: {texture_path.replace('/', os.sep)}")
                continue

            if not os.path.exists(abs_texture_path):
                print(f"  [!] Material-Textur für {mat_name} nicht gefunden: {abs_texture_path}")
                continue

            try:
                # Lade Textur
                if abs_texture_path.lower().endswith(".dds"):
                    try:
                        import importlib

                        imageio = importlib.import_module("imageio.v2")
                        img_array = imageio.imread(abs_texture_path)
                        # Konvertiere RGBA zu RGB (entferne Alpha-Kanal für volle Opazität)
                        if img_array.ndim == 3 and img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                        print(f"  [✓] Material-Textur geladen: {mat_name} -> {abs_texture_path.replace('/', os.sep)}")
                    except ImportError:
                        print(f"  [!] imageio nicht verfügbar, überspringe {mat_name} DDS Textur")
                        continue
                else:
                    img = Image.open(abs_texture_path)
                    img_array = np.array(img.convert("RGB"))
                    print(f"  [✓] Material-Textur geladen: {mat_name} -> {abs_texture_path.replace('/', os.sep)}")

                if img_array.ndim == 2:  # Grauwerte -> RGB
                    img_array = np.stack([img_array] * 3, axis=-1)

                texture = pv.Texture(img_array)
                # Aktiviere Mipmap und Interpolation für bessere Qualität
                texture.mipmap = True
                texture.interpolate = True
                material_textures[mat_name] = texture

            except Exception as e:
                print(f"  [!] Fehler beim Laden der Material-Textur {mat_name}: {e}")

        return material_textures

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
        if texture_path.startswith("/levels/") or texture_path.startswith(config.RELATIVE_DIR):
            return _resolve_beamng_path(texture_path)

        # 2. Asset-Pfade (/assets/materials/...) -> suche in data/assets/
        if texture_path.startswith("/assets/"):
            rel_path = texture_path[1:].replace("/", os.sep)  # Entferne nur führenden /, behalte "assets"
            # Suche relativ zum aktuellen Verzeichnis
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            abs_path = os.path.join(data_dir, rel_path)
            abs_path = os.path.abspath(abs_path)  # Normalisiere Pfad
            return abs_path if os.path.exists(abs_path) else None

        return None

    def _load_buildings(self):
        """Lade buildings_*.dae aus dem buildings-Verzeichnis."""
        buildings_dir = os.path.join(config.BEAMNG_DIR_SHAPES, "buildings")

        if not os.path.exists(buildings_dir):
            return

        building_files = list(Path(buildings_dir).glob("buildings_tile_*.dae"))

        if not building_files:
            return

        print(f"  [Buildings] Lade {len(building_files)} building DAEs...")
        print(f"  [Buildings DEBUG] Material_textures verfügbar: {list(self.material_textures.keys())}")
        print(f"  [Buildings DEBUG] use_textures: {self.use_textures}")

        for building_path in building_files:
            try:
                # Lade DAE mit dae_loader
                from tools.dae_loader import load_dae_tile

                building_data = load_dae_tile(building_path)

                vertices = building_data.get("vertices", [])
                faces = building_data.get("faces", [])
                materials = building_data.get("materials", [])

                if len(vertices) == 0:
                    continue

                print(f"\n  [Buildings DEBUG] {building_path.stem}:")
                print(f"    Vertices: {len(vertices)}")
                print(f"    Faces: {len(faces)}")
                print(f"    Materials: {set(materials)}")

                # Rendering-Ansicht: Nutze Materials (lod2_wall_white, lod2_roof_red)
                if self.use_textures:
                    # Gruppiere Faces nach Material
                    wall_faces = []
                    roof_faces = []

                    for face_idx, material in enumerate(materials):
                        if "wall" in material.lower():
                            wall_faces.append(faces[face_idx])
                        elif "roof" in material.lower():
                            roof_faces.append(faces[face_idx])

                    print(f"    Wall faces: {len(wall_faces)}, Roof faces: {len(roof_faces)}")

                    # Hole UVs aus den Tiles
                    tiles_info = building_data.get("tiles", {})
                    all_uvs = None

                    # Sammle alle UVs aus allen Tiles
                    for tile_name, tile_data in tiles_info.items():
                        tile_uvs = tile_data.get("uvs")
                        if tile_uvs is not None and len(tile_uvs) > 0:
                            if all_uvs is None:
                                all_uvs = tile_uvs
                            else:
                                all_uvs = np.vstack([all_uvs, tile_uvs])
                            print(
                                f"    Tile {tile_name}: UVs shape={tile_uvs.shape}, min={tile_uvs.min(axis=0)}, max={tile_uvs.max(axis=0)}"
                            )
                            break  # Für Gebäude verwenden wir meist nur ein Tile

                    if all_uvs is None:
                        print(f"    [!] KEINE UVs gefunden!")

                    # Rendere Walls mit Textur
                    if wall_faces:
                        wall_mesh = self._create_mesh(vertices, wall_faces)

                        # Setze UVs wenn vorhanden
                        if all_uvs is not None:
                            wall_mesh.active_texture_coordinates = all_uvs % 1.0
                            print(f"    UVs auf Wall-Mesh gesetzt: {wall_mesh.active_texture_coordinates.shape}")

                        # Finde Wall-Material-Textur
                        wall_material = "lod2_wall_white"
                        if wall_material in self.material_textures:
                            texture = self.material_textures[wall_material]
                            print(f"    ✓ Wall-Textur gefunden: {wall_material}")
                            actor = self.plotter.add_mesh(
                                wall_mesh,
                                texture=texture,
                                opacity=1.0,
                                label=f"{building_path.stem}_walls",
                                show_edges=False,
                                lighting=True,
                                ambient=self.material_ambient,
                                diffuse=self.material_diffuse,
                                specular=self.material_specular,
                            )
                            print(f"    ✓ Wall mit Textur gerendert")
                        else:
                            # Fallback: Farbe (weiß)
                            print(f"    [!] Wall-Textur NICHT gefunden: {wall_material}")
                            print(f"        Verfügbare Materials: {list(self.material_textures.keys())}")
                            actor = self.plotter.add_mesh(
                                wall_mesh,
                                color="white",
                                opacity=1.0,
                                label=f"{building_path.stem}_walls",
                                show_edges=False,
                                lighting=True,
                                ambient=self.material_ambient,
                                diffuse=self.material_diffuse,
                                specular=self.material_specular,
                            )
                            print(f"    [!] Wall mit Farb-Fallback gerendert")
                        self.building_actors.append(actor)

                    # Rendere Roofs mit Textur
                    if roof_faces:
                        roof_mesh = self._create_mesh(vertices, roof_faces)

                        # Setze UVs wenn vorhanden
                        if all_uvs is not None:
                            roof_mesh.active_texture_coordinates = all_uvs % 1.0
                            print(f"    UVs auf Roof-Mesh gesetzt: {roof_mesh.active_texture_coordinates.shape}")

                        # Finde Roof-Material-Textur
                        roof_material = "lod2_roof_red"
                        if roof_material in self.material_textures:
                            texture = self.material_textures[roof_material]
                            print(f"    ✓ Roof-Textur gefunden: {roof_material}")
                            actor = self.plotter.add_mesh(
                                roof_mesh,
                                texture=texture,
                                opacity=1.0,
                                label=f"{building_path.stem}_roofs",
                                show_edges=False,
                                lighting=True,
                                ambient=self.material_ambient,
                                diffuse=self.material_diffuse,
                                specular=self.material_specular,
                            )
                            print(f"    ✓ Roof mit Textur gerendert")
                        else:
                            # Fallback: Farbe (dunkelrot)
                            print(f"    [!] Roof-Textur NICHT gefunden: {roof_material}")
                            actor = self.plotter.add_mesh(
                                roof_mesh,
                                color=[0.6, 0.2, 0.1],
                                opacity=1.0,
                                label=f"{building_path.stem}_roofs",
                                show_edges=False,
                                lighting=True,
                                ambient=self.material_ambient,
                                diffuse=self.material_diffuse,
                                specular=self.material_specular,
                            )
                            print(f"    [!] Roof mit Farb-Fallback gerendert")
                        self.building_actors.append(actor)

                # Grid-Ansicht: Nutze grid_colors mit Edges
                else:
                    wall_faces = []
                    roof_faces = []

                    for face_idx, material in enumerate(materials):
                        if "wall" in material.lower():
                            wall_faces.append(faces[face_idx])
                        elif "roof" in material.lower():
                            roof_faces.append(faces[face_idx])

                    # Walls mit Grid-Farben
                    if wall_faces:
                        wall_colors = self.grid_colors.get("building_wall", {})
                        mesh = self._create_mesh(vertices, wall_faces)
                        actor = self.plotter.add_mesh(
                            mesh,
                            color=wall_colors.get("face", [0.95, 0.95, 0.95]),
                            opacity=0.5,  # Semi-transparent in grid mode
                            show_edges=True,
                            edge_color=wall_colors.get("edge", [0.3, 0.3, 0.3]),
                            line_width=1.0,
                            label=f"{building_path.stem}_walls",
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        self.building_actors.append(actor)

                    # Roofs mit Grid-Farben
                    if roof_faces:
                        roof_colors = self.grid_colors.get("building_roof", {})
                        mesh = self._create_mesh(vertices, roof_faces)
                        actor = self.plotter.add_mesh(
                            mesh,
                            color=roof_colors.get("face", [0.6, 0.2, 0.1]),
                            opacity=0.5,  # Semi-transparent in grid mode
                            show_edges=True,
                            edge_color=roof_colors.get("edge", [0.3, 0.1, 0.05]),
                            line_width=1.0,
                            label=f"{building_path.stem}_roofs",
                            lighting=True,
                            ambient=self.material_ambient,
                            diffuse=self.material_diffuse,
                            specular=self.material_specular,
                        )
                        self.building_actors.append(actor)

            except Exception as e:
                print(f"  [!] Fehler beim Laden von {building_path.name}: {e}")

        if self.building_actors:
            print(f"  [Buildings] {len(self.building_actors)} Gebäude-Meshes geladen")

    def _update_active_layers_text(self):
        """Aktualisiere Aktive-Layer-Text oben rechts."""
        active_items = []
        if self.show_terrain:
            active_items.append("T")
        if self.show_roads:
            active_items.append("S")  # S für Straßen
        if self.show_buildings:
            active_items.append("H")  # H für Häuser
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
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_config(self, data):
        """Speichere Config-Datei."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
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
            print(f"  [Reload] Lade alle DAE-Dateien aus main.items.json...")

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

            # Lade Items neu
            items_path = os.path.join(config.BEAMNG_DIR, config.ITEMS_JSON)
            with open(items_path, "r", encoding="utf-8") as f:
                self.items = json.load(f)

            # Extrahiere alle DAE-Dateien aus Items
            self.dae_files = []
            for item_name, item_data in self.items.items():
                # item_data kann ein Dict oder ein String sein
                if isinstance(item_data, dict):
                    shape_name = item_data.get("shapeName")
                elif isinstance(item_data, str):
                    # Direkter String-Wert (z.B. einfache Pfad-Struktur)
                    shape_name = item_data
                else:
                    continue
                    
                if shape_name and shape_name.endswith(".dae"):
                    dae_path = _resolve_beamng_path(shape_name)
                    if dae_path and os.path.exists(dae_path):
                        self.dae_files.append((item_name, dae_path))

            # Lade alle DAE-Dateien neu
            from tools.dae_loader import load_dae_tile

            self.tile_data = []
            for item_name, dae_path in self.dae_files:
                data = load_dae_tile(dae_path)
                if data:
                    self.tile_data.append((item_name, data))

            # Lade Texturen neu
            self.textures = self._load_textures()

            print(f"  ✓ {len(self.tile_data)} DAE-Dateien neu geladen")

            # Debug-Layer zurücksetzen (wird neu geladen wenn aktiv)
            self.debug_loaded = False
            self.debug_actors = []

            self.update_view()

            # Stelle Debug-Layer wieder her wenn er aktiv war
            if debug_was_visible:
                self.show_debug = True
                self._update_debug_visibility()

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
        """Lade Debug-Layer mit Junctions und Centerlines (einmalig)."""
        print("  [Debug] Lade Debug-Layer...")

        # Lade Junction-Daten aus cache/debug_network.json (lokales Project-Verzeichnis)
        debug_network_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "debug_network.json")

        if not os.path.exists(debug_network_path):
            print(f"  [Debug] Keine Junction-Daten gefunden: {debug_network_path}")
            return

        try:
            with open(debug_network_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)
        except Exception as e:
            print(f"  [!] Fehler beim Laden der Debug-Daten: {e}")
            return

        junctions = debug_data.get("junctions", [])
        roads = debug_data.get("roads", [])

        print(f"  [Debug] Lade {len(junctions)} Junctions, {len(roads)} Roads")

        # === JUNCTIONS: Kombiniere alle Spheres in ein MultiBlock ===
        junction_blocks = pv.MultiBlock()
        label_positions = []
        label_texts = []

        for j_idx, junction in enumerate(junctions):
            pos = junction.get("position")
            if not pos or len(pos) < 3:
                continue

            # Validiere Position (keine NaN/Inf)
            try:
                if not all(isinstance(v, (int, float)) and not (abs(v) == float("inf") or v != v) for v in pos):
                    print(f"  [!] Junction[{j_idx}]: ungültige Position {pos}, überspringe")
                    continue
            except:
                print(f"  [!] Junction[{j_idx}]: Position-Validierung fehlgeschlagen, überspringe")
                continue

            # Füge Sphere zum MultiBlock hinzu
            sphere = pv.Sphere(radius=2.0, center=pos)
            junction_blocks.append(sphere)

            # Sammle Label-Position und Text
            label_positions.append([pos[0], pos[1], pos[2] + 3.0])
            label_texts.append(str(j_idx))

        # Rendere alle Junctions als EINEN Actor
        if len(junction_blocks) > 0:
            junction_color = self.grid_colors.get("junction", {}).get("color", [0.0, 0.0, 1.0])
            junction_opacity = self.grid_colors.get("junction", {}).get("opacity", 0.5)
            actor = self.plotter.add_mesh(
                junction_blocks, color=junction_color, opacity=junction_opacity, label="Junctions"
            )
            self.debug_actors.append(actor)

        # Rendere alle Labels als EINEN Actor
        if len(label_positions) > 0:
            try:
                actor = self.plotter.add_point_labels(
                    label_positions,
                    label_texts,
                    point_size=0,
                    font_size=12,
                    text_color="blue",
                    shape_opacity=0.0,
                )
                self.debug_actors.append(actor)
            except Exception as e:
                print(f"  [!] Fehler beim Rendern der Junction-Labels: {e}")

        # === CENTERLINES: Kombiniere alle Linien in ein PolyData ===
        all_points = []
        all_lines = []
        point_offset = 0
        road_label_positions = []
        road_label_texts = []

        for road in roads:
            coords = road.get("coords", [])
            if len(coords) < 2:
                continue

            coords_array = np.array(coords)
            n = len(coords_array)

            # Füge Punkte hinzu
            all_points.extend(coords_array)

            # Erstelle Line-Connectivity
            for i in range(n - 1):
                all_lines.append([2, point_offset + i, point_offset + i + 1])

            point_offset += n

            # Berechne Mittelpunkt der Centerline für Label
            mid_idx = n // 2
            mid_point = coords_array[mid_idx]
            road_label_positions.append([mid_point[0], mid_point[1], mid_point[2] + 2.0])  # 2m über Straße
            # Sichere Konvertierung der road_id zu String
            road_id = road.get("road_id")
            road_label_texts.append(str(road_id) if road_id is not None else "?")

        # Rendere alle Centerlines als EINEN Actor
        if len(all_points) > 0:
            all_points = np.array(all_points)
            all_lines = np.array(all_lines)

            centerline_color = self.grid_colors.get("centerline", {}).get("color", [0.0, 0.0, 1.0])
            centerline_width = self.grid_colors.get("centerline", {}).get("line_width", 2.0)
            centerline_opacity = self.grid_colors.get("centerline", {}).get("opacity", 1.0)

            centerlines_mesh = pv.PolyData(all_points, lines=all_lines)
            actor = self.plotter.add_mesh(
                centerlines_mesh,
                color=centerline_color,
                line_width=centerline_width,
                opacity=centerline_opacity,
                label="Centerlines",
            )
            self.debug_actors.append(actor)

        # Rendere Road-Labels
        if len(road_label_positions) > 0:
            try:
                actor = self.plotter.add_point_labels(
                    road_label_positions,
                    road_label_texts,
                    point_size=0,
                    font_size=10,
                    text_color="black",
                    shape_opacity=0.0,
                )
                self.debug_actors.append(actor)
            except Exception as e:
                print(f"  [!] Fehler beim Rendern der Road-Labels: {e}")

        print(
            f"  [Debug] {len(self.debug_actors)} Debug-Actors gerendert ({len(junctions)} Junctions, {len(roads)} Centerlines)"
        )

        # === BOUNDARY-POLYGONE aus debug_network.json ===
        boundary_polygons = debug_data.get("boundary_polygons", [])

        if boundary_polygons:
            print(f"  [Debug] Lade {len(boundary_polygons)} Boundary-Polygone...")

            # Kombiniere alle Boundary-Polygone in ein PolyData (wie Centerlines)
            all_boundary_points = []
            all_boundary_lines = []
            boundary_point_offset = 0

            for boundary in boundary_polygons:
                poly_type = boundary.get("type", "unknown")
                coords = boundary.get("coords", [])

                if len(coords) < 2:
                    continue

                coords_array = np.array(coords)
                n = len(coords_array)

                # Füge Punkte hinzu
                all_boundary_points.extend(coords_array)

                # Erstelle Line-Connectivity
                for i in range(n - 1):
                    all_boundary_lines.append([2, boundary_point_offset + i, boundary_point_offset + i + 1])

                # Schließe Polygon wenn es kein search_circle ist
                if poly_type != "search_circle":
                    all_boundary_lines.append([2, boundary_point_offset + n - 1, boundary_point_offset])

                boundary_point_offset += n

            # Rendere alle Boundary-Polygone als EINEN Actor
            if len(all_boundary_points) > 0:
                try:
                    all_boundary_points = np.array(all_boundary_points)
                    all_boundary_lines = np.array(all_boundary_lines)

                    boundary_color = self.grid_colors.get("boundary", {}).get("color", [1.0, 0.0, 1.0])
                    boundary_width = self.grid_colors.get("boundary", {}).get("line_width", 2.0)

                    boundary_mesh = pv.PolyData(all_boundary_points, lines=all_boundary_lines)
                    actor = self.plotter.add_mesh(
                        boundary_mesh,
                        color=boundary_color,
                        line_width=boundary_width,
                        opacity=1.0,
                        label="Boundaries",
                        render_lines_as_tubes=False,
                    )
                    self.debug_actors.append(actor)
                    print(f"  [Debug] Boundary-Polygone kombiniert gerendert (1 Actor)")
                except Exception as e:
                    print(f"  [!] Fehler beim Rendern der Boundary-Polygone: {e}")

        # === COMPONENT-LINIEN aus debug_network.json ===
        component_lines = debug_data.get("component_lines", [])

        if component_lines:
            print(f"  [Debug] Lade {len(component_lines)} Component-Linien...")

            # Gruppiere Components nach Typ (terrain/road)
            terrain_components = []
            road_components = []

            for component in component_lines:
                label = component.get("label", "component")
                if "road" in label:
                    road_components.append(component)
                else:
                    terrain_components.append(component)

            # Rendere Terrain-Components (grün)
            if terrain_components:
                all_points = []
                all_lines = []
                point_offset = 0

                for component in terrain_components:
                    coords = component.get("coords", [])
                    if len(coords) < 2:
                        continue

                    coords_array = np.array(coords)
                    n = len(coords_array)
                    all_points.extend(coords_array)

                    for i in range(n - 1):
                        all_lines.append([2, point_offset + i, point_offset + i + 1])
                    point_offset += n

                if len(all_points) > 0:
                    try:
                        all_points = np.array(all_points)
                        all_lines = np.array(all_lines)

                        terrain_color = self.grid_colors.get("component_terrain", {}).get("color", [0.2, 0.8, 0.2])
                        terrain_width = self.grid_colors.get("component_terrain", {}).get("line_width", 3.0)

                        terrain_mesh = pv.PolyData(all_points, lines=all_lines)
                        actor = self.plotter.add_mesh(
                            terrain_mesh,
                            color=terrain_color,
                            line_width=terrain_width,
                            opacity=1.0,
                            label="Terrain Component Lines",
                            render_lines_as_tubes=False,
                        )
                        self.debug_actors.append(actor)
                        print(f"  [Debug] {len(terrain_components)} Terrain-Component-Linien (grün)")
                    except Exception as e:
                        print(f"  [!] Fehler beim Rendern der Terrain-Component-Linien: {e}")

            # Rendere Road-Components (rot)
            if road_components:
                all_points = []
                all_lines = []
                point_offset = 0

                for component in road_components:
                    coords = component.get("coords", [])
                    if len(coords) < 2:
                        continue

                    coords_array = np.array(coords)
                    n = len(coords_array)
                    all_points.extend(coords_array)

                    for i in range(n - 1):
                        all_lines.append([2, point_offset + i, point_offset + i + 1])
                    point_offset += n

                if len(all_points) > 0:
                    try:
                        all_points = np.array(all_points)
                        all_lines = np.array(all_lines)

                        road_color = self.grid_colors.get("component_road", {}).get("color", [0.8, 0.2, 0.2])
                        road_width = self.grid_colors.get("component_road", {}).get("line_width", 3.0)

                        road_mesh = pv.PolyData(all_points, lines=all_lines)
                        actor = self.plotter.add_mesh(
                            road_mesh,
                            color=road_color,
                            line_width=road_width,
                            opacity=1.0,
                            label="Road Component Lines",
                            render_lines_as_tubes=False,
                        )
                        self.debug_actors.append(actor)
                        print(f"  [Debug] {len(road_components)} Road-Component-Linien (rot)")
                    except Exception as e:
                        print(f"  [!] Fehler beim Rendern der Road-Component-Linien: {e}")

    def show(self):
        """Zeige das Viewer-Fenster."""
        self.plotter.show()


if __name__ == "__main__":
    viewer = DAETileViewer()
    if hasattr(viewer, "plotter") and viewer.plotter is not None:
        viewer.show()
    else:
        print("[!] Kein Plotter initialisiert (vermutlich keine DAE-Dateien geladen).")
