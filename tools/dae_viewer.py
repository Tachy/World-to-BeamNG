"""
DAE Viewer - Visualisiere die exportierte terrain.dae mit allen Tiles

Steuerung:
    G = Alle Tiles anzeigen (Merge-View)
    M = Nur Material Terrain anzeigen
    R = Nur Material Road anzeigen
    T = Toggle Alle Materials    X = Toggle Texturen (An/Aus)    Space = Tile-Liste anzeigen/verstecken
    Maus:
        Rechtsklick-Drag = Kamera drehen
        Scroll = Zoom
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


class DAETileViewer:
    def __init__(self):
        self.dae_path = os.path.join(config.BEAMNG_DIR_SHAPES, "terrain.dae")

        if not os.path.exists(self.dae_path):
            print(f"Fehler: {self.dae_path} nicht gefunden!")
            return

        print(f"Lade DAE von: {self.dae_path}")
        self.tile_data = load_dae_tile(self.dae_path)

        if not self.tile_data:
            print("Keine Geometrie in DAE gefunden!")
            return

        # Sichtbarkeits-Flags
        self.show_terrain = True
        self.show_roads = True
        self.use_textures = True  # Texturen standardmäßig an
        self.show_debug = False  # Debug-Layer (Junctions, Centerlines)

        # Speichere Actor-Referenzen für Sichtbarkeits-Toggles
        self.terrain_actors = []  # Liste von Terrain-Mesh-Actors
        self.road_actors = []  # Liste von Road-Mesh-Actors
        self.debug_actors = []  # Liste von Debug-Actors (Junctions, Centerlines)
        self.debug_loaded = False  # Flag: Debug-Layer bereits geladen?

        # Lade Texturen
        self.textures_dir = os.path.join(config.BEAMNG_DIR_SHAPES, "textures")
        self.textures = self._load_textures()

        if self.textures:
            print(f"  -> {len(self.textures)} Texturen geladen")

        # Status-Actors
        self._reload_actor = None
        self._camera_status_actor = None
        self._active_layers_actor = None
        self._render_update_counter = 0  # Für RenderEvent Drosselung

        # PyVista Setup
        self.plotter = pv.Plotter()
        self.plotter.set_background("skyblue")  # Himmelblau
        self.config_path = os.path.join(os.path.dirname(__file__), "dae_viewer.cfg")

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
        print("  M = Nur Terrain")
        print("  R = Nur Roads")
        print("  T = Toggle Materials")
        print("  X = Toggle Texturen")
        print("  D = Toggle Debug-Layer (Junctions, Centerlines)")
        print("  K = Kamera laden | Shift+K = Kamera speichern")
        print("  L = Reload DAE")
        print("  Up/Down = Zoom ändern")

        self.update_view()
        self._apply_saved_camera_state()

    def _on_key_press(self, obj, event):
        """KeyPress Event Handler."""
        key = obj.GetKeySym()
        key_lower = key.lower()

        if key_lower == "m":
            self.show_terrain = True
            self.show_roads = False
            print("\n[Material] Nur Terrain")
            self._update_visibility()

        elif key_lower == "r":
            self.show_terrain = False
            self.show_roads = True
            print("\n[Material] Nur Roads")
            self._update_visibility()

        elif key_lower == "t":
            self.show_terrain = not self.show_terrain
            self.show_roads = not self.show_roads
            print(
                f"\n[Material] Terrain: {'AN' if self.show_terrain else 'AUS'}, Roads: {'AN' if self.show_roads else 'AUS'}"
            )
            self._update_visibility()

        elif key_lower == "x":
            self.use_textures = not self.use_textures
            print(f"\n[Texturen] {'AN' if self.use_textures else 'AUS'}")
            self.update_view()

        elif key_lower == "d":
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
        """Aktualisiere Sichtbarkeit der Terrain/Road Actors ohne Reload."""
        for actor in self.terrain_actors:
            actor.SetVisibility(self.show_terrain)
        for actor in self.road_actors:
            actor.SetVisibility(self.show_roads)

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

    def update_view(self):
        """Aktualisiere 3D-View."""
        # Speichere Kamera vor clear()
        camera_pos = None
        camera_focal = None
        camera_up = None
        try:
            cam = self.plotter.camera
            if cam is not None:
                camera_pos = list(cam.position)
                camera_focal = list(cam.focal_point)
                camera_up = list(cam.up)
        except Exception as e:
            print(f"[!] Fehler beim Speichern der Kamera vor update_view: {e}")

        self.plotter.clear()

        # Leere Actor-Listen
        self.terrain_actors = []
        self.road_actors = []
        self.debug_actors = []

        vertices = self.tile_data.get("vertices", [])
        faces = self.tile_data.get("faces", [])
        materials = self.tile_data.get("materials", [])
        tiles_info = self.tile_data.get("tiles", {})  # {tile_name: {"faces": [...], "uvs": [...]}}

        if len(vertices) == 0:
            print("Keine Vertices gefunden!")
            self.plotter.show()
            return

        # Farben und Edge-Farben pro Material
        face_colors = {
            "terrain": [0.8, 0.95, 0.8],  # Hellgrün
            "road": [0.9, 0.9, 0.9],  # Hellgrau
        }
        edge_colors = {
            "terrain": [0.2, 0.5, 0.2],  # Dunkelgrün
            "road": [1.0, 0.0, 0.0],  # Rot
        }

        # Debug: Zeige welche Materialien vorhanden sind
        unique_materials = set(materials) if materials else set()
        if unique_materials:
            print(f"  Materialien im File: {', '.join(sorted(unique_materials))}")
        else:
            print("  [!] Keine Material-Informationen gefunden")

        # Wenn Texturen aktiviert sind, rendere pro Tile mit Textur
        if self.use_textures and tiles_info:
            for tile_name, tile_info in tiles_info.items():
                tile_vertices_local = tile_info.get("vertices", [])
                tile_faces_local = tile_info.get("faces_local", [])
                tile_uvs = tile_info.get("uvs", [])

                if len(tile_faces_local) == 0 or len(tile_vertices_local) == 0:
                    continue

                # tile_vertices_local ist jetzt bereits NumPy Array
                if not isinstance(tile_vertices_local, np.ndarray):
                    tile_vertices_local = np.array(tile_vertices_local)

                # Erstelle Mesh für dieses Tile mit lokalen Vertices
                mesh = self._create_mesh_with_uvs(tile_vertices_local, tile_faces_local, tile_uvs)

                # Lade Textur für dieses Tile
                texture = self.textures.get(tile_name)

                # Nur Textur anwenden, wenn UVs vorhanden sind
                if texture is not None and len(tile_uvs) > 0:
                    try:
                        self.plotter.add_mesh(mesh, texture=texture, opacity=1.0, label=tile_name)
                    except Exception as e:
                        # Fallback bei Textur-Fehler
                        print(f"  [!] Textur-Fehler für {tile_name}: {e}")
                        self.plotter.add_mesh(mesh, color=[0.6, 0.5, 0.4], opacity=0.8, label=tile_name)
                else:
                    # Fallback: Farbe
                    self.plotter.add_mesh(mesh, color=[0.6, 0.5, 0.4], opacity=0.8, label=tile_name)
        else:
            # Material-basiertes Rendering mit Faces und Edges
            terrain_faces = []
            road_faces = []

            for face_idx, material in enumerate(materials):
                # Erkenne Material-Typ aus Namen: "road" oder "terrain"
                if "road" in material.lower():
                    road_faces.append(faces[face_idx])
                elif "terrain" in material.lower():
                    terrain_faces.append(faces[face_idx])
                else:
                    # Fallback: alles andere ist Terrain
                    terrain_faces.append(faces[face_idx])

            print(f"  Faces: {len(terrain_faces)} Terrain, {len(road_faces)} Roads")

            # Terrain Mesh (wie im mesh_viewer: einfach show_edges=True)
            if terrain_faces:
                terrain_mesh = self._create_mesh(vertices, terrain_faces)
                actor = self.plotter.add_mesh(
                    terrain_mesh,
                    color=face_colors.get("terrain", [0.8, 0.95, 0.8]),
                    label="Terrain",
                    opacity=0.9,
                    show_edges=True,
                    edge_color=edge_colors.get("terrain", [0.2, 0.5, 0.2]),
                    line_width=1.0,
                )
                self.terrain_actors.append(actor)
                actor.SetVisibility(self.show_terrain)

            # Road Mesh (wie im mesh_viewer: einfach show_edges=True)
            if road_faces:
                road_mesh = self._create_mesh(vertices, road_faces)
                actor = self.plotter.add_mesh(
                    road_mesh,
                    color=face_colors.get("road", [0.9, 0.9, 0.9]),
                    label="Roads",
                    opacity=0.9,
                    show_edges=True,
                    edge_color=edge_colors.get("road", [1.0, 0.0, 0.0]),
                    line_width=2.0,
                )
                self.road_actors.append(actor)
                actor.SetVisibility(self.show_roads)

        # Statuszeilen
        # Oben links: Bedienungsanleitung
        bedienung = "M/R/T: Materials | X: Texturen | D: Debug | K: Cam | L: Reload | Up/Down: Zoom"
        self.plotter.add_text(
            bedienung,
            position="upper_left",
            font_size=10,
        )

        # Oben rechts: Aktive Layer
        self._update_active_layers_text()

        self.plotter.view_xy()
        self._update_camera_status()

        # Stelle Kamera wieder her
        if camera_pos is not None:
            try:
                cam = self.plotter.camera
                cam.position = camera_pos
                cam.focal_point = camera_focal
                cam.up = camera_up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
            except Exception as e:
                print(f"[!] Fehler beim Wiederherstellen der Kamera: {e}")

    def _create_mesh(self, vertices, faces):
        """Erstelle ein PyVista PolyData Mesh aus Vertices und Faces."""
        # PyVista erwartet: [num_points_in_face, pt0, pt1, pt2, ...]
        pyvista_faces = []
        for face in faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(vertices, pyvista_faces)
        return mesh

    def _create_mesh_with_uvs(self, vertices, faces, uvs):
        """Erstelle ein PyVista PolyData Mesh mit UV-Koordinaten."""
        pyvista_faces = []
        for face in faces:
            pyvista_faces.extend([3, face[0], face[1], face[2]])

        mesh = pv.PolyData(vertices, pyvista_faces)

        # Füge UV-Koordinaten hinzu (als texture coordinates)
        # uvs ist jetzt NumPy Array - prüfe Länge statt Wahrheitswert
        if len(uvs) > 0 and len(uvs) == len(vertices):
            # PyVista erwartet 2D UV coords als "Texture Coordinates"
            uv_array = np.array(uvs) if not isinstance(uvs, np.ndarray) else uvs
            mesh.active_texture_coordinates = uv_array

        return mesh

    def _load_textures(self):
        """Lade alle Tile-Texturen aus dem textures-Verzeichnis."""
        textures = {}

        if not os.path.exists(self.textures_dir):
            print(f"  [!] Textures-Verzeichnis nicht gefunden: {self.textures_dir}")
            return textures

        # Finde alle tile_*.jpg Dateien
        texture_files = list(Path(self.textures_dir).glob("tile_*.jpg"))

        for texture_path in texture_files:
            tile_name = texture_path.stem  # z.B. "tile_0_0"

            try:
                # Lade Bild mit PIL
                img = Image.open(texture_path)

                # Konvertiere zu RGB array
                img_array = np.array(img)

                # PyVista Texture erwartet RGB im 0-255 Bereich
                textures[tile_name] = pv.Texture(img_array)

            except Exception as e:
                print(f"  [!] Fehler beim Laden von {texture_path.name}: {e}")

        return textures

    def _update_active_layers_text(self):
        """Aktualisiere Aktive-Layer-Text oben rechts."""
        active_items = []
        if self.show_terrain:
            active_items.append("T")
        if self.show_roads:
            active_items.append("R")
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
        """Lade DAE neu (L-Taste)."""
        self._show_reload_overlay()
        try:
            print(f"  [Reload] {self.dae_path}")

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

            # Lade DAE neu
            from tools.dae_loader import load_dae_tile

            self.tile_data = load_dae_tile(self.dae_path)

            # Lade Texturen neu
            self.textures = self._load_textures()

            print(f"  ✓ Neu geladen")

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

            # Füge Sphere zum MultiBlock hinzu
            sphere = pv.Sphere(radius=2.0, center=pos)
            junction_blocks.append(sphere)

            # Sammle Label-Position und Text
            label_positions.append([pos[0], pos[1], pos[2] + 3.0])
            label_texts.append(str(j_idx))

        # Rendere alle Junctions als EINEN Actor
        if len(junction_blocks) > 0:
            actor = self.plotter.add_mesh(junction_blocks, color="blue", opacity=0.8, label="Junctions")
            self.debug_actors.append(actor)

        # Rendere alle Labels als EINEN Actor
        if len(label_positions) > 0:
            actor = self.plotter.add_point_labels(
                label_positions,
                label_texts,
                point_size=0,
                font_size=12,
                text_color="blue",
                shape_opacity=0.0,
            )
            self.debug_actors.append(actor)

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
            road_label_texts.append(str(road.get("road_id", "?")))

        # Rendere alle Centerlines als EINEN Actor
        if len(all_points) > 0:
            all_points = np.array(all_points)
            all_lines = np.array(all_lines)

            centerlines_mesh = pv.PolyData(all_points, lines=all_lines)
            actor = self.plotter.add_mesh(
                centerlines_mesh,
                color="blue",
                line_width=2.0,
                opacity=0.6,
                label="Centerlines",
            )
            self.debug_actors.append(actor)

        # Rendere Road-Labels
        if len(road_label_positions) > 0:
            actor = self.plotter.add_point_labels(
                road_label_positions,
                road_label_texts,
                point_size=0,
                font_size=10,
                text_color="black",
                shape_opacity=0.0,
            )
            self.debug_actors.append(actor)

        print(
            f"  [Debug] {len(self.debug_actors)} Debug-Actors gerendert ({len(junctions)} Junctions, {len(roads)} Centerlines)"
        )

        # === Boundary-Polygone (lokales Stitching): lade boundary_polygons_local.obj falls vorhanden ===
        boundary_polys_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cache", "boundary_polygons_local.obj"
        )
        if os.path.exists(boundary_polys_path):
            try:
                print(f"  [Debug] Lade Boundary-Polygone aus {boundary_polys_path}")
                boundary_mesh = pv.read(boundary_polys_path)

                # Rendere als magenta Linien
                actor = self.plotter.add_mesh(
                    boundary_mesh,
                    color="magenta",
                    line_width=3.0,
                    opacity=1.0,
                    label="Boundary-Polygone (lokal)",
                    render_lines_as_tubes=False,
                )
                self.debug_actors.append(actor)
                print(f"  [Debug] Boundary-Polygone geladen")
            except Exception as e:
                print(f"  [!] Fehler beim Laden der Boundary-Polygone: {e}")

    def show(self):
        """Zeige das Viewer-Fenster."""
        self.plotter.show()


if __name__ == "__main__":
    viewer = DAETileViewer()
    viewer.show()
