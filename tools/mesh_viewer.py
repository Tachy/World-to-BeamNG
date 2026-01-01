"""
Mesh Viewer - Visualisiere Gelände, Straßen und Debug-Ebenen
Steuerung:
    1-6 = Toggle Debug-Ebenen
    7 = Toggle Terrain
    8 = Toggle Roads
    9 = Toggle Slopes
    K/Shift+K = Kamera laden/speichern
    L = Reload OBJ

    Maus-Steuerung:
        Rechtsklick-Drag = Kamera drehen
        Scroll = Zoom
        Mittelklick-Drag = Verschieben
        F = Fly-To Mode (VTK Standard)
"""

import json
import pyvista as pv
import numpy as np
import os
import sys

# Füge Parent-Verzeichnis zum Path hinzu für world_to_beamng import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Konfigurationswerte für Suchkreise/Sampling
try:
    from world_to_beamng import config

    _CENTERLINE_SEARCH_RADIUS = config.CENTERLINE_SEARCH_RADIUS
    _CENTERLINE_SAMPLE_SPACING = config.CENTERLINE_SAMPLE_SPACING
except Exception:
    _CENTERLINE_SEARCH_RADIUS = 10.0
    _CENTERLINE_SAMPLE_SPACING = 10.0


class MeshViewer:
    def __init__(
        self,
        obj_file="beamng.obj",
        fallback_obj="roads.obj",
        label_distance_threshold=1e9,
        label_max_points=50000000,
        dolly_step=10.0,
        show_point_labels=False,
        label_click_px_tol=30.0,
    ):
        # Wechsle zum Parent-Verzeichnis (wo beamng.obj liegt)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(self.base_dir)

        print(f"Lade OBJ: {obj_file}")
        (
            self.vertices,
            self.road_faces,
            self.slope_faces,
            self.terrain_faces,
            self.junction_faces,
            self.road_face_to_idx,
        ) = self._parse_obj(obj_file, fallback_obj)

        # Lade Debug-Ebenen (ebene1.obj bis ebene9.obj)
        self.debug_layers = {}
        self.debug_layer_visible = {}
        self._load_debug_layers()

        if len(self.vertices) > 0:
            print(
                f"  Mesh Bounds: X=[{self.vertices[:, 0].min():.1f}, {self.vertices[:, 0].max():.1f}], "
                f"Y=[{self.vertices[:, 1].min():.1f}, {self.vertices[:, 1].max():.1f}], "
                f"Z=[{self.vertices[:, 2].min():.1f}, {self.vertices[:, 2].max():.1f}]"
            )

        self.show_roads = True
        self.show_slopes = False
        self.show_terrain = False
        self.label_distance_threshold = label_distance_threshold
        self.label_max_points = label_max_points
        self.show_point_labels = show_point_labels
        self.current_label_points = None
        self.label_click_px_tol = label_click_px_tol
        self.obj_file = obj_file
        self.config_path = os.path.join(self.base_dir, "tools", "mesh_viewer.cfg")

        print(
            f"Gefunden: {len(self.vertices)} Vertices, {len(self.road_faces)} Road Faces, "
            f"{len(self.slope_faces)} Slope Faces, {len(self.terrain_faces)} Terrain Faces, "
            f"{len(self.junction_faces)} Junction Faces"
        )
        print("\nSteuerung:")
        print("  1-6 = Toggle Debug-Ebenen")
        print("  7 = Toggle Terrain")
        print("  8 = Toggle Roads (Strassen)")
        print("  9 = Toggle Slopes (Boeschungen)")
        print("  K = Kamera laden | Shift+K = Kamera speichern")
        print("  L = Reload OBJ")
        print("\n  Maus-Steuerung:")
        print("    Rechtsklick-Drag = Kamera drehen")
        print("    Scroll = Zoom")
        print("    Mittelklick-Drag = Verschieben")
        print("    F = Fly-To Mode (VTK Standard)")

        self.plotter = pv.Plotter()
        self._reload_actor = None
        self._camera_status_actor = None
        self._active_layers_actor = None
        self._render_update_counter = 0  # Für RenderEvent Drosslung
        self._debug_layer_actors = {}  # Speichere Debug-Layer Actors für Performance
        self._terrain_actors = []  # Speichere Terrain Actors
        self._roads_actors = []  # Speichere Roads Actors
        self._slopes_actors = []  # Speichere Slopes Actors
        # Stelle zuletzt gespeicherte Fensterposition/-größe wieder her (falls vorhanden)
        self._apply_saved_window_state()

        # Verwende VTK Standard Interactor (Trackball-Modus, Zoom, etc.)

        # Mouse-Observer für Label-Klick (Bildschirm-Koordinaten)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self.on_left_click)

        # KeyPressEvent-Observer für unsere Custom Keys (1-6, 7-9, K, L)
        self.plotter.iren.add_observer("KeyPressEvent", self._on_key_press)
        self.plotter.iren.add_observer("ExitEvent", self._on_close_save_window_state)

        # Observer für Kamera-Änderungen (nur bei expliziten Aktionen, nicht bei Maus-Move)
        self.plotter.iren.add_observer("ScrollEvent", self._on_camera_change)
        self.plotter.iren.add_observer("EndInteractionEvent", self._on_camera_change)
        self.plotter.iren.add_observer("InteractionEvent", self._on_camera_change)
        # RenderEvent für Fly-To und andere kontinuierliche Kamera-Änderungen (mit Drosslung)
        self.plotter.iren.add_observer("RenderEvent", self._on_render_event)

        self.update_view()
        self._apply_saved_camera_state()

    def _load_debug_layers(self):
        """Suche und lade Debug-Ebenen (ebene1.obj bis ebene9.obj)"""
        for i in range(1, 10):
            layer_file = f"ebene{i}.obj"
            if os.path.exists(layer_file):
                print(f"Lade Debug-Ebene: {layer_file}")
                try:
                    vertices, edges, points, point_labels = self._load_line_obj(
                        layer_file
                    )
                    if len(vertices) > 0:
                        self.debug_layers[f"ebene{i}"] = {
                            "vertices": vertices,
                            "edges": edges,
                            "points": points,
                            "point_labels": point_labels,
                            "file": layer_file,
                        }
                        self.debug_layer_visible[f"ebene{i}"] = (
                            True  # Standardmäßig sichtbar
                        )
                        print(
                            f"  -> {len(vertices)} vertices, {len(edges)} edges, {len(points)} points"
                        )
                except Exception as e:
                    print(f"  Fehler beim Laden von {layer_file}: {e}")

    def _reload_debug_layers(self):
        """Lade Debug-Layer neu und erhalte Sichtbarkeits-Status nach Namen."""
        previous_visibility = dict(self.debug_layer_visible)
        self.debug_layers = {}
        self.debug_layer_visible = {}
        self._load_debug_layers()
        for name, vis in previous_visibility.items():
            if name in self.debug_layer_visible:
                self.debug_layer_visible[name] = vis

    def _load_line_obj(self, obj_file):
        """Lade OBJ-Datei mit Linien und optionalen Punkten (p-Statements)."""
        vertices = []
        edges = []
        points = []
        point_labels = []

        try:
            with open(obj_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        vertices.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )
                    elif line.startswith("l "):
                        # Linie: "l v1 v2 ..." oder "l v1 v2"
                        parts = line.split("#", 1)[0].strip().split()[1:]
                        indices = [int(p.split("/")[0]) - 1 for p in parts]
                        for i in range(len(indices) - 1):
                            edges.append([indices[i], indices[i + 1]])
                    elif line.startswith("p "):
                        # Punkte: "p v1" (optional Kommentar: # Junction X)
                        payload, *comment = line.split("#", 1)
                        parts = payload.strip().split()[1:]
                        indices = [int(p.split("/")[0]) - 1 for p in parts]
                        points.extend(indices)

                        label = None
                        if comment:
                            text = comment[0].strip().lower()
                            if text.startswith("junction"):
                                try:
                                    label = int(text.split()[1])
                                except Exception:
                                    label = None
                        # Falls mehrere Punkte in einer Zeile: gleiche Label-Reihenfolge
                        for _ in indices:
                            point_labels.append(label)
                    elif line.startswith("f "):
                        # Falls Faces vorhanden: überspringe fürs Erste
                        pass

            return (
                np.array(vertices) if vertices else np.array([]),
                edges,
                points,
                point_labels,
            )
        except FileNotFoundError:
            return np.array([]), [], [], []

    def toggle_debug_layer(self, layer_num):
        """Toggle Sichtbarkeit einer Debug-Ebene ohne kompletten Rebuild."""
        layer_name = f"ebene{layer_num}"
        if layer_name not in self.debug_layer_visible:
            print(f"\nDebug-Ebene {layer_name} nicht gefunden")
            return

        # Toggle Sichtbarkeit
        self.debug_layer_visible[layer_name] = not self.debug_layer_visible[layer_name]
        state = "AN" if self.debug_layer_visible[layer_name] else "AUS"
        print(f"\nDebug-Ebene {layer_name}: {state}")

        # Stelle sicher, dass die Actors für diesen Layer existieren (lazy build)
        if layer_name not in self._debug_layer_actors:
            self._build_debug_layer_actors(layer_name)

        # Visibility schalten, auch wenn Build nichts ergeben hat (kein Rebuild mehr)
        actors = self._debug_layer_actors.get(layer_name, [])
        visible = self.debug_layer_visible[layer_name]
        for actor in actors:
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass
        self._update_active_layers_text()
        self._update_camera_status()
        try:
            self.plotter.render()
        except Exception:
            pass

    def _get_layer_color(self, layer_name):
        layer_colors = {
            "ebene1": "red",
            "ebene2": "blue",
            "ebene3": "yellow",
            "ebene4": "cyan",
            "ebene5": "magenta",
            "ebene6": "orange",
            "ebene7": "purple",
            "ebene8": "pink",
            "ebene9": "brown",
        }
        return layer_colors.get(layer_name, "white")

    def _build_debug_layer_actors(self, layer_name):
        """Erzeuge Actors für einen Debug-Layer einmalig, ohne komplette Szene zu löschen."""
        data = self.debug_layers.get(layer_name)
        if not data:
            return

        vertices = data.get("vertices", [])
        edges = data.get("edges", [])
        points = data.get("points", [])
        point_labels = data.get("point_labels", [])

        if len(vertices) == 0:
            return

        color = self._get_layer_color(layer_name)
        actors_for_layer = []
        visible = self.debug_layer_visible.get(layer_name, False)

        if len(edges) > 0:
            layer_mesh = pv.PolyData()
            layer_mesh.points = vertices
            try:
                layer_mesh.lines = np.hstack([[2, edge[0], edge[1]] for edge in edges])
            except Exception:
                layer_mesh.lines = None

            actor = self.plotter.add_mesh(
                layer_mesh,
                color=color,
                line_width=3,
                label=layer_name,
            )
            actor.SetVisibility(visible)
            actors_for_layer.append(actor)

        if len(points) > 0:
            pts = vertices[points]
            actor = self.plotter.add_points(
                pts,
                color=color,
                point_size=12,
                render_points_as_spheres=True,
            )
            actor.SetVisibility(visible)
            actors_for_layer.append(actor)

            if (
                point_labels
                and len(point_labels) == len(points)
                and len(point_labels) <= 2000
            ):
                labels = [str(lbl) if lbl is not None else "" for lbl in point_labels]
                actor = self.plotter.add_point_labels(
                    pts,
                    labels,
                    point_size=0,
                    font_size=10,
                    text_color=color,
                    shape_color="white",
                    shape_opacity=0.6,
                )
                actor.SetVisibility(visible)
                actors_for_layer.append(actor)

        if actors_for_layer:
            self._debug_layer_actors[layer_name] = actors_for_layer

    def _update_active_layers_text(self):
        """Aktualisiere den Status-Text oben rechts (aktive Ebenen)."""
        active_items = []
        if self.show_terrain:
            active_items.append("T")
        if self.show_slopes:
            active_items.append("S")
        if self.show_roads:
            active_items.append("R")

        for i in range(1, 7):
            layer_name = f"ebene{i}"
            if self.debug_layer_visible.get(layer_name, False):
                active_items.append(str(i))

        active_text = " ".join(active_items) if active_items else "-"

        try:
            self.plotter.remove_actor("active_layers_text")
        except Exception:
            pass

        try:
            self._active_layers_actor = self.plotter.add_text(
                active_text,
                position="upper_right",
                font_size=10,
                name="active_layers_text",
            )
        except Exception:
            self._active_layers_actor = None

    def _parse_obj(self, obj_file, fallback_obj="roads.obj"):
        """Parse OBJ file and extract vertices, faces, etc. (optimized read)."""

        def parse_file(path):
            vertices = []
            road_faces = []
            slope_faces = []
            terrain_faces = []
            junction_faces = []
            road_face_to_idx = []
            current_material = None
            current_road_idx = None

            with open(path, "r", encoding="utf-8", buffering=32 * 1024 * 1024) as f:
                for line in f:
                    if not line:
                        continue
                    prefix = line[0]
                    if prefix == "v" and line.startswith("v "):
                        parts = line.split()
                        vertices.append(
                            (float(parts[1]), float(parts[2]), float(parts[3]))
                        )
                        continue

                    if prefix == "u" and line.startswith("usemtl"):
                        current_material = line.split()[1].lower()
                        current_road_idx = None
                        tokens = current_material.replace("-", "_").split("_")
                        for t in tokens:
                            if t.isdigit():
                                current_road_idx = int(t)
                                break
                        continue

                    if prefix != "f" or not line.startswith("f "):
                        continue

                    parts = line.split()[1:]
                    face = [int(p.split("/")[0]) - 1 for p in parts]

                    if current_material:
                        if "terrain" in current_material:
                            terrain_faces.append(face)
                        elif "slope" in current_material:
                            slope_faces.append(face)
                        elif (
                            "junction" in current_material or "quad" in current_material
                        ):
                            junction_faces.append(face)
                        elif "road" in current_material:
                            road_faces.append(face)
                            road_face_to_idx.append(current_road_idx)

            return (
                vertices,
                road_faces,
                slope_faces,
                terrain_faces,
                junction_faces,
                road_face_to_idx,
            )

        try:
            (
                vertices,
                road_faces,
                slope_faces,
                terrain_faces,
                junction_faces,
                road_face_to_idx,
            ) = parse_file(obj_file)
        except FileNotFoundError:
            print(
                f"  Warnung: {obj_file} nicht gefunden. Nutze Fallback {fallback_obj}"
            )
            if fallback_obj:
                (
                    vertices,
                    road_faces,
                    slope_faces,
                    terrain_faces,
                    junction_faces,
                    road_face_to_idx,
                ) = parse_file(fallback_obj)
            else:
                raise

        if len(terrain_faces) == 0:
            print("  Info: Keine Terrain-Faces gefunden (nutze ggf. terrain-only OBJ).")
        if len(slope_faces) == 0:
            print("  Info: Keine Slope-Faces gefunden.")
        if len(junction_faces) > 0:
            print(f"  Info: {len(junction_faces)} Junction-Quad-Faces gefunden!")

        # Konvertiere zu NumPy für spätere schnelle Remaps
        vertex_array = np.asarray(vertices, dtype=np.float32)
        road_faces = np.asarray(road_faces, dtype=np.int32)
        slope_faces = np.asarray(slope_faces, dtype=np.int32)
        terrain_faces = np.asarray(terrain_faces, dtype=np.int32)
        junction_faces = np.asarray(junction_faces, dtype=np.int32)
        road_face_to_idx = list(road_face_to_idx)

        return (
            vertex_array,
            road_faces.tolist(),
            slope_faces.tolist(),
            terrain_faces.tolist(),
            junction_faces.tolist(),
            road_face_to_idx,
        )

    def toggle_roads(self):
        self.show_roads = not self.show_roads
        state = "AN" if self.show_roads else "AUS"
        print(f"\nRoads (Strassen): {state}")
        self._update_active_layers_text()
        self._toggle_actors_visibility(self._roads_actors, self.show_roads)

    def toggle_slopes(self):
        self.show_slopes = not self.show_slopes
        state = "AN" if self.show_slopes else "AUS"
        print(f"\nBoeschungen: {state}")
        self._update_active_layers_text()
        self._toggle_actors_visibility(self._slopes_actors, self.show_slopes)

    def toggle_terrain(self):
        self.show_terrain = not self.show_terrain
        state = "AN" if self.show_terrain else "AUS"
        print(f"\nTerrain: {state}")
        self._update_active_layers_text()
        self._toggle_actors_visibility(self._terrain_actors, self.show_terrain)

    def _toggle_actors_visibility(self, actors, visible):
        """Setze Visibility für Liste von Actors (performant)"""
        for actor in actors:
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass

        # Immer rendern, auch wenn Liste leer (harmlos)
        try:
            self._update_camera_status()
            self.plotter.render()
        except Exception:
            pass

    def _on_camera_change(self, obj, event):
        """Update Statuszeile nach Kamera-Änderungen (Maus, Scroll, etc.)"""
        try:
            self._update_camera_status()
        except Exception:
            pass

    def _on_render_event(self, obj, event):
        """Update Statuszeile bei RenderEvent mit Drosslung (für Fly-To etc.)"""
        try:
            # Drossle auf jeden 5. Frame (ca. 12 FPS statt 60 FPS)
            self._render_update_counter += 1
            if self._render_update_counter >= 5:
                self._render_update_counter = 0
                self._update_camera_status()
        except Exception:
            pass

    def _on_key_press(self, obj, event):
        """Key-Handler für Custom Keys: 1-6, 7, 8, 9, K, L"""
        try:
            key = obj.GetKeySym()
        except Exception:
            return

        # === Debug-Ebenen (1-6) ===
        if key in "123456":
            layer_num = int(key)
            self.toggle_debug_layer(layer_num)

        # === 7 = Terrain an/aus ===
        elif key == "7":
            self.toggle_terrain()

        # === 8 = Roads an/aus ===
        elif key == "8":
            self.toggle_roads()

        # === 9 = Slopes an/aus ===
        elif key == "9":
            self.toggle_slopes()

        # === Kamera-State ===
        elif key == "k":
            self.load_camera_state()
        elif key == "K":
            self.save_camera_state()

        # === Cursor Up/Down = Zoom ändern ===
        elif key == "Up":
            self._adjust_zoom(-5.0)  # Zoom rein (kleinerer Winkel)
        elif key == "Down":
            self._adjust_zoom(5.0)  # Zoom raus (größerer Winkel)

        # === Reload OBJ ===
        elif key == "l" or key == "L":
            self.reload_obj_file()

    def update_view(self, reset_camera=True):
        self.plotter.clear()
        self.current_label_points = None

        # Zeige ALLE Faces (keine Einzelne-Straße-Auswahl mehr)
        subset_road_faces = self.road_faces
        # Erzeuge alle Geometrien, Sichtbarkeit wird später per Actor gesetzt
        subset_slope_faces = self.slope_faces
        subset_terrain_faces = self.terrain_faces

        print(
            f"Zeige alle {len(subset_road_faces)} Road, {len(subset_slope_faces)} Slope, "
            f"{len(subset_terrain_faces)} Terrain Faces..."
        )

        subset_road_faces_array = np.array(subset_road_faces)
        used_vertices = set(subset_road_faces_array.flatten().tolist())

        # Füge Junction-Face Vertices hinzu
        if hasattr(self, "junction_faces") and len(self.junction_faces) > 0:
            junction_faces_array = np.array(self.junction_faces)
            used_vertices.update(junction_faces_array.flatten().tolist())

        if len(subset_slope_faces) > 0:
            subset_slope_faces_array = np.array(subset_slope_faces)
            used_vertices.update(subset_slope_faces_array.flatten().tolist())

        if len(subset_terrain_faces) > 0:
            subset_terrain_faces_array = np.array(subset_terrain_faces)
            used_vertices.update(subset_terrain_faces_array.flatten().tolist())

        used_vertices = np.array(sorted(used_vertices))
        vertex_map = {old: new for new, old in enumerate(used_vertices)}
        subset_points = self.vertices[used_vertices]

        # Falls keine Punkte übrig sind: Szene leeren, Status updaten, rendern und zurück
        if subset_points.size == 0:
            self.plotter.clear()
            self.plotter.render()
            self._update_camera_status()
            print("[Hinweis] Keine Geometrie vorhanden (alle Ebenen leer/ausgeblendet)")
            return
        # Roads rendern (immer erstellen, Visibility nach show_roads)
        self._roads_actors = []
        remapped_road_faces = np.array(
            [[vertex_map[v] for v in face] for face in subset_road_faces]
        )
        pyvista_road_faces = np.hstack(
            [[3] + face.tolist() for face in remapped_road_faces]
        )
        road_mesh = pv.PolyData(subset_points, pyvista_road_faces)

        actor = self.plotter.add_mesh(
            road_mesh,
            color="lightgray",
            show_edges=True,
            edge_color="red",
            line_width=2,
            opacity=0.5,
            label="Road",
        )
        actor.SetVisibility(self.show_roads)
        self._roads_actors.append(actor)

        # Junction-Quads (immer erstellen, gleiche Visibility wie Roads)
        if hasattr(self, "junction_faces") and len(self.junction_faces) > 0:
            subset_junction_faces = self.junction_faces
            if len(subset_junction_faces) > 0:
                remapped_junction_faces = np.array(
                    [[vertex_map[v] for v in face] for face in subset_junction_faces]
                )
                pyvista_junction_faces = np.hstack(
                    [[3] + face.tolist() for face in remapped_junction_faces]
                )
                junction_points = subset_points.copy()
                junction_points[:, 2] += 0.1
                junction_mesh = pv.PolyData(junction_points, pyvista_junction_faces)

                actor = self.plotter.add_mesh(
                    junction_mesh,
                    color="limegreen",
                    show_edges=True,
                    edge_color="darkgreen",
                    line_width=3,
                    opacity=1.0,
                    label="Junction-Quads",
                )
                actor.SetVisibility(self.show_roads)
                self._roads_actors.append(actor)

        # Terrain rendern (immer erstellen, Visibility nach show_terrain)
        self._terrain_actors = []
        if len(subset_terrain_faces) > 0:
            remapped_terrain_faces = np.array(
                [[vertex_map[v] for v in face] for face in subset_terrain_faces]
            )
            pyvista_terrain_faces = np.hstack(
                [[3] + face.tolist() for face in remapped_terrain_faces]
            )
            terrain_mesh = pv.PolyData(subset_points, pyvista_terrain_faces)

            actor = self.plotter.add_mesh(
                terrain_mesh,
                color="green",
                show_edges=False,
                opacity=0.5,
                label="Terrain",
            )
            actor.SetVisibility(self.show_terrain)
            self._terrain_actors.append(actor)

        # Slopes rendern (immer erstellen, Visibility nach show_slopes)
        self._slopes_actors = []
        if len(subset_slope_faces) > 0:
            remapped_slope_faces = np.array(
                [[vertex_map[v] for v in face] for face in subset_slope_faces]
            )
            pyvista_slope_faces = np.hstack(
                [[3] + face.tolist() for face in remapped_slope_faces]
            )
            slope_mesh = pv.PolyData(subset_points, pyvista_slope_faces)

            actor = self.plotter.add_mesh(
                slope_mesh,
                color="tan",
                show_edges=True,
                edge_color="darkgreen",
                line_width=1,
                opacity=0.7,
                label="Slope",
            )
            actor.SetVisibility(self.show_slopes)
            self._slopes_actors.append(actor)

        # Debug-Ebenen rendern und Actors speichern für Performance
        self._debug_layer_actors = {}  # Reset actors
        for layer_name, layer_data in self.debug_layers.items():
            actors_for_layer = []
            visible = self.debug_layer_visible.get(layer_name, False)

            vertices = layer_data["vertices"]
            edges = layer_data["edges"]
            points = layer_data.get("points", [])
            point_labels = layer_data.get("point_labels", [])

            if len(vertices) > 0 and len(edges) > 0:
                layer_mesh = pv.PolyData()
                layer_mesh.points = vertices
                lines = np.hstack([[2, edge[0], edge[1]] for edge in edges])
                layer_mesh.lines = lines

                color = self._get_layer_color(layer_name)
                actor = self.plotter.add_mesh(
                    layer_mesh,
                    color=color,
                    line_width=3,
                    label=layer_name,
                )
                actor.SetVisibility(visible)
                actors_for_layer.append(actor)

            if len(vertices) > 0 and len(points) > 0:
                pts = vertices[points]
                color = self._get_layer_color(layer_name)
                actor = self.plotter.add_points(
                    pts,
                    color=color,
                    point_size=12,
                    render_points_as_spheres=True,
                )
                actor.SetVisibility(visible)
                actors_for_layer.append(actor)

                if (
                    point_labels
                    and len(point_labels) == len(points)
                    and len(point_labels) <= 2000
                ):
                    labels = [
                        str(lbl) if lbl is not None else "" for lbl in point_labels
                    ]
                    actor = self.plotter.add_point_labels(
                        pts,
                        labels,
                        point_size=0,
                        font_size=10,
                        text_color=color,
                        shape_color="white",
                        shape_opacity=0.6,
                    )
                    actor.SetVisibility(visible)
                    actors_for_layer.append(actor)

            # Speichere Actors für diesen Layer
            if actors_for_layer:
                self._debug_layer_actors[layer_name] = actors_for_layer

        # Linke Seite: Bedienungsanleitung (statisch)
        bedienung = (
            "1-6: Debug | 7: Terrain | 8: Roads | 9: Slopes | K: Cam | L: Reload"
        )
        self.plotter.add_text(
            bedienung,
            position="upper_left",
            font_size=10,
        )

        # Rechte Seite: Aktive Ebenen mit gleicher Schriftgröße wie links
        self._update_active_layers_text()

        self.plotter.show_grid()

        center = subset_points.mean(axis=0)
        bounds = [
            subset_points[:, 0].min(),
            subset_points[:, 0].max(),
            subset_points[:, 1].min(),
            subset_points[:, 1].max(),
            subset_points[:, 2].min(),
            subset_points[:, 2].max(),
        ]

        if reset_camera:
            padding = 50
            bounds_with_padding = [
                bounds[0] - padding,
                bounds[1] + padding,
                bounds[2] - padding,
                bounds[3] + padding,
                bounds[4] - padding,
                bounds[5] + padding,
            ]
            self.plotter.camera_position = None
            self.plotter.view_isometric()
            self.plotter.camera.focal_point = center
            cam_dist = (
                max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
                * 1.5
            )
            self.plotter.camera.position = [
                center[0] + cam_dist,
                center[1] + cam_dist,
                center[2] + cam_dist,
            ]

        cam_pos = (
            np.array(self.plotter.camera.position) if self.plotter.camera else None
        )
        cam_dist = np.linalg.norm(cam_pos - center) if cam_pos is not None else np.inf

        show_labels = (
            self.show_point_labels and cam_dist < self.label_distance_threshold
        )

        if show_labels and len(subset_points) > 0:
            if len(subset_points) > self.label_max_points:
                label_indices = np.linspace(
                    0, len(subset_points) - 1, num=self.label_max_points, dtype=int
                )
                sampled_points = subset_points[label_indices]
                sampled_labels = [str(i) for i in label_indices]
            else:
                sampled_points = subset_points
                sampled_labels = [str(i) for i in range(len(subset_points))]

            self.plotter.add_point_labels(
                sampled_points,
                sampled_labels,
                point_size=10,
                font_size=12,
                text_color="blue",
                shape_color="white",
                shape_opacity=0.7,
            )

        self._update_camera_status()
        self.plotter.render()

    def show(self):
        """Starte Viewer. Reload erfolgt manuell über L-Taste."""
        try:
            self.plotter.show()
        except Exception as e:
            print(f"[FEHLER] PyVista show() ist fehlgeschlagen: {e}")
            import traceback

            traceback.print_exc()

    def reload_obj_file(self):
        """Lade OBJ + Debug-Layer manuell neu (L-Taste), Kamera bleibt erhalten."""

        self._show_reload_overlay()
        try:
            print(f"  [Reload] {self.obj_file}")

            camera_pos = None
            camera_focal = None
            camera_up = None
            try:
                camera_pos = self.plotter.camera.position
                camera_focal = self.plotter.camera.focal_point
                camera_up = self.plotter.camera.up_vector
            except Exception:
                pass

            (
                self.vertices,
                self.road_faces,
                self.slope_faces,
                self.terrain_faces,
                self.junction_faces,
                self.road_face_to_idx,
            ) = self._parse_obj(self.obj_file, "roads.obj")

            self._reload_debug_layers()

            print(
                f"  ✓ Neu geladen: {len(self.vertices)} Vertices, {len(self.road_faces)} Faces"
            )

            self.update_view(reset_camera=False)

            if camera_pos is not None:
                try:
                    self.plotter.camera.position = camera_pos
                    self.plotter.camera.focal_point = camera_focal
                    self.plotter.camera.up_vector = camera_up
                    print("  ✓ Kamera-Position beibehalten")
                except Exception:
                    pass

            return True

        except Exception as e:
            print(f"  ✗ Fehler beim Reload: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self._hide_reload_overlay()

    def _world_to_display(self, pts):
        ren = self.plotter.renderer
        coords = []
        for p in pts:
            ren.SetWorldPoint(p[0], p[1], p[2], 1.0)
            ren.WorldToDisplay()
            coords.append(ren.GetDisplayPoint())
        return np.array(coords)

    def _show_reload_overlay(self):
        """Zeige zentrierten Reload-Hinweis während des Nachladens."""
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
        except Exception:
            self._reload_actor = None

    def _hide_reload_overlay(self):
        if self._reload_actor is not None:
            try:
                self.plotter.remove_actor(self._reload_actor)
                self.plotter.render()
            except Exception:
                pass
            self._reload_actor = None

    # _update_camera_status entfernt - keine Status-Zeile mehr gewünscht

    def _update_camera_status(self):
        """Zeige Kamera-Status-Zeile mit Position, Tilt, Roll, Yaw, Zoom"""
        cam = self.plotter.camera
        if cam is None:
            return

        try:
            # Setze Roll auf 0 und Zoom auf 30° immer
            try:
                cam.SetViewUp(0.0, 0.0, 1.0)
                cam.view_angle = 30.0
            except Exception:
                try:
                    # Fallback: versuche direkt zu setzen
                    cam.up_vector = [0.0, 0.0, 1.0]
                    cam.view_angle = 30.0
                except Exception:
                    pass

            pos = np.array(cam.position, dtype=float)
            focal = np.array(cam.focal_point, dtype=float)

            # up_vector korrekt auslesen
            try:
                up = np.array(cam.up_vector, dtype=float)
            except AttributeError:
                # Fallback: verwende GetViewUp() oder default [0, 0, 1]
                try:
                    up = np.array(cam.GetViewUp(), dtype=float)
                except Exception:
                    up = np.array([0.0, 0.0, 1.0], dtype=float)

            forward = focal - pos
            f_norm = np.linalg.norm(forward)
            if f_norm > 1e-9:
                forward = forward / f_norm
            else:
                forward = np.array([0.0, 0.0, 1.0])

            yaw = np.degrees(np.arctan2(forward[1], forward[0]))
            tilt = np.degrees(
                np.arctan2(forward[2], np.linalg.norm(forward[:2]) + 1e-9)
            )

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
            except Exception:
                zoom = 30.0

            text = (
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                f"Tilt: {tilt:.1f}° | Roll: {roll:.1f}° | Yaw: {yaw:.1f}° | Zoom: {zoom:.1f}°"
            )

            # Entferne alten Text-Actor
            try:
                self.plotter.remove_actor("camera_status_text")
            except Exception:
                pass

            try:
                self._camera_status_actor = self.plotter.add_text(
                    text,
                    position="lower_left",
                    font_size=10,
                    color="black",
                    shadow=True,
                    name="camera_status_text",
                )
            except Exception:
                self._camera_status_actor = None
        except Exception:
            pass

    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return {}
        return {}

    def _save_config(self, data):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_camera_state(self):
        cfg = self._load_config()
        return cfg.get("camera")

    def load_camera_state(self):
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
                cam.up_vector = up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
                print("[Kamera] Geladen")
            else:
                print("[Kamera] Ungültiger Kamera-State")
        except Exception as e:
            print(f"[Kamera] Fehler beim Laden: {e}")

    def _apply_saved_camera_state(self):
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
                cam.up_vector = up
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
        except Exception:
            pass

    def save_camera_state(self):
        cam = self.plotter.camera
        if cam is None:
            print("[Kamera] Kamera nicht verfügbar")
            return
        try:
            state = {
                "position": list(cam.position),
                "focal_point": list(cam.focal_point),
                "up_vector": list(cam.up_vector),
            }
            cfg = self._load_config()
            cfg["camera"] = state
            self._save_config(cfg)
            print(f"[Kamera] Gespeichert nach {self.config_path}")
        except Exception as e:
            print(f"[Kamera] Fehler beim Speichern: {e}")
        self._save_window_state()  # Konfig konsistent halten

    def _load_window_state(self):
        cfg = self._load_config()
        return cfg.get("window")

    def _apply_saved_window_state(self):
        state = self._load_window_state()
        if not state:
            return
        try:
            x = int(state.get("x", 0))
            y = int(state.get("y", 0))
            w = int(state.get("w", 0))
            h = int(state.get("h", 0))
        except Exception:
            return

        # Mindestgrößen und einfache Bounds-Prüfung
        if w < 200 or h < 150:
            return
        if x < -5000 or y < -5000:
            return

        try:
            win = self.plotter.ren_win
            win.SetSize(w, h)
            win.SetPosition(x, y)
        except Exception:
            pass

    def _save_window_state(self):
        try:
            win = self.plotter.ren_win
            pos = win.GetPosition()
            size = win.GetSize()
            state = {
                "x": int(pos[0]),
                "y": int(pos[1]),
                "w": int(size[0]),
                "h": int(size[1]),
            }
            cfg = self._load_config()
            cfg["window"] = state
            # Speichere auch aktuelle Kamera, um immer synchron zu bleiben
            cam = self.plotter.camera
            if cam is not None:
                cfg["camera"] = {
                    "position": list(cam.position),
                    "focal_point": list(cam.focal_point),
                    "up_vector": list(cam.up_vector),
                }
            self._save_config(cfg)
        except Exception:
            pass

    def _on_close_save_window_state(self, *args, **kwargs):
        self._save_window_state()

    def on_left_click(self, obj, event):
        """Label-Klick-Handler (nicht mehr verwendet, da keine road_idx-Labels)"""
        pass


if __name__ == "__main__":
    viewer = MeshViewer(
        "beamng.obj",
        fallback_obj="roads.obj",
    )
    viewer.show()
