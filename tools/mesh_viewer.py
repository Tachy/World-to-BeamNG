"""
Mesh Viewer - Visualisiere Gelände, Straßen und Debug-Ebenen
Steuerung:
  - SPACE: Nächste Straße (Einzeln-Modus)
  - B: Vorherige Straße
  - R: Toggle Roads (Straßen) an/aus
  - S: Toggle Slopes (Böschungen) an/aus
  - T: Toggle Terrain an/aus
  - A: Toggle Alle Straßen / Einzelne Straße
  - L: Toggle Punkt-Labels an/aus
  - I: Toggle road_idx-Labels an/aus
  - 1-9: Toggle Debug-Ebenen (ebene1.obj - ebene9.obj)
  - Q: Beenden
"""

import pyvista as pv
import numpy as np
import os
import glob
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
        faces_per_road=10000,
        label_distance_threshold=1e9,
        label_max_points=50000000,
        dolly_step=10.0,
        show_point_labels=False,
        road_idx_label_max=5000000,
        show_road_idx_labels=False,
        label_click_px_tol=30.0,
        preselect_road_idx=None,
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

        self.faces_per_road = faces_per_road
        self.current_road = 0
        self.max_roads = max(1, len(self.road_faces) // faces_per_road)
        self.show_roads = True
        self.show_slopes = False
        self.show_terrain = False
        self.show_all = True
        self.label_distance_threshold = label_distance_threshold
        self.label_max_points = label_max_points
        self.dolly_step = dolly_step
        self.show_point_labels = show_point_labels
        self.road_idx_label_max = road_idx_label_max
        self.show_road_idx_labels = show_road_idx_labels
        self.selected_road_idx = None
        self.current_label_points = None
        self.current_label_ridx = None
        self.label_click_px_tol = label_click_px_tol
        self.preselect_road_idx = preselect_road_idx

        print(
            f"Gefunden: {len(self.vertices)} Vertices, {len(self.road_faces)} Road Faces, "
            f"{len(self.slope_faces)} Slope Faces, {len(self.terrain_faces)} Terrain Faces, "
            f"{len(self.junction_faces)} Junction Faces"
        )
        print(f"Ca. {self.max_roads} Strassen (je {faces_per_road} Faces)")
        print("\nSteuerung:")
        print("  SPACE = Naechste Strasse")
        print("  B = Vorherige Strasse")
        print("  R = Toggle Roads (Strassen)")
        print("  S = Toggle Slopes (Boeschungen)")
        print("  T = Toggle Terrain")
        print("  A = Toggle ALLE Strassen / Einzelne Strasse")
        print("  Pfeil hoch/runter = Kamera vor/zurueck (statt Zoom)")
        print("  L = Toggle Punkt-Labels an/aus")
        print("  I = Toggle road_idx-Labels an/aus")
        if len(self.debug_layers) > 0:
            print(
                f"  1-9 = Toggle Debug-Ebenen ({', '.join(sorted(self.debug_layers.keys()))})"
            )
        print("  Q = Beenden")

        self.plotter = pv.Plotter()
        self.plotter.add_key_event("space", self.next_road)
        self.plotter.add_key_event("b", self.prev_road)
        self.plotter.add_key_event("r", self.toggle_roads)
        self.plotter.add_key_event("s", self.toggle_slopes)
        self.plotter.add_key_event("t", self.toggle_terrain)
        self.plotter.add_key_event("a", self.toggle_show_all)
        self.plotter.add_key_event("Up", self.dolly_forward)
        self.plotter.add_key_event("Down", self.dolly_backward)
        self.plotter.add_key_event("l", self.toggle_labels)
        self.plotter.add_key_event("i", self.toggle_road_idx_labels)

        # Füge Key-Handler für Debug-Ebenen hinzu (1-9)
        for i in range(1, 10):
            self.plotter.add_key_event(
                str(i), lambda layer=i: self.toggle_debug_layer(layer)
            )

        # Mouse-Observer für Label-Klick (Bildschirm-Koordinaten)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self.on_left_click)

        # Optional vorselektieren: springe in den Road-Slot und markiere
        if self.preselect_road_idx is not None and len(self.road_face_to_idx) > 0:
            try:
                idx = self.road_face_to_idx.index(self.preselect_road_idx)
                self.current_road = idx // self.faces_per_road
                self.selected_road_idx = self.preselect_road_idx
                print(
                    f"Vorauswahl road_idx {self.preselect_road_idx} in Slot {self.current_road}"
                )
            except ValueError:
                print(
                    f"Warnung: road_idx {self.preselect_road_idx} nicht im Mapping gefunden"
                )

        self.update_view()

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
        """Toggle Sichtbarkeit einer Debug-Ebene"""
        layer_name = f"ebene{layer_num}"
        if layer_name in self.debug_layer_visible:
            self.debug_layer_visible[layer_name] = not self.debug_layer_visible[
                layer_name
            ]
            state = "AN" if self.debug_layer_visible[layer_name] else "AUS"
            print(f"\nDebug-Ebene {layer_name}: {state}")
            self.update_view(reset_camera=False)
        else:
            print(f"\nDebug-Ebene {layer_name} nicht gefunden")

    def _parse_obj(self, obj_file, fallback_obj="roads.obj"):
        """Parse OBJ file and extract vertices, faces, etc."""
        vertices = []
        road_faces = []
        slope_faces = []
        terrain_faces = []
        junction_faces = []
        road_face_to_idx = []
        current_material = None
        current_road_idx = None

        # Versuche primär beamng.obj zu laden (unified mesh)
        try:
            with open(obj_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        vertices.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )
                    elif line.startswith("usemtl"):
                        current_material = line.strip().split()[1].lower()
                        # Versuche road-Index aus dem Materialnamen zu extrahieren (z.B. road_12)
                        current_road_idx = None
                        tokens = current_material.replace("-", "_").split("_")
                        for t in tokens:
                            if t.isdigit():
                                current_road_idx = int(t)
                                break
                    elif line.startswith("f "):
                        parts = line.strip().split()[1:]
                        face = [int(p.split("/")[0]) - 1 for p in parts]
                        if current_material:
                            if "terrain" in current_material:
                                terrain_faces.append(face)
                            elif "slope" in current_material:
                                slope_faces.append(face)
                            elif (
                                "junction" in current_material
                                or "quad" in current_material
                            ):
                                junction_faces.append(face)
                            elif "road" in current_material:
                                road_faces.append(face)
                                road_face_to_idx.append(current_road_idx)
        except FileNotFoundError:
            print(
                f"  Warnung: {obj_file} nicht gefunden. Nutze Fallback {fallback_obj}"
            )
            # Fallback auf roads.obj
            if fallback_obj:
                try:
                    with open(fallback_obj, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("v "):
                                parts = line.strip().split()
                                vertices.append(
                                    [float(parts[1]), float(parts[2]), float(parts[3])]
                                )
                            elif line.startswith("usemtl"):
                                current_material = line.strip().split()[1].lower()
                                current_road_idx = None
                                tokens = current_material.replace("-", "_").split("_")
                                for t in tokens:
                                    if t.isdigit():
                                        current_road_idx = int(t)
                                        break
                            elif line.startswith("f "):
                                parts = line.strip().split()[1:]
                                face = [int(p.split("/")[0]) - 1 for p in parts]
                                if current_material:
                                    if "terrain" in current_material:
                                        terrain_faces.append(face)
                                    elif "slope" in current_material:
                                        slope_faces.append(face)
                                    elif (
                                        "junction" in current_material
                                        or "quad" in current_material
                                    ):
                                        junction_faces.append(face)
                                    elif "road" in current_material:
                                        road_faces.append(face)
                                        road_face_to_idx.append(current_road_idx)
                except FileNotFoundError:
                    print(f"  Fehler: Weder {obj_file} noch {fallback_obj} gefunden!")
                    raise

        if len(terrain_faces) == 0:
            print("  Info: Keine Terrain-Faces gefunden (nutze ggf. terrain-only OBJ).")
        if len(slope_faces) == 0:
            print("  Info: Keine Slope-Faces gefunden.")
        if len(junction_faces) > 0:
            print(f"  Info: {len(junction_faces)} Junction-Quad-Faces gefunden!")

        return (
            np.array(vertices),
            road_faces,
            slope_faces,
            terrain_faces,
            junction_faces,
            road_face_to_idx,
        )

    def toggle_roads(self):
        self.show_roads = not self.show_roads
        state = "AN" if self.show_roads else "AUS"
        print(f"\nRoads (Strassen): {state}")
        self.update_view(reset_camera=False)

    def toggle_slopes(self):
        self.show_slopes = not self.show_slopes
        state = "AN" if self.show_slopes else "AUS"
        print(f"\nBoeschungen: {state}")
        self.update_view(reset_camera=False)

    def toggle_terrain(self):
        self.show_terrain = not self.show_terrain
        state = "AN" if self.show_terrain else "AUS"
        print(f"\nTerrain: {state}")
        self.update_view(reset_camera=False)

    def toggle_show_all(self):
        self.show_all = not self.show_all
        state = "ALLE" if self.show_all else "EINZELN"
        print(f"\nAnzeige-Modus: {state}")
        self.update_view(reset_camera=False)

    def toggle_labels(self):
        self.show_point_labels = not self.show_point_labels
        state = "AN" if self.show_point_labels else "AUS"
        print(f"\nPunkt-Labels: {state}")
        self.update_view(reset_camera=False)

    def toggle_road_idx_labels(self):
        self.show_road_idx_labels = not self.show_road_idx_labels
        state = "AN" if self.show_road_idx_labels else "AUS"
        print(f"\nroad_idx Labels: {state}")
        self.update_view(reset_camera=False)

    def next_road(self):
        self.current_road = min(self.current_road + 1, self.max_roads - 1)
        print(f"\nStrasse {self.current_road}/{self.max_roads}")
        self.update_view()

    def prev_road(self):
        self.current_road = max(self.current_road - 1, 0)
        print(f"\nStrasse {self.current_road}/{self.max_roads}")
        self.update_view()

    def dolly_forward(self):
        self._dolly(self.dolly_step)

    def dolly_backward(self):
        self._dolly(-self.dolly_step)

    def _dolly(self, step):
        cam = self.plotter.camera
        if cam is None:
            return

        pos = np.array(cam.position)
        focal = np.array(cam.focal_point)
        direction = focal - pos
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return

        direction = direction / norm
        pos_new = pos + direction * step
        focal_new = focal + direction * step

        cam.position = pos_new.tolist()
        cam.focal_point = focal_new.tolist()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def update_view(self, reset_camera=True):
        self.plotter.clear()
        self.current_label_points = None
        self.current_label_ridx = None

        if self.show_all:
            start_face = 0
            end_face = len(self.road_faces)
            subset_road_faces = self.road_faces
            subset_slope_faces = self.slope_faces if self.show_slopes else []
            subset_terrain_faces = self.terrain_faces if self.show_terrain else []
            current_road_idx = None
            print(
                f"Zeige alle {len(subset_road_faces)} Road, {len(subset_slope_faces)} Slope, "
                f"{len(subset_terrain_faces)} Terrain Faces..."
            )
        else:
            start_face = self.current_road * self.faces_per_road
            end_face = min(start_face + self.faces_per_road, len(self.road_faces))
            subset_road_faces = self.road_faces[start_face:end_face]

            # Bestimme road_idx aus Mapping (falls vorhanden)
            if len(self.road_face_to_idx) == len(self.road_faces) and start_face < len(
                self.road_face_to_idx
            ):
                current_road_idx = self.road_face_to_idx[start_face]
            else:
                current_road_idx = self.current_road

            if len(subset_road_faces) == 0:
                print("Keine Faces mehr!")
                return

            subset_slope_faces = []
            if self.show_slopes:
                slope_start = start_face * 2
                slope_end = min(
                    slope_start + self.faces_per_road * 2, len(self.slope_faces)
                )
                subset_slope_faces = self.slope_faces[slope_start:slope_end]

            subset_terrain_faces = []
            if self.show_terrain:
                terrain_start = start_face * 10
                terrain_end = min(
                    terrain_start + self.faces_per_road * 10, len(self.terrain_faces)
                )
                subset_terrain_faces = self.terrain_faces[terrain_start:terrain_end]

        # Mapping für den aktuellen Ausschnitt (Road-Indices pro Face)
        if len(self.road_face_to_idx) > 0:
            face_indices = self.road_face_to_idx[
                start_face : min(end_face, len(self.road_face_to_idx))
            ]
        else:
            face_indices = []

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

        if self.show_roads:
            remapped_road_faces = np.array(
                [[vertex_map[v] for v in face] for face in subset_road_faces]
            )
            pyvista_road_faces = np.hstack(
                [[3] + face.tolist() for face in remapped_road_faces]
            )
            road_mesh = pv.PolyData(subset_points, pyvista_road_faces)

            self.plotter.add_mesh(
                road_mesh,
                color="lightgray",
                show_edges=True,
                edge_color="red",
                line_width=2,
                opacity=0.5,
                label="Road",
            )

        # Junction-Quads (grün, deutlich sichtbar, leicht erhöht)
        if (
            self.show_roads
            and hasattr(self, "junction_faces")
            and len(self.junction_faces) > 0
        ):
            subset_junction_faces = self.junction_faces
            if len(subset_junction_faces) > 0:
                remapped_junction_faces = np.array(
                    [[vertex_map[v] for v in face] for face in subset_junction_faces]
                )
                pyvista_junction_faces = np.hstack(
                    [[3] + face.tolist() for face in remapped_junction_faces]
                )
                # Erstelle Junction-Mesh mit leichtem Z-Offset (0.1m höher)
                junction_points = subset_points.copy()
                junction_points[:, 2] += 0.1  # 10cm höher für bessere Sichtbarkeit
                junction_mesh = pv.PolyData(junction_points, pyvista_junction_faces)

                self.plotter.add_mesh(
                    junction_mesh,
                    color="limegreen",
                    show_edges=True,
                    edge_color="darkgreen",
                    line_width=3,
                    opacity=1.0,
                    label="Junction-Quads",
                )

        if self.show_terrain and len(subset_terrain_faces) > 0:
            remapped_terrain_faces = np.array(
                [[vertex_map[v] for v in face] for face in subset_terrain_faces]
            )
            pyvista_terrain_faces = np.hstack(
                [[3] + face.tolist() for face in remapped_terrain_faces]
            )
            terrain_mesh = pv.PolyData(subset_points, pyvista_terrain_faces)

            self.plotter.add_mesh(
                terrain_mesh,
                color="green",
                show_edges=False,
                opacity=0.5,
                label="Terrain",
            )

        if self.show_slopes and len(subset_slope_faces) > 0:
            remapped_slope_faces = np.array(
                [[vertex_map[v] for v in face] for face in subset_slope_faces]
            )
            pyvista_slope_faces = np.hstack(
                [[3] + face.tolist() for face in remapped_slope_faces]
            )
            slope_mesh = pv.PolyData(subset_points, pyvista_slope_faces)

            self.plotter.add_mesh(
                slope_mesh,
                color="tan",
                show_edges=True,
                edge_color="darkgreen",
                line_width=1,
                opacity=0.7,
                label="Slope",
            )

        # Hervorhebung einer selektierten Straße (gleicher Vertex-Ausschnitt)
        if self.selected_road_idx is not None and len(face_indices) > 0:
            highlight_faces = []
            for face, ridx in zip(subset_road_faces, face_indices):
                if ridx == self.selected_road_idx:
                    highlight_faces.append([vertex_map[v] for v in face])

            if len(highlight_faces) > 0:
                pyvista_highlight_faces = np.hstack([[3] + f for f in highlight_faces])
                highlight_mesh = pv.PolyData(subset_points, pyvista_highlight_faces)
                self.plotter.add_mesh(
                    highlight_mesh,
                    color="magenta",
                    show_edges=True,
                    edge_color="black",
                    line_width=3,
                    opacity=0.75,
                    label="Selected Road",
                )
            else:
                print(
                    f"Keine Faces für selektierte road_idx {self.selected_road_idx} im aktuellen Ausschnitt gefunden"
                )

        # Debug-Ebenen rendern (mit verschiedenen Farben)
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

        for layer_name, layer_data in self.debug_layers.items():
            if self.debug_layer_visible.get(layer_name, False):
                vertices = layer_data["vertices"]
                edges = layer_data["edges"]
                points = layer_data.get("points", [])
                point_labels = layer_data.get("point_labels", [])

                if len(vertices) > 0 and len(edges) > 0:
                    layer_mesh = pv.PolyData()
                    layer_mesh.points = vertices

                    # Konvertiere edges zu PyVista line format
                    lines = np.hstack([[2, edge[0], edge[1]] for edge in edges])
                    layer_mesh.lines = lines

                    color = layer_colors.get(layer_name, "white")
                    self.plotter.add_mesh(
                        layer_mesh,
                        color=color,
                        line_width=3,
                        label=layer_name,
                    )

                if len(vertices) > 0 and len(points) > 0:
                    pts = vertices[points]
                    color = layer_colors.get(layer_name, "white")
                    self.plotter.add_points(
                        pts,
                        color=color,
                        point_size=12,
                        render_points_as_spheres=True,
                    )

                    if (
                        point_labels
                        and len(point_labels) == len(points)
                        and len(point_labels)
                        <= 2000  # Erhöht für Junction-Labels (war: 500)
                    ):
                        labels = [
                            str(lbl) if lbl is not None else "" for lbl in point_labels
                        ]
                        self.plotter.add_point_labels(
                            pts,
                            labels,
                            point_size=0,
                            font_size=10,
                            text_color=color,
                            shape_color="white",
                            shape_opacity=0.6,
                        )

        roads_info = "ON" if self.show_roads else "OFF"
        slopes_info = "ON" if self.show_slopes else "OFF"
        terrain_info = "ON" if self.show_terrain else "OFF"
        mode_info = "ALLE" if self.show_all else f"{self.current_road}/{self.max_roads}"
        road_info = "ALLE" if self.show_all else f"road_idx ~ {current_road_idx}"
        label_state = "ON" if self.show_point_labels else "OFF"
        road_label_state = "ON" if self.show_road_idx_labels else "AUS"

        # Info-Text mit Debug-Layer Status
        debug_info = ""
        if len(self.debug_layers) > 0:
            visible_layers = [
                name for name, vis in self.debug_layer_visible.items() if vis
            ]
            if visible_layers:
                debug_info = f" | Debug: {', '.join(visible_layers)}"

        self.plotter.add_text(
            f"Straße: {mode_info} | Roads: {roads_info} | Terrain: {terrain_info} | Slopes: {slopes_info} | {road_info} | Labels: {label_state} | road_idx: {road_label_state}{debug_info}",
            position="upper_left",
            font_size=10,
        )
        self.plotter.show_grid()

        # Berechne Camera-Bounds
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

        # Road-Index-Labels
        if self.show_road_idx_labels and len(self.road_face_to_idx) > 0:
            road_to_vertices = {}

            for face, ridx in zip(subset_road_faces, face_indices):
                if ridx is None:
                    continue
                if ridx not in road_to_vertices:
                    road_to_vertices[ridx] = []
                road_to_vertices[ridx].extend(face)

            if len(road_to_vertices) > 0:
                labels = []
                points = []
                for ridx, face_vertices in road_to_vertices.items():
                    verts_unique = np.unique(face_vertices)
                    pts = self.vertices[verts_unique]
                    centroid = pts.mean(axis=0)
                    labels.append(str(ridx))
                    points.append(centroid)

                self.plotter.add_point_labels(
                    np.array(points),
                    labels,
                    point_size=12,
                    font_size=12,
                    text_color="black",
                    shape_color="white",
                    shape_opacity=0.6,
                )

                self.current_label_points = np.array(points)
                self.current_label_ridx = labels
            else:
                self.current_label_points = None
                self.current_label_ridx = None
        else:
            self.current_label_points = None
            self.current_label_ridx = None

        self.plotter.render()

    def show(self):
        self.plotter.show()

    def _world_to_display(self, pts):
        ren = self.plotter.renderer
        coords = []
        for p in pts:
            ren.SetWorldPoint(p[0], p[1], p[2], 1.0)
            ren.WorldToDisplay()
            coords.append(ren.GetDisplayPoint())
        return np.array(coords)

    def on_left_click(self, obj, event):
        if self.current_label_points is None or len(self.current_label_points) == 0:
            return

        try:
            x, y = obj.GetEventPosition()
        except AttributeError:
            return
        disp = self._world_to_display(self.current_label_points)
        diffs = disp[:, :2] - np.array([x, y])
        dist2 = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dist2))
        if dist2[idx] > self.label_click_px_tol**2:
            return

        chosen = self.current_label_ridx[idx]
        try:
            chosen_idx = int(chosen)
        except Exception:
            return

        self.selected_road_idx = chosen_idx
        print(f"Selektierte road_idx: {chosen_idx}")
        self.update_view(reset_camera=False)


if __name__ == "__main__":
    viewer = MeshViewer(
        "beamng.obj",
        fallback_obj="roads.obj",
        faces_per_road=10000,
        preselect_road_idx=None,
    )
    viewer.show()
