"""
Debug-Tool: Visualisiere Straßen und Böschungen mit Vertex-Nummern und Edges
Steuerung:
  - SPACE: Nächste Straße
  - B: Vorherige Straße
  - S: Toggle Slopes (Böschungen) ein/aus
  - A: Toggle Alle Straßen / Einzelne Straße
  - Q: Beenden
"""

import pyvista as pv
import numpy as np


class RoadViewer:
    def __init__(
        self,
        obj_file="roads.obj",
        faces_per_road=10000,
        label_distance_threshold=1e9,
        label_max_points=50000000,
        dolly_step=10.0,
        show_point_labels=True,
        road_idx_label_max=5000000,
        show_road_idx_labels=True,
        label_click_px_tol=30.0,
        preselect_road_idx=None,
    ):
        print("Lade OBJ und extrahiere road_surface + slope_surface Faces...")
        (
            self.vertices,
            self.road_faces,
            self.slope_faces,
            self.road_face_to_idx,
        ) = self._parse_obj(obj_file)

        self.faces_per_road = faces_per_road
        self.current_road = 0
        self.max_roads = max(1, len(self.road_faces) // faces_per_road)
        self.show_slopes = True
        self.show_all = False
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
            f"Gefunden: {len(self.vertices)} Vertices, {len(self.road_faces)} Road Faces, {len(self.slope_faces)} Slope Faces"
        )
        print(f"Ca. {self.max_roads} Straßen (je {faces_per_road} Faces)")
        print("\nSteuerung:")
        print("  SPACE = Nächste Straße")
        print("  B = Vorherige Straße")
        print("  S = Toggle Slopes (Böschungen)")
        print("  A = Toggle ALLE Straßen / Einzelne Straße")
        print("  Pfeil hoch/runter = Kamera vor/zurück (statt Zoom)")
        print("  L = Toggle Punkt-Labels an/aus")
        print("  R = Toggle road_idx-Labels an/aus")
        print("  Q = Beenden")

        self.plotter = pv.Plotter()
        self.plotter.add_key_event("space", self.next_road)
        self.plotter.add_key_event("b", self.prev_road)
        self.plotter.add_key_event("s", self.toggle_slopes)
        self.plotter.add_key_event("a", self.toggle_show_all)
        self.plotter.add_key_event("Up", self.dolly_forward)
        self.plotter.add_key_event("Down", self.dolly_backward)
        self.plotter.add_key_event("l", self.toggle_labels)
        self.plotter.add_key_event("r", self.toggle_road_idx_labels)
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

    def _parse_obj(self, obj_file):
        vertices = []
        road_faces = []
        slope_faces = []
        road_face_to_idx = []
        current_material = None
        current_road_idx = None

        with open(obj_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
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
                        if "slope" in current_material:
                            slope_faces.append(face)
                        elif "road" in current_material:
                            road_faces.append(face)
                            road_face_to_idx.append(current_road_idx)

        if len(slope_faces) == 0:
            print(
                "Warnung: Keine slope*-Materialien gefunden. Prüfe 'usemtl' Namen im OBJ."
            )

        return np.array(vertices), road_faces, slope_faces, road_face_to_idx

    def toggle_slopes(self):
        self.show_slopes = not self.show_slopes
        state = "AN" if self.show_slopes else "AUS"
        print(f"\nBöschungen: {state}")
        self.update_view()

    def toggle_show_all(self):
        self.show_all = not self.show_all
        state = "ALLE" if self.show_all else "EINZELN"
        print(f"\nAnzeige-Modus: {state}")
        self.update_view()

    def toggle_labels(self):
        self.show_point_labels = not self.show_point_labels
        state = "AN" if self.show_point_labels else "AUS"
        print(f"\nPunkt-Labels: {state}")
        self.update_view()

    def toggle_road_idx_labels(self):
        self.show_road_idx_labels = not self.show_road_idx_labels
        state = "AN" if self.show_road_idx_labels else "AUS"
        print(f"\nroad_idx Labels: {state}")
        self.update_view()

    def next_road(self):
        self.current_road = min(self.current_road + 1, self.max_roads - 1)
        print(f"\nStraße {self.current_road}/{self.max_roads}")
        self.update_view()

    def prev_road(self):
        self.current_road = max(self.current_road - 1, 0)
        print(f"\nStraße {self.current_road}/{self.max_roads}")
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
            current_road_idx = None
            print(
                f"Zeige alle {len(subset_road_faces)} Road Faces und {len(subset_slope_faces)} Slope Faces..."
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

        # Mapping für den aktuellen Ausschnitt (Road-Indices pro Face)
        if len(self.road_face_to_idx) > 0:
            face_indices = self.road_face_to_idx[
                start_face : min(end_face, len(self.road_face_to_idx))
            ]
        else:
            face_indices = []

        subset_road_faces_array = np.array(subset_road_faces)
        used_vertices = set(subset_road_faces_array.flatten().tolist())

        if len(subset_slope_faces) > 0:
            subset_slope_faces_array = np.array(subset_slope_faces)
            used_vertices.update(subset_slope_faces_array.flatten().tolist())

        used_vertices = np.array(sorted(used_vertices))
        vertex_map = {old: new for new, old in enumerate(used_vertices)}
        subset_points = self.vertices[used_vertices]

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
            label="Road",
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

        slopes_info = "ON" if self.show_slopes else "OFF"
        mode_info = "ALLE" if self.show_all else f"{self.current_road}/{self.max_roads}"
        road_info = "ALLE" if self.show_all else f"road_idx ~ {current_road_idx}"
        label_state = "ON" if self.show_point_labels else "OFF"
        road_label_state = "ON" if self.show_road_idx_labels else "AUS"
        self.plotter.add_text(
            f"Straße: {mode_info} | Slopes: {slopes_info} | {road_info} | Labels: {label_state} | road_idx: {road_label_state} | A=toggle all, SPACE=next, B=prev, S=slopes, L=labels, R=road_idx, Q=quit",
            position="upper_left",
            font_size=10,
        )
        self.plotter.show_grid()

        center = subset_points.mean(axis=0)
        if reset_camera:
            self.plotter.camera.focal_point = center
            self.plotter.reset_camera()

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

        # Road-Index-Labels: immer darstellen, keine Kappung
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
    # Falls du direkt eine road_idx hervorheben willst, hier setzen
    viewer = RoadViewer(
        "roads.obj",
        faces_per_road=10000,
        preselect_road_idx=84076071,
    )
    viewer.show()
