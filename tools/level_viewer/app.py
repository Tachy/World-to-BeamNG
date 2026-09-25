"""
Viewer shell: plotter, key bindings, layer registry, picking with an info panel, reload and state persistence.

Navigation uses VTK's terrain style: left drag turns around the vertical axis and tilts the view, but never rolls,
so the terrain stays level; a double-click inspects the object under the cursor and flies the camera to it.

Keys avoid VTK's built-in single-letter shortcuts (w/s wireframe/surface, e/q exit, f fly-to, p pick, r reset,
j/t joystick/trackball, u, 3), which would otherwise fire as well.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv
import vtk

from .layers import LAYER_CLASSES, Layer, TerrainLayer, ViewerContext
from .level_data import load_level
from .state import CONFIG_PATH, apply_camera, camera_to_dict, load_state, save_state

FLY_DISTANCE_M = 40.0  # camera distance after a double-click (as in the old dae_viewer)

# Help panel (lower right, toggled with "i"); the layer keys are listed in the layer panel (upper left)
HELP_SECTIONS = [
    ("Mouse", [
        ("left drag", "turn around / tilt (never rolls)"),
        ("shift+left, middle", "pan"),
        ("wheel, right drag", "zoom"),
        ("double-click", "inspect object + fly there (40 m)"),
    ]),
    ("Camera", [
        ("space", "frame the selected object"),
        ("f", "fly to the point under the cursor"),
        ("r", "reset: show everything"),
        ("v", "oblique overview"),
        ("Up / Down", "zoom in / out"),
    ]),
    ("Selection", [
        ("Esc", "clear selection and info"),
    ]),
    ("View", [
        ("x", "terrain: aerial photo / elevation colors"),
        ("layer keys", "show / hide a layer (list above)"),
        ("+ / -", "thicker / thinner lines and points"),
    ]),
    ("Other", [
        ("l", "reload the level from disk"),
        ("i", "show / hide this help"),
        ("q", "quit (window, camera, layers are saved)"),
    ]),
]


def help_text() -> str:
    """The key and mouse reference as aligned plain text (rendered in a monospace font)."""
    width = max(len(key) for _, entries in HELP_SECTIONS for key, _ in entries)
    lines = []
    for title, entries in HELP_SECTIONS:
        if lines:
            lines.append("")
        lines.append(title)
        lines.extend(f"  {key:<{width}}  {text}" for key, text in entries)
    return "\n".join(lines)


class LevelViewer:
    def __init__(self, options, off_screen: bool = False):
        self.options = options
        self.state = {} if off_screen else load_state()
        window = self.state.get("window", {}).get("size", [1600, 1000])
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window, title="World-to-BeamNG level viewer")
        self.plotter.set_background("#1d2330", top="#44516b")
        self.plotter.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)
        self.plotter.add_axes()
        self.show_help = True
        self.selection_actor = None
        self.selected = None  # (layer, actor, cell_id, point)
        self._load()

    # -- loading ------------------------------------------------------------------------------------------------------
    def _load(self) -> None:
        start = time.perf_counter()
        level = load_level(self.options.level_dir)
        self.ctx = ViewerContext(
            level=level,
            terrain_step=self.options.terrain_step,
            full_photo=self.options.full_photo,
            debug_network_path=self.options.debug_network,
        )
        self.layers: List[Layer] = [cls(self.ctx) for cls in LAYER_CLASSES]
        saved = self.state.get("layers", {})
        for layer in self.layers:
            if layer.key in self.options.layer_overrides:
                layer.visible = self.options.layer_overrides[layer.key]
            elif layer.title in saved:
                layer.visible = bool(saved[layer.title])
        for layer in self.layers:
            layer.ensure(self.plotter)
        self.load_seconds = time.perf_counter() - start
        self._update_panel()

    def reload(self) -> None:
        self._clear_selection()
        for layer in self.layers:
            layer.clear(self.plotter)
        self._load()
        self.plotter.render()

    def layer(self, key: str) -> Optional[Layer]:
        return next((layer for layer in self.layers if layer.key == key), None)

    # -- keys ---------------------------------------------------------------------------------------------------------
    def _bind_keys(self) -> None:
        # pyvista's default "b" adds a mouse observer on every press (box picking) and would fire together with the
        # structures layer key; its other defaults (q, v, Up/Down zoom, +/- line width) are kept and listed in the help
        self.plotter.iren.clear_events_for_key("b")
        for layer in self.layers:
            self.plotter.add_key_event(layer.key, lambda layer=layer: self._toggle(layer))
        self.plotter.add_key_event("x", self._toggle_texture)
        self.plotter.add_key_event("l", self.reload)
        self.plotter.add_key_event("i", self._toggle_help)
        self.plotter.add_key_event("space", self._focus_selection)
        self.plotter.add_key_event("Escape", lambda: (self._clear_selection(), self.plotter.render()))

    def _toggle(self, layer: Layer) -> None:
        layer.set_visible(self.plotter, not layer.visible)
        self._update_panel()
        self.plotter.render()

    def _toggle_texture(self) -> None:
        terrain = self.layer(TerrainLayer.key)
        if terrain is not None:
            terrain.toggle_texture(self.plotter)
            self.plotter.render()

    def _toggle_help(self) -> None:
        self.show_help = not self.show_help
        self._update_panel()
        self.plotter.render()

    # -- panels -------------------------------------------------------------------------------------------------------
    def _update_panel(self) -> None:
        lines = [
            f"{self.ctx.level.level_dir.name}  ({len(self.ctx.level.items)} items, loaded in {self.load_seconds:.1f} s)",
            "",
            "Layers (press the key to show / hide):",
        ]
        for layer in self.layers:
            mark = "x" if layer.visible else " "
            lines.append(f" [{mark}] {layer.key}  {layer.title}" + (f"  - {layer.summary}" if layer.summary and layer.built else ""))
        lines.append("")
        if self.show_help:
            lines.append("")
            lines.append(help_text())
        else:
            lines.append("i: show keys and mouse help")
        self._add_panel_text("\n".join(lines), "upper_left", "panel", "white")

    def _add_panel_text(self, text: str, position: str, name: str, color: str) -> None:
        """Monospace text block with a half-transparent dark background (readable on bright terrain)."""
        actor = self.plotter.add_text(text, position=position, font_size=9, name=name, color=color, font="courier")
        prop = actor.GetTextProperty()
        prop.SetBackgroundColor(0.05, 0.07, 0.1)
        prop.SetBackgroundOpacity(0.6)

    def _show_info(self, lines: List[str]) -> None:
        self._add_panel_text("\n".join(lines), "upper_right", "info", "yellow")

    # -- picking ------------------------------------------------------------------------------------------------------
    def _actor_layers(self) -> Dict[int, Layer]:
        return {id(actor): layer for layer in self.layers if layer.visible for actor in layer.actors}

    def pick(self, display_x: int, display_y: int) -> Optional[List[str]]:
        """
        Picks the visible object under a display position and shows its item in the info panel.

        A hardware prop picker finds the actor and the 3D point (fast even for the full-resolution terrain); the cell
        picker, which tests cell by cell, then runs on that one actor only - and not at all for the terrain, whose
        info comes from the picked point.
        """
        owners = self._actor_layers()
        candidates = [a for layer in self.layers if layer.visible for a in layer.actors if a.GetPickable() and a.GetVisibility()]
        prop_picker = vtk.vtkPropPicker()
        prop_picker.PickFromListOn()
        for actor in candidates:
            prop_picker.AddPickList(actor)
        if not prop_picker.Pick(display_x, display_y, 0, self.plotter.renderer):
            return None
        actor = prop_picker.GetActor()
        layer = owners.get(id(actor))
        if layer is None:
            return None
        point = np.array(prop_picker.GetPickPosition())
        cell_id = -1
        if not isinstance(layer, TerrainLayer):
            cell_picker = vtk.vtkCellPicker()
            cell_picker.SetTolerance(0.0005)
            cell_picker.PickFromListOn()
            cell_picker.AddPickList(actor)
            if cell_picker.Pick(display_x, display_y, 0, self.plotter.renderer):
                cell_id = cell_picker.GetCellId()
                point = np.array(cell_picker.GetPickPosition())
        lines = layer.describe(actor, cell_id, point)
        lines.append(f"picked at: {point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}")
        grid = self.ctx.terrain if not isinstance(layer, TerrainLayer) else None
        if grid is not None:
            height = grid.height_at(point[0], point[1])
            if height is not None:
                lines.append(f"terrain below: {height:.2f} m ({point[2] - height:+.2f} m)")
        self._show_info(lines)
        self._select(layer, actor, cell_id, point)
        return lines

    def _select(self, layer: Layer, actor, cell_id: int, point) -> None:
        self._clear_selection()
        outline = layer.highlight(actor, cell_id)
        self.selected = (layer, actor, cell_id, point, outline)
        if outline is not None and outline.n_points:
            self.selection_actor = self.plotter.add_mesh(outline, color="yellow", line_width=4, pickable=False, reset_camera=False, render=False)

    def _clear_selection(self) -> None:
        if self.selection_actor is not None:
            self.plotter.remove_actor(self.selection_actor, render=False)
        self.selection_actor = None
        self.selected = None
        self.plotter.remove_actor("info", render=False)

    def _focus_selection(self) -> None:
        if self.selected is None:
            return
        outline, point = self.selected[4], self.selected[3]
        center = np.array(outline.center) if outline is not None and outline.n_points else point
        radius = max(20.0, float(np.linalg.norm(np.ptp(outline.points, axis=0))) if outline is not None and outline.n_points else 20.0)
        camera = self.plotter.camera
        direction = np.array(camera.position) - np.array(camera.focal_point)
        direction /= max(np.linalg.norm(direction), 1e-9)
        camera.focal_point = center
        camera.position = center + direction * radius * 1.5
        camera.up = (0.0, 0.0, 1.0)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def fly_to(self, point, distance: float = FLY_DISTANCE_M) -> None:
        """Puts the focal point on `point` and the camera `distance` meters away along the current viewing direction."""
        camera = self.plotter.camera
        direction = np.array(camera.focal_point, dtype=float) - np.array(camera.position, dtype=float)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
        point = np.asarray(point, dtype=float)
        camera.focal_point = point
        camera.position = point - direction * distance
        camera.up = (0.0, 0.0, 1.0)
        self.plotter.renderer.ResetCameraClippingRange()

    def _on_double_click(self, *_args) -> None:
        x, y = self.plotter.iren.get_event_position()
        if self.pick(x, y) is not None:
            self.fly_to(self.selected[3])
        self.plotter.render()

    # -- run ----------------------------------------------------------------------------------------------------------
    def _save(self, *_args) -> None:
        state = {
            "window": {"size": list(self.plotter.window_size)},
            "camera": camera_to_dict(self.plotter.camera),
            "layers": {layer.title: layer.visible for layer in self.layers},
        }
        save_state(state, CONFIG_PATH)

    def _initial_camera(self) -> None:
        if apply_camera(self.plotter, self.state.get("camera")):
            return
        grid = self.ctx.terrain
        if grid is not None:
            cx, cy = float(grid.x.mean()), float(grid.y.mean())
            cz = float(np.nanmedian(grid.z))
            extent = float(max(np.ptp(grid.x), np.ptp(grid.y)))
            self.plotter.camera_position = [(cx, cy - extent * 0.9, cz + extent * 0.7), (cx, cy, cz), (0, 0, 1)]
        else:
            self.plotter.reset_camera()

    def show(self) -> None:
        self._bind_keys()
        self._initial_camera()
        self.plotter.track_click_position(self._on_double_click, side="left", double=True, viewport=False)
        self.plotter.show(before_close_callback=self._save)

    def screenshot(self, path) -> None:
        self._initial_camera()
        self.plotter.show(screenshot=str(path), auto_close=True)
