"""
Viewer layers: each layer builds its pyvista meshes lazily on first display, can be toggled by a key and explains
a picked cell. Merged meshes carry the cell array "item_id" (index into LevelData.items), so a pick maps back to the
item of the level file.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv

from . import geometry as geo
from .dae import read_dae
from .level_data import LevelData, LevelItem, TerrainGrid, load_debug_network, load_forest, load_terrain, resolve_asset
from .materials import load_image, material_color

ITEM_ID = "item_id"


@dataclass
class ViewerContext:
    """Shared state of all layers: the loaded level plus lazily loaded terrain."""

    level: LevelData
    terrain_step: int = 1
    full_photo: bool = True
    debug_network_path: Optional[object] = None
    _terrain: Optional[TerrainGrid] = field(default=None, repr=False)
    _terrain_loaded: bool = field(default=False, repr=False)

    @property
    def terrain(self) -> Optional[TerrainGrid]:
        if not self._terrain_loaded:
            self._terrain = load_terrain(self.level, self.terrain_step)
            self._terrain_loaded = True
        return self._terrain


def polydata(points, faces, item_ids=None) -> pv.PolyData:
    mesh = pv.PolyData(np.asarray(points, dtype=float), faces=geo.to_vtk_faces(faces))
    if item_ids is not None:
        mesh.cell_data[ITEM_ID] = np.asarray(item_ids, dtype=np.int64)
    return mesh


def polylines(lines, item_ids) -> pv.PolyData:
    """One PolyData of several polylines [(n, 3) arrays] with one item id per line."""
    points, cells, ids = [], [], []
    offset = 0
    for coords, item_id in zip(lines, item_ids):
        coords = np.asarray(coords, dtype=float)
        if len(coords) < 2:
            continue
        points.append(coords)
        cells.append(np.concatenate([[len(coords)], np.arange(offset, offset + len(coords))]))
        ids.append(item_id)
        offset += len(coords)
    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.vstack(points), lines=np.concatenate(cells))
    mesh.cell_data[ITEM_ID] = np.asarray(ids, dtype=np.int64)
    return mesh


def _texture(image: np.ndarray) -> pv.Texture:
    """Smoothly filtered, mipmapped texture that is clamped at its border (no wrap-around at tile seams)."""
    texture = pv.numpy_to_texture(image)
    texture.interpolate = True
    texture.mipmap = True
    texture.repeat = False
    texture.SetEdgeClamp(True)
    return texture


def _offset_coincident(actor):
    """Draw on top of coplanar surfaces (roads on the embedded terrain)."""
    mapper = actor.GetMapper()
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -2)


class Layer:
    key = ""
    title = ""
    visible_by_default = True

    def __init__(self, ctx: ViewerContext):
        self.ctx = ctx
        self.actors: List = []
        self.meshes: Dict[int, pv.DataSet] = {}  # id(actor) -> mesh (for picking)
        self.built = False
        self.visible = self.visible_by_default
        self.summary = ""

    # -- building -----------------------------------------------------------------------------------------------------
    def build(self, plotter: pv.Plotter) -> None:
        """Creates the actors (called once, on first display)."""
        raise NotImplementedError

    def add(self, plotter: pv.Plotter, mesh: pv.DataSet, pickable: bool = True, **kwargs):
        if mesh is None or mesh.n_points == 0:
            return None
        actor = plotter.add_mesh(mesh, reset_camera=False, pickable=pickable, render=False, **kwargs)
        self.actors.append(actor)
        self.meshes[id(actor)] = mesh
        return actor

    def ensure(self, plotter: pv.Plotter) -> None:
        if self.visible and not self.built:
            self.build(plotter)
            self.built = True
        for actor in self.actors:
            actor.SetVisibility(self.visible)

    def set_visible(self, plotter: pv.Plotter, visible: bool) -> None:
        self.visible = visible
        self.ensure(plotter)

    def clear(self, plotter: pv.Plotter) -> None:
        for actor in self.actors:
            plotter.remove_actor(actor, render=False)
        self.actors, self.meshes, self.built = [], {}, False

    # -- inspection ---------------------------------------------------------------------------------------------------
    def item_for(self, actor, cell_id: int) -> Optional[LevelItem]:
        mesh = self.meshes.get(id(actor))
        if mesh is None or ITEM_ID not in mesh.cell_data or cell_id < 0 or cell_id >= mesh.n_cells:
            return None
        index = int(mesh.cell_data[ITEM_ID][cell_id])
        return self.ctx.level.items[index] if 0 <= index < len(self.ctx.level.items) else None

    def describe(self, actor, cell_id: int, point) -> List[str]:
        item = self.item_for(actor, cell_id)
        return describe_item(item) if item is not None else [self.title]

    def highlight(self, actor, cell_id: int) -> Optional[pv.PolyData]:
        """Outline of everything that belongs to the picked item (yellow selection)."""
        mesh = self.meshes.get(id(actor))
        item = self.item_for(actor, cell_id)
        if mesh is None or item is None:
            return None
        cells = np.flatnonzero(np.asarray(mesh.cell_data[ITEM_ID]) == item.index)
        part = mesh.extract_cells(cells).extract_surface()
        edges = part.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
        return edges if edges.n_points else part


def describe_item(item: LevelItem) -> List[str]:
    lines = [f"{item.name}  [{item.cls}]", f"file: {item.source}  line {item.index + 1}"]
    if item.material:
        lines.append(f"material: {item.material}")
    x, y, z = item.position
    lines.append(f"position: {x:.2f}, {y:.2f}, {z:.2f}")
    if not np.allclose(item.scale, 1.0):
        lines.append("scale: " + ", ".join(f"{v:.2f}" for v in item.scale))
    if item.rotation is not None and item.cls == "SpawnSphere":
        lines.append(f"heading (local -Y): {geo.heading_deg(geo.forward_direction(item.rotation)):.1f} deg")
    for key in ("renderPriority", "drivability", "shapeName", "terrainFile", "dataFile"):
        if key in item.raw:
            lines.append(f"{key}: {item.raw[key]}")
    nodes = item.raw.get("nodes")
    if isinstance(nodes, list) and nodes:
        widths = [n[3] for n in nodes if len(n) > 3]
        lines.append(f"nodes: {len(nodes)}" + (f", width {min(widths):.2f}-{max(widths):.2f} m" if widths else ""))
    return lines


# ---------------------------------------------------------------------------------------------------------------------


class TerrainLayer(Layer):
    """
    Terrain as structured grids (implicit connectivity, so even the full-resolution 4000x4000 samples fit in memory):
    one grid per aerial photo tile with that photo as texture, or one grid with the minimap. The elevation-colored
    variant ([x]) reuses the same grids (a second actor coloring the heights), built when it is first shown.
    """

    key, title = "g", "Terrain"
    PHOTO_MAX_PX = 8192  # the exported photo tiles are 8192 px; GPUs handle that texture size

    def __init__(self, ctx):
        super().__init__(ctx)
        self.textured = True
        self.photo_actors: List = []
        self.elevation_actors: List = []
        self.grids: List[pv.PolyData] = []

    def _grid(self, row_slice=slice(None), col_slice=slice(None)) -> Optional[pv.PolyData]:
        grid = self.ctx.terrain
        x, y = grid.x[col_slice], grid.y[row_slice]
        if len(x) < 2 or len(y) < 2:
            return None
        # float32 is plenty for level-local coordinates (+-4 km, mm precision) and halves the memory
        z = grid.z[row_slice, col_slice]
        xx, yy = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
        mesh = pv.StructuredGrid(xx, yy, z.astype(np.float32))
        # Point order of the structured grid is VTK's; map every point back to its sample via its coordinates
        # (exact lookups: the grid points are the same float32 values as x/y)
        cols = np.searchsorted(x.astype(np.float32), mesh.points[:, 0]).clip(0, len(x) - 1)
        rows = np.searchsorted(y.astype(np.float32), mesh.points[:, 1]).clip(0, len(y) - 1)
        # Vertex normals from the slopes -> smooth (Gouraud) shading instead of one flat shade per terrain quad
        mesh.point_data["Normals"] = geo.height_grid_normals(x, y, z)[rows, cols]
        mesh.point_data.active_normals_name = "Normals"
        hole = grid.hole[row_slice, col_slice]
        if hole.any():
            mesh.point_data["hole"] = hole[rows, cols].astype(np.float32)
            cell_hole = mesh.point_data_to_cell_data(pass_point_data=True).cell_data["hole"] > 0
            mesh.hide_cells(np.flatnonzero(cell_hole), inplace=True)
        # Surface once as PolyData: the photo and the elevation actor then share it (a structured grid would be
        # turned into its own surface copy by every actor's mapper); hidden (hole) cells are dropped here
        surface = mesh.extract_surface(pass_pointid=False, pass_cellid=False)
        for name in ("hole", "vtkGhostType"):
            if name in surface.point_data:
                del surface.point_data[name]
            if name in surface.cell_data:
                del surface.cell_data[name]
        return surface

    def build(self, plotter):
        grid = self.ctx.terrain
        if grid is None:
            self.summary = "no terrain"
            return
        for mesh, texture in self._textured_parts():
            self.grids.append(mesh)
            actor = self.add(plotter, mesh, texture=texture)
            if actor is not None:
                actor.prop.interpolation = "gouraud"  # use the vertex normals (smooth shading)
                self.photo_actors.append(actor)
        if not self.grids:
            mesh = self._grid()
            if mesh is not None:
                self.grids.append(mesh)
        if not self.photo_actors:
            self.textured = False
        source = "photo tiles" if self.ctx.full_photo and self.ctx.level.photos else "minimap"
        self.summary = f"{len(grid.x)}x{len(grid.y)} samples (step {self.ctx.terrain_step}), {source}"

    def _textured_parts(self):
        level, grid = self.ctx.level, self.ctx.terrain
        if self.ctx.full_photo and level.photos:
            with ThreadPoolExecutor(max_workers=4) as pool:
                images = list(pool.map(lambda p: load_image(p.path, self.PHOTO_MAX_PX), level.photos))
            for photo, image in zip(level.photos, images):
                x_min, x_max, y_min, y_max = photo.bounds
                spacing = grid.square_size * self.ctx.terrain_step
                cols = geo.tile_sample_range(grid.x, x_min, x_max, spacing)
                rows = geo.tile_sample_range(grid.y, y_min, y_max, spacing)
                if image is None or len(cols) < 2 or len(rows) < 2:
                    continue
                mesh = self._grid(slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
                if mesh is None:
                    continue
                mesh.active_texture_coordinates = geo.terrain_tcoords(mesh.points[:, 0], mesh.points[:, 1], x_min, y_max, x_max - x_min, y_max - y_min)
                yield mesh, _texture(image)
            return
        minimap = level.minimap
        image = load_image(minimap.path, 4096) if minimap is not None else None
        if image is None:
            return
        mesh = self._grid()
        mesh.active_texture_coordinates = geo.terrain_tcoords(mesh.points[:, 0], mesh.points[:, 1], minimap.x_min, minimap.y_max, minimap.size_x, minimap.size_y)
        yield mesh, _texture(image)

    def _ensure_elevation(self, plotter):
        if self.elevation_actors or not self.grids:
            return
        grid = self.ctx.terrain
        clim = (float(np.nanmin(grid.z)), float(np.nanmax(grid.z)))  # one color scale across all photo tiles
        for mesh in self.grids:
            mesh.point_data["elevation"] = mesh.points[:, 2]
            actor = self.add(plotter, mesh, scalars="elevation", cmap="gist_earth", clim=clim, show_scalar_bar=False)
            if actor is not None:
                actor.prop.interpolation = "gouraud"
                self.elevation_actors.append(actor)

    def ensure(self, plotter):
        super().ensure(plotter)
        if self.visible and self.built and not self.textured:
            self._ensure_elevation(plotter)
        for actor in self.photo_actors:
            actor.SetVisibility(self.visible and self.textured)
        for actor in self.elevation_actors:
            actor.SetVisibility(self.visible and not self.textured)

    def toggle_texture(self, plotter):
        if self.photo_actors or not self.textured:
            self.textured = not self.textured if self.photo_actors else False
        self.ensure(plotter)

    def clear(self, plotter):
        super().clear(plotter)
        self.photo_actors, self.elevation_actors, self.grids = [], [], []
        self.textured = True

    def describe(self, actor, cell_id, point):
        grid = self.ctx.terrain
        lines = ["Terrain  [TerrainBlock]"]
        if grid is not None and point is not None:
            height = grid.height_at(point[0], point[1])
            lines.append(f"x, y: {point[0]:.2f}, {point[1]:.2f}")
            lines.append("height: " + (f"{height:.2f} m" if height is not None else "(hole)"))
            lines.append(f"layer: {grid.material_at(point[0], point[1])}")
        return lines

    def highlight(self, actor, cell_id):
        return None


class RoadLayer(Layer):
    """DecalRoads without the marking lines, one merged mesh per material."""

    key, title = "a", "Roads"
    markings = False

    def _selected(self, item: LevelItem) -> bool:
        is_marking = item.name.startswith("marking_") or item.raw.get("drivability") == -1
        return is_marking == self.markings

    def build(self, plotter):
        groups: Dict[str, list] = {}
        count = 0
        for item in self.ctx.level.of("DecalRoad"):
            if not self._selected(item):
                continue
            z = geo.road_z_offset(item.raw.get("renderPriority")) + (0.03 if self.markings else 0.0)
            points, faces = geo.ribbon(item.raw.get("nodes") or [], z_offset=z)
            if len(faces):
                groups.setdefault(item.material or "?", []).append((points, faces, item.index))
                count += 1
        for material, meshes in sorted(groups.items()):
            points, faces, ids = geo.merge_meshes(meshes)
            actor = self.add(plotter, polydata(points, faces, ids), color=material_color(material, self.ctx.level.materials))
            if actor is not None:
                _offset_coincident(actor)
        self.summary = f"{count} in {len(groups)} materials"


class MarkingLayer(RoadLayer):
    key, title = "m", "Markings"
    markings = True


class WaterLayer(Layer):
    key, title = "o", "Water"

    def build(self, plotter):
        rivers = []
        for item in self.ctx.level.of("River"):
            points, faces = geo.ribbon(item.raw.get("nodes") or [], z_offset=0.05)
            rivers.append((points, faces, item.index))
        if rivers:
            actor = self.add(plotter, polydata(*geo.merge_meshes(rivers)), color=(0.15, 0.4, 0.85), opacity=0.8)
            if actor is not None:
                _offset_coincident(actor)
        blocks = []
        for item in self.ctx.level.of("WaterBlock"):
            points, faces = geo.water_block_box(item.position, item.scale)
            blocks.append((points, faces, item.index))
        if blocks:
            self.add(plotter, polydata(*geo.merge_meshes(blocks)), color=(0.1, 0.35, 0.8), opacity=0.55)
        self.summary = f"{len(rivers)} rivers, {len(blocks)} water blocks"


class StaticsLayer(Layer):
    """TSStatic DAE meshes (bridges, tunnels, walls, buildings), one merged mesh per material."""

    key, title = "b", "Structures"
    horizon = False

    def _selected(self, item: LevelItem) -> bool:
        return ("horizon" in item.name.lower()) == self.horizon

    def build(self, plotter):
        items = [i for i in self.ctx.level.of("TSStatic") if self._selected(i)]
        paths = {i.index: resolve_asset(self.ctx.level.level_dir, i.raw.get("shapeName", "")) for i in items}
        unique = sorted({p for p in paths.values() if p is not None})
        with ThreadPoolExecutor(max_workers=4) as pool:
            parsed = dict(zip(unique, pool.map(read_dae, unique)))

        groups: Dict[str, list] = {}
        textured = []
        missing = 0
        for item in items:
            path = paths[item.index]
            if path is None:
                missing += 1
                continue
            for dae_mesh in parsed[path]:
                world = geo.transform_points(dae_mesh.positions, item.position, item.scale, item.rotation)
                for material, tris in dae_mesh.triangles.items():
                    if self.horizon:
                        textured.append((world, tris, item.index, dae_mesh.uvs))
                    else:
                        groups.setdefault(material, []).append((world, tris, item.index))
        for material, meshes in sorted(groups.items()):
            self.add(plotter, polydata(*geo.merge_meshes(meshes)), color=material_color(material, self.ctx.level.materials))
        for world, tris, item_id, uvs in textured:
            self._add_horizon(plotter, world, tris, item_id, uvs)
        self.summary = f"{len(items) - missing} shapes" + (f", {missing} not resolvable" if missing else "")

    def _add_horizon(self, plotter, world, tris, item_id, uvs):
        mesh = polydata(world, tris, np.full(len(tris), item_id))
        texture_path = self.ctx.level.level_dir / "art" / "shapes" / "textures" / "horizon_sentinel2.dds"
        image = load_image(texture_path, 4096) if uvs is not None and texture_path.exists() else None
        if image is not None:
            mesh.active_texture_coordinates = uvs
            # Image row 0 = north and the exporter's v grows toward north, i.e. the same convention as the minimap
            self.add(plotter, mesh, texture=_texture(image))
        else:
            self.add(plotter, mesh, scalars=world[:, 2], cmap="gist_earth", show_scalar_bar=False)


class HorizonLayer(StaticsLayer):
    key, title = "h", "Horizon"
    horizon = True
    visible_by_default = False


class ForestLayer(Layer):
    key, title = "c", "Forest"

    def build(self, plotter):
        forest = load_forest(self.ctx.level)
        if forest is None:
            self.summary = "no forest"
            return
        self.forest = forest
        cloud = pv.PolyData(forest.positions.astype(float))
        cloud.point_data["type"] = forest.type_idx
        # Green shades per type (distinguishable up close, calm from afar)
        greens = np.column_stack([
            0.10 + 0.25 * ((forest.type_idx * 7) % 11) / 10.0,
            0.35 + 0.45 * ((forest.type_idx * 5) % 13) / 12.0,
            0.08 + 0.20 * ((forest.type_idx * 3) % 7) / 6.0,
        ])
        cloud.point_data["rgb"] = greens
        self.add(plotter, cloud, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=False)
        self.summary = f"{len(forest.positions)} trees, {len(forest.types)} types"

    def describe(self, actor, cell_id, point):
        forest = getattr(self, "forest", None)
        if forest is None or cell_id < 0 or cell_id >= len(forest.positions):
            return ["Forest"]
        x, y, z = forest.positions[cell_id]
        return [f"tree #{cell_id}  [forest instance]", f"type: {forest.types[forest.type_idx[cell_id]]}", f"position: {x:.2f}, {y:.2f}, {z:.2f}"]

    def highlight(self, actor, cell_id):
        forest = getattr(self, "forest", None)
        if forest is None or cell_id < 0 or cell_id >= len(forest.positions):
            return None
        return pv.Sphere(radius=1.5, center=forest.positions[cell_id].astype(float))


class MarkerLayer(Layer):
    """Tunnel darkness zones, portals and spawn points with their facing direction."""

    key, title = "z", "Zones + spawns"

    def build(self, plotter):
        level = self.ctx.level
        for cls, color, opacity in (("Zone", (0.15, 0.15, 0.2), 0.25), ("Portal", (1.0, 0.55, 0.1), 0.45)):
            boxes = [(*geo.oriented_box(i.position, i.scale, i.rotation), i.index) for i in level.of(cls)]
            if boxes:
                mesh = polydata(*geo.merge_meshes(boxes))
                self.add(plotter, mesh, color=color, opacity=opacity)
                self.add(plotter, mesh.extract_all_edges(), color=color, line_width=1, pickable=False)

        spawns = level.of("SpawnSphere")
        if spawns:
            spheres, arrows = [], []
            for item in spawns:
                sphere = pv.Sphere(radius=1.5, center=item.position + [0, 0, 1.5], theta_resolution=12, phi_resolution=8)
                sphere.cell_data[ITEM_ID] = np.full(sphere.n_cells, item.index)
                spheres.append(sphere)
                arrow = pv.Arrow(start=item.position + [0, 0, 1.5], direction=geo.forward_direction(item.rotation), scale=8.0)
                arrow.cell_data[ITEM_ID] = np.full(arrow.n_cells, item.index)
                arrows.append(arrow)
            self.add(plotter, pv.merge(spheres), color=(0.9, 0.1, 0.6))
            self.add(plotter, pv.merge(arrows), color=(1.0, 0.9, 0.1))
        self.summary = f"{len(level.of('Zone'))} zones, {len(level.of('Portal'))} portals, {len(spawns)} spawns"


class LabelLayer(Layer):
    key, title = "n", "Labels"
    visible_by_default = False

    def build(self, plotter):
        items = self.ctx.level.of("SpawnSphere", "TSStatic")
        if not items:
            return
        points = np.array([i.position + [0, 0, 4] for i in items])
        actor = plotter.add_point_labels(points, [i.name for i in items], font_size=12, text_color="white", shape_opacity=0.4, render=False, pickable=False, always_visible=True)
        self.actors.append(actor)
        self.summary = f"{len(items)} labels"


class DebugNetworkLayer(Layer):
    key, title = "d", "Debug network"
    visible_by_default = False

    def build(self, plotter):
        data = load_debug_network(self.ctx.debug_network_path)
        self.primitives = data.get("primitives", [])
        if not self.primitives:
            self.summary = "no debug_network.json"
            return
        lines, ids, colors, points, point_ids = [], [], [], [], []
        for index, prim in enumerate(self.primitives):
            kind, coords = prim.get("type", "line"), prim.get("coords") or []
            if kind in ("line", "polygon") and len(coords) >= 2:
                coords = list(coords) + ([coords[0]] if kind == "polygon" else [])
                lines.append(coords)
                ids.append(index)
                colors.append(prim.get("color", [0, 0, 1]))
            elif kind in ("point", "circle") and coords:
                points.append(coords[0])
                point_ids.append(index)
        if lines:
            mesh = polylines(lines, ids)
            mesh.cell_data["rgb"] = np.clip(np.asarray(colors, dtype=float)[: mesh.n_cells], 0, 1)
            self.add(plotter, mesh, scalars="rgb", rgb=True, line_width=2)
        if points:
            cloud = pv.PolyData(np.asarray(points, dtype=float))
            cloud.cell_data[ITEM_ID] = np.asarray(point_ids)
            self.add(plotter, cloud, color=(0.1, 0.3, 1.0), point_size=10, render_points_as_spheres=True)
        self.summary = f"{len(lines)} lines, {len(points)} points"

    def describe(self, actor, cell_id, point):
        mesh = self.meshes.get(id(actor))
        if mesh is None or cell_id < 0 or cell_id >= mesh.n_cells:
            return ["Debug network"]
        prim = self.primitives[int(mesh.cell_data[ITEM_ID][cell_id])]
        return [f"{prim.get('label') or prim.get('type')}  [debug primitive]", f"type: {prim.get('type')}, points: {len(prim.get('coords') or [])}"]

    def highlight(self, actor, cell_id):
        mesh = self.meshes.get(id(actor))
        if mesh is None or cell_id < 0 or cell_id >= mesh.n_cells:
            return None
        return mesh.extract_cells([cell_id]).extract_surface()


LAYER_CLASSES = (TerrainLayer, RoadLayer, MarkingLayer, WaterLayer, StaticsLayer, HorizonLayer, ForestLayer, MarkerLayer, LabelLayer, DebugNetworkLayer)
