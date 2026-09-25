"""
Walls of the LOD2 buildings: seamless plaster plus windows/doors as separate faces.

Each wall stays ONE uncut polygon with a metrically tiled plaster texture (no cells, no seams in the plaster).
Windows, doors and basement windows are small faces (sprites from the window atlas) that sit
config.FACADE_WINDOW_OFFSET_M in front of the wall.

Storeys are counted DOWNWARD from the eave (config.FACADE_STOREY_HEIGHT_M). If a remainder of at least
config.FACADE_BASEMENT_MIN_REMAINDER_M is left at the bottom, it is a raised basement: basement windows are placed
there near ground level. Doors exist only where the ground-floor level is (almost) at terrain height.

Church towers (`building["tower_walls"]`, see church_towers.py) get no windows, doors or basement windows. Their
front wall carries a tower clock instead: the tower wall that points away from the nave (west if there is no nave).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon

from .. import config
from .facade_styles import building_key, choice, plaster_index
from .ring_geometry import newell_normal, open_ring, unit_or_none
from .triangulate import triangulate_ccw
from .window_atlas import WindowAtlasLayout, WindowSprite

_MIN_WALL_AREA_M2 = 0.05
_HORIZONTAL_FACE_NZ = 0.95  # |Nz| above this: almost horizontal "wall", without wall frame
_WINDOW_MAX_NZ = 0.2  # |Nz| from here on a wall counts as inclined: no windows
_EDGE_MARGIN_M = 0.05  # windows must lie this far inside the wall edge
_CLOCK_EDGE_MARGIN_M = 0.3  # the tower clock needs more clearance to the wall edge
_GROUND_CLEARANCE_M = 0.25  # lower edge of a window at least this high above the wall base
_DOOR_BELOW_GROUND_M = 0.3  # ground-floor level may be this far below the wall base for a door to still be created
_SHUTTER_SIDE_MARGIN_M = 0.3  # bay must be this much wider than a sprite with shutters
_WINDOW_KIND_PLAIN_PERMILLE = 450
_WINDOW_KIND_TRANSOM_PERMILLE = 750  # above: shutters (as far as the bay is wide enough)


@dataclass
class FacadeMesh:
    """Wall geometry of a building: plaster face and window faces (separate materials)."""

    plaster: int  # index into PLASTER_COLORS
    vertices: np.ndarray  # (N, 3)
    uvs: np.ndarray  # (N, 2)
    wall_faces: List[List[int]]
    window_faces: List[List[int]]

    @staticmethod
    def empty(plaster: int = 0) -> "FacadeMesh":
        return FacadeMesh(plaster, np.zeros((0, 3)), np.zeros((0, 2)), [], [])


@dataclass
class _Wall:
    index: int
    verts: np.ndarray  # original vertices (ring with closing point); index base of `faces`
    faces: np.ndarray
    ring: np.ndarray  # ring without closing point
    out: np.ndarray  # outward-pointing unit normal
    x: np.ndarray  # meters along the wall (viewer-right)
    y: np.ndarray  # absolute height of the ring points
    to_world: np.ndarray  # (3, 3): [x, y, 1] @ to_world = point in space
    width: float
    min_z: float
    windows_allowed: bool
    flat: bool


@dataclass
class _Building:
    key: str
    z_eave: float
    storeys: int
    has_basement: bool
    flips_winding: bool
    front_index: Optional[int]
    clock_wall: Optional[int] = None


class FacadeMapper:
    """Generates wall geometry including UVs for a building dict (`walls`, `roofs`, `id`, `bounds`)."""

    def __init__(
        self,
        layout: Optional[WindowAtlasLayout] = None,
        storey_height_m: float = config.FACADE_STOREY_HEIGHT_M,
        bay_width_m: float = config.FACADE_BAY_WIDTH_M,
        narrow_wall_m: float = config.FACADE_NARROW_WALL_M,
    ):
        self._layout = layout or WindowAtlasLayout()
        self._storey_h = storey_height_m
        self._bay_w = bay_width_m
        self._narrow = narrow_wall_m

    def map_building(self, building: Dict) -> FacadeMesh:
        rings = [open_ring(verts) for verts, _ in building.get("walls", [])]
        key = building_key(building)
        plaster = plaster_index(key)
        if not rings:
            return FacadeMesh.empty(plaster)

        points = np.vstack(rings)
        z_base, z_eave = float(points[:, 2].min()), self._eave_z(building, points)
        storeys, remainder = self._storeys(z_eave - z_base)
        normals = [newell_normal(ring) for ring in rings]
        flips = self._rings_point_inward(rings, normals, points)
        walls = self._build_walls(building["walls"], rings, normals, flips)
        tower = set(building.get("tower_walls", ()))
        for wall in walls:
            if wall.index in tower:
                wall.windows_allowed = False  # church tower: no windows, doors, basement windows
        context = _Building(
            key=key,
            z_eave=z_eave,
            storeys=storeys,
            has_basement=storeys > 0 and remainder >= config.FACADE_BASEMENT_MIN_REMAINDER_M,
            flips_winding=flips,
            front_index=self._front_wall_index(walls),
        )
        context.clock_wall = next((w.index for w in self._clock_candidates(walls, tower) if self._clock_box(w) is not None), None)

        vertices: List[np.ndarray] = []
        uvs: List[np.ndarray] = []
        wall_faces: List[List[int]] = []
        window_faces: List[List[int]] = []
        offset = 0
        for wall in walls:
            wall_v, wall_uv, wall_f = self._plaster(wall, context)
            if len(wall_v):
                vertices.append(wall_v)
                uvs.append(wall_uv)
                wall_faces.extend([[i + offset for i in face] for face in wall_f])
                offset += len(wall_v)

            window_v, window_uv, window_f = self._windows(wall, context)
            if len(window_v):
                vertices.append(window_v)
                uvs.append(window_uv)
                window_faces.extend([[i + offset for i in face] for face in window_f])
                offset += len(window_v)

        if not vertices:
            return FacadeMesh.empty(plaster)
        return FacadeMesh(plaster, np.vstack(vertices), np.vstack(uvs), wall_faces, window_faces)

    # ------------------------------------------------------------------ Building

    @staticmethod
    def _eave_z(building: Dict, wall_points: np.ndarray) -> float:
        """Eave height: lowest roof point; without a roof, the highest wall edge."""
        roofs = [verts for verts, _ in building.get("roofs", [])]
        if roofs:
            return float(np.vstack(roofs)[:, 2].min())
        return float(wall_points[:, 2].max())

    def _storeys(self, height: float) -> Tuple[int, float]:
        """
        Number of full storeys counted down from the eave, and the remainder below.

        The lowest storey may be too short by config.FACADE_STOREY_ROUNDING storey heights. The remainder (can be
        slightly negative) is the raised basement, provided it is large enough.
        """
        storeys = max(0, int(math.floor(height / self._storey_h + config.FACADE_STOREY_ROUNDING)))
        return storeys, height - storeys * self._storey_h

    @staticmethod
    def _rings_point_inward(rings: List[np.ndarray], normals: List[np.ndarray], points: np.ndarray) -> bool:
        """
        CityGML does not guarantee the ring direction. Majority vote by area over the whole building:
        if the ring normals (area-weighted) point toward the centroid, all rings are oriented inward.
        """
        centre = points[:, :2].mean(axis=0)
        score = 0.0
        for ring, normal in zip(rings, normals):  # normal magnitude = 2 * area
            score += float(normal[:2] @ (ring[:, :2].mean(axis=0) - centre))
        return score < 0

    # ------------------------------------------------------------------ Walls

    def _build_walls(self, source_walls, rings, normals, flip: bool) -> List[_Wall]:
        walls = []
        for index, ((verts, faces), ring, raw_normal) in enumerate(zip(source_walls, rings, normals)):
            normal = unit_or_none(raw_normal)
            if normal is None or np.linalg.norm(raw_normal) / 2 < _MIN_WALL_AREA_M2:
                continue
            wall = self._wall_frame(index, verts, np.asarray(faces), ring, -normal if flip else normal)
            if wall is not None:
                walls.append(wall)
        return walls

    def _wall_frame(self, index, verts, faces, ring, out) -> Optional[_Wall]:
        u_axis = unit_or_none(np.array([-out[1], out[0], 0.0]))  # up x out = viewer-right
        flat = abs(out[2]) >= _HORIZONTAL_FACE_NZ or u_axis is None
        if flat:
            zeros = np.zeros(len(ring))
            return _Wall(index, verts, faces, ring, out, zeros, zeros, np.eye(3), 0.0, float(ring[:, 2].min()), False, True)

        x = (ring - ring[0]) @ u_axis
        y = ring[:, 2]
        to_world, _, rank, _ = np.linalg.lstsq(np.column_stack([x, y, np.ones(len(ring))]), ring, rcond=None)
        if rank < 3:  # a line in the (x, y) frame
            return None
        return _Wall(
            index=index,
            verts=verts,
            faces=faces,
            ring=ring,
            out=out,
            x=x,
            y=y,
            to_world=to_world,
            width=float(x.max() - x.min()),
            min_z=float(y.min()),
            windows_allowed=abs(out[2]) < _WINDOW_MAX_NZ,
            flat=False,
        )

    def _front_wall_index(self, walls: List[_Wall]) -> Optional[int]:
        """The longest wall with windows carries the door."""
        candidates = [w for w in walls if not w.flat and w.windows_allowed and w.width >= self._narrow]
        return max(candidates, key=lambda w: w.width).index if candidates else None

    @staticmethod
    def _clock_candidates(walls: List[_Wall], tower: set) -> List[_Wall]:
        """
        Tower walls for the tower clock, front wall first: the wall that points away from the nave. Without a nave
        (free-standing tower) or with a centered tower the front points west (-x), as is usual for churches.
        """
        candidates = [w for w in walls if w.index in tower and not w.flat and w.width >= config.CHURCH_CLOCK_MIN_WALL_M]
        if not candidates:
            return []

        tower_walls = [w for w in walls if w.index in tower]
        nave_walls = [w for w in walls if w.index not in tower]
        front = np.array([-1.0, 0.0])
        if nave_walls:
            away = np.mean([w.ring[:, :2].mean(axis=0) for w in tower_walls], axis=0) - np.mean(
                [w.ring[:, :2].mean(axis=0) for w in nave_walls], axis=0
            )
            if np.linalg.norm(away) >= 1.0:
                front = away / np.linalg.norm(away)
        return sorted(candidates, key=lambda w: -float(w.out[:2] @ front))

    # ------------------------------------------------------------------ Plaster

    def _plaster(self, wall: _Wall, building: _Building):
        """Wall as one polygon; UVs metric (config.FACADE_PLASTER_REPEAT_M), offset against each other per wall."""
        repeat = config.FACADE_PLASTER_REPEAT_M
        shift = np.array([choice(building.key, f"u{wall.index}", 1000), choice(building.key, f"v{wall.index}", 1000)]) / 1000.0

        if wall.flat:  # underside/ledge: plan-view coordinates
            uv = wall.verts[:, :2] / repeat + shift
            faces = np.asarray(wall.faces, dtype=np.int64).reshape(-1, 3)
            return wall.verts, uv, faces.tolist()

        outline = np.column_stack([wall.x, wall.y])
        triangles = triangulate_ccw(outline)
        if not triangles:
            return np.zeros((0, 3)), np.zeros((0, 2)), []
        uv = (outline - outline.min(axis=0)) / repeat + shift
        if building.flips_winding:
            triangles = [[a, c, b] for a, b, c in triangles]
        return wall.ring, uv, triangles

    # ------------------------------------------------------------------ Windows

    def _windows(self, wall: _Wall, building: _Building):
        """Windows, doors and basement windows of a wall as rectangles in front of the wall."""
        empty = np.zeros((0, 3)), np.zeros((0, 2)), []
        if wall.index == building.clock_wall:
            return self._clock(wall, building) or empty
        if wall.flat or not wall.windows_allowed or wall.width < self._narrow or building.storeys == 0:
            return empty

        bays = max(1, int(math.floor(wall.width / self._bay_w + 0.5)))
        bay_w = wall.width / bays
        centres = float(wall.x.min()) + (np.arange(bays) + 0.5) * bay_w
        boxes, sprites = self._boxes(wall, building, centres, bay_w)
        if not boxes:
            return empty

        boxes = np.array(boxes)
        keep = self._inside_wall(wall, boxes)
        if not keep.any():
            return empty
        return self._quads(wall, building, boxes[keep], [sprite for sprite, use in zip(sprites, keep) if use])

    def _clock_box(self, wall: _Wall) -> Optional[np.ndarray]:
        """
        Rectangle of the tower clock: centered on the wall, as high as the wall outline allows (at most
        CHURCH_CLOCK_BELOW_TOP_M below the top edge, at least CHURCH_CLOCK_MIN_HEIGHT_M above the wall base); None if
        it fits nowhere.
        """
        width, height = self._layout.size_m(WindowSprite.TOWER_CLOCK)
        centre_x = (float(wall.x.min()) + float(wall.x.max())) / 2
        top = float(wall.y.max()) - config.CHURCH_CLOCK_BELOW_TOP_M
        bottom = wall.min_z + config.CHURCH_CLOCK_MIN_HEIGHT_M
        centres = np.arange(top, bottom - 1e-9, -0.25)
        if not len(centres):
            return None
        boxes = np.column_stack(
            [np.full(len(centres), centre_x - width / 2), centres - height / 2, np.full(len(centres), centre_x + width / 2), centres + height / 2]
        )
        fits = np.flatnonzero(self._inside_wall(wall, boxes, margin=_CLOCK_EDGE_MARGIN_M))
        return boxes[fits[0]] if len(fits) else None

    def _clock(self, wall: _Wall, building: _Building):
        box = self._clock_box(wall)
        return self._quads(wall, building, box[None, :], [WindowSprite.TOWER_CLOCK]) if box is not None else None

    def _boxes(self, wall: _Wall, building: _Building, centres: np.ndarray, bay_w: float):
        """Rectangles (x0, y0, x1, y1) and sprite per window/door/basement window; not yet checked against the outline."""
        boxes: List[Tuple[float, float, float, float]] = []
        sprites: List[WindowSprite] = []
        key, storey_h = building.key, self._storey_h

        def add(sprite: WindowSprite, centre: float, bottom: float) -> None:
            width, height = self._layout.size_m(sprite)
            boxes.append((centre - width / 2, bottom, centre + width / 2, bottom + height))
            sprites.append(sprite)

        window_sprite = self._window_sprite(wall, building, bay_w)
        ground_floor = building.z_eave - building.storeys * storey_h
        door_bay = self._door_bay(len(centres), key)
        door_here = (
            wall.index == building.front_index
            and -_DOOR_BELOW_GROUND_M <= ground_floor - wall.min_z <= config.FACADE_DOOR_MAX_HEIGHT_ABOVE_BASE_M
        )

        for storey in range(building.storeys):  # 0 = top storey (eave), from top to bottom
            floor = building.z_eave - (storey + 1) * storey_h
            is_ground = storey == building.storeys - 1
            for bay, centre in enumerate(centres):
                if is_ground and door_here and bay == door_bay:
                    door = WindowSprite.DOOR_WOOD if choice(key, "door", 2) == 0 else WindowSprite.DOOR_WHITE
                    add(door, float(centre), max(floor, wall.min_z))
                    continue
                bottom = floor + config.FACADE_WINDOW_SILL_M
                if bottom - wall.min_z < _GROUND_CLEARANCE_M:  # storey lies (partly) below the terrain
                    continue
                add(window_sprite, float(centre), bottom)

        if building.has_basement and ground_floor - wall.min_z >= config.FACADE_BASEMENT_MIN_EXPOSED_M:
            basement = WindowSprite.BASEMENT_BARS if choice(key, "basement", 2) == 0 else WindowSprite.BASEMENT_PLAIN
            for centre in centres:
                add(basement, float(centre), wall.min_z + config.FACADE_BASEMENT_SILL_M)
        return boxes, sprites

    def _window_sprite(self, wall: _Wall, building: _Building, bay_w: float) -> WindowSprite:
        """Window type per wall (shutters only if the bay is wide enough)."""
        roll = choice(building.key, f"window{wall.index}", 1000)
        if roll < _WINDOW_KIND_PLAIN_PERMILLE:
            return WindowSprite.WINDOW_PLAIN
        if roll < _WINDOW_KIND_TRANSOM_PERMILLE:
            return WindowSprite.WINDOW_TRANSOM
        green = choice(building.key, "shutters", 2) == 0
        shutters = WindowSprite.WINDOW_SHUTTER_GREEN if green else WindowSprite.WINDOW_SHUTTER_BROWN
        if self._layout.size_m(shutters)[0] + _SHUTTER_SIDE_MARGIN_M <= bay_w:
            return shutters
        return WindowSprite.WINDOW_PLAIN

    @staticmethod
    def _door_bay(bays: int, key: str) -> int:
        """Door not in the outer bays, provided the wall is wide enough for that."""
        if bays >= 3:
            return 1 + choice(key, "door_bay", bays - 2)
        if bays == 2:
            return choice(key, "door_bay", 2)
        return 0

    @staticmethod
    def _inside_wall(wall: _Wall, boxes: np.ndarray, margin: float = _EDGE_MARGIN_M) -> np.ndarray:
        """Which rectangles lie completely (with edge margin) inside the wall outline?"""
        polygon = Polygon(np.column_stack([wall.x, wall.y]))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return np.zeros(len(boxes), dtype=bool)
        m = margin
        # A door stands with its threshold on the wall base: no edge margin there
        bottom_margin = np.where(boxes[:, 1] <= wall.min_z + 1e-6, 0.0, m)
        grown = shapely.box(boxes[:, 0] - m, boxes[:, 1] - bottom_margin, boxes[:, 2] + m, boxes[:, 3] + m)
        shapely.prepare(polygon)
        return np.asarray(shapely.contains(polygon, grown))

    def _quads(self, wall: _Wall, building: _Building, boxes: np.ndarray, sprites: List[WindowSprite]):
        """Four corner points per window, placed in front of the wall, with the sprite's atlas UVs."""
        xs = np.stack([boxes[:, 0], boxes[:, 2], boxes[:, 2], boxes[:, 0]], axis=1)
        ys = np.stack([boxes[:, 1], boxes[:, 1], boxes[:, 3], boxes[:, 3]], axis=1)
        points = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3) @ wall.to_world
        points = points + wall.out * config.FACADE_WINDOW_OFFSET_M

        rects = np.array([self._layout.uv_rect(sprite) for sprite in sprites])  # (K, 4): u_min, v_min, u_max, v_max
        us = np.stack([rects[:, 0], rects[:, 2], rects[:, 2], rects[:, 0]], axis=1)
        vs = np.stack([rects[:, 1], rects[:, 1], rects[:, 3], rects[:, 3]], axis=1)
        uvs = np.stack([us, vs], axis=-1).reshape(-1, 2)

        base = 4 * np.arange(len(boxes))[:, None]
        faces = np.vstack([base + np.array([0, 1, 2]), base + np.array([0, 2, 3])])
        if building.flips_winding:
            faces = faces[:, [0, 2, 1]]
        return points, uvs, faces.tolist()
