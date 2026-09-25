"""
Window atlas: sprites for windows, doors and basement windows.

The sprites are separate, small faces that lie 3 cm in front of the plaster wall (see FacadeMapper). Each sprite has
its real size in meters (resolution config.FACADE_WINDOW_ATLAS_PX_PER_M); there are no cells and no plaster in the
sprite. Component relief (reveal, frame, sill) is stored in the normal map.
"""

from enum import Enum
from typing import Dict, Tuple

import numpy as np

from .. import config
from .texture_utils import gaussian_blur_wrap, gray_to_rgb, normal_from_height, to_uint8

WINDOW_ATLAS_VERSION = 2  # increase when the sprites change: forces the files to be regenerated

RGB = Tuple[int, int, int]
_GLASS_TOP = (128, 150, 170)
_GLASS_BOTTOM = (52, 68, 84)
_REVEAL = (46, 46, 48)
_FRAME = (241, 241, 238)
_SILL = (226, 223, 216)
_SHUTTER_GREEN = (84, 108, 78)
_SHUTTER_BROWN = (112, 66, 48)
_DOOR_WOOD = (108, 72, 46)
_DOOR_WHITE = (232, 232, 228)
_METAL = (58, 60, 64)
_CLOCK_GOLD = (205, 168, 72)
_CLOCK_PLATE = (30, 36, 56)
_CLOCK_DIAL = (20, 26, 44)
_ATLAS_WIDTH_PX = 2048
_EDGE_PAD_PX = 2  # border so that the normal map/cavity darkening do not "wrap" around the sprite edge


class WindowSprite(str, Enum):
    WINDOW_PLAIN = "window_plain"
    WINDOW_TRANSOM = "window_transom"
    WINDOW_SHUTTER_GREEN = "window_shutter_green"
    WINDOW_SHUTTER_BROWN = "window_shutter_brown"
    DOOR_WOOD = "door_wood"
    DOOR_WHITE = "door_white"
    BASEMENT_PLAIN = "basement_plain"
    BASEMENT_BARS = "basement_bars"
    TOWER_CLOCK = "tower_clock"


# (width, height) in meters. Windows include the sill at the bottom; with shutters the sprite is wider.
SPRITE_SIZES_M: Dict[WindowSprite, Tuple[float, float]] = {
    WindowSprite.WINDOW_PLAIN: (1.10, 1.46),
    WindowSprite.WINDOW_TRANSOM: (1.10, 1.46),
    WindowSprite.WINDOW_SHUTTER_GREEN: (1.78, 1.46),
    WindowSprite.WINDOW_SHUTTER_BROWN: (1.78, 1.46),
    WindowSprite.DOOR_WOOD: (1.00, 2.15),
    WindowSprite.DOOR_WHITE: (1.00, 2.15),
    WindowSprite.BASEMENT_PLAIN: (0.86, 0.56),
    WindowSprite.BASEMENT_BARS: (0.86, 0.56),
    WindowSprite.TOWER_CLOCK: (2.00, 2.00),
}


class WindowAtlasLayout:
    """Packs the sprites into rows and converts sprite -> pixel/UV rectangle."""

    def __init__(self, px_per_m: int = config.FACADE_WINDOW_ATLAS_PX_PER_M, gutter_px: int = config.FACADE_GUTTER_PX):
        self.px_per_m = px_per_m
        self.gutter_px = gutter_px
        self.width_px = _ATLAS_WIDTH_PX
        self._placement: Dict[WindowSprite, Tuple[int, int, int, int]] = {}

        x = y = row_height = 0
        for sprite in WindowSprite:
            w, h = self.size_px(sprite)
            block_w, block_h = w + 2 * gutter_px, h + 2 * gutter_px
            if x + block_w > self.width_px:
                x, y, row_height = 0, y + row_height, 0
            self._placement[sprite] = (x + gutter_px, y + gutter_px, w, h)
            x += block_w
            row_height = max(row_height, block_h)
        self.height_px = 1 << max(2, int(np.ceil(np.log2(y + row_height))))

    def size_m(self, sprite: WindowSprite) -> Tuple[float, float]:
        return SPRITE_SIZES_M[sprite]

    def size_px(self, sprite: WindowSprite) -> Tuple[int, int]:
        w, h = SPRITE_SIZES_M[sprite]
        return int(round(w * self.px_per_m)), int(round(h * self.px_per_m))

    def rect_px(self, sprite: WindowSprite) -> Tuple[int, int, int, int]:
        """(x, y, width, height) of the sprite content (without gutter); y from the top."""
        return self._placement[sprite]

    def uv_rect(self, sprite: WindowSprite) -> Tuple[float, float, float, float]:
        """(u_min, v_min, u_max, v_max) of the sprite content; V points up."""
        x, y, w, h = self._placement[sprite]
        return x / self.width_px, 1.0 - (y + h) / self.height_px, (x + w) / self.width_px, 1.0 - y / self.height_px


class _Canvas:
    """A sprite being drawn; coordinates in meters, origin bottom left, y up."""

    def __init__(self, width_m: float, height_m: float, px_per_m: int):
        self.w_m, self.h_m, self.ppm = width_m, height_m, px_per_m
        self.w, self.h = int(round(width_m * px_per_m)), int(round(height_m * px_per_m))
        self.albedo = np.zeros((self.h, self.w, 3))
        self.height = np.zeros((self.h, self.w))
        self.rough = np.full((self.h, self.w), 0.6)

    def _slice(self, x0: float, y0: float, x1: float, y1: float):
        c0, c1 = int(round(x0 * self.ppm)), int(round(x1 * self.ppm))
        r0, r1 = self.h - int(round(y1 * self.ppm)), self.h - int(round(y0 * self.ppm))
        return slice(max(r0, 0), min(r1, self.h)), slice(max(c0, 0), min(c1, self.w))

    def rect(self, box, color: RGB = None, height: float = None, lift: float = None, rough: float = None) -> None:
        rows, cols = self._slice(*box)
        if color is not None:
            self.albedo[rows, cols] = np.array(color, dtype=np.float64) / 255.0
        if height is not None:
            self.height[rows, cols] = height
        if lift is not None:
            self.height[rows, cols] += lift
        if rough is not None:
            self.rough[rows, cols] = rough

    def glass(self, box, height: float) -> None:
        rows, cols = self._slice(*box)
        count = rows.stop - rows.start
        if count <= 0 or cols.stop <= cols.start:
            return
        ramp = np.linspace(0.0, 1.0, count)[:, None, None]
        top, bottom = np.array(_GLASS_TOP) / 255.0, np.array(_GLASS_BOTTOM) / 255.0
        self.albedo[rows, cols] = (top * (1 - ramp) + bottom * ramp) * np.ones((1, cols.stop - cols.start, 1))
        self.height[rows, cols] = height
        self.rough[rows, cols] = 0.08

    def slats(self, box, count: int, color: RGB, depth: float) -> None:
        """Horizontal slats (window shutter)."""
        rows, cols = self._slice(*box)
        h, w = rows.stop - rows.start, cols.stop - cols.start
        if h <= 0 or w <= 0:
            return
        span = np.arange(h) / float(h) * count
        line = (span - np.floor(span)) < 0.30
        mask = np.broadcast_to(line[:, None], (h, w))
        self.albedo[rows, cols][mask] = np.array(color, dtype=np.float64) / 255.0
        self.height[rows, cols][mask] -= depth

    def bars(self, box, count: int, color: RGB) -> None:
        """Vertical grille bars."""
        rows, cols = self._slice(*box)
        h, w = rows.stop - rows.start, cols.stop - cols.start
        if h <= 0 or w <= 0:
            return
        span = (np.arange(w) + 0.5) / float(w) * count
        line = np.abs(span - np.round(span)) < 0.16
        inner = ((np.round(span) >= 1) & (np.round(span) <= count - 1))[None, :]  # bars only between the edges
        mask = np.broadcast_to(line[None, :], (h, w)) & inner
        self.albedo[rows, cols][mask] = np.array(color, dtype=np.float64) / 255.0
        self.height[rows, cols][mask] += 2.5
        self.rough[rows, cols][mask] = 0.4


class WindowAtlasGenerator:
    """Draws the window atlas (albedo, normal map, roughness)."""

    def __init__(self, layout: WindowAtlasLayout = None):
        self._layout = layout or WindowAtlasLayout()

    def generate(self) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8 RGB images (height_px, width_px, 3)
        """
        layout, gutter = self._layout, self._layout.gutter_px
        albedo = np.full((layout.height_px, layout.width_px, 3), 128, np.uint8)
        normal = np.tile(np.array([128, 128, 255], np.uint8), (layout.height_px, layout.width_px, 1))
        roughness = np.full_like(albedo, 200)

        for sprite in WindowSprite:
            content = self._finish(*self._draw(sprite))
            x, y, w, h = layout.rect_px(sprite)
            for target, image in zip((albedo, normal, roughness), content):
                # Fill the gutter with the edge color: bilinear filtering/mips only see the sprite itself
                block = np.pad(image, ((gutter, gutter), (gutter, gutter), (0, 0)), mode="edge")
                target[y - gutter : y + h + gutter, x - gutter : x + w + gutter] = block
        return {"albedo": albedo, "normal": normal, "roughness": roughness}

    # ------------------------------------------------------------------ Sprites

    def _draw(self, sprite: WindowSprite):
        width_m, height_m = self._layout.size_m(sprite)
        canvas = _Canvas(width_m, height_m, self._layout.px_per_m)
        drawers = {
            WindowSprite.WINDOW_PLAIN: lambda: self._window(canvas, transom=False, shutters=None),
            WindowSprite.WINDOW_TRANSOM: lambda: self._window(canvas, transom=True, shutters=None),
            WindowSprite.WINDOW_SHUTTER_GREEN: lambda: self._window(canvas, transom=False, shutters=_SHUTTER_GREEN),
            WindowSprite.WINDOW_SHUTTER_BROWN: lambda: self._window(canvas, transom=False, shutters=_SHUTTER_BROWN),
            WindowSprite.DOOR_WOOD: lambda: self._door(canvas, _DOOR_WOOD, glazed=True),
            WindowSprite.DOOR_WHITE: lambda: self._door(canvas, _DOOR_WHITE, glazed=False),
            WindowSprite.BASEMENT_PLAIN: lambda: self._basement(canvas, bars=False),
            WindowSprite.BASEMENT_BARS: lambda: self._basement(canvas, bars=True),
            WindowSprite.TOWER_CLOCK: lambda: self._clock(canvas),
        }
        drawers[sprite]()
        return canvas.albedo, canvas.height, canvas.rough

    @staticmethod
    def _window(c: _Canvas, transom: bool, shutters) -> None:
        window_w = 1.10
        x0 = (c.w_m - window_w) / 2
        x1 = x0 + window_w
        top = c.h_m

        c.rect((0, 0, c.w_m, top), color=_REVEAL, height=0.0)  # nothing below the sprite shows through
        if shutters is not None:
            for sx0, sx1 in ((0.0, x0 - 0.02), (x1 + 0.02, c.w_m)):
                c.rect((sx0, 0.06, sx1, top - 0.02), color=shutters, height=1.5, rough=0.6)
                c.slats((sx0, 0.06, sx1, top - 0.02), 16, tuple(int(v * 0.62) for v in shutters), depth=0.8)

        c.rect((x0, 0.06, x1, top), color=_REVEAL, height=-6.0)  # reveal
        c.rect((x0, 0.0, x1, 0.06), color=_SILL, height=2.0, rough=0.7)  # sill
        c.rect((x0 + 0.04, 0.10, x1 - 0.04, top - 0.04), color=_FRAME, height=-4.0, rough=0.5)  # frame
        glass = (x0 + 0.09, 0.15, x1 - 0.09, top - 0.09)
        c.glass(glass, height=-6.5)
        mid = (x0 + x1) / 2
        c.rect((mid - 0.025, 0.15, mid + 0.025, top - 0.09), color=_FRAME, height=-4.5, rough=0.5)  # mullion
        if transom:
            bar = 0.15 + 0.70 * (top - 0.09 - 0.15)
            c.rect((x0 + 0.09, bar - 0.025, x1 - 0.09, bar + 0.025), color=_FRAME, height=-4.5, rough=0.5)  # transom bar

    @staticmethod
    def _door(c: _Canvas, leaf: RGB, glazed: bool) -> None:
        w, h = c.w_m, c.h_m
        dark = tuple(int(v * 0.6) for v in _REVEAL)
        c.rect((0, 0, w, h), color=dark, height=-7.0)  # reveal
        c.rect((0.04, 0.0, w - 0.04, h - 0.04), color=_FRAME, height=-5.0, rough=0.5)  # door frame
        c.rect((0.09, 0.0, w - 0.09, h - 0.09), color=leaf, height=-4.0, rough=0.55)  # door leaf
        shade = tuple(int(v * 0.86) for v in leaf)
        c.rect((0.17, 0.12, w - 0.17, 0.95), color=shade, height=-5.0)  # lower panel
        if glazed:
            c.glass((0.17, 1.15, w - 0.17, h - 0.25), height=-5.5)
        else:
            c.rect((0.17, 1.15, w - 0.17, h - 0.25), color=shade, height=-5.0)
        c.rect((w - 0.21, 0.98, w - 0.16, 1.08), color=(200, 190, 150), lift=2.0, rough=0.3)  # handle

    @staticmethod
    def _basement(c: _Canvas, bars: bool) -> None:
        w, h = c.w_m, c.h_m
        c.rect((0, 0, w, h), color=_REVEAL, height=-6.0)
        c.rect((0.03, 0.03, w - 0.03, h - 0.03), color=(210, 210, 206), height=-4.0, rough=0.6)  # frame
        c.glass((0.08, 0.08, w - 0.08, h - 0.08), height=-6.0)
        c.rect((w / 2 - 0.02, 0.08, w / 2 + 0.02, h - 0.08), color=(210, 210, 206), height=-4.5, rough=0.6)
        if bars:
            c.bars((0.08, 0.08, w - 0.08, h - 0.08), 4, _METAL)

    @staticmethod
    def _clock(c: _Canvas) -> None:
        """Tower clock: dark plate with golden frame, ring, hour marks and hands (10:10)."""
        rows, cols = np.indices((c.h, c.w))
        x = (cols + 0.5) / c.ppm - c.w_m / 2  # meters to the right of the center
        y = c.h_m / 2 - (rows + 0.5) / c.ppm  # meters upward
        radius = np.hypot(x, y)
        angle = np.arctan2(x, y)  # clockwise from 12 o'clock

        gold = np.array(_CLOCK_GOLD) / 255.0
        c.albedo[:] = np.array(_CLOCK_PLATE) / 255.0
        c.height[:] = 0.0
        c.rough[:] = 0.55

        border = (np.abs(x) > c.w_m / 2 - 0.07) | (np.abs(y) > c.h_m / 2 - 0.07)
        dial = radius < 0.80
        ring = (radius >= 0.80) & (radius < 0.85)
        c.albedo[dial] = np.array(_CLOCK_DIAL) / 255.0
        c.height[dial] = -1.0

        marks = np.zeros_like(border)
        for hour in range(12):
            centre = np.radians(30 * hour)
            offset = np.abs((angle - centre + np.pi) % (2 * np.pi) - np.pi) * radius  # lateral distance in meters
            quarter = hour % 3 == 0
            marks |= (radius >= (0.55 if quarter else 0.62)) & (radius <= 0.76) & (offset <= (0.045 if quarter else 0.025))

        def hand(degrees: float, length: float, width: float) -> np.ndarray:
            direction = np.array([np.sin(np.radians(degrees)), np.cos(np.radians(degrees))])
            along = np.clip(x * direction[0] + y * direction[1], 0.0, length)
            return np.hypot(x - along * direction[0], y - along * direction[1]) <= width / 2

        hands = hand(305.0, 0.45, 0.07) | hand(60.0, 0.68, 0.05) | (radius <= 0.06)
        for mask in (border, ring, marks, hands):
            c.albedo[mask] = gold
            c.height[mask] = 2.5
            c.rough[mask] = 0.35

    @staticmethod
    def _finish(albedo: np.ndarray, height: np.ndarray, rough: np.ndarray):
        """Normal map and cavity darkening; pad the border first so that nothing "wraps" around the sprite."""
        pad = ((_EDGE_PAD_PX, _EDGE_PAD_PX), (_EDGE_PAD_PX, _EDGE_PAD_PX))
        padded_height = np.pad(height, pad, mode="edge")
        smooth = gaussian_blur_wrap(padded_height, 0.8)
        cavity = np.clip(gaussian_blur_wrap(padded_height, 4.0) - smooth, 0.0, 8.0) / 8.0
        normal = normal_from_height(smooth, 1.0)[_EDGE_PAD_PX:-_EDGE_PAD_PX, _EDGE_PAD_PX:-_EDGE_PAD_PX]
        cavity = cavity[_EDGE_PAD_PX:-_EDGE_PAD_PX, _EDGE_PAD_PX:-_EDGE_PAD_PX]
        return to_uint8(albedo * (1.0 - 0.45 * cavity)[..., None]), normal, gray_to_rgb(rough)
