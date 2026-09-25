"""
Plaster colors of the walls and deterministic choice per building.

The houses are plastered: mostly white, occasionally beige tones, very rarely red tones. Each color is its own
wall material (same normal/roughness texture, own albedo texture).
"""

import zlib
from dataclasses import dataclass
from typing import Dict, Tuple

RGB = Tuple[int, int, int]
_WEIGHT_TOTAL = 1000


@dataclass(frozen=True)
class PlasterColor:
    name: str
    rgb: RGB
    weight: int  # per mille of all buildings


PLASTER_COLORS: Tuple[PlasterColor, ...] = (
    PlasterColor("white", (238, 236, 230), 580),
    PlasterColor("cream", (238, 229, 205), 160),
    PlasterColor("beige", (222, 205, 172), 120),
    PlasterColor("sand", (207, 184, 146), 80),
    PlasterColor("salmon", (214, 160, 138), 35),
    PlasterColor("terracotta", (184, 108, 84), 25),
)

if sum(color.weight for color in PLASTER_COLORS) != _WEIGHT_TOTAL:
    raise ValueError("The plaster color weights must add up to 1000 per mille")


def stable_hash(key: str) -> int:
    """
    Reproducible hash (crc32). Python `hash()` is salted per process and would reshuffle the colors on every
    export.
    """
    return zlib.crc32(key.encode("utf-8"))


def building_key(building: Dict) -> str:
    """Key of a building: gml:id, otherwise the centroid from the bounds rounded to 10 cm."""
    building_id = building.get("id")
    if building_id and building_id != "unknown":
        return str(building_id)
    b = building["bounds"]
    return f"{(b[0] + b[3]) / 2:.1f}_{(b[1] + b[4]) / 2:.1f}"


def plaster_index(key: str) -> int:
    """Index into PLASTER_COLORS; the probability follows the weights (crc32 of the key mod 1000)."""
    point = stable_hash(key) % _WEIGHT_TOTAL
    upper = 0
    for index, color in enumerate(PLASTER_COLORS):
        upper += color.weight
        if point < upper:
            return index
    raise AssertionError("unreachable: weights add up to 1000")


def choice(key: str, salt: str, count: int) -> int:
    """Reproducible random choice 0..count-1 from building key and purpose (`salt`)."""
    return stable_hash(f"{key}|{salt}") % count
