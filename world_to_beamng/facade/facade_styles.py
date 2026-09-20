"""
Putzfarben der Wände und deterministische Wahl je Gebäude.

Die Häuser sind verputzt: vorwiegend weiß, vereinzelt Beigetöne, ganz vereinzelt Rottöne. Jede Farbe ist ein eigenes
Wandmaterial (dieselbe Normal-/Roughness-Textur, eigene Albedo-Textur).
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
    weight: int  # Promille aller Gebäude


PLASTER_COLORS: Tuple[PlasterColor, ...] = (
    PlasterColor("white", (238, 236, 230), 580),
    PlasterColor("cream", (238, 229, 205), 160),
    PlasterColor("beige", (222, 205, 172), 120),
    PlasterColor("sand", (207, 184, 146), 80),
    PlasterColor("salmon", (214, 160, 138), 35),
    PlasterColor("terracotta", (184, 108, 84), 25),
)

if sum(color.weight for color in PLASTER_COLORS) != _WEIGHT_TOTAL:
    raise ValueError("Die Gewichte der Putzfarben müssen sich zu 1000 Promille ergänzen")


def stable_hash(key: str) -> int:
    """
    Reproduzierbarer Hash (crc32). Python-`hash()` ist je Prozess gesalzen und würde die Farben bei jedem Export
    neu würfeln.
    """
    return zlib.crc32(key.encode("utf-8"))


def building_key(building: Dict) -> str:
    """Schlüssel eines Gebäudes: gml:id, sonst der auf 10 cm gerundete Schwerpunkt aus den bounds."""
    building_id = building.get("id")
    if building_id and building_id != "unknown":
        return str(building_id)
    b = building["bounds"]
    return f"{(b[0] + b[3]) / 2:.1f}_{(b[1] + b[4]) / 2:.1f}"


def plaster_index(key: str) -> int:
    """Index in PLASTER_COLORS; die Wahrscheinlichkeit folgt den Gewichten (crc32 des Schlüssels mod 1000)."""
    point = stable_hash(key) % _WEIGHT_TOTAL
    upper = 0
    for index, color in enumerate(PLASTER_COLORS):
        upper += color.weight
        if point < upper:
            return index
    raise AssertionError("unerreichbar: Gewichte ergeben 1000")


def choice(key: str, salt: str, count: int) -> int:
    """Reproduzierbare Zufallswahl 0..count-1 aus Gebäudeschlüssel und Zweck (`salt`)."""
    return stable_hash(f"{key}|{salt}") % count
