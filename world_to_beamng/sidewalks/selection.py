"""
Which side of a road way has a sidewalk, from its OSM `sidewalk*` tags.

OSM has no kerb lines along roads (on the Baden-Wuerttemberg test map: no barrier=kerb way, kerb=* only on a few crossing
nodes), but a sidewalk beside a carriageway practically always has a kerb - so the sidewalk tags decide. Sides are
relative to the drawing direction of the way. `separate` (the sidewalk is drawn as its own footway) counts like `yes`:
the kerb still runs along the carriageway edge.

Precedence: `sidewalk` first, then `sidewalk:both`, then `sidewalk:left` / `sidewalk:right` for the side they name.
"""

from typing import Dict, FrozenSet, Mapping, Optional

SIDES = ("left", "right")
_PRESENT = frozenset({"yes", "separate"})
_BASE = {"both": SIDES, "yes": SIDES, "separate": SIDES, "left": ("left",), "right": ("right",)}


def sidewalk_sides(tags: Mapping[str, str]) -> Dict[str, bool]:
    """{"left": bool, "right": bool} - whether the road has a sidewalk on that side."""
    present = {side: side in _BASE.get(tags.get("sidewalk", ""), ()) for side in SIDES}
    both = tags.get("sidewalk:both")
    if both is not None:
        present = {side: both in _PRESENT for side in SIDES}
    for side in SIDES:
        value = tags.get(f"sidewalk:{side}")
        if value is not None:
            present[side] = value in _PRESENT
    return present


def _surface(tags: Mapping[str, str], side: str) -> Optional[str]:
    for key in (f"sidewalk:{side}:surface", "sidewalk:both:surface", "sidewalk:surface"):
        if tags.get(key):
            return tags[key]
    return None


def select_sidewalk_sides(tags: Mapping[str, str], excluded_highways: FrozenSet[str], mapping: Mapping) -> Dict[str, str]:
    """
    {side: surface type name} for every side with a sidewalk; empty for excluded highway types.

    Args:
        mapping: the "sidewalks" section of data/osm_to_beamng.json - "surface_materials" maps an OSM surface value to
            a surface type of "surface_types", "default_surface_material" is used without or with an unknown value
    """
    if tags.get("highway") in excluded_highways:
        return {}
    surfaces = mapping.get("surface_materials", {})
    default = mapping.get("default_surface_material", "asphalt_road_standard")
    present = sidewalk_sides(tags)
    return {side: surfaces.get(_surface(tags, side), default) for side in SIDES if present[side]}
