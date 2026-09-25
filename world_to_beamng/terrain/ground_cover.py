"""
Ground vegetation (grass, flowers, fern, weeds) as BeamNG `GroundCover` objects.

Grass blades are not part of the terrain texture in BeamNG: a GroundCover object has
ONE billboard material (texture atlas, shared assets under
/assets/materials/foliage/...) and several `Types` (atlas region, size,
clumping), which are bound to the name of a terrain material via `layer`.
The templates come from BeamNG's own levels (see
tools/extract_ground_cover_templates.py -> data/ground_cover_templates.json).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from uuid import uuid4

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "data" / "ground_cover_templates.json"

# Object fields taken over from the template (radius/maxElements are
# limited or set separately).
_PASSTHROUGH_FIELDS = (
    "gridSize",
    "maxBillboardTiltAngle",
    "windGustFrequency",
    "windGustLength",
    "windGustStrength",
    "windTurbulenceFrequency",
    "windTurbulenceStrength",
    "zOffset",
    "reflectScale",
)


def load_ground_cover_templates(path: Optional[Path] = None) -> Dict:
    """Loads data/ground_cover_templates.json ({"billboard_materials": ..., "templates": ...})."""
    return json.loads(Path(path or TEMPLATES_PATH).read_text(encoding="utf-8"))


def build_ground_cover_items(
    landuse_mappings: Dict,
    used_layers: Sequence[str],
    templates_data: Dict,
    max_elements: int,
    max_radius: float,
    layer_variants: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """
    Builds one GroundCover object per terrain layer and template.

    Args:
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"] (per category
            "internal_name" = layer name and "groundCover" = list of template names)
        used_layers: layer names that actually occur in the layer map
            (the objects would be useless for layers that are not painted)
        templates_data: result of load_ground_cover_templates()
        max_elements: upper limit of simultaneously drawn elements per object
        max_radius: upper limit for the view distance (meters) per object
        layer_variants: four-image mode: layer -> its tile variants (e.g. {"mat_grass": ["mat_grass_t0",
            "mat_grass_t1"]}). Each variant gets its OWN object (gc_<variant>_<template>) with the types of the
            template. The types must not be multiplied for several variants into ONE object: an object
            carries at most 8 types (all 229 objects in BeamNG's original levels have exactly 8), with more the
            grass is missing completely.

    Returns:
        List of item fields ("name", "material", "radius", "Types", ...)
        for ItemManager.add_ground_cover().
    """
    templates = templates_data["templates"]
    layer_variants = layer_variants or {}
    used = set(used_layers)
    items = []

    for category in landuse_mappings.values():
        layer = category.get("internal_name")
        if category.get("keep_photo") or not layer or layer not in used:
            continue
        for template_name in category.get("groundCover", []):
            template = templates.get(template_name)
            if template is None:
                continue

            radius = min(float(template.get("radius", max_radius)), float(max_radius))
            # One object per layer or (four-image mode) per tile variant of the layer
            for bound in layer_variants.get(layer, [layer]):
                item = {
                    "name": f"gc_{bound}_{template_name}",
                    "material": template["material"],
                    "radius": radius,
                    "maxElements": int(max_elements),
                    # Without layer, a type would grow on ALL terrain materials
                    "Types": [dict(t, layer=bound) for t in template["types"]],
                }
                for field in _PASSTHROUGH_FIELDS:
                    if field in template:
                        item[field] = template[field]
                for field in ("dissolveRadius", "shapeCullRadius"):
                    if field in template:
                        item[field] = min(float(template[field]), radius)
                items.append(item)

    return items


def build_billboard_material_entries(items: Sequence[Dict], templates_data: Dict) -> Dict[str, Dict]:
    """
    Returns the billboard materials (name -> material JSON) of all atlases used by
    `items`, each with a new persistentId.
    """
    materials = {}
    for item in items:
        name = item["material"]
        if name in materials:
            continue
        entry = json.loads(json.dumps(templates_data["billboard_materials"][name]))
        entry["persistentId"] = str(uuid4())
        materials[name] = entry
    return materials
