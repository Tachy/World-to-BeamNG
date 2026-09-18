"""
Bodenbewuchs (Gras, Blumen, Farn, Unkraut) als BeamNG-`GroundCover`-Objekte.

Grashalme sind in BeamNG kein Teil der Terrain-Textur: ein GroundCover-Objekt hat
EIN Billboard-Material (Textur-Atlas, gemeinsame Assets unter
/assets/materials/foliage/...) und mehrere `Types` (Atlas-Ausschnitt, Größe,
Klumpung), die über `layer` an den Namen eines Terrain-Materials gebunden sind.
Die Vorlagen stammen aus BeamNGs eigenen Levels (siehe
tools/extract_ground_cover_templates.py -> data/ground_cover_templates.json).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from uuid import uuid4

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "data" / "ground_cover_templates.json"

# Objekt-Felder, die aus der Vorlage übernommen werden (radius/maxElements werden
# separat begrenzt bzw. gesetzt).
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
    """Lädt data/ground_cover_templates.json ({"billboard_materials": ..., "templates": ...})."""
    return json.loads(Path(path or TEMPLATES_PATH).read_text(encoding="utf-8"))


def build_ground_cover_items(
    landuse_mappings: Dict,
    used_layers: Sequence[str],
    templates_data: Dict,
    max_elements: int,
    max_radius: float,
) -> List[Dict]:
    """
    Baut je Terrain-Layer und Vorlage ein GroundCover-Objekt.

    Args:
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"] (pro Kategorie
            "internal_name" = Layer-Name und "groundCover" = Liste von Vorlagennamen)
        used_layers: Layer-Namen, die in der Layer-Map tatsächlich vorkommen
            (für nicht gemalte Layer wären die Objekte nutzlos)
        templates_data: Ergebnis von load_ground_cover_templates()
        max_elements: Obergrenze gleichzeitig gezeichneter Elemente je Objekt
        max_radius: Obergrenze für die Sichtweite (Meter) je Objekt

    Returns:
        Liste von Item-Feldern ("name", "material", "radius", "Types", ...)
        für ItemManager.add_ground_cover().
    """
    templates = templates_data["templates"]
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
            item = {
                "name": f"gc_{layer}_{template_name}",
                "material": template["material"],
                "radius": radius,
                "maxElements": int(max_elements),
                # Ohne layer würde ein Typ auf ALLEN Terrain-Materialien wachsen
                "Types": [dict(t, layer=layer) for t in template["types"]],
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
    Liefert die Billboard-Materialien (Name -> Material-JSON) aller von `items`
    verwendeten Atlanten, jeweils mit neuer persistentId.
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
