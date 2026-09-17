"""
Baut die .ter-Layer-Map (Material-Index pro Rasterzelle) aus zwei Quellen:

1. Luftbild-Fallback: pro 500m-Kachel (config.TILE_SIZE) ein eigenes Foto-
   Material, mit denselben Dateinamen ("tile_<x>_<y>"), die
   io/dae.py:create_terrain_materials_json() bereits für den bisherigen
   Mesh-Ansatz erzeugt hat.
2. OSM-Landnutzung: Polygone aus data/osm_to_beamng.json["landuse_mappings"]
   werden priorisiert in die Layer-Map gebrannt (Spec Abschnitt 6).
"""

from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from affine import Affine
from rasterio.features import rasterize

# Bewusst konservativer Startsatz (siehe Spec Abschnitt 6/Global Constraints).
# water/industrial/commercial/residential/vineyard/greenhouse_horticulture/
# orchard bleiben trotz vorhandenem landuse_mappings-Eintrag Foto-Fallback,
# bis ihre Texturpfade verifiziert und bewusst aktiviert werden.
ACTIVE_LANDUSE_CATEGORIES = {"forest", "meadow", "farmland"}

EMPTY_RASTER_VALUE = 255


def get_landuse_category(osm_tags: Dict, landuse_mappings: Dict) -> Optional[str]:
    """
    Ermittelt die landuse_mappings-Kategorie für ein OSM-Element.

    landuse_mappings ist flach nach Kategorienamen organisiert (z.B. "forest",
    "meadow") - der Kategoriename IST der OSM-Tag-Wert. Prüft landuse-, dann
    natural-, dann leisure-Tag.

    Returns:
        Kategoriename oder None, falls kein aktiver Treffer
    """
    for tag_key in ("landuse", "natural", "leisure"):
        value = osm_tags.get(tag_key)
        if value in landuse_mappings and value in ACTIVE_LANDUSE_CATEGORIES:
            return value
    return None


def build_photo_fallback_layer(
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    tile_size: float,
    real_max_x: float,
    real_max_y: float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Baut die Basis-Layer-Map: pro tile_size-Kachel (config.TILE_SIZE, Default
    500m) ein eigenes Foto-Material, benannt wie die bestehenden
    Terrain-Tile-Texturen ("tile_<x>_<y>").

    Das Heightmap-Array (size x size) ist auf die nächste Zweierpotenz
    aufgefüllt (siehe heightmap.py:build_heightmap()) und reicht daher über
    die echten Höhendaten hinaus. Für Zellen jenseits von real_max_x/
    real_max_y (Padding-Bereich) wird - analog zu build_heightmap()'s eigener
    Rand-Fortsetzung der Höhenwerte - dieselbe Kachel wie am echten Datenrand
    verwendet, statt eine neue Kachel zu erzeugen, für die es keine
    Luftbild-Textur auf der Festplatte gibt.

    Args:
        real_max_x, real_max_y: Welt-Koordinaten des Endes der ECHTEN
            (nicht gepaddeten) Höhendaten (origin_x/origin_y sind bereits die
            echte Min-Grenze, da Padding laut build_heightmap() nur nach
            rechts/unten erweitert, nie davor)

    Returns:
        (layer_map, material_names) - layer_map ist (size, size) uint8,
        material_names[i] ist der Materialname für layer_map-Wert i
    """
    layer_map = np.zeros((size, size), dtype=np.uint8)
    material_names: List[str] = []
    tile_index: Dict[Tuple[int, int], int] = {}

    for row in range(size):
        world_y = origin_y + row * square_size
        world_y_clamped = min(world_y, real_max_y)
        tile_y = int(np.floor(world_y_clamped / tile_size)) * int(tile_size)
        for col in range(size):
            world_x = origin_x + col * square_size
            world_x_clamped = min(world_x, real_max_x)
            tile_x = int(np.floor(world_x_clamped / tile_size)) * int(tile_size)

            key = (tile_x, tile_y)
            if key not in tile_index:
                if len(material_names) >= 254:
                    raise ValueError(
                        f"Mehr als 254 Foto-Kacheln im Terrain-Bereich "
                        f"({len(material_names)} bereits erzeugt, weitere Kachel "
                        f"tile_{tile_x}_{tile_y} würde das Limit überschreiten) - "
                        f"TERRAIN_SQUARE_SIZE oder TILE_SIZE erhöhen"
                    )
                tile_index[key] = len(material_names)
                material_names.append(f"tile_{tile_x}_{tile_y}")

            layer_map[row, col] = tile_index[key]

    return layer_map, material_names


def paint_landuse_materials(
    layer_map: np.ndarray,
    material_names: List[str],
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    landuse_polygons: List[Dict],
    landuse_mappings: Dict,
) -> Tuple[np.ndarray, List[str]]:
    """
    Brennt OSM-Landnutzungs-Polygone in die Layer-Map, priorisiert nach
    landuse_mappings[category]["priority"] (höhere Priorität gewinnt bei
    Überlappung, siehe Spec Abschnitt 6/8).

    Args:
        layer_map: (size, size) uint8, wird NICHT verändert (Kopie wird zurückgegeben)
        material_names: bisherige Materialliste (Foto-Fallback-Namen)
        landuse_polygons: Liste von {"osm_tags": Dict, "geometry": shapely.Polygon}
                          in lokalen (Grid-)Koordinaten
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]

    Returns:
        (neue layer_map, erweiterte material_names)
    """
    result = layer_map.copy()
    names = list(material_names)
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(square_size, square_size)

    scored = []
    for poly in landuse_polygons:
        category = get_landuse_category(poly["osm_tags"], landuse_mappings)
        if category is None:
            continue
        category_data = landuse_mappings[category]
        scored.append((category_data.get("priority", 0), poly["geometry"], category_data["internal_name"]))

    # Aufsteigend nach Priorität sortieren -> hohe Priorität wird zuletzt (obenauf) gebrannt
    scored.sort(key=lambda item: item[0])

    for _priority, geometry, internal_name in scored:
        if internal_name not in names:
            if len(names) >= 254:
                raise ValueError(
                    f"Mehr als 254 Materialien ({len(names)} bereits vorhanden, "
                    f"weitere Landnutzungs-Kategorie '{internal_name}' würde das Limit "
                    f"überschreiten) - Landnutzungs-Kategorien reduzieren"
                )
            names.append(internal_name)
        material_index = names.index(internal_name)

        mask = rasterize(
            [(geometry, material_index)],
            out_shape=(size, size),
            transform=transform,
            fill=EMPTY_RASTER_VALUE,
            dtype="uint8",
        )
        hit = mask != EMPTY_RASTER_VALUE
        result[hit] = mask[hit]

    return result, names


def build_terrain_material_entries(
    material_names: List[str],
    photo_tile_names: List[str],
    landuse_mappings: Dict,
    level_name: str,
    tile_size: float,
) -> Dict[str, Dict]:
    """
    Baut TerrainMaterial-JSON-Einträge für materials.json (Schema verifiziert
    gegen BeamNGs eigenes template-Level, siehe Task 7).

    Args:
        material_names: alle Layer-Map-Materialnamen in Index-Reihenfolge
        photo_tile_names: Teilmenge von material_names, die Foto-Kacheln sind
                          (Namen wie "tile_<x>_<y>")
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        level_name, tile_size: für den Foto-Textur-Pfad

    Returns:
        {material_name: {...TerrainMaterial JSON...}}
    """
    entries: Dict[str, Dict] = {}
    landuse_by_internal_name = {v["internal_name"]: v for v in landuse_mappings.values()}
    photo_tile_set = set(photo_tile_names)

    for name in material_names:
        if name in photo_tile_set:
            entries[name] = {
                "internalName": name,
                "class": "TerrainMaterial",
                "persistentId": str(uuid4()),
                "baseColorBaseTex": f"/levels/{level_name}/art/shapes/textures/{name}.dds",
                "baseColorBaseTexSize": tile_size,
            }
            continue

        category_data = landuse_by_internal_name.get(name)
        if category_data is None:
            continue

        entry = {
            "internalName": name,
            "class": "TerrainMaterial",
            "persistentId": str(uuid4()),
            "baseColorBaseTex": category_data["baseColorMap"],
            "baseColorBaseTexSize": 4.0,
        }
        if category_data.get("normalMap"):
            entry["normalBaseTex"] = category_data["normalMap"]
        entries[name] = entry

    return entries


def build_terrain_material_texture_set(name: str, base_tex_size: int = 512) -> Dict[str, Dict]:
    """
    Baut den TerrainMaterialTextureSet-Eintrag, den TerrainBlock.materialTextureSet
    referenziert (Schema verifiziert gegen BeamNGs template/gridmap_v2-Level:
    {"class": "TerrainMaterialTextureSet", "baseTexSize": [w,h], "detailTexSize": [...], "macroTexSize": [...]}).

    Args:
        name: Name des Sets (muss exakt mit TerrainBlock.materialTextureSet übereinstimmen)
        base_tex_size: Atlas-Auflösung in Pixel (quadratisch) für baseColor-Texturen

    Returns:
        {name: {...TerrainMaterialTextureSet JSON...}}
    """
    return {
        name: {
            "class": "TerrainMaterialTextureSet",
            "internalName": name,
            "baseTexSize": [base_tex_size, base_tex_size],
            "detailTexSize": [1024, 1024],
            "macroTexSize": [1024, 1024],
        }
    }
