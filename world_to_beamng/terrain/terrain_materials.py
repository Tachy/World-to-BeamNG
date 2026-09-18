"""
Baut die .ter-Layer-Map (Material-Index pro Rasterzelle) aus zwei Quellen:

1. Luftbild-Fallback: EIN Foto-Material (AERIAL_PHOTO_MATERIAL_NAME) für die
   gesamte Fläche - io/aerial.py setzt das Gesamtfoto selbst aus allen
   DOP20-Quellbildern zusammen (siehe Recherche 2026-09-18: viele einzigartige
   500m-Kachel-Materialien überfordern BeamNGs Terrain-Atlas-Packer).
2. OSM-Landnutzung: Polygone aus data/osm_to_beamng.json["landuse_mappings"]
   werden priorisiert in die Layer-Map gebrannt (Spec Abschnitt 6).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from affine import Affine
from PIL import Image
from rasterio.features import rasterize

# Flache Platzhalter für die Pflicht-Texturslots, die BeamNGs v1.5-Terrain-
# Material-System für JEDEN Layer verlangt (siehe ensure_flat_pbr_placeholders()).
# Werte sind die PBR-"neutral"-Konvention: Normal zeigt gerade nach oben,
# Roughness/Height sind mittelgrau (neutral), AO ist weiß (keine zusätzliche
# Verschattung), baseColor ist beliebig (wird immer mit Strength=0 benutzt).
_FLAT_PLACEHOLDER_COLORS = {
    "baseColor": (128, 128, 128),
    "normal": (128, 128, 255),
    "roughness": (128, 128, 128),
    "ao": (255, 255, 255),
    "height": (128, 128, 128),
}

# Pixelgröße der Detail-Texturen: muss exakt der detailTexSize der
# TerrainMaterialTextureSet entsprechen (siehe build_terrain_material_texture_set()).
DETAIL_TEX_SIZE = 1024

# Stärke, mit der die graue Detail-Textur über das Luftbild gelegt wird
# (BeamNGs eigene Gras-Materialien nutzen 0.15-0.2, siehe east_coast_usa).
DEFAULT_DETAIL_STRENGTH = 0.25

EMPTY_RASTER_VALUE = 255


def get_landuse_category(osm_tags: Dict, landuse_mappings: Dict) -> Optional[str]:
    """
    Ermittelt die landuse_mappings-Kategorie für ein OSM-Element.

    Jede Kategorie listet unter "osm_tags" ihre zugehörigen Tag-Werte, z.B.
    {"landuse": ["meadow", "grass"], "natural": ["grassland"]}. Passen mehrere
    Kategorien (z.B. landuse=meadow + natural=wood), gewinnt die mit der
    höchsten "priority". Kategorien ohne "osm_tags" (z.B. der "base"-
    Fallback-Eintrag) oder mit "active": false werden nie zugeordnet.

    Returns:
        Kategoriename oder None, falls kein Treffer
    """
    best_category = None
    best_priority = None
    for category, data in landuse_mappings.items():
        category_tags = data.get("osm_tags")
        if not category_tags or data.get("active", True) is False:
            continue
        if not any(osm_tags.get(key) in values for key, values in category_tags.items()):
            continue
        priority = data.get("priority", 0)
        if best_priority is None or priority > best_priority:
            best_category, best_priority = category, priority
    return best_category


AERIAL_PHOTO_MATERIAL_NAME = "aerial_photo"


def build_photo_fallback_layer(size: int) -> Tuple[np.ndarray, List[str]]:
    """
    Baut die Basis-Layer-Map: die GESAMTE Fläche bekommt EIN einziges
    Luftbild-Material (AERIAL_PHOTO_MATERIAL_NAME), nicht mehr ein eigenes
    Material pro 500m-Kachel.

    Hintergrund (Recherche 2026-09-18): BeamNGs v1.5-Terrain-Material-System
    ist für eine kleine Anzahl wiederholender Materialien ausgelegt, nicht für
    viele (16-25) einzigartige 4096px-Texturen - der Atlas-Packer hat dabei
    einzelne Kacheln sichtbar verdreht dargestellt, obwohl die Quelldateien
    nachweislich korrekt waren. io/aerial.py setzt das Luftbild jetzt selbst
    zu EINEM Gesamtfoto zusammen (siehe process_aerial_images()); die
    Layer-Map muss daher nur noch überall auf denselben Material-Index (0)
    zeigen.

    Returns:
        (layer_map, material_names) - layer_map ist (size, size) uint8 mit
        lauter Nullen, material_names = [AERIAL_PHOTO_MATERIAL_NAME]
    """
    layer_map = np.zeros((size, size), dtype=np.uint8)
    return layer_map, [AERIAL_PHOTO_MATERIAL_NAME]


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
        # keep_photo-Kategorien (Wohn-/Gewerbegebiete, Wasser) haben kein eigenes
        # Material: sie stellen das Luftbild (Index 0) über darunterliegenden
        # Layern wieder her.
        internal_name = None if category_data.get("keep_photo") else category_data["internal_name"]
        scored.append((category_data.get("priority", 0), poly["geometry"], internal_name))

    # Aufsteigend nach Priorität sortieren -> hohe Priorität wird zuletzt (obenauf) gebrannt
    scored.sort(key=lambda item: item[0])

    for _priority, geometry, internal_name in scored:
        if internal_name is None:
            material_index = 0
        else:
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


def mask_layer_map_with_photo(
    layer_map: np.ndarray,
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    geometries: List,
    buffer: float = 0.0,
) -> np.ndarray:
    """
    Setzt die Layer-Map unter den Geometrien auf das Luftbild (Index 0) zurück.

    Grund: Bodenbewuchs (GroundCover) wächst auf dem Terrain-LAYER. Straßen sind
    Decals über dem Terrain - ohne diese Maskierung würde Gras durch Straßen und
    Häuser wachsen, sobald darunter ein Landnutzungs-Layer liegt. Auf dem
    Foto-Layer wächst nichts.

    Args:
        geometries: shapely-Geometrien in lokalen Koordinaten (z.B. Straßenflächen,
            Gebäudegrundrisse)
        buffer: Puffer in Metern um jede Geometrie (z.B. Straßenschulter)

    Returns:
        Neue layer_map (Eingabe bleibt unverändert)
    """
    result = layer_map.copy()
    shapes = []
    for geometry in geometries:
        if buffer:
            geometry = geometry.buffer(buffer)
        if geometry is not None and not geometry.is_empty:
            shapes.append((geometry, 1))
    if not shapes:
        return result

    transform = Affine.translation(origin_x, origin_y) * Affine.scale(square_size, square_size)
    mask = rasterize(shapes, out_shape=(size, size), transform=transform, fill=0, dtype="uint8")
    result[mask == 1] = 0
    return result


DETAIL_TEXTURE_KEYS = ("detailColorMap", "detailNormalMap")


def ensure_landuse_detail_textures_sized(
    landuse_mappings: Dict,
    detail_tex_size: int,
    beamng_dir: Path,
    textures_dir: Path,
    level_name: str,
) -> Dict:
    """
    Skaliert detailColorMap/detailNormalMap aller aktiven Landnutzungs-Kategorien
    auf detail_tex_size und gibt eine Kopie von landuse_mappings mit den
    (ggf. neuen) Pfaden zurück.

    Grund: baseColorDetailTex/normalDetailTex ALLER TerrainMaterial-Einträge
    müssen exakt die detailTexSize der TerrainMaterialTextureSet haben -
    sonst meldet BeamNG "dont have required size" und rendert für das
    GESAMTE Material die "warning texture" (einheitlich grauer Boden), siehe
    Recherche 2026-09-18. BeamNGs Terrain-Texturen sind bereits 1024 px groß;
    die Skalierung greift nur bei abweichenden Größen.

    Nur level-lokale Texturen (Pfad beginnt mit "levels/{level_name}/",
    also von tools/vendor_shared_textures.py bereits hierher kopiert) werden
    angefasst; andere Pfade bleiben unverändert.
    """
    target_size = (detail_tex_size, detail_tex_size)
    level_prefix = f"levels/{level_name}/"
    textures_dir.mkdir(parents=True, exist_ok=True)
    result: Dict = {}

    for category, data in landuse_mappings.items():
        data = dict(data)
        if data.get("active", True) is not False:
            for key in DETAIL_TEXTURE_KEYS:
                rel_path = data.get(key)
                if not rel_path or not rel_path.lstrip("/").startswith(level_prefix):
                    continue
                fs_path = beamng_dir / Path(rel_path.lstrip("/")).relative_to(level_prefix)
                if not fs_path.is_file():
                    continue
                with Image.open(fs_path) as img:
                    if img.size == target_size:
                        continue
                    resized_name = f"_terrain_detail_{fs_path.stem}_{detail_tex_size}.png"
                    resized_path = textures_dir / resized_name
                    if not resized_path.exists():
                        img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS).save(resized_path, "PNG")
                data[key] = f"/levels/{level_name}/art/shapes/textures/{resized_name}"
        result[category] = data

    return result


def ensure_flat_pbr_placeholders(
    textures_dir: Path,
    level_name: str,
    base_tex_size: int,
    detail_tex_size: int = 1024,
    macro_tex_size: int = 1024,
) -> Dict[str, Dict[str, str]]:
    """
    Erzeugt (einmalig) flache Platzhalter-PNGs für die Pflicht-Texturslots und
    gibt ihre Level-Pfade zurück, verschachtelt nach Tier (base/detail/macro).

    Grund: BeamNGs v1.5-Terrain-Material-Editor speichert laut offizieller Doku
    (https://documentation.beamng.com/modding/levels/level_formats/terrain/)
    KEIN TerrainMaterial mit leerem Texturslot - alle 5 Kanäle (baseColor,
    normal, roughness, ao, height) brauchen Base-, Detail- UND Macro-Textur,
    sonst rendert BeamNG das Material als "warning texture" (einheitlich
    dunkelgrauer Boden). Wir haben nur echte baseColor-Base-Daten (Luftbild/
    Landuse-Textur) - für die restlichen Slots reichen neutrale Platzhalter,
    deren Detail-/Macro-Anteil über *DetailStrength/*MacroStrength=0
    zusätzlich stummgeschaltet wird (siehe build_terrain_material_entries()).

    WICHTIG: Die Platzhalter müssen exakt die in der TerrainMaterialTextureSet
    deklarierte Pixelgröße je Tier haben (baseTexSize/detailTexSize/
    macroTexSize) - sonst loggt BeamNG "dont have required size of W-H" und
    rendert ebenfalls die "warning texture" (siehe Recherche 2026-09-18: ein
    generisches 8x8-Bild reichte NICHT, obwohl der Texturslot selbst gefüllt war).

    Returns:
        {"base": {"normal": "/levels/.../_flat_normal_4096.png", ...},
         "detail": {"baseColor": ..., "normal": ..., ...},
         "macro": {...}}
    """
    textures_dir.mkdir(parents=True, exist_ok=True)
    tier_sizes = {"base": base_tex_size, "detail": detail_tex_size, "macro": macro_tex_size}
    paths: Dict[str, Dict[str, str]] = {}

    for tier, size in tier_sizes.items():
        paths[tier] = {}
        for channel, color in _FLAT_PLACEHOLDER_COLORS.items():
            filename = f"_flat_{channel}_{size}.png"
            filepath = textures_dir / filename
            if not filepath.exists():
                Image.new("RGB", (size, size), color).save(filepath, "PNG")
            paths[tier][channel] = f"/levels/{level_name}/art/shapes/textures/{filename}"

    return paths


def _add_required_pbr_slots(entry: Dict, placeholders: Dict[str, Dict[str, str]]) -> None:
    """
    Ergänzt ein TerrainMaterial-Dict um die 13 Pflichtfelder (baseColor-Detail/
    -Macro sowie normal/roughness/ao/height je Base+Detail+Macro), die
    build_terrain_material_entries() nicht aus echten Daten hat - siehe
    ensure_flat_pbr_placeholders().

    Detail-/Macro-Anteil wird per *Strength=[0, 0] auf null gedämpft, damit
    der neutrale Platzhalterinhalt so oder so keine sichtbare Rolle spielt.
    baseColorBaseTex/-Size wird vom Aufrufer bereits gesetzt.
    """
    zero = [0.0, 0.0]
    # setdefault: bereits gesetzte echte Texturen (z.B. Detail-Textur einer
    # Landnutzungs-Kategorie) dürfen nicht durch Platzhalter überschrieben werden.
    entry.setdefault("baseColorDetailTex", placeholders["detail"]["baseColor"])
    entry.setdefault("baseColorDetailStrength", zero)
    entry.setdefault("baseColorMacroTex", placeholders["macro"]["baseColor"])
    entry.setdefault("baseColorMacroStrength", zero)

    for channel in ("normal", "roughness", "ao", "height"):
        entry.setdefault(f"{channel}BaseTex", placeholders["base"][channel])
        entry.setdefault(f"{channel}DetailTex", placeholders["detail"][channel])
        entry.setdefault(f"{channel}DetailStrength", zero)
        entry.setdefault(f"{channel}MacroTex", placeholders["macro"][channel])
        entry.setdefault(f"{channel}MacroStrength", zero)


def build_terrain_material_entries(
    material_names: List[str],
    photo_tile_names: List[str],
    landuse_mappings: Dict,
    level_name: str,
    photo_extent_size: float,
    placeholders: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Baut TerrainMaterial-JSON-Einträge für materials.json (Schema verifiziert
    gegen BeamNGs offizielle Doku, siehe _add_required_pbr_slots()).

    Args:
        material_names: alle Layer-Map-Materialnamen in Index-Reihenfolge
        photo_tile_names: Teilmenge von material_names, die auf das
                          zusammengesetzte Luftbild verweisen (aktuell nur
                          [AERIAL_PHOTO_MATERIAL_NAME], siehe
                          build_photo_fallback_layer())
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        level_name: für den Foto-Textur-Pfad
        photo_extent_size: Kantenlänge (Meter) der gesamten exportierten
                          Fläche, die das EINE Luftbild abdeckt (nicht mehr
                          eine 500m-Kachelgröße - das Foto wiederholt sich
                          nicht, sondern deckt den kompletten Bereich einmal ab)
        placeholders: von ensure_flat_pbr_placeholders() - Pflicht-Texturslots,
                      für die wir keine echten Daten haben

    Returns:
        {material_name: {...TerrainMaterial JSON...}}
    """
    entries: Dict[str, Dict] = {}
    landuse_by_internal_name = {v["internal_name"]: v for v in landuse_mappings.values() if v.get("internal_name")}
    photo_tile_set = set(photo_tile_names)

    for name in material_names:
        if name in photo_tile_set:
            entry = {
                "internalName": name,
                "class": "TerrainMaterial",
                "persistentId": str(uuid4()),
                # .png, nicht .dds: BeamNGs Terrain-Atlas-Packer erwartet eine PNG-
                # Quelltextur und cached sie selbst zu DDS (siehe io/aerial.py).
                "baseColorBaseTex": f"/levels/{level_name}/art/shapes/textures/{name}.png",
                "baseColorBaseTexSize": photo_extent_size,
            }
            _add_required_pbr_slots(entry, placeholders)
            entries[name] = entry
            continue

        category_data = landuse_by_internal_name.get(name)
        if category_data is None:
            continue

        # Farbe aus dem Luftbild (gleiche Basis-Textur wie das Foto-Material),
        # die Landnutzung liegt als graue Detail-Textur darüber. BeamNGs
        # Terrain-Texturen sind Detail-Texturen (near-greyscale) - als Basis
        # ergäben sie einheitlich graue Flächen.
        photo_name = photo_tile_names[0]
        strength = float(category_data.get("detailStrength", DEFAULT_DETAIL_STRENGTH))
        entry = {
            "internalName": name,
            "class": "TerrainMaterial",
            "persistentId": str(uuid4()),
            "baseColorBaseTex": f"/levels/{level_name}/art/shapes/textures/{photo_name}.png",
            "baseColorBaseTexSize": photo_extent_size,
            "baseColorDetailTex": category_data["detailColorMap"],
            "baseColorDetailStrength": [strength, 0.0],
        }
        if category_data.get("detailNormalMap"):
            entry["normalDetailTex"] = category_data["detailNormalMap"]
            entry["normalDetailStrength"] = [1.0, 0.0]
        if category_data.get("groundModelName"):
            # BeamNGs groundmodels.json kennt nur GROSSGESCHRIEBENE Namen; ohne
            # groundmodelName loggt BeamNG "ground model not found ... using asphalt".
            entry["groundmodelName"] = str(category_data["groundModelName"]).upper()
        _add_required_pbr_slots(entry, placeholders)
        entries[name] = entry

    return entries


def build_terrain_material_texture_set(name: str, base_tex_size: int = 512) -> Dict[str, Dict]:
    """
    Baut den TerrainMaterialTextureSet-Eintrag, den TerrainBlock.materialTextureSet
    referenziert. TerrainBlock löst diesen Namen über das SimObject-"name"-Feld auf
    (NICHT "internalName" - das ist nur für TerrainMaterial-Layer-Referenzen aus der
    .ter-Datei relevant), siehe offizielles Schema-Beispiel unter
    https://documentation.beamng.com/modding/levels/level_formats/terrain/:
    {"class": "TerrainMaterialTextureSet", "name": "...", "baseTexSize": [w,h], ...}.
    Ohne "name"-Feld findet BeamNG das Set nicht ("Failed to find
    TerrainMaterialTextureSet with name: ...") und stürzt beim ersten Terrain-Draw
    mit "D3D12: root cbv with 0 gpu va" ab, weil die Terrain-Material-Konstanten nie
    gebunden wurden.

    Args:
        name: Name des Sets (muss exakt mit TerrainBlock.materialTextureSet übereinstimmen)
        base_tex_size: Atlas-Auflösung in Pixel (quadratisch) für baseColor-Texturen

    Returns:
        {name: {...TerrainMaterialTextureSet JSON...}}
    """
    return {
        name: {
            "class": "TerrainMaterialTextureSet",
            "name": name,
            "baseTexSize": [base_tex_size, base_tex_size],
            "detailTexSize": [DETAIL_TEX_SIZE, DETAIL_TEX_SIZE],
            "macroTexSize": [1024, 1024],
        }
    }
