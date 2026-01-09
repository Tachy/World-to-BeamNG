"""
LoD2-Gebäudedaten Verarbeitung für BeamNG.

Lädt 3D-Gebäudemodelle aus CityGML-Dateien (LGL Baden-Württemberg),
transformiert sie ins lokale Koordinatensystem und exportiert sie
als TSStatic-Objekte für BeamNG.

Format: CityGML 2km x 2km Kacheln in ZIP-Archiven
Ausgabe: .dae-Dateien pro Tile + main.items.json Einträge
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from lxml import etree
import zipfile


def load_citygml_from_zip(zip_path: Path) -> List[etree.Element]:
    """
    Extrahiert CityGML-Dateien aus einem ZIP-Archiv.

    Args:
        zip_path: Pfad zum ZIP-Archiv

    Returns:
        Liste von XML-ElementTree-Wurzeln
    """
    buildings = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith((".gml", ".xml")):
                    with zip_ref.open(file_info) as gml_file:
                        tree = etree.parse(gml_file)
                        buildings.append(tree.getroot())
    except Exception as e:
        print(f"[!] Fehler beim Laden von {zip_path}: {e}")

    return buildings


def parse_citygml_buildings(gml_root: etree.Element, local_offset=None) -> List[Dict]:
    """
    Parst CityGML-Daten und extrahiert Gebäudegeometrie.

    WICHTIG: Diese Funktion gibt NUR die RAW UTM-Koordinaten zurück, OHNE Normalisierung!
    Die Normalisierung erfolgt später in normalize_buildings_full() mit vollständiger 3D-Offset.

    Args:
        gml_root: XML-Wurzel des CityGML-Dokuments
        local_offset: NICHT VERWENDET - nur für Rückwärtskompatibilität

    Returns:
        Liste von Gebäude-Dicts mit:
        - 'id': Gebäude-ID
        - 'walls': List of (vertices, faces) für Wände (UTM-Koordinaten!)
        - 'roofs': List of (vertices, faces) für Dächer (UTM-Koordinaten!)
        - 'bounds': (min_x, min_y, min_z, max_x, max_y, max_z) in UTM
    """
    # Namespaces für CityGML 1.0 (LGL Baden-Württemberg)
    namespaces = {
        "gml": "http://www.opengis.net/gml",
        "bldg": "http://www.opengis.net/citygml/building/1.0",
        "core": "http://www.opengis.net/citygml/1.0",
    }

    buildings = []

    # Finde alle Building-Objekte
    for city_object in gml_root.findall(".//core:cityObjectMember", namespaces):
        building_elem = city_object.find("bldg:Building", namespaces)
        if building_elem is None:
            continue

        building_id = building_elem.get("{http://www.opengis.net/gml}id", "unknown")

        walls = []
        roofs = []
        all_vertices = []

        # Finde alle boundedBy-Elemente
        for bounded in building_elem.findall(".//bldg:boundedBy", namespaces):
            # WallSurface
            wall_surface = bounded.find("bldg:WallSurface", namespaces)
            if wall_surface is not None:
                wall_geom = _extract_surface_geometry(wall_surface, namespaces, local_offset)
                if wall_geom:
                    walls.extend(wall_geom)
                    for verts, _ in wall_geom:
                        all_vertices.append(verts)

            # RoofSurface
            roof_surface = bounded.find("bldg:RoofSurface", namespaces)
            if roof_surface is not None:
                roof_geom = _extract_surface_geometry(roof_surface, namespaces, local_offset)
                if roof_geom:
                    roofs.extend(roof_geom)
                    for verts, _ in roof_geom:
                        all_vertices.append(verts)

        # Berechne Bounding Box
        if all_vertices:
            all_verts_combined = np.vstack(all_vertices)
            bounds = (
                float(np.min(all_verts_combined[:, 0])),
                float(np.min(all_verts_combined[:, 1])),
                float(np.min(all_verts_combined[:, 2])),
                float(np.max(all_verts_combined[:, 0])),
                float(np.max(all_verts_combined[:, 1])),
                float(np.max(all_verts_combined[:, 2])),
            )

            buildings.append({"id": building_id, "walls": walls, "roofs": roofs, "bounds": bounds})

    return buildings


def _extract_surface_geometry(
    surface_elem: etree.Element, namespaces: Dict, local_offset: Tuple[float, float]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extrahiert Geometrie aus WallSurface oder RoofSurface.

    WICHTIG: Gibt RAW UTM-Koordinaten zurück, OHNE Normalisierung!
    Die Normalisierung erfolgt später zentral in normalize_buildings_full().

    Args:
        surface_elem: XML-Element für Oberfläche
        namespaces: XML-Namespaces
        local_offset: WIRD NICHT VERWENDET - nur für Kompatibilität

    Returns:
        Liste von (vertices, faces) Tupeln mit UTM-Koordinaten
    """
    geometries = []

    # Finde alle Polygon-Elemente
    for polygon in surface_elem.findall(".//gml:Polygon", namespaces):
        # Exterior Ring (Hauptpolygon)
        exterior = polygon.find(".//gml:exterior//gml:posList", namespaces)
        if exterior is None or not exterior.text:
            continue

        # Parse Koordinaten (x y z x y z ...)
        coords_text = exterior.text.strip().split()
        coords = np.array([float(c) for c in coords_text]).reshape(-1, 3)

        if len(coords) < 3:
            continue

        # WICHTIG: KEINE Normalisierung hier! Gib RAW UTM-Koordinaten zurück
        coords_utm = coords.astype(np.float64)

        # Erstelle Faces (Triangulation für Polygone)
        n_verts = len(coords_utm)
        if n_verts == 3:
            # Dreieck
            faces = np.array([[0, 1, 2]])
        elif n_verts == 4:
            # Viereck -> 2 Dreiecke
            faces = np.array([[0, 1, 2], [0, 2, 3]])
        else:
            # Polygon -> Fan-Triangulation
            faces = []
            for i in range(1, n_verts - 1):
                faces.append([0, i, i + 1])
            faces = np.array(faces)

        geometries.append((coords_utm, faces))

    return geometries


def transform_to_local_coords(vertices: np.ndarray, local_offset: Tuple[float, float]) -> np.ndarray:
    """
    Transformiert UTM-Koordinaten ins lokale System.

    Args:
        vertices: (N, 3) Array mit UTM-Koordinaten
        local_offset: (x_offset, y_offset)

    Returns:
        (N, 3) Array mit lokalen Koordinaten
    """
    result = vertices.copy()
    result[:, 0] -= local_offset[0]
    result[:, 1] -= local_offset[1]
    return result


def snap_buildings_to_terrain_batch(
    buildings: List[Dict], height_points: np.ndarray, height_elevations: np.ndarray
) -> List[Dict]:
    """
    Setzt mehrere Gebäude auf das Terrain (Batch-Normalisierung).

    ZENTRALE Z-KOORDINATEN-NORMALISIERUNG (Optimierte Version):
    Diese Funktion wird zentral nach dem Import aller Daten aufgerufen.
    Erstellt den KDTree EINMAL für alle Gebäude (sehr effizient).

    Args:
        buildings: Liste von Gebäude-Dicts
        height_points: (N, 2) Array mit XY-Koordinaten des Terrains (lokale Koordinaten)
        height_elevations: (N,) Array mit Z-Werten des Terrains (normalisiert)

    Returns:
        Liste von modifizierten Gebäude-Dicts mit normalisierter Z-Höhe
    """
    from scipy.spatial import cKDTree

    # Erstelle KDTree EINMAL (sehr teuer!)
    print("  [i] Erstelle KDTree für Terrain-Abfragen...")
    tree = cKDTree(height_points)
    print(f"  [✓] KDTree erstellt ({len(height_points):,} Punkte)")

    snapped_buildings = []
    for building_idx, building in enumerate(buildings):
        # Alle Vertices sammeln
        all_verts = []
        for verts, _ in building.get("walls", []):
            all_verts.append(verts)
        for verts, _ in building.get("roofs", []):
            all_verts.append(verts)

        if not all_verts:
            snapped_buildings.append(building)
            continue

        all_verts = np.vstack(all_verts)
        min_z = np.min(all_verts[:, 2])

        # Ermittle Terrain-Höhe am niedrigsten Punkt
        min_point_xy = all_verts[np.argmin(all_verts[:, 2]), :2]

        _, idx = tree.query(min_point_xy)
        terrain_z = height_elevations[idx]

        # Z-Offset berechnen
        z_offset = terrain_z - min_z

        # Alle Vertices anpassen
        building_copy = {
            "id": building.get("id"),
            "walls": [],
            "roofs": [],
            "bounds": building.get("bounds"),
        }

        for verts, faces in building.get("walls", []):
            building_copy["walls"].append((verts + [0, 0, z_offset], faces))

        for verts, faces in building.get("roofs", []):
            building_copy["roofs"].append((verts + [0, 0, z_offset], faces))

        snapped_buildings.append(building_copy)

        # Progress-Ausgabe alle 100 Gebäude
        if (building_idx + 1) % 100 == 0:
            print(f"    {building_idx + 1}/{len(buildings)} Gebäude normalisiert")

    return snapped_buildings


def snap_building_to_terrain(building: Dict, height_points: np.ndarray, height_elevations: np.ndarray) -> Dict:
    """
    Setzt Gebäude auf das Terrain (Snap-to-Terrain).

    Ermittelt den niedrigsten Punkt des Gebäudes und passt alle Vertices
    so an, dass dieser Punkt auf dem Terrain liegt.

    DEPRECATED: Verwende stattdessen snap_buildings_to_terrain_batch() für bessere Performance!

    Args:
        building: Gebäude-Dict mit 'walls' und 'roofs' Geometrien
        height_points: (N, 2) Array mit XY-Koordinaten des Terrains (lokale Koordinaten)
        height_elevations: (N,) Array mit Z-Werten des Terrains (normalisiert)

    Returns:
        Modifiziertes Gebäude-Dict mit Z-Koordinaten ans Terrain angepasst
    """
    from scipy.spatial import cKDTree

    # Alle Vertices sammeln
    all_verts = []
    for verts, _ in building.get("walls", []):
        all_verts.append(verts)
    for verts, _ in building.get("roofs", []):
        all_verts.append(verts)

    if not all_verts:
        return building

    all_verts = np.vstack(all_verts)
    min_z = np.min(all_verts[:, 2])

    # Ermittle Terrain-Höhe am niedrigsten Punkt
    min_point_xy = all_verts[np.argmin(all_verts[:, 2]), :2]

    tree = cKDTree(height_points)
    _, idx = tree.query(min_point_xy)
    terrain_z = height_elevations[idx]

    # Z-Offset berechnen
    z_offset = terrain_z - min_z

    # Alle Vertices anpassen
    for i, (verts, faces) in enumerate(building.get("walls", [])):
        building["walls"][i] = (verts + [0, 0, z_offset], faces)

    for i, (verts, faces) in enumerate(building.get("roofs", [])):
        building["roofs"][i] = (verts + [0, 0, z_offset], faces)

    return building


def normalize_buildings_full(
    buildings: List[Dict],
    local_offset: Tuple[float, float, float],
) -> List[Dict]:
    """
    Normalisiert Gebäude-Vertices ins lokale Koordinatensystem (X, Y, Z).

    ZENTRALE NORMALISIERUNG nach dem Import:
    - Alle Vertices werden sofort mit global_offset (inkl. Z) normalisiert
    - X, Y, Z-Koordinaten werden konsistent transformiert
    - Gebäude stehen direkt auf dem korrekten Höhenniveau

    Args:
        buildings: Liste von Gebäude-Dicts
        local_offset: (origin_x, origin_y, z_min) - 3D-Offset

    Returns:
        Liste von normalisierten Gebäude-Dicts
    """
    if not buildings or len(local_offset) < 3:
        return buildings

    ox, oy, oz = local_offset[0], local_offset[1], local_offset[2]
    normalized = []

    for building in buildings:
        # Kopiere Gebäude-Struktur
        building_norm = {
            "id": building.get("id"),
            "walls": [],
            "roofs": [],
            "bounds": building.get("bounds"),
        }

        # Normalisiere alle Wand-Vertices
        for verts, faces in building.get("walls", []):
            # Verts: (N, 3) array - konvertiere zu float für Subtraktion
            verts_norm = np.asarray(verts, dtype=np.float64).copy()
            verts_norm[:, 0] -= ox
            verts_norm[:, 1] -= oy
            verts_norm[:, 2] -= oz
            building_norm["walls"].append((verts_norm, faces))

        # Normalisiere alle Dach-Vertices
        for verts, faces in building.get("roofs", []):
            # Verts: (N, 3) array - konvertiere zu float für Subtraktion
            verts_norm = np.asarray(verts, dtype=np.float64).copy()
            verts_norm[:, 0] -= ox
            verts_norm[:, 1] -= oy
            verts_norm[:, 2] -= oz
            building_norm["roofs"].append((verts_norm, faces))

        # WICHTIG: Bounds NEU BERECHNEN aus normalisierten Vertices!
        # (Nicht einfach vom Offset subtrahieren, da parse_citygml_buildings schon
        # X/Y teilweise normalisiert hat)
        all_normalized_verts = []
        for verts, _ in building_norm["walls"]:
            all_normalized_verts.append(verts)
        for verts, _ in building_norm["roofs"]:
            all_normalized_verts.append(verts)

        if all_normalized_verts:
            all_verts_combined = np.vstack(all_normalized_verts)
            bounds = (
                float(np.min(all_verts_combined[:, 0])),
                float(np.min(all_verts_combined[:, 1])),
                float(np.min(all_verts_combined[:, 2])),
                float(np.max(all_verts_combined[:, 0])),
                float(np.max(all_verts_combined[:, 1])),
                float(np.max(all_verts_combined[:, 2])),
            )
            building_norm["bounds"] = bounds

        normalized.append(building_norm)

    return normalized


def cache_lod2_buildings(
    lod2_dir: str,
    bbox: Tuple[float, float, float, float],
    local_offset: Tuple[float, float, float],
    cache_dir: str,
    height_hash: str,
) -> str:
    """
    Lädt und cached LoD2-Gebäudedaten - mit sofortiger 3D-Normalisierung.

    Args:
        lod2_dir: Verzeichnis mit ZIP-Dateien
        bbox: (min_lat, min_lon, max_lat, max_lon) in WGS84
        local_offset: (x_offset, y_offset, z_offset) in lokalen Koordinaten - 3D!
        cache_dir: Cache-Verzeichnis
        height_hash: Hash für Cache-Validierung

    Returns:
        Pfad zur Cache-Datei
    """
    import hashlib
    from pyproj import Transformer

    # BBOX von WGS84 (Lat/Lon) zu UTM konvertieren
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    min_x_utm, min_y_utm = transformer.transform(bbox[1], bbox[0])  # lon, lat
    max_x_utm, max_y_utm = transformer.transform(bbox[3], bbox[2])
    bbox_utm = (min_x_utm, min_y_utm, max_x_utm, max_y_utm)

    bbox_utm = (min_x_utm, min_y_utm, max_x_utm, max_y_utm)

    # Cache-Key: Verwende direkt height_hash (tile_hash) für einheitliche Konsistenz
    # Alle Cache-Files für diesen Tile (OSM, LoD2, elevations, grid) nutzen dasselbe Hash
    cache_file = Path(cache_dir) / f"lod2_{height_hash}.pkl"

    if cache_file.exists():
        print(f"  [i] LoD2-Cache gefunden: {cache_file.name}")
        return str(cache_file)

    print(f"[9] Lade LoD2-Gebäudedaten aus {lod2_dir}...")

    lod2_path = Path(lod2_dir)
    if not lod2_path.exists():
        print(f"  [!] LoD2-Verzeichnis nicht gefunden: {lod2_dir}")
        return None

    # Sammle alle ZIP-Dateien
    zip_files = list(lod2_path.glob("*.zip"))
    if not zip_files:
        print(f"  [!] Keine ZIP-Dateien in {lod2_dir} gefunden")
        return None

    print(f"  [i] {len(zip_files)} ZIP-Archive gefunden")

    # ZENTRALE PIPELINE: Parse → BBOX-Filter (UTM) → Normalisierung (EINMAL!)
    all_buildings_raw_utm = []  # RAW UTM-Gebäude vor Filterung
    total_parsed = 0

    # PHASE 1: Parse alle Gebäude aus ZIPs (RAW UTM-Koordinaten)
    for zip_path in zip_files:
        gml_roots = load_citygml_from_zip(zip_path)
        for gml_root in gml_roots:
            buildings_raw = parse_citygml_buildings(gml_root, None)
            all_buildings_raw_utm.extend(buildings_raw)
            total_parsed += len(buildings_raw)

    print(f"  [i] {total_parsed} Gebäude aus ZIPs geparst")

    # PHASE 2: BBOX-Filterung in UTM-Koordinaten (VOR Normalisierung!)
    buildings_in_bbox_utm = []
    for building in all_buildings_raw_utm:
        bounds = building.get("bounds")
        if not bounds:
            continue
        # bounds = (min_x, min_y, min_z, max_x, max_y, max_z) in UTM
        center_x = (bounds[0] + bounds[3]) / 2
        center_y = (bounds[1] + bounds[4]) / 2
        # Prüfe ob Zentrum in BBOX liegt
        if bbox_utm[0] <= center_x <= bbox_utm[2] and bbox_utm[1] <= center_y <= bbox_utm[3]:
            buildings_in_bbox_utm.append(building)

    print(f"  [i] {len(buildings_in_bbox_utm)} Gebäude in UTM-BBOX gefunden")

    # PHASE 3: ZENTRALE Normalisierung EINMAL (danach NIE WIEDER!)
    all_buildings = normalize_buildings_full(buildings_in_bbox_utm, local_offset)
    print(f"  [✓] {len(all_buildings)} Gebäude normalisiert")

    # Pickle-Cache schreiben
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(all_buildings, f)

    print(f"  [✓] {len(all_buildings)} Gebäude gecached")

    return str(cache_file)


def load_buildings_from_cache(cache_file: str) -> List[Dict]:
    """Lädt Gebäude aus Cache."""
    if not cache_file or not Path(cache_file).exists():
        return []

    with open(cache_file, "rb") as f:
        return pickle.load(f)


def cache_normalized_buildings(
    buildings: List[Dict],
    height_points: np.ndarray,
    height_elevations: np.ndarray,
    cache_dir: str,
    tile_hash: str,
) -> Optional[str]:
    """
    Normalisiert Gebäude ans Terrain und cached sie.

    Args:
        buildings: Rohe Gebäude-Dicts
        height_points: (N, 2) Array mit XY-Koordinaten des Terrains
        height_elevations: (N,) Array mit Z-Werten des Terrains
        cache_dir: Cache-Verzeichnis
        tile_hash: Hash für Cache-Validierung

    Returns:
        Pfad zur normalisierten Cache-Datei oder None
    """
    if not buildings:
        return None

    cache_file = Path(cache_dir) / f"lod2_normalized_{tile_hash}.pkl"

    if cache_file.exists():
        print(f"  [i] Normalisierte LoD2-Cache gefunden: {cache_file.name}")
        return str(cache_file)

    # Normalisiere Gebäude
    normalized_buildings = snap_buildings_to_terrain_batch(buildings, height_points, height_elevations)

    # Cache schreiben
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(normalized_buildings, f)

    print(f"  [✓] {len(normalized_buildings)} normalisierte Gebäude gecached")

    return str(cache_file)


def export_buildings_to_dae(
    buildings: List[Dict],
    output_dir: str,
    tile_x: int,
    tile_y: int,
    wall_color: Tuple[float, float, float],
    roof_color: Tuple[float, float, float],
) -> Optional[str]:
    """
    Exportiert Gebäude als .dae-Datei.

    Args:
        buildings: Liste von Gebäude-Dicts
        output_dir: Ausgabeverzeichnis
        tile_x, tile_y: Tile-Koordinaten
        wall_color: RGB-Farbe für Wände (0-1) [UNUSED - via MaterialManager]
        roof_color: RGB-Farbe für Dächer (0-1) [UNUSED - via MaterialManager]

    Returns:
        Pfad zur erzeugten .dae-Datei oder None
    """
    from ..managers import DAEExporter
    from .. import config

    if not buildings:
        return None

    # Filtere Gebäude: Nur solche, deren Schwerpunkt innerhalb der Grid-Bounds liegt
    bounds = config.GRID_BOUNDS_LOCAL
    if bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        buildings_in_bounds = []
        for building in buildings:
            b = building.get("bounds")
            if b:
                centroid_x = (b[0] + b[3]) / 2.0
                centroid_y = (b[1] + b[4]) / 2.0
                if min_x <= centroid_x <= max_x and min_y <= centroid_y <= max_y:
                    buildings_in_bounds.append(building)

        if len(buildings) != len(buildings_in_bounds):
            print(f"  [LoD2] {len(buildings) - len(buildings_in_bounds)} Gebäude außerhalb Grid-Bounds übersprungen")

        buildings = buildings_in_bounds

    if not buildings:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dae_file = output_path / f"buildings_tile_{tile_x}_{tile_y}.dae"

    # Konvertiere Buildings zu DAEExporter Format
    meshes = []
    for bldg_idx, building in enumerate(buildings):
        all_vertices = []
        vertex_offset = 0
        wall_faces = []
        roof_faces = []

        # Wände
        for verts, faces in building.get("walls", []):
            for face in faces:
                wall_faces.append([f + vertex_offset for f in face])
            all_vertices.append(verts)
            vertex_offset += len(verts)

        # Dächer
        for verts, faces in building.get("roofs", []):
            for face in faces:
                roof_faces.append([f + vertex_offset for f in face])
            all_vertices.append(verts)
            vertex_offset += len(verts)

        if not all_vertices:
            continue

        vertices_combined = np.vstack(all_vertices)

        meshes.append(
            {
                "id": f"building_{bldg_idx}",
                "vertices": vertices_combined,
                "faces": {"lod2_wall_white": wall_faces, "lod2_roof_red": roof_faces},
            }
        )

    # Export mit DAEExporter
    exporter = DAEExporter()
    exporter.export_multi_mesh(output_path=str(dae_file), meshes=meshes, with_uv=False)

    print(f"  [✓] DAE exportiert: {dae_file.name} ({len(buildings)} Gebäude)")

    return str(dae_file)


def create_materials_json() -> Dict:
    """
    Erstellt die main.materials.json Einträge für LoD2-Gebäude.

    Returns:
        Dict mit Material-Definitionen
    """
    from ..managers import MaterialManager

    manager = MaterialManager(beamng_dir="")

    # Wall-Material (weiss)
    manager.add_building_material(
        "lod2_wall_white",
        color=[0.95, 0.95, 0.95, 1.0],
        groundType="STONE",
        materialTag0="beamng",
        materialTag1="Building",
    )

    # Roof-Material (rot)
    manager.add_building_material(
        "lod2_roof_red",
        color=[0.6, 0.2, 0.1, 1.0],
        groundType="ROOF_TILES",
        materialTag0="beamng",
        materialTag1="Building",
    )

    return manager.materials


def export_materials_json(output_dir: str) -> str:
    """
    Exportiert main.materials.json für LoD2-Gebäude.

    Args:
        output_dir: Ausgabeverzeichnis (BeamNG Level-Root)

    Returns:
        Pfad zur erstellten Datei
    """
    import json

    output_path = Path(output_dir)
    materials_file = output_path / "main.materials.json"

    materials = create_materials_json()

    # Wenn Datei existiert, merge mit existing
    if materials_file.exists():
        with open(materials_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(materials)
        materials = existing

    with open(materials_file, "w", encoding="utf-8") as f:
        json.dump(materials, f, indent=2)

    print(f"  [✓] Materials JSON: {materials_file.name}")
    return str(materials_file)


def create_items_json_entry(dae_path: str, tile_x: int, tile_y: int) -> Dict:
    """
    Erstellt einen main.items.json-Eintrag für ein Gebäude-Tile.

    Args:
        dae_path: Relativer Pfad zur .dae-Datei
        tile_x, tile_y: Tile-Koordinaten (Welt-Koordinaten der oberen linken Ecke)

    Returns:
        Dict für items.json
    """
    from ..managers import ItemManager
    import os

    manager = ItemManager(beamng_dir="")
    dae_filename = os.path.basename(dae_path)
    item_name = f"buildings_tile_{tile_x}_{tile_y}"

    manager.add_building(item_name, dae_filename, position=(0, 0, 0))

    return manager.items[item_name]
