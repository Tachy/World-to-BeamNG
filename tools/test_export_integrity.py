#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit-Test für die Integrität aller exportierten BeamNG-Daten.

Prüft:
- DAE-Dateien (Terrain + Gebäude)
- JSON-Dateien (materials + items)
- Texturen
- Material/Shape-Referenzen
- XML-Validität
- Texture-Mapping (Tile-Namen zu Texturdateien)
"""

import sys
import os
import io
from pathlib import Path

# Setze UTF-8 Encoding für stdout
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json
import xml.etree.ElementTree as ET
from lxml import etree as lxml_etree

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config


class ExportIntegrityTest:
    """Test-Suite für Export-Integrität."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.dae_materials = set()  # Aus library_materials
        self.triangle_materials = set()  # Aus triangles material-Attributen
        self.beamng_dir = Path(config.BEAMNG_DIR)
        self.shapes_dir = Path(config.BEAMNG_DIR_SHAPES)
        self.textures_dir = Path(config.BEAMNG_DIR_TEXTURES)
        self.buildings_dir = Path(config.BEAMNG_DIR_BUILDINGS)
        self.cache_dir = Path(config.CACHE_DIR)
        self.debug_network_path = self.cache_dir / "debug_network.json"

    def _resolve_relative_path(self, relative_path):
        r"""
        Konvertiert einen relativen BeamNG-Pfad zu absolutem Windows-Pfad.

        Args:
            relative_path: Pfad im Format '\levels\World_to_BeamNG\art\shapes\textures\file.dds'

        Returns:
            Absoluter Path-Objekt
        """
        if relative_path.startswith("\\levels\\"):
            # Entferne Präfix und konvertiere zu Windows-Pfad
            level_path = relative_path.replace("\\levels\\", "").lstrip("\\")
            # Extrahiere Level-Namen (z.B. "World_to_BeamNG")
            level_name = level_path.split("\\")[0]
            # Restlicher Pfad
            rest_path = "\\".join(level_path.split("\\")[1:])
            # Kombiniere mit BEAMNG_DIR
            full_path = self.beamng_dir / rest_path
            return full_path
        elif relative_path.startswith("/levels/"):
            # Unix-Format
            level_path = relative_path.replace("/levels/", "").lstrip("/")
            level_name = level_path.split("/")[0]
            rest_path = "/".join(level_path.split("/")[1:])
            full_path = self.beamng_dir / rest_path.replace("/", "\\")
            return full_path
        else:
            # Direkt im BEAMNG_DIR suchen
            return self.beamng_dir / relative_path

    def error(self, msg):
        """Registriere kritischen Fehler."""
        self.errors.append(f"❌ {msg}")
        print(f"  ❌ {msg}")

    def warning(self, msg):
        """Registriere Warnung."""
        self.warnings.append(f"⚠️ {msg}")
        print(f"  ⚠️ {msg}")

    def success(self, msg):
        """Registriere Erfolg."""
        print(f"  ✅ {msg}")

    def test_terrain_dae(self):
        """Teste terrain.dae Integrität."""
        print("\n[1] Teste terrain.dae...")

        # Suche dynamisch nach terrain_<x>_<y>.dae
        terrain_daes = list(self.shapes_dir.glob("terrain_*.dae"))

        if not terrain_daes:
            self.error(f"Keine terrain_*.dae gefunden in: {self.shapes_dir}")
            return

        terrain_dae = terrain_daes[0]
        self.success(f"terrain.dae gefunden: {terrain_dae.name}")

        # Parse XML
        try:
            tree = ET.parse(terrain_dae)
            root = tree.getroot()

            # Prüfe COLLADA-Root
            if root.tag != "{http://www.collada.org/2005/11/COLLADASchema}COLLADA":
                self.error("Kein valides COLLADA-Root-Element")
                return

            self.success("Valides COLLADA XML")

            # Zähle Geometrien
            geometries = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}geometry")
            if len(geometries) == 0:
                self.error("Keine Geometrien gefunden")
            else:
                self.success(f"{len(geometries)} Geometrien gefunden")

            # Zähle Materialien
            materials = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}material")
            if len(materials) == 0:
                self.warning("Keine Materialien in DAE gefunden")
            else:
                self.success(f"{len(materials)} Materialien gefunden")

            material_ids = [m.get("id") for m in materials if m.get("id")]
            self.dae_materials.update(material_ids)

            # Sammle Material-Referenzen aus triangles
            triangles = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}triangles")
            tri_materials = [t.get("material") for t in triangles if t.get("material")]
            if tri_materials:
                self.triangle_materials.update(tri_materials)
                self.success(f"{len(set(tri_materials))} Material-Referenzen in triangles")

            # Prüfe Vertices/Faces
            float_arrays = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}float_array")
            if float_arrays:
                total_floats = sum(int(fa.get("count", 0)) for fa in float_arrays)
                self.success(f"{len(float_arrays)} Float-Arrays, {total_floats:,} Werte total")

        except ET.ParseError as e:
            self.error(f"XML-Parse-Fehler: {e}")
        except Exception as e:
            self.error(f"Fehler beim Testen: {e}")

    def test_terrain_face_materials(self):
        """Teste dass JEDES Face in terrain_x_y.dae ein Material hat."""
        print("\n[1b] Teste Terrain Face-Materialien...")

        # Suche dynamisch nach terrain_<x>_<y>.dae
        terrain_daes = list(self.shapes_dir.glob("terrain_*.dae"))

        if not terrain_daes:
            self.warning("Keine terrain_*.dae gefunden - überspringe Face-Material-Test")
            return

        terrain_dae = terrain_daes[0]
        self.success(f"Teste Face-Materialien in: {terrain_dae.name}")

        try:
            tree = ET.parse(terrain_dae)
            root = tree.getroot()
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Sammle alle definierten Materialien
            materials = root.findall(".//collada:material", ns)
            defined_materials = set()
            for mat in materials:
                mat_id = mat.get("id")
                if mat_id:
                    defined_materials.add(mat_id)

            self.success(f"{len(defined_materials)} Materialien in library_materials definiert: {defined_materials}")

            # Prüfe alle triangles und polylist Primitive
            triangles_list = root.findall(".//collada:triangles", ns)
            polylists = root.findall(".//collada:polylist", ns)

            faces_without_material = []
            faces_with_undefined_material = []
            material_face_count = {}  # material -> anzahl faces

            # Prüfe triangles
            for tri_idx, tri in enumerate(triangles_list):
                material = tri.get("material")
                count = int(tri.get("count", 0))

                # Zähle Faces pro Material
                if material:
                    material_face_count[material] = material_face_count.get(material, 0) + count

                if material is None or material == "":
                    faces_without_material.append(f"triangles[{tri_idx}]: material-Attribut FEHLT")
                elif material not in defined_materials:
                    faces_with_undefined_material.append(
                        f"triangles[{tri_idx}]: material='{material}' nicht in library_materials definiert"
                    )

            # Prüfe polylist
            for poly_idx, poly in enumerate(polylists):
                material = poly.get("material")
                count = int(poly.get("count", 0))

                # Zähle Faces pro Material
                if material:
                    material_face_count[material] = material_face_count.get(material, 0) + count

                if material is None or material == "":
                    faces_without_material.append(f"polylist[{poly_idx}]: material-Attribut FEHLT")
                elif material not in defined_materials:
                    faces_with_undefined_material.append(
                        f"polylist[{poly_idx}]: material='{material}' nicht in library_materials definiert"
                    )

            # Berichte Fehler
            if faces_without_material:
                self.error(f"{len(faces_without_material)} Faces ohne Material-Attribut:")
                for err in faces_without_material[:10]:
                    self.error(f"  - {err}")
                if len(faces_without_material) > 10:
                    self.error(f"  ... und {len(faces_without_material) - 10} weitere")

            if faces_with_undefined_material:
                self.error(f"{len(faces_with_undefined_material)} Faces mit undefined Material:")
                for err in faces_with_undefined_material[:10]:
                    self.error(f"  - {err}")
                if len(faces_with_undefined_material) > 10:
                    self.error(f"  ... und {len(faces_with_undefined_material) - 10} weitere")

            if not faces_without_material and not faces_with_undefined_material:
                total_faces = len(triangles_list) + len(polylists)
                self.success(f"✓ Alle {total_faces} Faces haben definierte Materialien")

            # === Material-Statistik ===
            if material_face_count:
                print("\n  [Material-Verteilung] Dreiecke pro Material:")

                # Sortiere nach Anzahl (absteigend)
                sorted_materials = sorted(material_face_count.items(), key=lambda x: x[1], reverse=True)

                for material_name, face_count in sorted_materials:
                    # Unterscheide zwischen Tile-Materialien und Road-Materialien
                    if material_name.startswith("tile_"):
                        mat_type = "Terrain-Tile"
                    elif material_name.startswith("italy_"):
                        mat_type = "Straße"
                    else:
                        mat_type = "Sonstiges"

                    print(f"    • {material_name:30s} ({mat_type:15s}): {face_count:6d} Faces")

                # Zusammenfassung
                total_faces_counted = sum(material_face_count.values())
                terrain_tile_faces = sum(c for m, c in material_face_count.items() if m.startswith("tile_"))
                road_faces = sum(c for m, c in material_face_count.items() if m.startswith("italy_"))

                print(f"\n  [Zusammenfassung]")
                print(f"    Total:      {total_faces_counted:6d} Faces")
                print(
                    f"    Terrain:    {terrain_tile_faces:6d} Faces ({100*terrain_tile_faces/total_faces_counted:.1f}%)"
                )
                print(f"    Straßen:    {road_faces:6d} Faces ({100*road_faces/total_faces_counted:.1f}%)")

        except ET.ParseError as e:
            self.error(f"XML-Parse-Fehler: {e}")
        except Exception as e:
            self.error(f"Fehler beim Face-Material-Test: {e}")

    def test_building_daes(self):
        """Teste buildings/*.dae Integrität."""
        print("\n[2] Teste buildings/*.dae...")

        if not self.buildings_dir.exists():
            self.warning(f"Buildings-Verzeichnis nicht gefunden: {self.buildings_dir}")
            return

        building_daes = list(self.buildings_dir.glob("*.dae"))

        if len(building_daes) == 0:
            self.warning("Keine Gebäude-DAEs gefunden")
            return

        self.success(f"{len(building_daes)} Gebäude-DAEs gefunden")

        errors = 0
        total_geometries = 0

        for dae_file in building_daes:
            try:
                tree = ET.parse(dae_file)
                root = tree.getroot()

                # Zähle Geometrien
                geometries = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}geometry")
                total_geometries += len(geometries)

                # Prüfe ob lod2_wall_white und lod2_roof_red vorhanden
                materials = root.findall(".//{http://www.collada.org/2005/11/COLLADASchema}material")
                material_ids = [m.get("id") for m in materials if m.get("id")]
                self.dae_materials.update(material_ids)

                has_wall = "lod2_wall_white" in material_ids
                has_roof = "lod2_roof_red" in material_ids

                if not has_wall or not has_roof:
                    self.warning(f"{dae_file.name}: Fehlende Materialien (wall={has_wall}, roof={has_roof})")

            except Exception as e:
                errors += 1
                self.error(f"{dae_file.name}: Parse-Fehler - {e}")

        if errors == 0:
            self.success(f"Alle {len(building_daes)} DAEs erfolgreich geparst")
            self.success(f"{total_geometries} Gebäude-Geometrien total")
        else:
            self.error(f"{errors}/{len(building_daes)} DAEs fehlerhaft")

    def test_materials_json(self):
        """Teste main.materials.json Integrität."""
        print("\n[3] Teste main.materials.json...")

        materials_json = self.beamng_dir / "main.materials.json"

        if not materials_json.exists():
            self.error(f"main.materials.json nicht gefunden: {materials_json}")
            return

        self.success(f"main.materials.json gefunden")

        try:
            with open(materials_json, "r", encoding="utf-8") as f:
                materials = json.load(f)

            self.success(f"{len(materials)} Materialien definiert")

            # Prüfe LoD2-Materialien
            has_wall = "lod2_wall_white" in materials
            has_roof = "lod2_roof_red" in materials

            if has_wall and has_roof:
                self.success("LoD2-Materialien vorhanden (wall + roof)")
            else:
                self.warning(f"LoD2-Materialien fehlen (wall={has_wall}, roof={has_roof})")

            # Prüfe Material-Struktur
            invalid_materials = []
            for mat_name, mat_data in materials.items():
                if "mapTo" not in mat_data:
                    invalid_materials.append(f"{mat_name}: fehlt 'mapTo'")
                if "class" not in mat_data or mat_data["class"] != "Material":
                    invalid_materials.append(f"{mat_name}: fehlt/falsche 'class'")
                if "Stages" not in mat_data or not isinstance(mat_data["Stages"], list):
                    invalid_materials.append(f"{mat_name}: fehlt/falsche 'Stages'")

            if invalid_materials:
                for err in invalid_materials[:5]:  # Zeige max 5 Fehler
                    self.error(err)
                if len(invalid_materials) > 5:
                    self.error(f"... und {len(invalid_materials)-5} weitere")
            else:
                self.success("Alle Materialien haben valide Struktur")

            # Prüfe Textur-Referenzen
            texture_refs = []
            for mat_name, mat_data in materials.items():
                if "Stages" in mat_data and mat_data["Stages"]:
                    stage = mat_data["Stages"][0]
                    if "colorMap" in stage:
                        texture_refs.append(stage["colorMap"])

            if texture_refs:
                self.success(f"{len(texture_refs)} Textur-Referenzen gefunden")

                # Prüfe ob referenzierte Texturen existieren (mit relativen Pfaden)
                missing_textures = []
                for tex_path in texture_refs:
                    # Konvertiere relativen Pfad zu absolutem Windows-Pfad
                    full_path = self._resolve_relative_path(tex_path)

                    if not full_path.exists():
                        missing_textures.append(f"{tex_path} -> {full_path}")

                if missing_textures:
                    self.warning(f"{len(missing_textures)} referenzierte Texturen fehlen:")
                    for tex in missing_textures[:5]:
                        self.warning(f"  - {tex}")
                    if len(missing_textures) > 5:
                        self.warning(f"  ... und {len(missing_textures)-5} weitere")
                else:
                    self.success("Alle referenzierten Texturen existieren")

                # Bidirektionaler Abgleich DAE <-> materials.json
                # Materialien, die in DAEs referenziert werden, müssen hier existieren
                used_materials = set(self.dae_materials) | set(self.triangle_materials)
                json_material_names = set(materials.keys())
                json_mapto = {m.get("mapTo") for m in materials.values() if isinstance(m, dict) and "mapTo" in m}

                missing_in_json = [m for m in used_materials if m not in json_material_names and m not in json_mapto]
                if missing_in_json:
                    for m in missing_in_json[:5]:
                        self.error(f"Material in DAE referenziert, aber nicht in materials.json: {m}")
                    if len(missing_in_json) > 5:
                        self.error(f"... und {len(missing_in_json)-5} weitere")
                else:
                    self.success("Alle DAE-Material-Referenzen in materials.json vorhanden")

                # Optionale Warnung für ungenutzte Materialien in JSON
                # Ignoriere Tile-Materialien (tile_<x>_<y>), Terrain-Basis-Materialien und Horizon-Materialien
                unused = [m for m in json_material_names if m not in used_materials]
                unused = [m for m in unused if m not in ("unknown",)]  # allow fallback
                # Filtere Tile-Materialien heraus (normal für BeamNG Material-System)
                unused = [m for m in unused if not m.startswith("tile_")]
                # Filtere Horizon-Materialien heraus (werden vom Item direkt verwendet, nicht von Geometrien)
                unused = [m for m in unused if not m.startswith("horizon_")]
                if unused:
                    self.warning(f"{len(unused)} Strassenmaterialien in materials.json ungenutzt (z.B. {unused[:3]})")

        except json.JSONDecodeError as e:
            self.error(f"JSON-Parse-Fehler: {e}")
        except Exception as e:
            self.error(f"Fehler beim Testen: {e}")

    def test_items_json(self):
        """Teste items.json Integrität."""
        print("\n[4] Teste items.json...")

        items_json = self.beamng_dir / "main" / "items.json"

        if not items_json.exists():
            self.error(f"items.json nicht gefunden: {items_json}")
            return

        self.success(f"items.json gefunden")

        try:
            with open(items_json, "r", encoding="utf-8") as f:
                items = json.load(f)

            self.success(f"{len(items)} Items definiert")

            # Prüfe Terrain-Item (dynamisch: "terrain_<x>_<y>")
            terrain_items = [k for k in items.keys() if k.startswith("terrain_")]
            if terrain_items:
                terrain_item_name = terrain_items[0]
                terrain_item = items[terrain_item_name]
                self.success(f"Terrain-Mesh-Item vorhanden: {terrain_item_name}")
                if terrain_item.get("class") != "TSStatic":
                    self.error(f"Terrain-Item {terrain_item_name}: class != TSStatic")
                if terrain_item.get("collisionType") != "Visible Mesh":
                    self.warning(f"Terrain-Item {terrain_item_name}: collisionType != Visible Mesh")
            else:
                self.warning("Kein terrain_<x>_<y> Item gefunden")

            # Zähle Gebäude-Items
            building_items = [k for k in items.keys() if k.startswith("buildings_tile_")]
            if building_items:
                self.success(f"{len(building_items)} Gebäude-Tile-Items")
            else:
                self.warning("Keine Gebäude-Items gefunden")

            # Prüfe Item-Struktur und Shape-Referenzen
            invalid_items = []
            missing_shapes = []

            for item_name, item_data in items.items():
                if "__name" not in item_data:
                    invalid_items.append(f"{item_name}: fehlt '__name'")
                if "class" not in item_data:
                    invalid_items.append(f"{item_name}: fehlt 'class'")
                if "shapeName" not in item_data:
                    invalid_items.append(f"{item_name}: fehlt 'shapeName'")
                else:
                    # Prüfe ob Shape-Datei existiert (mit relativen Pfaden)
                    shape_name = item_data["shapeName"]
                    # Konvertiere relativen Pfad zu absolutem Windows-Pfad
                    shape_file = self._resolve_relative_path(shape_name)

                    if not shape_file.exists():
                        missing_shapes.append(f"{item_name} → {shape_name} ({shape_file})")

            if invalid_items:
                for err in invalid_items[:5]:
                    self.error(err)
                if len(invalid_items) > 5:
                    self.error(f"... und {len(invalid_items)-5} weitere")
            else:
                self.success("Alle Items haben valide Struktur")

            if missing_shapes:
                self.error(f"{len(missing_shapes)} referenzierte Shapes fehlen:")
                for shape in missing_shapes[:5]:
                    self.error(f"  - {shape}")
                if len(missing_shapes) > 5:
                    self.error(f"  ... und {len(missing_shapes)-5} weitere")
            else:
                self.success("Alle referenzierten Shapes existieren")

        except json.JSONDecodeError as e:
            self.error(f"JSON-Parse-Fehler: {e}")
        except Exception as e:
            self.error(f"Fehler beim Testen: {e}")

    def test_textures(self):
        """Teste Texturen."""
        print("\n[5] Teste Texturen...")

        if not self.textures_dir.exists():
            self.warning(f"Textures-Verzeichnis nicht gefunden: {self.textures_dir}")
            return

        textures = list(self.textures_dir.glob("tile_*.dds"))

        if len(textures) == 0:
            self.warning("Keine Tile-Texturen gefunden")
            return

        self.success(f"{len(textures)} Tile-Texturen gefunden")

        # Prüfe Dateigrößen
        empty_textures = []
        large_textures = []

        for tex in textures:
            size = tex.stat().st_size
            if size == 0:
                empty_textures.append(tex.name)
            elif size > 50 * 1024 * 1024:  # > 50 MB
                large_textures.append(f"{tex.name} ({size/1024/1024:.1f}MB)")

        if empty_textures:
            self.error(f"{len(empty_textures)} leere Texturen:")
            for tex in empty_textures[:5]:
                self.error(f"  - {tex}")
        else:
            self.success("Keine leeren Texturen")

        if large_textures:
            self.warning(f"{len(large_textures)} sehr große Texturen (>10MB):")
            for tex in large_textures[:5]:
                self.warning(f"  - {tex}")

        # Berechne Gesamt-Größe
        total_size = sum(tex.stat().st_size for tex in textures)
        self.success(f"Gesamt-Textur-Größe: {total_size/1024/1024:.1f} MB")

    def _index_to_coords(self, tile_index_x, tile_index_y):
        """
        Konvertiere Tile-Indizes zu absoluten Koordinaten.
        Index -2, -1, 0, 1 correspond zu Koordinaten -1000, -500, 0, 500.
        """
        x_coord = tile_index_x * 500
        y_coord = tile_index_y * 500
        return (x_coord, y_coord)

    def test_texture_mapping(self):
        """Teste Texture-Mapping: DAE Geometry-Namen zu Texturdateien."""
        print("\n[5b] Teste Texture-Mapping (DAE <-> Texturen)...")

        # Lade verfügbare Texturdateien
        if not self.textures_dir.exists():
            self.warning("Textures-Verzeichnis nicht gefunden, überspringe Mapping-Test")
            return

        texture_files = list(self.textures_dir.glob("tile_*.dds"))
        texture_keys = set(f.stem for f in texture_files)

        if not texture_keys:
            self.warning("Keine Tile-Texturen vorhanden")
            return

        self.success(f"{len(texture_keys)} Texturdateien vorhanden")

        # Teste alle Terrain-DAE Dateien
        terrain_daes = list(self.shapes_dir.glob("terrain_*.dae"))

        if not terrain_daes:
            self.warning("Keine terrain_*.dae gefunden")
            return

        total_geometries = 0
        unmapped_geometries = []

        for dae_file in terrain_daes:
            try:
                parser = lxml_etree.XMLParser(huge_tree=True)
                tree = lxml_etree.parse(str(dae_file), parser)
                root = tree.getroot()
                ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

                # Extrahiere alle Geometry-Namen
                geometries = root.findall(".//collada:geometry", ns)

                for geometry in geometries:
                    geom_name = geometry.get("name", "unknown")
                    total_geometries += 1

                    # Konvertiere Geometry-Name zu Texture-Key
                    if geom_name.startswith("tile_"):
                        parts = geom_name.split("_")
                        if len(parts) == 3:  # "tile_X_Y"
                            try:
                                idx_x = int(parts[1])
                                idx_y = int(parts[2])
                                coords = self._index_to_coords(idx_x, idx_y)
                                texture_key = f"tile_{coords[0]}_{coords[1]}"

                                if texture_key not in texture_keys:
                                    unmapped_geometries.append((dae_file.name, geom_name, texture_key))
                            except (ValueError, IndexError):
                                unmapped_geometries.append((dae_file.name, geom_name, "PARSE_ERROR"))

            except Exception as e:
                self.error(f"{dae_file.name}: Konnte nicht geparst werden - {e}")
                continue

        if unmapped_geometries:
            self.error(f"{len(unmapped_geometries)}/{total_geometries} Geometrien haben keine Texturen:")
            for dae_name, geom_name, tex_key in unmapped_geometries[:10]:
                self.error(f"  {dae_name}: {geom_name} → {tex_key} [MISSING]")
            if len(unmapped_geometries) > 10:
                self.error(f"  ... und {len(unmapped_geometries)-10} weitere")
        else:
            self.success(f"Alle {total_geometries} Geometrien haben Texturen")

    def test_debug_network_json(self):
        """Teste debug_network.json Integrität und Konsistenz."""
        print("\n[6] Teste debug_network.json...")

        if not self.debug_network_path.exists():
            self.warning(f"debug_network.json nicht gefunden: {self.debug_network_path}")
            return

        self.success(f"debug_network.json gefunden")

        try:
            with open(self.debug_network_path, "r", encoding="utf-8") as f:
                debug_data = json.load(f)

            self.success("debug_network.json erfolgreich geparst")

            # Prüfe notwendige Top-Level Keys
            required_keys = ["roads", "junctions", "grid_colors", "boundary_polygons"]
            missing_keys = [k for k in required_keys if k not in debug_data]

            if missing_keys:
                self.error(f"Erforderliche Keys fehlen: {missing_keys}")
            else:
                self.success("Alle erforderlichen Sections vorhanden")

            # === Prüfe ROADS ===
            roads = debug_data.get("roads", [])
            self.success(f"{len(roads)} Roads definiert")

            if roads:
                invalid_roads = []
                missing_colors = []
                for idx, road in enumerate(roads):
                    if "road_id" not in road:
                        invalid_roads.append(f"Road[{idx}]: fehlt 'road_id'")
                    if "coords" not in road or not isinstance(road["coords"], list):
                        invalid_roads.append(f"Road[{idx}]: fehlt/invalid 'coords'")
                    elif len(road["coords"]) < 2:
                        invalid_roads.append(f"Road[{idx}]: weniger als 2 Koordinaten")
                    else:
                        # Prüfe Koordinaten-Format
                        for coord_idx, coord in enumerate(road["coords"]):
                            if not isinstance(coord, list) or len(coord) != 3:
                                invalid_roads.append(
                                    f"Road[{idx}].coords[{coord_idx}]: ungültiges Format (erwartet [x,y,z])"
                                )
                                break

                    if "junction_start_id" not in road or "junction_end_id" not in road:
                        invalid_roads.append(f"Road[{idx}]: fehlen Junction-IDs")

                    # Prüfe Farbdefinitionen (Debug-Layer-Objekte)
                    required_color_fields = ["color", "line_width", "opacity"]
                    missing_fields = [f for f in required_color_fields if f not in road]
                    if missing_fields:
                        missing_colors.append(f"Road[{idx}]: fehlen {missing_fields}")

                if invalid_roads:
                    self.error(f"{len(invalid_roads)} Roads mit Fehlern:")
                    for err in invalid_roads[:5]:
                        self.error(f"  - {err}")
                    if len(invalid_roads) > 5:
                        self.error(f"  ... und {len(invalid_roads)-5} weitere")
                else:
                    self.success("Alle Roads haben valide Struktur")

                if missing_colors:
                    self.error(f"{len(missing_colors)} Roads ohne vollständige Farbdefinition:")
                    for err in missing_colors[:5]:
                        self.error(f"  - {err}")
                    if len(missing_colors) > 5:
                        self.error(f"  ... und {len(missing_colors)-5} weitere")
                else:
                    self.success("Alle Roads haben Farbdefinitionen (color, line_width, opacity)")

            # === Prüfe JUNCTIONS ===
            junctions = debug_data.get("junctions", [])
            self.success(f"{len(junctions)} Junctions definiert")

            if junctions:
                invalid_junctions = []
                missing_colors = []
                for idx, junction in enumerate(junctions):
                    if "position" not in junction:
                        invalid_junctions.append(f"Junction[{idx}]: fehlt 'position'")
                    elif not isinstance(junction["position"], list) or len(junction["position"]) != 3:
                        invalid_junctions.append(f"Junction[{idx}]: ungültiges Position-Format")

                    # Prüfe Farbdefinitionen (Debug-Layer-Objekte)
                    required_color_fields = ["color", "opacity"]
                    missing_fields = [f for f in required_color_fields if f not in junction]
                    if missing_fields:
                        missing_colors.append(f"Junction[{idx}]: fehlen {missing_fields}")

                if invalid_junctions:
                    self.error(f"{len(invalid_junctions)} Junctions mit Fehlern:")
                    for err in invalid_junctions[:5]:
                        self.error(f"  - {err}")
                else:
                    self.success("Alle Junctions haben valide Struktur")

                if missing_colors:
                    self.error(f"{len(missing_colors)} Junctions ohne vollständige Farbdefinition:")
                    for err in missing_colors[:5]:
                        self.error(f"  - {err}")
                    if len(missing_colors) > 5:
                        self.error(f"  ... und {len(missing_colors)-5} weitere")
                else:
                    self.success("Alle Junctions haben Farbdefinitionen (color, opacity)")

            # === Prüfe GRID_COLORS ===
            grid_colors = debug_data.get("grid_colors", {})
            if grid_colors:
                self.success(f"grid_colors vorhanden ({len(grid_colors)} Einträge)")

                required_color_keys = ["building_wall", "building_roof"]
                missing_color_keys = [k for k in required_color_keys if k not in grid_colors]

                if missing_color_keys:
                    self.warning(f"grid_colors: Fehlende Standard-Keys: {missing_color_keys}")
                else:
                    # Prüfe Struktur der Color-Definitionen
                    # Mesh-Layer (terrain, road, building_*) brauchen: face, edge, face_opacity, edge_opacity
                    # Debug-Layer (junction, centerline, boundary) brauchen: color, opacity
                    invalid_colors = []
                    for color_name, color_def in grid_colors.items():
                        # Debug-Layer haben andere Struktur
                        if color_name in ["junction", "centerline", "boundary"]:
                            required_fields = ["color", "opacity"]
                        else:
                            # Mesh-Layer
                            required_fields = ["face", "edge", "face_opacity", "edge_opacity"]

                        missing_fields = [f for f in required_fields if f not in color_def]
                        if missing_fields:
                            invalid_colors.append(f"{color_name}: fehlen {missing_fields}")

                    if invalid_colors:
                        for err in invalid_colors[:5]:
                            self.warning(f"  - {err}")
                    else:
                        self.success("grid_colors haben valide Struktur")
            else:
                self.warning("grid_colors nicht vorhanden")

            # === Prüfe BOUNDARY_POLYGONS ===
            boundary_polygons = debug_data.get("boundary_polygons", [])
            self.success(f"{len(boundary_polygons)} Boundary-Polygone definiert")

            if boundary_polygons:
                invalid_polygons = []
                polygon_types = {}

                for idx, poly in enumerate(boundary_polygons):
                    poly_type = poly.get("type", "unknown")
                    polygon_types[poly_type] = polygon_types.get(poly_type, 0) + 1

                    if "type" not in poly:
                        invalid_polygons.append(f"Polygon[{idx}]: fehlt 'type'")
                    elif poly_type not in ["boundary", "search_circle"]:
                        invalid_polygons.append(f"Polygon[{idx}]: unbekannter type '{poly_type}'")

                    if "coords" not in poly or not isinstance(poly["coords"], list):
                        invalid_polygons.append(f"Polygon[{idx}]: fehlt/invalid 'coords'")
                    elif len(poly["coords"]) < 2:
                        invalid_polygons.append(f"Polygon[{idx}]: weniger als 2 Koordinaten")

                    if "color" not in poly or not isinstance(poly["color"], list) or len(poly["color"]) != 3:
                        invalid_polygons.append(f"Polygon[{idx}]: fehlt/invalid 'color' (RGB [0-1])")

                if polygon_types:
                    self.success(f"Polygon-Typen: {polygon_types}")

                if invalid_polygons:
                    self.error(f"{len(invalid_polygons)} Polygone mit Fehlern:")
                    for err in invalid_polygons[:5]:
                        self.error(f"  - {err}")
                    if len(invalid_polygons) > 5:
                        self.error(f"  ... und {len(invalid_polygons)-5} weitere")
                else:
                    self.success("Alle Boundary-Polygone haben valide Struktur")

            # === Konsistenz-Prüfung: Roads vs. Junctions ===
            if roads and junctions:
                max_junction_id = len(junctions) - 1
                invalid_refs = []

                for road_idx, road in enumerate(roads):
                    jstart = road.get("junction_start_id")
                    jend = road.get("junction_end_id")

                    if jstart is not None and (jstart < 0 or jstart > max_junction_id):
                        invalid_refs.append(
                            f"Road[{road_idx}]: junction_start_id {jstart} außerhalb Range [0-{max_junction_id}]"
                        )
                    if jend is not None and (jend < 0 or jend > max_junction_id):
                        invalid_refs.append(
                            f"Road[{road_idx}]: junction_end_id {jend} außerhalb Range [0-{max_junction_id}]"
                        )

                if invalid_refs:
                    self.error(f"{len(invalid_refs)} ungültige Junction-Referenzen:")
                    for err in invalid_refs[:5]:
                        self.error(f"  - {err}")
                    if len(invalid_refs) > 5:
                        self.error(f"  ... und {len(invalid_refs)-5} weitere")
                else:
                    self.success("Alle Road-Junction-Referenzen sind konsistent")

            # === INTEGRITÄT: Z-Koordinaten der Debug-Daten ===
            print("  [Debug-Layer] Validiere Z-Koordinaten...")
            z_errors = []

            # Prüfe Junction-Positionen
            for idx, junction in enumerate(junctions):
                pos = junction.get("position", [])
                if len(pos) == 3:
                    x, y, z = pos
                    if not all(
                        isinstance(v, (int, float)) and not (abs(v) == float("inf") or v != v) for v in [x, y, z]
                    ):
                        z_errors.append(f"Junction[{idx}]: ungültige Koordinaten (NaN/Inf)")

            # Prüfe Road-Centerline-Koordinaten
            for road_idx, road in enumerate(roads):
                coords = road.get("coords", [])
                for coord_idx, coord in enumerate(coords):
                    if len(coord) == 3:
                        x, y, z = coord
                        if not all(
                            isinstance(v, (int, float)) and not (abs(v) == float("inf") or v != v) for v in [x, y, z]
                        ):
                            z_errors.append(f"Road[{road_idx}].coords[{coord_idx}]: ungültige Koordinaten (NaN/Inf)")
                            break

            # Prüfe Boundary-Polygon-Koordinaten
            for poly_idx, poly in enumerate(boundary_polygons):
                coords = poly.get("coords", [])
                for coord_idx, coord in enumerate(coords):
                    if len(coord) == 3:
                        x, y, z = coord
                        if not all(
                            isinstance(v, (int, float)) and not (abs(v) == float("inf") or v != v) for v in [x, y, z]
                        ):
                            z_errors.append(
                                f"Boundary[{poly_idx}].coords[{coord_idx}]: ungültige Koordinaten (NaN/Inf)"
                            )
                            break

            if z_errors:
                self.error(f"{len(z_errors)} Z-Koordinaten-Fehler im Debug-Layer:")
                for err in z_errors[:5]:
                    self.error(f"  - {err}")
                if len(z_errors) > 5:
                    self.error(f"  ... und {len(z_errors)-5} weitere")
            else:
                self.success("Alle Debug-Layer Z-Koordinaten sind valide (keine NaN/Inf)")

            # === INTEGRITÄT: Z-Range Konsistenz ===
            print("  [Debug-Layer] Prüfe Z-Koordinaten-Bereiche...")
            junction_zs = [j["position"][2] for j in junctions if "position" in j and len(j["position"]) == 3]
            road_zs = []
            for road in roads:
                road_zs.extend([c[2] for c in road.get("coords", []) if len(c) == 3])
            boundary_zs = []
            for poly in boundary_polygons:
                boundary_zs.extend([c[2] for c in poly.get("coords", []) if len(c) == 3])

            if junction_zs:
                junc_z_range = (min(junction_zs), max(junction_zs))
                self.success(f"Junctions Z-Range: [{junc_z_range[0]:.2f}, {junc_z_range[1]:.2f}]")

            if road_zs:
                road_z_range = (min(road_zs), max(road_zs))
                self.success(f"Roads (Centerlines) Z-Range: [{road_z_range[0]:.2f}, {road_z_range[1]:.2f}]")

            if boundary_zs:
                boundary_z_range = (min(boundary_zs), max(boundary_zs))
                self.success(f"Boundaries Z-Range: [{boundary_z_range[0]:.2f}, {boundary_z_range[1]:.2f}]")

            # === Gesamtstatistik ===
            debug_json_size = self.debug_network_path.stat().st_size
            self.success(f"debug_network.json Größe: {debug_json_size/1024:.1f} KB")

        except json.JSONDecodeError as e:
            self.error(f"JSON-Parse-Fehler: {e}")
        except Exception as e:
            self.error(f"Fehler beim Testen: {e}")

    def test_xyz_normalization(self):
        """Teste XYZ-Koordinaten-Normalisierung aller Objekte."""
        print("\n[7] Teste XYZ-Koordinaten-Normalisierung...")

        import numpy as np
        from tools.dae_loader import load_dae_tile

        # === LADE TERRAIN-DAE ===
        terrain_dae = self.shapes_dir / "terrain.dae"
        terrain_z_values = []

        if terrain_dae.exists():
            try:
                terrain_data = load_dae_tile(terrain_dae)
                terrain_vertices = terrain_data.get("vertices", np.array([]))

                if isinstance(terrain_vertices, np.ndarray) and len(terrain_vertices) > 0:
                    terrain_z_values = terrain_vertices[:, 2].tolist()
                    z_min, z_max = min(terrain_z_values), max(terrain_z_values)
                    z_mean = np.mean(terrain_z_values)

                    self.success(
                        f"Terrain: {len(terrain_z_values)} Vertices, Z=[{z_min:.2f}, {z_max:.2f}], M={z_mean:.2f}"
                    )
            except Exception as e:
                self.warning(f"Terrain-DAE-Analyse: {e}")

        # === LADE GEBÄUDE-DAEs ===
        building_daes = list(self.buildings_dir.glob("*.dae"))
        building_z_values = []
        building_z_ranges = {}

        if building_daes:
            print(f"  [Gebäude] Analysiere {len(building_daes)} DAE-Dateien...")

            for dae_file in building_daes:
                try:
                    dae_data = load_dae_tile(dae_file)
                    vertices = dae_data.get("vertices", np.array([]))

                    if isinstance(vertices, np.ndarray) and len(vertices) > 0:
                        z_coords = vertices[:, 2].tolist()
                        building_z_values.extend(z_coords)

                        z_min, z_max = min(z_coords), max(z_coords)
                        building_z_ranges[dae_file.name] = {"min": z_min, "max": z_max, "count": len(z_coords)}
                except Exception as e:
                    self.warning(f"{dae_file.name}: {e}")

            if building_z_values:
                z_min, z_max = min(building_z_values), max(building_z_values)
                z_mean = np.mean(building_z_values)

                self.success(
                    f"Gebäude: {len(building_z_values)} Vertices, Z=[{z_min:.2f}, {z_max:.2f}], M={z_mean:.2f}"
                )

                # Detaillierte Analyse
                print(f"    Z-Range pro Gebäude:")
                sorted_buildings = sorted(building_z_ranges.items(), key=lambda x: x[1]["min"])
                for name, zrange in sorted_buildings[:5]:
                    print(f"      • {name}: [{zrange['min']:.2f}, {zrange['max']:.2f}] ({zrange['count']} Verts)")
                if len(building_z_ranges) > 5:
                    print(f"      ... und {len(building_z_ranges)-5} weitere")

        # === LADE DEBUG_NETWORK.JSON ===
        debug_z_values = {"roads": [], "junctions": [], "boundaries": []}

        if self.debug_network_path.exists():
            try:
                with open(self.debug_network_path, "r", encoding="utf-8") as f:
                    debug_data = json.load(f)

                # Roads
                roads = debug_data.get("roads", [])
                for road in roads:
                    coords = road.get("coords", [])
                    for coord in coords:
                        if len(coord) >= 3:
                            debug_z_values["roads"].append(coord[2])

                # Junctions
                junctions = debug_data.get("junctions", [])
                for junction in junctions:
                    pos = junction.get("position", [])
                    if len(pos) >= 3:
                        debug_z_values["junctions"].append(pos[2])

                # Boundary Polygons
                boundary_polygons = debug_data.get("boundary_polygons", [])
                for poly in boundary_polygons:
                    coords = poly.get("coords", [])
                    for coord in coords:
                        if len(coord) >= 3:
                            debug_z_values["boundaries"].append(coord[2])

                # Statistik
                for key, z_values in debug_z_values.items():
                    if z_values:
                        z_min, z_max = min(z_values), max(z_values)
                        z_mean = np.mean(z_values)
                        count = len(z_values)
                        self.success(
                            f"Debug {key.capitalize()}: {count} Coordinates, Z=[{z_min:.2f}, {z_max:.2f}], M={z_mean:.2f}"
                        )

            except Exception as e:
                self.warning(f"Debug-Daten-Analyse: {e}")

        # === KONSISTENZ-PRÜFUNG ===
        print("\n  [Konsistenz] Vergleiche Z-Koordinaten zwischen Objekten:")

        all_z_values = terrain_z_values + building_z_values + debug_z_values["roads"] + debug_z_values["junctions"]

        if all_z_values:
            z_min, z_max = min(all_z_values), max(all_z_values)
            z_range = z_max - z_min

            print(f"    • Gesamt Z-Range: [{z_min:.2f}, {z_max:.2f}] (Spanne: {z_range:.2f}m)")

            # Prüfe ob Gebäude auf Terrain positioniert sind
            if terrain_z_values and building_z_values:
                terrain_min = min(terrain_z_values)
                building_min = min(building_z_values)
                terrain_max = max(terrain_z_values)
                building_max = max(building_z_values)

                # Prüfe ob Gebäude-Basis im Terrain-Bereich oder knapp darüber liegt
                # (es ist normal, dass Gebäude etwas höher sind als der Terrain-Min,
                #  weil sie auf variablem Terrain stehen)
                overlap_min = max(terrain_min, building_min)
                overlap_max = min(terrain_max, building_max)

                if overlap_max > overlap_min:
                    self.success(f"Gebäude sind auf Terrain positioniert (Höhen überlappen)")
                    print(f"      Terrain Z: [{terrain_min:.2f}, {terrain_max:.2f}]")
                    print(f"      Gebäude Z: [{building_min:.2f}, {building_max:.2f}]")
                    print(f"      Überlappung: [{overlap_min:.2f}, {overlap_max:.2f}]")
                else:
                    self.warning(f"Gebäude-Höhen und Terrain-Höhen überlappen nicht")

            # Prüfe ob Gebäude-Basis nicht extrem unter Terrain liegt
            if terrain_z_values and building_z_values:
                terrain_min = min(terrain_z_values)
                building_min = min(building_z_values)
                diff = building_min - terrain_min

                if diff >= -5.0:  # 5m Toleranz für Fundamente
                    self.success(f"Gebäude-Basishöhe relativ zu Terrain: {diff:.2f}m")
                else:
                    self.error(f"Gebäude zu tief unter Terrain: {diff:.2f}m")

            # Prüfe ob Koordinatensystem konsistent ist (keine wilden Ausreißer)
            if len(all_z_values) > 100:
                mean = np.mean(all_z_values)
                std = np.std(all_z_values)

                # Erlauben Sie große Varianz (unterschiedliche Terrainerhöhungen sind normal)
                # Suche nur nach extremen Ausreißern (z.B. 10σ)
                outlier_threshold_high = mean + 10 * std
                outlier_threshold_low = mean - 10 * std

                outliers_high = sum(1 for z in all_z_values if z > outlier_threshold_high)
                outliers_low = sum(1 for z in all_z_values if z < outlier_threshold_low)

                if outliers_high + outliers_low == 0:
                    self.success(f"Keine extremen Z-Koordinaten-Ausreißer gefunden")
                else:
                    self.warning(f"{outliers_high + outliers_low} extreme Ausreisser (>M±10s) gefunden")
        else:
            self.warning("Keine Z-Koordinaten zum Vergleich vorhanden")

    def test_horizon_dae(self):
        """Teste terrain_horizon.dae Integrität."""
        print("\n[Horizon] Teste terrain_horizon.dae...")

        horizon_dae = self.shapes_dir / "terrain_horizon.dae"

        if not horizon_dae.exists():
            self.warning("terrain_horizon.dae nicht gefunden - Horizon-Layer nicht generiert")
            return

        self.success(f"terrain_horizon.dae gefunden")

        # Parse XML
        try:
            tree = ET.parse(horizon_dae)
            root = tree.getroot()

            # Definiere Namespace
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Prüfe COLLADA-Root
            if "COLLADA" not in root.tag:
                self.error("Kein valides COLLADA-Root-Element in horizon DAE")
                return

            self.success("Valides COLLADA XML in horizon DAE")

            # Zähle Geometrien
            geometries = root.findall(".//collada:geometry", ns)
            if len(geometries) == 0:
                self.error("Keine Geometrien in horizon DAE gefunden")
                return
            else:
                self.success(f"{len(geometries)} Geometrie(n) in horizon DAE")

            # Prüfe Vertices
            sources = root.findall(".//collada:source", ns)
            if len(sources) < 2:  # Mindestens Vertices + UVs
                self.warning(f"Nur {len(sources)} Source(n) gefunden (erwartet: Vertices + UVs)")

            # Prüfe UV-Mapping vorhanden
            uv_sources = [s for s in sources if "uv" in s.get("id", "").lower()]
            if uv_sources:
                self.success("UV-Mapping in horizon DAE gefunden")
            else:
                self.warning("Keine UV-Sources in horizon DAE gefunden")

            # Zähle Faces (triangles/polylist)
            triangles = root.findall(".//collada:triangles", ns)
            polylists = root.findall(".//collada:polylist", ns)

            total_faces = len(triangles) + len(polylists)
            if total_faces == 0:
                self.error("Keine Faces (triangles/polylist) in horizon DAE")
            else:
                self.success(f"{total_faces} Face-Primitive in horizon DAE")

            # Extrahiere Vertex-Anzahl aus float_array count
            float_arrays = root.findall(".//collada:float_array", ns)
            for fa in float_arrays:
                count_str = fa.get("count", "")
                if count_str:
                    try:
                        count = int(count_str)
                        if "vertex" in fa.get("id", "").lower():
                            vertex_count = count // 3  # 3 Koordinaten pro Vertex
                            self.success(f"{vertex_count} Vertices in horizon DAE")
                        elif "uv" in fa.get("id", "").lower():
                            uv_count = count // 2  # 2 UV-Koordinaten pro Vertex
                            self.success(f"{uv_count} UV-Koordinaten in horizon DAE")
                    except ValueError:
                        pass

        except ET.ParseError as e:
            self.error(f"Fehler beim Parsen horizon DAE: {e}")

    def test_horizon_materials(self):
        """Teste Horizon-Material in materials.json."""
        print("\n[Horizon] Teste Horizon-Material...")

        materials_path = self.beamng_dir / "main.materials.json"

        if not materials_path.exists():
            self.warning("main.materials.json nicht gefunden")
            return

        try:
            with open(materials_path, "r") as f:
                materials = json.load(f)

            # Prüfe horizon_terrain Material
            if "horizon_terrain" not in materials:
                self.warning("horizon_terrain Material nicht in materials.json")
                return

            horizon_mat = materials["horizon_terrain"]
            self.success("horizon_terrain Material gefunden")

            # Prüfe erforderliche Felder (mit Ausnahme diffuseMap für Phase 5)
            required_fields = ["name", "mapTo", "version"]
            for field in required_fields:
                if field in horizon_mat:
                    self.success(f"  - {field}: {horizon_mat[field]}")
                else:
                    self.warning(f"  - {field}: FEHLT")

            # Prüfe DDS-Texture
            diff_map = horizon_mat.get("diffuseMap", "")
            if not diff_map and "Stages" in horizon_mat and horizon_mat["Stages"]:
                diff_map = horizon_mat["Stages"][0].get("colorMap", "")

            if "horizon_sentinel2.dds" in diff_map or "horizon_sentinel2" in diff_map:
                # Konvertiere relativen Pfad zu absolutem Windows-Pfad
                dds_path = self._resolve_relative_path(diff_map)
                if dds_path.exists():
                    self.success(f"DDS-Texture existiert: {dds_path.name}")
                else:
                    self.error(f"DDS-Texture nicht gefunden: {diff_map} -> {dds_path}")
            else:
                self.warning(f"Unexpected diffuseMap: {diff_map}")

        except json.JSONDecodeError as e:
            self.error(f"Fehler beim Parsen materials.json: {e}")

    def test_horizon_items(self):
        """Teste Horizon-Item in items.json."""
        print("\n[Horizon] Teste Horizon-Item...")

        items_path = self.beamng_dir / "main" / "items.json"

        if not items_path.exists():
            self.warning("items.json nicht gefunden")
            return

        try:
            with open(items_path, "r") as f:
                items = json.load(f)

            # Prüfe Horizon Item
            if "Horizon" not in items:
                self.warning("Horizon Item nicht in items.json")
                return

            horizon_item = items["Horizon"]
            self.success("Horizon Item gefunden")

            # Prüfe erforderliche Felder
            required_fields = {
                "__name": "Horizon",
                "className": "TSStatic",
                "datablock": "DefaultStaticShape",
            }

            for field, expected_value in required_fields.items():
                if field in horizon_item:
                    actual = horizon_item[field]
                    if actual == expected_value:
                        self.success(f"  - {field}: {actual} [OK]")
                    else:
                        self.warning(f"  - {field}: {actual} (erwartet: {expected_value})")
                else:
                    self.warning(f"  - {field}: FEHLT")

            # Prüfe Position (sollte [0, 0, 0] sein für lokal)
            position = horizon_item.get("position", [])
            if position == [0, 0, 0]:
                self.success(f"  - position: {position} [OK] (lokal)")
            else:
                self.warning(f"  - position: {position} (erwartet: [0, 0, 0] für lokale Koordinaten)")

            # Prüfe Rotation (sollte [0, 0, 1, 0] sein)
            rotation = horizon_item.get("rotation", [])
            if rotation == [0, 0, 1, 0]:
                self.success(f"  - rotation: {rotation} [OK]")
            else:
                self.warning(f"  - rotation: {rotation} (erwartet: [0, 0, 1, 0])")

            # Prüfe Scale (sollte [1, 1, 1] sein)
            scale = horizon_item.get("scale", [])
            if scale == [1, 1, 1]:
                self.success(f"  - scale: {scale} [OK]")
            else:
                self.warning(f"  - scale: {scale} (erwartet: [1, 1, 1])")

            # Prüfe shapeName
            shape_name = horizon_item.get("shapeName", "")
            if "terrain_horizon.dae" in shape_name or "horizon" in shape_name.lower():
                # Konvertiere relativen Pfad zu absolutem Windows-Pfad
                horizon_dae = self._resolve_relative_path(shape_name)
                if horizon_dae.exists():
                    self.success(f"  - shapeName: {shape_name} [OK]")
                else:
                    self.error(f"  - shapeName referenziert nicht-existente DAE: {shape_name} ({horizon_dae})")
            else:
                self.warning(f"  - shapeName: {shape_name} (erwartet: terrain_horizon.dae)")

            # Prüfe meshCulling und originSort
            if horizon_item.get("meshCulling") == 0:
                self.success(f"  - meshCulling: 0 [OK]")
            if horizon_item.get("originSort") == 0:
                self.success(f"  - originSort: 0 [OK]")

        except json.JSONDecodeError as e:
            self.error(f"Fehler beim Parsen items.json: {e}")

    def test_horizon_uv_mapping(self):
        """Teste UV-Mapping Plausibilität in horizon DAE."""
        print("\n[Horizon] Teste UV-Mapping Plausibilität...")

        horizon_dae = self.shapes_dir / "terrain_horizon.dae"

        if not horizon_dae.exists():
            self.warning("terrain_horizon.dae nicht gefunden - überspringe UV-Test")
            return

        try:
            tree = ET.parse(horizon_dae)
            root = tree.getroot()
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Extrahiere UV-Koordinaten
            float_arrays = root.findall(".//collada:float_array", ns)
            uv_values = []

            for fa in float_arrays:
                if "uv" in fa.get("id", "").lower():
                    text = fa.text.strip()
                    if text:
                        values = [float(v) for v in text.split()]
                        uv_values.extend(values)

            if not uv_values:
                self.warning("Keine UV-Werte in horizon DAE gefunden")
                return

            # Analysiere UV-Bereich
            uv_min = min(uv_values)
            uv_max = max(uv_values)

            print(f"  [i] UV-Range: [{uv_min:.4f} .. {uv_max:.4f}]")

            # Prüfe ob UVs im erwarteten Bereich sind (mit Offset/Skalierung)
            # Normalerweise sollten sie zwischen -0.1 und 1.1 sein (mit kleinen Offsets)
            if uv_min >= -0.5 and uv_max <= 1.5:
                self.success(f"UV-Werte im erwarteten Bereich [{uv_min:.4f}..{uv_max:.4f}]")
            else:
                self.warning(f"UV-Werte außerhalb erwarteter Range: [{uv_min:.4f}..{uv_max:.4f}]")

            # Prüfe auf gültige UV-Dichte (sollten mehrere eindeutige Werte sein)
            unique_uvs = len(set(round(v, 6) for v in uv_values))
            if unique_uvs > 1:
                self.success(f"{unique_uvs} eindeutige UV-Werte gefunden")
            else:
                self.warning(f"Nur {unique_uvs} eindeutige UV-Wert(e) - möglicherweise fehler in UV-Mapping")

        except ET.ParseError as e:
            self.error(f"Fehler beim UV-Analyse in horizon DAE: {e}")

    def test_horizon_coordinates(self):
        """Teste Koordinaten-Plausibilität von Horizon-Mesh."""
        print("\n[Horizon] Teste Koordinaten-Plausibilität...")

        horizon_dae = self.shapes_dir / "terrain_horizon.dae"

        if not horizon_dae.exists():
            self.warning("terrain_horizon.dae nicht gefunden - überspringe Koordinaten-Test")
            return

        try:
            tree = ET.parse(horizon_dae)
            root = tree.getroot()
            ns = {"collada": "http://www.collada.org/2005/11/COLLADASchema"}

            # Extrahiere Vertices
            float_arrays = root.findall(".//collada:float_array", ns)
            vertices = []

            for fa in float_arrays:
                if "vertices" in fa.get("id", "").lower():
                    text = fa.text.strip() if fa.text else ""
                    if text:
                        values = [float(v) for v in text.split()]
                        # 3 Werte pro Vertex (X, Y, Z)
                        for i in range(0, len(values), 3):
                            if i + 2 < len(values):
                                vertices.append((values[i], values[i + 1], values[i + 2]))

            if not vertices:
                self.warning("Keine Vertices in horizon DAE gefunden")
                return

            # Analysiere Koordinaten-Bereiche
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)

            print(f"  [i] Mesh-Bounds:")
            print(f"      X: [{x_min:.0f}..{x_max:.0f}] ({x_max - x_min:.0f}m)")
            print(f"      Y: [{y_min:.0f}..{y_max:.0f}] ({y_max - y_min:.0f}m)")
            print(f"      Z: [{z_min:.0f}..{z_max:.0f}] ({z_max - z_min:.0f}m)")

            # Prüfe ob Mesh in lokalen Koordinaten ist (sollte um 0,0,0 sein)
            mesh_center_x = (x_min + x_max) / 2
            mesh_center_y = (y_min + y_max) / 2

            if abs(mesh_center_x) < 100000 and abs(mesh_center_y) < 100000:
                self.success(f"Mesh-Center in lokalen Koordinaten: ({mesh_center_x:.0f}, {mesh_center_y:.0f})")
            else:
                self.warning(f"Mesh-Center außerhalb erwartet: ({mesh_center_x:.0f}, {mesh_center_y:.0f})")

            # Prüfe ob Höhenvarianz plausibel ist (sollte nicht zu klein sein für Horizont)
            if z_max - z_min > 1.0:
                self.success(f"Höhenvarianz im Mesh: {z_max - z_min:.2f}m")
            else:
                self.warning(f"Kleine Höhenvarianz im Horizont-Mesh: {z_max - z_min:.2f}m (flach?)")

        except ET.ParseError as e:
            self.error(f"Fehler beim Parsen Koordinaten in horizon DAE: {e}")

    def run_all_tests(self):
        """Führe alle Tests aus."""
        print("=" * 60)
        print("EXPORT INTEGRITY TEST")
        print("=" * 60)
        print(f"BeamNG-Verzeichnis: {self.beamng_dir}")

        self.test_terrain_dae()
        self.test_terrain_face_materials()
        self.test_building_daes()
        self.test_materials_json()
        self.test_items_json()
        self.test_textures()
        self.test_texture_mapping()
        self.test_debug_network_json()
        self.test_xyz_normalization()

        # Horizon-Tests
        self.test_horizon_dae()
        self.test_horizon_materials()
        self.test_horizon_items()
        self.test_horizon_uv_mapping()
        self.test_horizon_coordinates()

        # Zusammenfassung
        print("\n" + "=" * 60)
        print("ZUSAMMENFASSUNG")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ {len(self.errors)} FEHLER gefunden:")
            for err in self.errors:
                print(f"  {err}")
        else:
            print("\n✅ Keine kritischen Fehler!")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNUNGEN:")
            for warn in self.warnings:
                print(f"  {warn}")
        else:
            print("\n✅ Keine Warnungen!")

        if not self.errors and not self.warnings:
            print("\n" + "=" * 60)
            print("🎉 ALLE TESTS BESTANDEN! Export ist valide.")
            print("=" * 60)
            return 0
        elif not self.errors:
            print("\n" + "=" * 60)
            print("[OK] Export ist funktionsfähig (mit Warnungen).")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ Export hat kritische Fehler!")
            print("=" * 60)
            return 1


def main():
    """Hauptfunktion."""
    tester = ExportIntegrityTest()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
