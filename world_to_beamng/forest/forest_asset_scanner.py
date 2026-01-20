"""
Forest Asset Scanner
====================

Scannt BeamNG's vorhandene Baum-Assets (DAE + DDS) und registriert:
1. Tree-Materials in materials.json (via MaterialManager)
2. ForestItemData in main/forestItemData.json (direkt als JSON Objekt)
3. Forest-Objekt in items.level.json (via ItemManager)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import uuid

logger = logging.getLogger(__name__)


class ForestAssetScanner:
    """Scannt und registriert BeamNG Tree-Assets."""

    def __init__(self, base_dir: Path, material_manager, item_manager):
        """
        Initialize scanner.

        Args:
            base_dir: BeamNG-Verzeichnis (config.BEAMNG_DIR)
            material_manager: MaterialManager instance
            item_manager: ItemManager instance
        """
        self.base_dir = Path(base_dir)
        self.material_manager = material_manager
        self.item_manager = item_manager

        self.foliage_dir = self.base_dir / "art" / "shapes" / "assets" / "meshes" / "foliage"
        self.tree_materials_dir = self.base_dir / "art" / "shapes" / "assets" / "materials" / "tree"

    def scan_and_register_trees(self) -> Dict[str, Dict]:
        """
        Scanne Assets und registriere Materials + forestItemData.json.

        Returns:
            Dict[tree_name, tree_info] mit allen registrierten Bäumen
        """
        logger.info("=== Forest Asset Scanner gestartet ===")

        # 1. Scanne DAE-Meshes
        dae_files = self._scan_tree_meshes()
        logger.info(f"Gefunden: {len(dae_files)} DAE-Dateien")

        # 2. Scanne DDS-Texturen
        textures_by_type = self._scan_tree_textures()
        logger.info(f"Gefunden: {len(textures_by_type)} Texture-Sets")

        # 3. Finde Kombinationen
        combinations = self._find_tree_combinations(dae_files, textures_by_type)
        logger.info(f"Erstellt: {len(combinations)} Tree-Kombinationen")

        # 4. Registriere Materials und sammle ForestItemData
        registered_trees = {}
        forest_item_data = {}  # Objekt für forestItemData.json (Key: tree_name)

        for combo in combinations:
            tree_info = self._register_tree_complete(combo)
            if tree_info:
                registered_trees[tree_info["name"]] = tree_info

                # Sammle ForestItemData (Dictionary mit Baumnamen als Keys)
                forest_item_data[tree_info["name"]] = {
                    "name": tree_info["name"],
                    "class": "TSForestItemData",
                    "shapeFile": tree_info["dae_path"],
                    "collidable": True,
                    "radius": tree_info["radius"],
                }

        logger.info(f"Registriert: {len(registered_trees)} Bäume")

        # 5. Schreibe forestItemData.json
        self._write_forest_item_data(forest_item_data)

        # 6. Registriere zentrales Forest-Objekt
        self._register_forest_object()

        logger.info("=== Forest Asset Scanner abgeschlossen ===")

        return registered_trees

    def _scan_tree_meshes(self) -> List[Path]:
        """Scanne DAE-Dateien im foliage-Verzeichnis."""
        if not self.foliage_dir.exists():
            logger.warning(f"Foliage-Verzeichnis nicht gefunden: {self.foliage_dir}")
            return []

        dae_files = list(self.foliage_dir.rglob("*.dae"))

        logger.debug(f"DAE-Meshes in {self.foliage_dir}:")
        by_category = {}
        for dae in dae_files:
            rel_path = dae.relative_to(self.foliage_dir)
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(dae)

        for category, files in sorted(by_category.items()):
            logger.debug(f"  {category}/: {len(files)} Dateien")

        return dae_files

    def _scan_tree_textures(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Scanne DDS-Texturen im tree-Verzeichnis.

        Returns:
            Dict[tree_type, Dict[texture_type, List[Path]]]
        """
        if not self.tree_materials_dir.exists():
            logger.warning(f"Tree-Materials-Verzeichnis nicht gefunden: {self.tree_materials_dir}")
            return {}

        dds_files = list(self.tree_materials_dir.rglob("*.dds"))

        textures_by_type = {}

        for dds in dds_files:
            rel_path = dds.relative_to(self.tree_materials_dir)

            # Extrahiere Baumtyp (erster Ordner)
            tree_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"

            # Extrahiere Textur-Typ aus Dateinamen
            filename = dds.stem

            texture_type = None
            if ".color" in filename or "_b." in filename or filename.endswith("_b"):
                texture_type = "baseColorMap"
            elif ".normal" in filename or "_nm." in filename or filename.endswith("_nm"):
                texture_type = "normalMap"
            elif "_o." in filename or filename.endswith("_o"):
                texture_type = "opacityMap"
            elif "_ao." in filename or filename.endswith("_ao"):
                texture_type = "ambientOcclusionMap"
            elif "_r." in filename or filename.endswith("_r"):
                texture_type = "roughnessMap"

            if tree_type not in textures_by_type:
                textures_by_type[tree_type] = {}

            if texture_type:
                if texture_type not in textures_by_type[tree_type]:
                    textures_by_type[tree_type][texture_type] = []
                textures_by_type[tree_type][texture_type].append(dds)

        logger.debug(f"Texture-Sets:")
        for tree_type, tex_types in sorted(textures_by_type.items()):
            logger.debug(f"  {tree_type}: {', '.join(tex_types.keys())}")

        return textures_by_type

    def _find_tree_combinations(
        self, dae_files: List[Path], textures_by_type: Dict[str, Dict[str, List[Path]]]
    ) -> List[Dict]:
        """
        Finde sinnvolle Kombinationen aus DAE + DDS.

        Returns:
            List[Dict] mit {"name", "dae_path", "textures": {...}}
        """
        combinations = []

        for tree_type, tex_types in sorted(textures_by_type.items()):
            # Suche passendes DAE - Bevorzuge "blocker" Variante (für Collision)
            matching_daes = [dae for dae in dae_files if tree_type.lower() in dae.stem.lower()]

            if not matching_daes:
                # Kein exakt passendes DAE gefunden - Fehler!
                logger.warning(f"Kein DAE für Tree-Typ '{tree_type}' gefunden (kein Fallback!)")
                continue  # ← WICHTIG: Skip statt Fallback!

            # Bevorzuge blocker_a Variante (Standard-Rendering)
            blocker_daes = [dae for dae in matching_daes if "blocker_a" in dae.stem.lower()]
            selected_dae = blocker_daes[0] if blocker_daes else matching_daes[0]

            logger.debug(f"Baum '{tree_type}': Gefunden {len(matching_daes)} DAEs, Wähle {selected_dae.name}")

            # Baue Textur-Dict (nimm jeweils die erste Datei)
            textures = {}
            for tex_type, files in tex_types.items():
                if files:
                    textures[tex_type] = files[0]

            combination = {"name": tree_type, "dae_path": selected_dae, "textures": textures}

            combinations.append(combination)

            logger.debug(f"Kombination: {tree_type} → {selected_dae.name} + {len(textures)} Texturen")

        return combinations

    def _register_tree_complete(self, combo: Dict) -> Dict:
        """
        Registriere Material für einen Baum (ForestItemData wird separat gesammelt).

        Args:
            combo: Dict mit "name", "dae_path", "textures"

        Returns:
            Dict mit tree_info oder None bei Fehler
        """
        tree_name = combo["name"]
        dae_path = combo["dae_path"]
        textures = combo["textures"]

        # Material-Name
        material_name = f"{tree_name}_leaves"

        # 1. Registriere Material
        try:
            stages = {}
            for tex_type, tex_path in textures.items():
                # Konvertiere zu relativem BeamNG-Pfad
                rel_path = tex_path.relative_to(self.base_dir)
                # Füge RELATIVE_DIR vorne an (levels/world_to_beamng/...)
                from world_to_beamng import config

                full_rel_path = config.RELATIVE_DIR / rel_path
                stages[tex_type] = str(full_rel_path).replace("\\", "/")

            # Setze useAnisotropic
            if stages:
                stages["useAnisotropic"] = True

            self.material_manager.add_material(
                name=material_name,
                template="tree",  # Nutzt Tree-Template
                mapTo=material_name,
                **{"class": "Material"},  # WICHTIG: BeamNG requires explicit class attribute!
                Stages=[stages],
                alphaTest=True,  # Wichtig für Blätter!
                alphaRef=128,  # Schwellenwert
                translucent=False,  # Performance
                doubleSided=True,  # Beide Seiten sichtbar
            )

            logger.info(f"✓ Material registriert: {material_name}")

        except Exception as e:
            logger.error(f"Fehler bei Material-Registrierung für {tree_name}: {e}")
            return None

        # 2. Präpariere DAE-Pfad für ForestItemData
        try:
            # Konvertiere DAE-Pfad zu relativem BeamNG-Pfad
            dae_rel_path = dae_path.relative_to(self.base_dir)
            # Füge RELATIVE_DIR vorne an (levels/world_to_beamng/...)
            from world_to_beamng import config

            full_dae_path = config.RELATIVE_DIR / dae_rel_path
            dae_beamng_path = str(full_dae_path).replace("\\", "/")

            # Berechne Radius aus Dateinamen (Heuristik)
            radius = 1.5  # Default
            if "large" in dae_path.stem.lower():
                radius = 2.0
            elif "small" in dae_path.stem.lower():
                radius = 1.0

            logger.info(f"✓ ForestItemData vorbereitet: {tree_name}")

        except Exception as e:
            logger.error(f"Fehler bei ForestItemData-Vorbereitung für {tree_name}: {e}")
            return None

        # Erfolg!
        return {
            "name": tree_name,
            "material": material_name,
            "dae_path": dae_beamng_path,
            "textures": {
                k: str(config.RELATIVE_DIR / v.relative_to(self.base_dir)).replace("\\", "/")
                for k, v in textures.items()
            },
            "radius": radius,
        }

    def _write_forest_item_data(self, forest_item_data: Dict[str, Dict]) -> None:
        """
        Schreibe ForestItemData in main/forestItemData.json (JSON Objekt-Format).

        Args:
            forest_item_data: Dict mit Baumnamen als Keys und TSForestItemData Objekten als Values
        """
        if not forest_item_data:
            logger.warning("Keine ForestItemData zum Schreiben vorhanden")
            return

        try:
            # Zielverzeichnis
            output_file = self.base_dir / "main" / "forestItemData.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Schreibe als JSON Objekt
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(forest_item_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ forestItemData.json geschrieben (JSON Objekt): {len(forest_item_data)} Einträge")
            logger.debug(f"  Pfad: {output_file}")

        except Exception as e:
            logger.error(f"Fehler beim Schreiben von forestItemData.json: {e}")

    def _register_forest_object(self):
        """Registriere zentrales Forest-Objekt."""
        try:
            self.item_manager.add_item(
                name="the_forest",
                item_class="Forest",
                dataFile="levels/world_to_beamng/main/forest.forest4.json",
                lodScale=1.0,
                overwrite=True,
            )

            logger.info("✓ Forest-Objekt registriert")

        except Exception as e:
            logger.error(f"Fehler bei Forest-Objekt-Registrierung: {e}")
