"""
Generate Forest Assets: managedItemData.json + forest_type_templates

Kombiniertes Script das:
1. DAE-Dateien scannt und managedItemData.json generiert (art/forest/, BeamNG-Item-Registry)
2. Aus den gescannten Items Waldtypen und Mappings generiert
3. osm_to_beamng.json aktualisiert
"""

from pathlib import Path
import json
import re
import shutil
import zipfile
import configparser
from collections import defaultdict
import sys

# Importiere config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config


# Mapping von Dateinamen-Patterns zu Baumarten
TREE_NAME_PATTERNS = {
    r"oak|eiche": "oak",
    r"pedunculate|sessile|quercus": "oak",
    r"beech|buche|fagus": "beech",
    r"birch|birke|betula": "birch",
    r"aspen|espe|tremuloides": "aspen",
    r"spruce|fichte|picea": "spruce",
    r"pine|scots|kiefer|sylvestris": "scots_pine",
    r"fir|tanne|abies": "fir",
    r"larch|lärche|larix": "larch",
    r"maple|ahorn": "maple",
    r"ash|esche|fraxinus": "ash",
    r"elm|ulme": "elm",
    r"poplar|pappel": "poplar",
    r"alder|erle|alnus": "alder",
    r"willow|weide|salix": "willow",
    r"rowan|eberesche|sorbus": "rowan",
    r"hazel|hasel|corylus": "hazel",
    r"elder|holunder|sambucus": "elder",
    r"cork|kork": "cork_oak",
    r"holm|steineiche": "holm_oak",
    r"olive|oliv": "olive",
}


def extract_tree_name_from_filename(filename: str) -> str:
    """Extrahiere Baumnamen aus DAE-Dateiname."""
    name = Path(filename).stem
    name_lower = name.lower()

    for pattern, tree_name in TREE_NAME_PATTERNS.items():
        if re.search(pattern, name_lower):
            return tree_name

    cleaned = re.sub(r"[_\-\d]", " ", name).strip()
    if not cleaned or cleaned.lower() in ["tree", "model", "asset"]:
        return "tree"
    return cleaned.lower()


def upgrade_to_canonical_meshes(dest_dir: Path, install_dir: Path) -> int:
    """
    Ersetzt Baum-Meshes, wo möglich, durch BeamNGs kanonische, level-unabhängige
    "trees_library" (+ "foliage/bushes") aus content/assets/meshes.zip.

    Hintergrund (Recherche 2026-09-17): east_coast_usa's eigener Baum-Ordner ist nur eine
    PER-LEVEL-KOPIE dieser gemeinsamen Bibliothek - die Dateinamen sind identisch
    (z.B. "tree_beech_bush_a.dae", "cork_oak_bush_large.dae"). Die Bibliothek deckt nicht
    alle Arten ab (keine Douglasie/Fichte, kein Italy-Ölbaum, keine Kiefer) - für die
    bleibt die level-lokale Kopie als Fallback bestehen, alles andere wird auf die
    kanonische, garantiert aktuelle Quelle "hochgestuft".

    Ersetzte Meshes bekommen ihre alten .cdae/.dae.asset.json/.imposter*.dds gelöscht,
    da diese zum ALTEN Mesh gehören und beim neuen nicht mehr passen - BeamNG erzeugt sie
    beim Laden automatisch neu.

    Returns:
        Anzahl der hochgestuften Meshes
    """
    meshes_zip_path = install_dir / "content" / "assets" / "meshes.zip"
    if not meshes_zip_path.is_file():
        print(f"[WARNUNG] meshes.zip nicht gefunden, überspringe kanonisches Upgrade: {meshes_zip_path}")
        return 0

    z = zipfile.ZipFile(meshes_zip_path)
    canonical_index = {}  # dae-Dateiname -> zip-Eintrag
    for entry in z.namelist():
        if entry.endswith(".dae") and ("foliage/trees_library/" in entry or "foliage/bushes/" in entry):
            canonical_index[entry.rsplit("/", 1)[-1]] = entry

    upgraded = 0
    for dae_file in dest_dir.rglob("*.dae"):
        entry = canonical_index.get(dae_file.name)
        if not entry:
            continue

        dae_file.write_bytes(z.read(entry))

        # Zum alten Mesh gehörende Caches sind jetzt ungültig -> löschen, BeamNG baut neu.
        for stale in (
            dae_file.with_suffix(".cdae"),
            dae_file.with_name(dae_file.name + ".asset.json"),
            dae_file.with_name(dae_file.name + ".imposter.dds"),
            dae_file.with_name(dae_file.name + ".imposter_normals.dds"),
        ):
            if stale.exists():
                stale.unlink()

        upgraded += 1

    print(f"[INFO] {upgraded} Meshes auf kanonische trees_library/foliage-Bibliothek hochgestuft")
    return upgraded


def copy_tree_assets(source_dir: Path, dest_dir: Path, install_dir: Path) -> int:
    """
    Kopiert den kompletten Baum-Asset-Ordner (DAE, Texturen, materials.json, ...)
    in den eigenen Level, damit world_to_beamng nicht mehr von einem fremden
    Level (z.B. east_coast_usa) abhängt.

    Nutzt east_coast_usa dabei nur noch als Fallback-Quelle für Geometrie + als Quelle
    für die Material-DEFINITIONEN (materials.json) - die Mesh-Dateien selbst werden im
    Anschluss über upgrade_to_canonical_meshes() wo möglich durch BeamNGs kanonische
    trees_library ersetzt (siehe dort).

    Schreibt anschließend die "levels/<altes_level>/..."-Pfadreferenzen in allen
    TEXT-Dateien (materials.json UND .dae, COLLADA embedded texture refs) auf den
    eigenen Level um. .cdae-Dateien sind ein kompiliertes Binär-Cache-Format, das
    den alten Pfad ebenfalls einbettet, aber NICHT sicher text-patchbar ist
    (Längen-präfixierte Strings) - diese werden stattdessen gelöscht, damit BeamNG
    sie beim nächsten Laden automatisch frisch aus der .dae neu kompiliert.

    Returns:
        Anzahl der umgeschriebenen Text-Dateien
    """
    print(f"[INFO] Kopiere Baum-Assets (Fallback-Basis): {source_dir} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    upgrade_to_canonical_meshes(dest_dir, install_dir)

    # Ermittle den alten Level-Namen aus dem Quellpfad (.../levels/<name>/art/...)
    parts = source_dir.parts
    old_level_name = parts[parts.index("levels") + 1] if "levels" in parts else None

    fixed = 0
    if old_level_name and old_level_name != config.LEVEL_NAME:
        old_ref = f"levels/{old_level_name}/art/shapes/trees"
        new_ref = f"levels/{config.LEVEL_NAME}/art/shapes/trees"

        text_files = list(dest_dir.rglob("*materials.json")) + list(dest_dir.rglob("*.dae"))
        for text_file in text_files:
            text = text_file.read_text(encoding="utf-8")
            if old_ref in text:
                text_file.write_text(text.replace(old_ref, new_ref), encoding="utf-8")
                fixed += 1

        # .cdae ist ein kompilierter Binär-Cache und bettet den alten Pfad ein.
        # Löschen -> BeamNG kompiliert beim nächsten Laden automatisch frisch aus der .dae.
        cdae_files = list(dest_dir.rglob("*.cdae"))
        for cdae_file in cdae_files:
            cdae_file.unlink()
        if cdae_files:
            print(f"[INFO] {len(cdae_files)} .cdae Cache-Dateien gelöscht (werden beim Laden neu kompiliert)")

    print(f"[INFO] {fixed} Text-Dateien (.dae/.materials.json) auf eigenen Level-Pfad umgeschrieben")
    return fixed


def get_beamng_install_dir() -> Path:
    """Liest den BeamNG-Installationspfad aus BeamNG.drive.ini."""
    ini_path = Path.home() / "AppData" / "Local" / "BeamNG" / "BeamNG.drive.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"BeamNG.drive.ini nicht gefunden: {ini_path}")

    raw = ini_path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser()
    parser.read_string("[main]\n" + raw)
    return Path(parser["main"]["installpath"].strip().strip('"'))


def resolve_tree_texture_links(dest_dir: Path, install_dir: Path) -> None:
    """
    Ersetzt die *.link Platzhalter (Zeiger auf einen gemeinsamen BeamNG-Textur-Pool) durch
    echte DDS-Bytes.

    Grund: Die .link-Dateien verweisen auf ein veraltetes Pfadschema
    ("/assets/materials/foliage/tree/<species>/...") - das gilt für ALLE Vanilla-Level
    (nicht nur east_coast_usa), passt aber nicht mehr zum aktuell installierten
    BeamNG-Content (der Baum-Texturpack liegt dort unter einem anderen Schema:
    "assets/materials/tree/<species>/<material>/..."). Deshalb per Dateiname (nicht per
    Pfad) im kompletten BeamNG-Materials-Content suchen und die echten Bytes lokal ablegen.

    Referenzen enden bisher auf ".png" (Cook-from-source Pfad); wir legen die Datei
    stattdessen als ".dds" ab (Direkt-Lade-Pfad, das Format, das die restliche Pipeline
    bereits für Terrain-/Straßen-Texturen verwendet) und schreiben alle betroffenen
    .dae/.materials.json Referenzen entsprechend um.
    """
    link_files = list(dest_dir.rglob("*.link"))
    if not link_files:
        return

    print(f"[INFO] Löse {len(link_files)} Textur-Links zu echten BeamNG-Assets auf...")

    materials_dir = install_dir / "content" / "assets" / "materials"

    # Index: Dateiname (ohne Verzeichnis) -> (zip_pfad, voller_eintrag_in_der_zip)
    name_index = {}
    for zip_path in sorted(materials_dir.glob("*.zip")):
        try:
            z = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            continue
        for entry in z.namelist():
            if entry.endswith("/"):
                continue
            fname = entry.rsplit("/", 1)[-1]
            name_index.setdefault(fname, (zip_path, entry))

    open_zips = {}
    rename_map = {}  # alter Logik-Name (....png) -> neuer physischer Name (....dds)
    resolved = 0
    unresolved = []

    for link_file in link_files:
        old_name = link_file.name[: -len(".link")]  # z.B. "t_beech_branch_o.data.png"
        base = old_name.rsplit(".", 1)[0]  # "t_beech_branch_o.data"

        # Suche eine physische Datei mit gleichem Basisnamen, beliebiges Bild-Suffix
        candidate = None
        # Manche Vanilla-Level lassen das "t_"-Präfix in ihren .link-Dateien weg, obwohl
        # der Asset-Pack es führt (z.B. "scots_pine_ao.data" -> "t_scots_pine_ao.data").
        for candidate_base in (base, f"t_{base}"):
            for ext in ("dds", "png", "tga"):
                fname = f"{candidate_base}.{ext}"
                if fname in name_index:
                    candidate = name_index[fname]
                    break
            if candidate:
                break

        if not candidate:
            unresolved.append(old_name)
            link_file.unlink()
            continue

        zip_path, entry = candidate
        if zip_path not in open_zips:
            open_zips[zip_path] = zipfile.ZipFile(zip_path)
        z = open_zips[zip_path]

        new_name = f"{base}.dds"
        dest_file = link_file.with_name(new_name)
        dest_file.write_bytes(z.read(entry))
        link_file.unlink()

        rename_map[old_name] = new_name
        resolved += 1

    print(f"[INFO] {resolved} Texturen aufgelöst und lokal abgelegt, {len(unresolved)} nicht gefunden")
    if unresolved:
        print("[WARNUNG] Nicht gefunden (Baum bleibt evtl. untexturiert):")
        for u in sorted(set(unresolved)):
            print("   ", u)

    # Referenzen in .dae/.materials.json auf die neuen physischen Dateinamen umschreiben
    if rename_map:
        rewritten = 0
        for text_file in list(dest_dir.rglob("*.dae")) + list(dest_dir.rglob("*materials.json")):
            text = text_file.read_text(encoding="utf-8")
            new_text = text
            for old_name, new_name in rename_map.items():
                if old_name in new_text:
                    new_text = new_text.replace(old_name, new_name)
            if new_text != text:
                text_file.write_text(new_text, encoding="utf-8")
                rewritten += 1
        print(f"[INFO] {rewritten} Dateien auf die neuen Textur-Dateinamen (.dds) umgeschrieben")


def scan_dae_files(dir_path: str, beamng_root: str) -> dict:
    """
    Scanne DAE-Dateien und generiere managedItemData.

    WICHTIG: dir_path MUSS bereits innerhalb von beamng_root liegen (siehe copy_tree_assets),
    damit shapeFile auf den eigenen Level zeigt statt auf die Quelle der Assets.

    Returns:
        {tree_key: {name, class, shapeFile, collidable, radius}}
    """
    dir_path_obj = Path(dir_path)
    beamng_root_obj = Path(beamng_root)

    if not dir_path_obj.is_dir():
        print(f"[ERROR] Verzeichnis nicht gefunden: {dir_path}")
        return None

    dae_files = sorted(dir_path_obj.rglob("*.dae"))

    print(f"[INFO] Gefundene DAE-Dateien: {len(dae_files)}")
    if not dae_files:
        print("[ERROR] Keine DAE-Dateien gefunden!")
        return None

    forest_item_data = {}
    tree_type_counts = defaultdict(int)

    for idx, dae_file in enumerate(dae_files, 1):
        tree_type = extract_tree_name_from_filename(dae_file.name)
        item_key = dae_file.stem

        # dae_file liegt unter beamng_root_obj (world_to_beamng) -> "levels/<level>/<rel>"
        relative_dae = dae_file.relative_to(beamng_root_obj)
        shape_file_path = f"levels/{config.LEVEL_NAME}/" + str(relative_dae).replace("\\", "/")

        radius = 2.0 if tree_type in ["cork_oak", "holm_oak"] else 1.5

        forest_item_data[item_key] = {
            "name": item_key,
            "class": "ForestItemData",
            "internalName": item_key,
            "shapeFile": shape_file_path,
            "collidable": True,
            "radius": radius,
        }

        tree_type_counts[tree_type] += 1
        if idx <= 10 or idx % 10 == 0:
            print(f"[{idx:3d}] {item_key:40s} → {tree_type:15s}")

    print(f"\n[INFO] Baum-Typen Übersicht:")
    for tree_type in sorted(tree_type_counts.keys()):
        print(f"       {tree_type:30s} : {tree_type_counts[tree_type]:3d}x")

    return forest_item_data


def categorize_trees(forest_item_data: dict) -> dict:
    """Kategorisiere Bäume nach Typ."""
    trees_by_type = defaultdict(list)
    for tree_key, tree_info in forest_item_data.items():
        tree_type = tree_info.get("name", "unknown")
        trees_by_type[tree_type].append(tree_key)
    return dict(trees_by_type)


def create_tree_distribution(preferred_trees: list) -> dict:
    """Erstelle tree_distribution Dictionary mit gleichmäßiger Verteilung."""
    if not preferred_trees:
        return {}
    probability = 1.0 / len(preferred_trees)
    return {tree: probability for tree in preferred_trees}


def generate_forest_types(trees_by_type: dict) -> dict:
    """Generiere sinnvolle Waldtypen für deutsche Wälder."""
    forest_types = {}
    all_tree_keys = []
    for tree_type, keys in trees_by_type.items():
        all_tree_keys.extend(keys)

    # 1. German Deciduous Dense
    deciduous_trees = [t for t in all_tree_keys if "forest" in t or "large" in t][:10]
    if deciduous_trees:
        forest_types["german_deciduous_dense"] = {
            "tree_density": 1.0,
            "average_height": [20.0, 30.0],
            "underground_material": "forest_floor",
            "lod_distance": 250.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(deciduous_trees),
            "comment": "Dichter Laubwald - Buchen und Eichen (klassischer deutscher Wald)",
        }

    # 2. German Mixed Forest
    mixed_trees = [t for t in all_tree_keys if "forest" in t or "group" in t][:10]
    if mixed_trees:
        forest_types["german_mixed_forest"] = {
            "tree_density": 0.85,
            "average_height": [18.0, 26.0],
            "underground_material": "forest_floor",
            "lod_distance": 220.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(mixed_trees),
            "comment": "Mischwald - Buchen, Eichen und Espen (vielfältiger Bestand)",
        }

    # 3. German Sparse Deciduous
    sparse_trees = [t for t in all_tree_keys if "bush" in t or ("small" in t and "forest" not in t)][:10]
    if sparse_trees:
        forest_types["german_sparse_deciduous"] = {
            "tree_density": 0.4,
            "average_height": [15.0, 23.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(sparse_trees),
            "comment": "Lichter Laubwald - überwiegend Busch- und kleinere Bäume",
        }

    # 4. Orchard Area
    orchard_trees = [t for t in all_tree_keys if "small" in t or "sml" in t][:10]
    if orchard_trees:
        forest_types["orchard_area"] = {
            "tree_density": 0.3,
            "average_height": [8.0, 15.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(orchard_trees),
            "comment": "Obstplantage - niedrige und kleine Bäume",
        }

    # 5. Hedgerow
    hedge_trees = [t for t in all_tree_keys if "wall" in t or ("small" in t and "bush" not in t)][:6]
    if hedge_trees:
        forest_types["hedgerow"] = {
            "tree_density": 0.2,
            "average_height": [10.0, 18.0],
            "underground_material": "grassland",
            "lod_distance": 120.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(hedge_trees),
            "comment": "Hecke/Feldgehölz - dünne, lineare Bestände",
        }

    # 6. Dead Forest
    dead_trees = [t for t in all_tree_keys if "dead" in t]
    if dead_trees:
        forest_types["dead_forest"] = {
            "tree_density": 0.3,
            "average_height": [15.0, 25.0],
            "underground_material": "forest_floor",
            "lod_distance": 200.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(dead_trees),
            "comment": "Totholz/Verfallender Wald - dürre, tote Bäume",
        }

    return forest_types


def generate_forest_mappings(forest_types: dict) -> dict:
    """Generiere forest_mappings basierend auf verfügbaren Waldtypen."""
    if not forest_types:
        return {
            "landuse": {"forest": "generic_forest", "wood": "generic_forest", "orchard": "generic_forest"},
            "natural": {
                "wood": "generic_forest",
                "forest": "generic_forest",
                "scrub": "generic_forest",
                "heath": "generic_forest",
                "tree_row": "generic_forest",
                "wetland": "generic_forest",
            },
            "leisure": {"nature_reserve": "generic_forest", "park": "generic_forest"},
            "tag_overrides": {
                "trees=conifer": "generic_forest",
                "trees=broadleaf": "generic_forest",
                "trees=mixed": "generic_forest",
                "leaf_type=needleleaf": "generic_forest",
                "leaf_type=broadleaved": "generic_forest",
                "leaf_type=mixed": "generic_forest",
            },
        }

    default_forest = "german_mixed_forest" if "german_mixed_forest" in forest_types else list(forest_types.keys())[0]

    return {
        "landuse": {
            "forest": default_forest,
            "wood": default_forest,
            "orchard": "orchard_area" if "orchard_area" in forest_types else default_forest,
        },
        "natural": {
            "wood": default_forest,
            "forest": default_forest,
            "scrub": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "heath": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "tree_row": "hedgerow" if "hedgerow" in forest_types else default_forest,
            "wetland": default_forest,
        },
        "leisure": {
            "nature_reserve": default_forest,
            "park": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
        },
        "tag_overrides": {
            "trees=conifer": default_forest,
            "trees=broadleaf": "german_deciduous_dense" if "german_deciduous_dense" in forest_types else default_forest,
            "trees=mixed": default_forest,
            "leaf_type=needleleaf": default_forest,
            "leaf_type=broadleaved": (
                "german_deciduous_dense" if "german_deciduous_dense" in forest_types else default_forest
            ),
            "leaf_type=mixed": default_forest,
        },
    }


def main():
    """Hauptfunktion: Generiere managedItemData.json und Waldtypen."""
    print("=" * 80)
    print("[START] Generiere Forest Assets (managedItemData + forest_type_templates)")
    print("=" * 80)

    install_dir = get_beamng_install_dir()

    # ===== PHASE 0: Baum-Assets in den eigenen Level kopieren =====
    print("\n[PHASE 0] Kopiere Baum-Assets in den eigenen Level (macht world_to_beamng unabhängig)")
    print("-" * 80)

    # east_coast_usa dient nur noch als Fallback-Quelle für Geometrie (Arten, die es in
    # BeamNGs kanonischer trees_library nicht gibt) und als Quelle der Material-Definitionen.
    userpath_levels_dir = config.BEAMNG_DIR.parent
    source_dir = userpath_levels_dir / "east_coast_usa" / "art" / "shapes" / "trees"

    if not source_dir.is_dir():
        print(f"[ERROR] Quellverzeichnis nicht gefunden: {source_dir}")
        return

    dest_dir = config.BEAMNG_DIR / "art" / "shapes" / "trees"
    copy_tree_assets(source_dir, dest_dir, install_dir)
    resolve_tree_texture_links(dest_dir, install_dir)

    # ===== PHASE 1: Scan DAE-Dateien (in der eigenen Kopie!) =====
    print("\n[PHASE 1] Scanne DAE-Dateien und generiere managedItemData.json")
    print("-" * 80)

    forest_item_data = scan_dae_files(str(dest_dir), str(config.BEAMNG_DIR))

    if not forest_item_data:
        print("[ERROR] Keine Forest-Items generiert")
        return

    # Speichere managedItemData.json (BeamNG erwartet die Item-Registry unter art/forest/)
    output_dir = config.BEAMNG_DIR / "art" / "forest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "managedItemData.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(forest_item_data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] managedItemData.json erstellt: {output_file}")
    print(f"       {len(forest_item_data)} Tree-Items")

    # ===== PHASE 2: Generiere Waldtypen =====
    print("\n[PHASE 2] Generiere Waldtypen und Mappings")
    print("-" * 80)

    trees_by_type = categorize_trees(forest_item_data)

    print("\n[INFO] Baum-Kategorisierung:")
    for tree_type, trees in sorted(trees_by_type.items()):
        print(f"       {tree_type:20s} : {len(trees):2d}x")

    forest_types = generate_forest_types(trees_by_type)

    print(f"\n[INFO] Generierte Waldtypen:")
    for forest_type in sorted(forest_types.keys()):
        tree_count = len(forest_types[forest_type].get("preferred_trees", {}))
        print(f"       {forest_type:30s} : {tree_count:3d} trees")

    forest_mappings = generate_forest_mappings(forest_types)

    # ===== PHASE 3: Update osm_to_beamng.json =====
    print(f"\n[PHASE 3] Aktualisiere osm_to_beamng.json")
    print("-" * 80)

    config_path = Path("data/osm_to_beamng.json")
    with open(config_path, "r", encoding="utf-8") as f:
        osm_config = json.load(f)

    osm_config["forest_type_templates"] = forest_types
    osm_config["forest_mappings"] = forest_mappings

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(osm_config, f, indent=4, ensure_ascii=False)

    print(f"[DONE] osm_to_beamng.json aktualisiert")
    print(f"       - {len(forest_types)} forest_type_templates")
    print(f"       - forest_mappings aktualisiert")

    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("[✓] ERFOLGREICH ABGESCHLOSSEN")
    print("=" * 80)
    print(f"managedItemData.json: {len(forest_item_data)} Tree-Items")
    print(f"Forest-Typen:        {len(forest_types)}")
    print(f"  - german_deciduous_dense")
    print(f"  - german_mixed_forest")
    print(f"  - german_sparse_deciduous")
    print(f"  - orchard_area")
    print(f"  - hedgerow")
    print(f"  - dead_forest")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
