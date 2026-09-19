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


# east_coast_usa's Bonus-Ordner "ECA_coast_bush" enthält neben generischen
# Filler-Büschen (generibush*) auch mediterrane Arten (Korkeiche, Pinie), die
# für einen deutschen Wald (Schwarzwald/Freiburg) fehl am Platz sind - siehe
# Recherche 2026-09-18: cork_oak_bush_* und maritime_pine_bush waren über die
# reinen Dateinamen-Muster in extract_tree_name_from_filename() fälschlich in
# den "deutschen" Waldmischungen gelandet. cork_oak_bush_* referenziert
# zusätzlich die kaputte "holm_oak_trunk"-Textur (existiert im aktuellen
# east_coast_usa.zip nicht mehr) - der Ausschluss behebt beides zugleich.
# generibush/generibush_small bleiben (klimaneutrale Filler-Büsche, keine
# Abhängigkeit zu den ausgeschlossenen Arten).
EXCLUDED_NON_NATIVE_SPECIES = ("cork_oak_bush_large", "cork_oak_bush_medium", "maritime_pine_bush")


def extract_tree_assets_from_zip(dest_dir: Path, install_dir: Path) -> int:
    """
    Entpackt den Baum-Asset-Ordner (DAE, kompilierte .cdae, Imposter-DDS,
    materials.json) direkt aus dem AKTUELL INSTALLIERTEN east_coast_usa.zip.

    Grund, warum nicht aus dem entpackten Userordner (AppData/.../levels/
    east_coast_usa) kopiert wird, wie früher: dieser Ordner ist ein Jahrzehnte
    alter Überbleibsel-Unpack (Dateien datiert 2013), den Steam-Updates nie
    anfassen - seine materials.json/.dae referenzieren Material-/Textur-Namen
    (z.B. "m_fir_merged_foliage" für Douglasie/Aspen), die im aktuell
    installierten Content-Pack längst umbenannt oder entfernt wurden (siehe
    Recherche 2026-09-18: Douglasie nutzt jetzt "m_fir_leaves_distant" +
    generierte .imposter.dds statt der alten merged_foliage-Textur). Das
    Content-Zip dagegen ist IMMER exakt so aktuell wie die installierte
    Spielversion und intern konsistent (DAE, Material und Texturen werden
    zusammen geshippt) - daher keine .link-Auflösung mehr nötig, das aktuelle
    Zip enthält gar keine .link-Platzhalter mehr.

    EXCLUDED_NON_NATIVE_SPECIES wird übersprungen (siehe dort).

    Returns:
        Anzahl der entpackten Dateien
    """
    zip_path = install_dir / "content" / "levels" / "east_coast_usa.zip"
    if not zip_path.is_file():
        print(f"[ERROR] east_coast_usa.zip nicht gefunden: {zip_path}")
        return 0

    prefix = "levels/east_coast_usa/art/shapes/trees/"
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    excluded = 0
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            if entry.endswith("/") or not entry.startswith(prefix):
                continue
            filename = entry.rsplit("/", 1)[-1]
            if any(filename.lower().startswith(species) for species in EXCLUDED_NON_NATIVE_SPECIES):
                excluded += 1
                continue
            out_path = dest_dir / entry[len(prefix) :]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(z.read(entry))
            count += 1

    print(f"[INFO] {count} Baum-Asset-Dateien aus {zip_path.name} entpackt -> {dest_dir}")
    if excluded:
        print(f"[INFO] {excluded} Dateien nicht-heimischer Arten übersprungen ({', '.join(EXCLUDED_NON_NATIVE_SPECIES)})")
    return count


def copy_tree_assets(dest_dir: Path, install_dir: Path) -> int:
    """
    Entpackt den kompletten Baum-Asset-Ordner in den eigenen Level, damit
    world_to_beamng nicht mehr von einem fremden Level (east_coast_usa)
    abhängt (siehe extract_tree_assets_from_zip()).

    Ersetzt die Meshes NICHT mehr durch BeamNGs "kanonische" trees_library
    (content/assets/meshes.zip) - Recherche 2026-09-18 ergab, dass deren
    beech/oak/birch-Meshes intern auf "m_ind_beech_leaves"/"m_ind_birch_leaves_01"-
    Materialien verweisen, die in KEINEM installierten Content-Pack definiert
    sind (Sackgasse, nicht das alte "stale path"-Problem: dieselbe Recherche,
    die east_coast_usa als Quelle fixte, zeigte, dass die trees_library selbst
    kaputt ist). east_coast_usa's eigene, per main.materials.json vollständig
    definierte Meshes (z.B. "m_birch_leaves_distant" statt "m_ind_birch_leaves_01")
    sind daher jetzt in JEDEM Fall die zuverlässigere Quelle.

    Schreibt anschließend "levels/east_coast_usa/..."-Pfadreferenzen in allen
    TEXT-Dateien (materials.json UND .dae, COLLADA embedded texture refs) auf den
    eigenen Level um. .cdae-Dateien sind ein kompiliertes Binär-Cache-Format, das
    den alten Pfad ebenfalls einbettet, aber NICHT sicher text-patchbar ist
    (Längen-präfixierte Strings) - diese werden stattdessen gelöscht, damit BeamNG
    sie beim nächsten Laden automatisch frisch aus der .dae neu kompiliert.

    Returns:
        Anzahl der entpackten Dateien (0 bei Fehlschlag)
    """
    extracted = extract_tree_assets_from_zip(dest_dir, install_dir)
    if extracted == 0:
        return 0

    old_ref = "levels/east_coast_usa/art/shapes/trees"
    new_ref = f"levels/{config.LEVEL_NAME}/art/shapes/trees"

    fixed = 0
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
    return extracted


def get_beamng_install_dir() -> Path:
    """Liest den BeamNG-Installationspfad aus BeamNG.drive.ini."""
    ini_path = Path.home() / "AppData" / "Local" / "BeamNG" / "BeamNG.drive.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"BeamNG.drive.ini nicht gefunden: {ini_path}")

    raw = ini_path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser()
    parser.read_string("[main]\n" + raw)
    return Path(parser["main"]["installpath"].strip().strip('"'))


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


# Niedrige Laubbäume (gemessene Modellhöhe 6-12,2 m). Explizit statt per Namensmuster:
# "low" im Namen ist unzuverlässig (tree_douglasfir_group_low ist 22,6 m hoch).
LOW_DECIDUOUS_TREES = [
    "tree_aspen_small_low",
    "tree_aspen_small_low_group",
    "tree_aspen_small_a",
    "tree_aspen_small_b",
    "tree_aspen_small_c",
    "tree_aspen_small_d",
    "tree_beech_small_b",
    "tree_beech_small_c",
    "tree_beech_small_d",
    "tree_oak_sml_a",
    "tree_oak_sml_b",
]


# Gärten/Wohngebiete: kleine Laubbäume (Obstbaum-Größe; BeamNG hat keine echten Obstbäume) und
# Büsche (gemessene Höhe 1,2-3,3 m). Explizit, weil Namensmuster bei den Höhen täuschen.
GARDEN_TREES = [
    "tree_aspen_small_low",
    "tree_aspen_small_a",
    "tree_aspen_small_b",
    "tree_aspen_small_c",
    "tree_aspen_small_d",
    "tree_beech_small_b",
    "tree_beech_small_c",
    "tree_beech_small_d",
    "tree_oak_sml_a",
    "tree_oak_sml_b",
]
GARDEN_BUSHES = [
    "tree_beech_bush_a",
    "tree_oak_bush_a",
    "tree_aspen_bush_a",
    "tree_aspen_bush_b",
    "tree_aspen_bush_c",
    "tree_oak_bush_c",
    "tree_beech_bush_b",
    "generibush_small",
]
SINGLE_TREES = [
    "tree_aspen_small_a",
    "tree_aspen_small_b",
    "tree_aspen_small_c",
    "tree_aspen_small_d",
    "tree_beech_small_b",
    "tree_beech_small_c",
    "tree_beech_small_d",
    "tree_oak_sml_a",
    "tree_oak_sml_b",
]


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

    # 2b. German Low Deciduous: alles mit Bäumen außer landuse=forest bekommt nur niedrige Laubbäume.
    # average_height wirkt als Skalierung (Zielhöhe / 20 m): 16-22 -> 0,8-1,1.
    low_trees = [t for t in LOW_DECIDUOUS_TREES if t in all_tree_keys]
    if low_trees:
        forest_types["german_low_deciduous"] = {
            "tree_density": 0.7,
            "average_height": [16.0, 22.0],
            "underground_material": "forest_floor",
            "lod_distance": 200.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(low_trees),
            "comment": "Niedriger Laubwald (6-13 m) - alles mit Bäumen außer landuse=forest; "
            "average_height wirkt als Skalierung (Zielhöhe/20 m)",
        }

    # 2c. Gärten/Kleingärten, Wohngebiete, Einzelbäume. Mindestabstand im ForestWorkflow ist 5 m:
    # Abstand = 5 / sqrt(tree_density) -> 0,3 ergibt ca. 9 m (lichte Bepflanzung).
    garden_trees = [t for t in GARDEN_TREES if t in all_tree_keys]
    garden_bushes = [t for t in GARDEN_BUSHES if t in all_tree_keys]
    if garden_trees and garden_bushes:
        weights = {t: 0.45 / len(garden_trees) for t in garden_trees}
        weights.update({t: 0.55 / len(garden_bushes) for t in garden_bushes})
        forest_types["garden_mixed"] = {
            "tree_density": 0.3,
            "average_height": [16.0, 22.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": weights,
            "comment": "Gärten/Kleingärten - lichte kleine Laubbäume (Obstbaum-Größe) und Büsche; "
            "es gibt keine echten Obstbaum-Assets in BeamNG",
        }
    if garden_bushes:
        forest_types["residential_green"] = {
            "tree_density": 0.3,
            "average_height": [16.0, 22.0],
            "underground_material": "grassland",
            "lod_distance": 150.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(garden_bushes),
            "comment": "Wohngebiete - lichte Büsche zwischen den Häusern (Straßen/Gebäude werden ausgespart)",
        }
    single_trees = [t for t in SINGLE_TREES if t in all_tree_keys]
    if single_trees:
        forest_types["single_tree"] = {
            "tree_density": 1.0,
            "average_height": [16.0, 26.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(single_trees),
            "comment": "Einzelbäume (OSM natural=tree als Punkt)",
        }
        # Baumreihe: natural=tree_row ist eine LINIE - Bäume im Abstand row_spacing entlang der Linie
        forest_types["tree_row"] = {
            "tree_density": 1.0,
            "row_spacing": 8.0,
            "average_height": [16.0, 24.0],
            "underground_material": "grassland",
            "lod_distance": 180.0,
            "collision_enabled": True,
            "preferred_trees": create_tree_distribution(single_trees),
            "comment": "Baumreihe (OSM natural=tree_row ist eine LINIE): Bäume im Abstand row_spacing entlang der Linie",
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
    # Nur landuse=forest bekommt den hohen Mischwald, alles andere mit Bäumen den niedrigen Laubwald
    low_forest = "german_low_deciduous" if "german_low_deciduous" in forest_types else default_forest

    garden = "garden_mixed" if "garden_mixed" in forest_types else None
    residential = "residential_green" if "residential_green" in forest_types else None

    mappings = {
        "landuse": {
            "forest": default_forest,
            "wood": low_forest,
            "orchard": "orchard_area" if "orchard_area" in forest_types else default_forest,
            **({"allotments": garden} if garden else {}),
            **({"residential": residential} if residential else {}),
        },
        "natural": {
            "wood": low_forest,
            "forest": low_forest,
            "scrub": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "heath": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            "tree_row": "tree_row" if "tree_row" in forest_types else ("hedgerow" if "hedgerow" in forest_types else default_forest),
            "wetland": low_forest,
        },
        "leisure": {
            "nature_reserve": low_forest,
            "park": "german_sparse_deciduous" if "german_sparse_deciduous" in forest_types else default_forest,
            **({"garden": garden} if garden else {}),
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
        # Lichtungen (innere Ringe von Wald-Relationen) bekommen nur niedrige Laubbäume
        # Lichtungen nur in Wald-Relationen - ein Loch im Wohngebiet ist etwas anderes
        "clearings": {
            "forest_type": low_forest,
            "only_for": ["landuse=forest", "landuse=wood", "natural=wood", "natural=forest"],
        },
        # Overrides (trees=conifer, ...) verfeinern nur landuse=forest - sonst würde z.B. ein
        # natural=wood mit Nadelbaum-Tag in einen hohen Waldtyp umgeleitet
        "tag_overrides_only_for": ["landuse=forest"],
    }
    if "single_tree" in forest_types:
        mappings["single_trees"] = {"forest_type": "single_tree"}  # OSM natural=tree (Punkte)
    return mappings


def main():
    """Hauptfunktion: Generiere managedItemData.json und Waldtypen."""
    print("=" * 80)
    print("[START] Generiere Forest Assets (managedItemData + forest_type_templates)")
    print("=" * 80)

    install_dir = get_beamng_install_dir()

    # ===== PHASE 0: Baum-Assets in den eigenen Level kopieren =====
    print("\n[PHASE 0] Kopiere Baum-Assets in den eigenen Level (macht world_to_beamng unabhängig)")
    print("-" * 80)

    # east_coast_usa.zip dient als Fallback-Quelle für Geometrie + Material-Definitionen
    # für Arten, die es in BeamNGs kanonischer trees_library nicht gibt (siehe
    # extract_tree_assets_from_zip()).
    dest_dir = config.BEAMNG_DIR / "art" / "shapes" / "trees"
    if copy_tree_assets(dest_dir, install_dir) == 0:
        return

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
