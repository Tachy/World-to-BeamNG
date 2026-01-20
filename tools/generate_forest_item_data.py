"""
Generate forestItemData.json from DAE files.

Durchsucht ein Verzeichnis rekursiv nach .dae-Dateien und erstellt
eine forestItemData.json in config.BEAMNG_DIR / main mit passenden Baum-Namen.
"""

from pathlib import Path
import json
import re
from collections import defaultdict
import sys

# Importiere config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config


# Mapping von Dateinamen-Patterns zu Baumarten
TREE_NAME_PATTERNS = {
    # Eichen (Oak)
    r"oak|eiche": "oak",
    r"pedunculate|sessile|quercus": "oak",
    # Buchen (Beech)
    r"beech|buche|fagus": "beech",
    # Birken (Birch)
    r"birch|birke|betula": "birch",
    # Espen (Aspen)
    r"aspen|espe|tremuloides": "aspen",
    # Fichten (Spruce)
    r"spruce|fichte|picea": "spruce",
    # Kiefern/Scots Pine (Pine)
    r"pine|scots|kiefer|sylvestris": "scots_pine",
    # Tannen (Fir)
    r"fir|tanne|abies": "fir",
    # Lärchen (Larch)
    r"larch|lärche|larix": "larch",
    # Ahorne (Maple)
    r"maple|ahorn": "maple",
    # Eschen (Ash)
    r"ash|esche|fraxinus": "ash",
    # Ulmen (Elm)
    r"elm|ulme": "elm",
    # Pappeln (Poplar)
    r"poplar|pappel": "poplar",
    # Erlen (Alder)
    r"alder|erle|alnus": "alder",
    # Weiden (Willow)
    r"willow|weide|salix": "willow",
    # Eberesche (Rowan)
    r"rowan|eberesche|sorbus": "rowan",
    # Hasel (Hazel)
    r"hazel|hasel|corylus": "hazel",
    # Holunder (Elder)
    r"elder|holunder|sambucus": "elder",
    # Korkeiche (Cork Oak)
    r"cork|kork": "cork_oak",
    # Steineiche (Holm Oak)
    r"holm|steineiche": "holm_oak",
    # Olive
    r"olive|oliv": "olive",
}


def extract_tree_name_from_filename(filename: str) -> str:
    """
    Extrahiere einen passenden Baumnamen aus dem Dateinamen.

    Args:
        filename: Name der DAE-Datei (mit oder ohne Extension)

    Returns:
        Baum-Name aus den Patterns oder der bereinigte Dateiname
    """
    # Entferne Extension
    name = Path(filename).stem

    # Konvertiere zu lowercase für Pattern-Matching
    name_lower = name.lower()

    # Versuche Pattern zu matchen
    for pattern, tree_name in TREE_NAME_PATTERNS.items():
        if re.search(pattern, name_lower):
            return tree_name

    # Fallback: Verwende bereinigten Dateinamen
    # Entferne Zahlen, Underscores und spezielle Zeichen
    cleaned = re.sub(r"[_\-\d]", " ", name).strip()

    # Wenn immer noch zu lang oder zu generisch, nutze generischen Namen
    if not cleaned or cleaned.lower() in ["tree", "model", "asset"]:
        return "tree"

    return cleaned.lower()


def generate_forest_item_data(dir_path: str, beamng_root: str) -> dict:
    """
    Durchsuche Verzeichnis nach DAE-Dateien und erstelle forestItemData.

    Pro DAE-Datei wird ein Eintrag angelegt (nicht pro Baumtyp).

    Args:
        dir_path: Pfad zum Suchverzeichnis mit DAE-Dateien
        beamng_root: Root-Pfad des BeamNG Verzeichnisses (für relative Pfade)

    Returns:
        {
            "tree_oak_01": {
                "name": "oak",
                "class": "TSForestItemData",
                "shapeFile": "levels/world_to_beamng/art/shapes/...",
                "collidable": true,
                "radius": 1.5
            },
            "tree_aspen_blocker_a": {
                "name": "aspen",
                "class": "TSForestItemData",
                "shapeFile": "levels/world_to_beamng/art/shapes/...",
                "collidable": true,
                "radius": 1.5
            },
            ...
        }
    """
    dir_path_obj = Path(dir_path)
    beamng_root_obj = Path(beamng_root)

    if not dir_path_obj.is_dir():
        print(f"[ERROR] Verzeichnis nicht gefunden: {dir_path}")
        return None

    # Finde alle DAE-Dateien
    dae_files = sorted(dir_path_obj.rglob("*.dae"))

    print(f"\n[INFO] Gefundene DAE-Dateien: {len(dae_files)}")
    print(f"[INFO] Suchverzeichnis: {dir_path}\n")

    if not dae_files:
        print("[WARNING] Keine DAE-Dateien gefunden!")
        return None

    # Erstelle forestItemData JSON (ein Entry pro DAE-Datei)
    forest_item_data = {}
    tree_type_counts = defaultdict(int)

    for idx, dae_file in enumerate(dae_files, 1):
        # Extrahiere Baumnamen aus Dateiname
        tree_type = extract_tree_name_from_filename(dae_file.name)

        # Erstelle eindeutigen Key aus Dateiname (ohne Extension)
        item_key = dae_file.stem

        # Konvertiere zu RelativePath vom BeamNG Root
        try:
            relative_dae = dae_file.relative_to(beamng_root_obj)
        except ValueError:
            # Falls nicht relativ, nutze absolute Path
            relative_dae = dae_file

        # Konvertiere zu BeamNG forward-slash Pfad und extrahiere ab "levels/"
        full_path = str(relative_dae).replace("\\", "/")

        # Extrahiere nur den Teil ab "levels/"
        if "levels/" in full_path:
            shape_file_path = "levels/" + full_path.split("levels/")[1]
        else:
            # Fallback: Verwende den kompletten Pfad
            shape_file_path = full_path

        # Bestimme Radius basierend auf Baumtyp (Default 1.5, größere für cork_oak)
        radius = 2.0 if tree_type in ["cork_oak", "holm_oak"] else 1.5

        # Erstelle Eintrag
        forest_item_data[item_key] = {
            "name": item_key,
            "class": "TSForestItemData",
            "shapeFile": shape_file_path,
            "collidable": True,
            "radius": radius,
        }

        tree_type_counts[tree_type] += 1

        print(f"[{idx:3d}] {item_key:40s} → type: {tree_type:15s} radius: {radius}")

    print(f"\n[INFO] Baum-Typen Übersicht:")
    for tree_type in sorted(tree_type_counts.keys()):
        count = tree_type_counts[tree_type]
        print(f"       {tree_type:30s} : {count:3d}x")

    return forest_item_data


def main():
    """Hauptfunktion."""
    # Suchverzeichnis für DAE-Dateien
    search_dir = Path(
        r"C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\east_coast_usa\art\shapes\trees"
    )

    # Prüfe ob Verzeichnis existiert
    if not search_dir.is_dir():
        print(f"[ERROR] Suchverzeichnis nicht gefunden: {search_dir}")
        return

    print(f"[START] Generiere forestItemData.json")
    print(f"        Quelle: {search_dir}\n")

    # Generiere Daten
    result = generate_forest_item_data(str(search_dir), str(config.BEAMNG_DIR))

    if not result:
        print("[ERROR] Keine Daten generiert")
        return

    # Zusammenfassung
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique tree types: {len(result)}\n")

    print("Tree types found:")
    print("-" * 80)
    for tree_type in sorted(result.keys()):
        print(f"  {tree_type:30s} → radius: {result[tree_type]['radius']}")

    # Erstelle Output-Datei in config.BEAMNG_DIR / main
    output_dir = config.BEAMNG_DIR / "main"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "forestItemData.json"

    # Schreibe JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[DONE] forestItemData.json erstellt: {output_file}")
    print(f"       {len(result)} Baumtypen")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
