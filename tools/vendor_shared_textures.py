"""
Vendor Shared Textures: Kopiert von data/osm_to_beamng.json referenzierte BeamNG-Standard-
Texturen (Asphalt, Putz, Dachziegel, ...) aus der BeamNG-Installation in den eigenen Level.

Hintergrund: osm_to_beamng.json referenziert diese Texturen unter
"levels/world_to_beamng/art/shapes/assets/materials/..." - also so, als lägen sie bereits
lokal im Level. Bisher hat sie dort aber niemand hingelegt, deshalb zeigt BeamNG beim Laden
"no Texture" für Straßen/Gebäude. Dieses Script holt die echten Dateien aus den BeamNG-
Content-ZIPs (content/assets/materials/*.zip) und kopiert sie an die erwartete Stelle.
"""

from pathlib import Path
import json
import sys
import zipfile

# Importiere config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config
from world_to_beamng.io.beamng_install import get_beamng_install_dir


def find_texture_paths(obj, level_prefix: str, found: set) -> None:
    """Sammelt rekursiv alle String-Werte, die unter dem Level-Textur-Prefix liegen."""
    if isinstance(obj, dict):
        for v in obj.values():
            find_texture_paths(v, level_prefix, found)
    elif isinstance(obj, list):
        for v in obj:
            find_texture_paths(v, level_prefix, found)
    elif isinstance(obj, str) and obj.startswith(level_prefix):
        found.add(obj)


def build_zip_index(materials_dir: Path) -> dict:
    """Baut ein Dict {virtueller_pfad_in_der_zip: zip_datei} über alle materials-ZIPs."""
    index = {}
    for zip_path in sorted(materials_dir.glob("*.zip")):
        try:
            z = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            continue
        for name in z.namelist():
            if not name.endswith("/"):
                index[name] = zip_path
    return index


def main():
    print("=" * 80)
    print("[START] Vendor Shared Textures (Straßen/Gebäude-Materialien lokal einbetten)")
    print("=" * 80)

    install_dir = get_beamng_install_dir()
    materials_dir = install_dir / "content" / "assets" / "materials"
    print(f"[INFO] BeamNG-Installation: {install_dir}")

    if not materials_dir.is_dir():
        print(f"[ERROR] Materials-Verzeichnis nicht gefunden: {materials_dir}")
        return

    osm_config_path = Path("data/osm_to_beamng.json")
    osm_config = json.loads(osm_config_path.read_text(encoding="utf-8"))

    level_prefix = f"levels/{config.LEVEL_NAME}/art/shapes/assets/materials/"
    found = set()
    find_texture_paths(osm_config, level_prefix, found)
    print(f"[INFO] {len(found)} referenzierte geteilte Texturen in osm_to_beamng.json gefunden")

    if not found:
        print("[INFO] Nichts zu tun.")
        return

    print("[INFO] Baue Index über BeamNG content/assets/materials/*.zip ...")
    zip_index = build_zip_index(materials_dir)
    print(f"[INFO] {len(zip_index)} Dateien in {len(list(materials_dir.glob('*.zip')))} ZIPs indiziert")

    level_shapes_prefix = f"levels/{config.LEVEL_NAME}/art/shapes/"
    open_zips = {}
    copied = 0
    missing = []

    for level_path in sorted(found):
        virtual_path = level_path[len(level_shapes_prefix):]  # z.B. "assets/materials/decalroad/..."
        zip_path = zip_index.get(virtual_path)

        if not zip_path:
            missing.append(level_path)
            continue

        if zip_path not in open_zips:
            open_zips[zip_path] = zipfile.ZipFile(zip_path)
        z = open_zips[zip_path]

        dest = config.BEAMNG_DIR / "art" / "shapes" / virtual_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(virtual_path))
        copied += 1

    print(f"\n[DONE] {copied} Texturen kopiert")
    if missing:
        print(f"[WARNUNG] {len(missing)} Texturen NICHT in den Content-ZIPs gefunden:")
        for m in missing:
            print("   ", m)

    print("=" * 80)
    print("[✓] ERFOLGREICH ABGESCHLOSSEN" if not missing else "[!] ABGESCHLOSSEN MIT WARNUNGEN")
    print("=" * 80)


if __name__ == "__main__":
    main()
