"""
Vendor Shared Textures: Copies the BeamNG default textures referenced by data/osm_to_beamng.json
(asphalt, plaster, roof tiles, ...) from the BeamNG installation into the level itself.

Background: osm_to_beamng.json references these textures under
"levels/world_to_beamng/art/shapes/assets/materials/..." - i.e. as if they already lay
locally in the level. Until now nobody put them there, which is why BeamNG shows
"no Texture" for roads/buildings on load. This script fetches the real files from the BeamNG
content ZIPs (content/assets/materials/*.zip) and copies them to the expected location.
"""

from pathlib import Path
import json
import sys
import zipfile

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config
from world_to_beamng.io.beamng_install import get_beamng_install_dir


def find_texture_paths(obj, level_prefix: str, found: set) -> None:
    """Recursively collects all string values that lie under the level texture prefix."""
    if isinstance(obj, dict):
        for v in obj.values():
            find_texture_paths(v, level_prefix, found)
    elif isinstance(obj, list):
        for v in obj:
            find_texture_paths(v, level_prefix, found)
    elif isinstance(obj, str) and obj.startswith(level_prefix):
        found.add(obj)


def build_zip_index(materials_dir: Path) -> dict:
    """Builds a dict {virtual_path_in_zip: zip_file} over all materials ZIPs."""
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
    print("[START] Vendor Shared Textures (embed road/building materials locally)")
    print("=" * 80)

    install_dir = get_beamng_install_dir()
    materials_dir = install_dir / "content" / "assets" / "materials"
    print(f"[INFO] BeamNG installation: {install_dir}")

    if not materials_dir.is_dir():
        print(f"[ERROR] Materials directory not found: {materials_dir}")
        return

    osm_config_path = Path("data/osm_to_beamng.json")
    osm_config = json.loads(osm_config_path.read_text(encoding="utf-8"))

    level_prefix = f"levels/{config.LEVEL_NAME}/art/shapes/assets/materials/"
    found = set()
    find_texture_paths(osm_config, level_prefix, found)
    print(f"[INFO] Found {len(found)} referenced shared textures in osm_to_beamng.json")

    if not found:
        print("[INFO] Nothing to do.")
        return

    print("[INFO] Building index over BeamNG content/assets/materials/*.zip ...")
    zip_index = build_zip_index(materials_dir)
    print(f"[INFO] Indexed {len(zip_index)} files in {len(list(materials_dir.glob('*.zip')))} ZIPs")

    level_shapes_prefix = f"levels/{config.LEVEL_NAME}/art/shapes/"
    open_zips = {}
    copied = 0
    missing = []

    for level_path in sorted(found):
        virtual_path = level_path[len(level_shapes_prefix):]  # e.g. "assets/materials/decalroad/..."
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

    print(f"\n[DONE] {copied} textures copied")
    if missing:
        print(f"[WARNING] {len(missing)} textures NOT found in the content ZIPs:")
        for m in missing:
            print("   ", m)

    print("=" * 80)
    print("[✓] COMPLETED SUCCESSFULLY" if not missing else "[!] COMPLETED WITH WARNINGS")
    print("=" * 80)


if __name__ == "__main__":
    main()
