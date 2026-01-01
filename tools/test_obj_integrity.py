"""
OBJ/MTL Integritäts-Tester
Überprüft beamng.obj und beamng.mtl auf Korrektheit
"""

import os
import re


def test_mtl_file(mtl_path):
    """Testet MTL-Datei auf Integrität."""
    print(f"\n{'=' * 60}")
    print(f"TESTE MTL-DATEI: {mtl_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(mtl_path):
        print(f"❌ FEHLER: Datei nicht gefunden!")
        return False

    file_size = os.path.getsize(mtl_path)
    print(f"[OK] Dateigroesse: {file_size:,} Bytes")

    try:
        with open(mtl_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        print(f"[OK] Datei lesbar: {len(lines)} Zeilen")

        # Finde alle Materialien
        materials = re.findall(r"^newmtl\s+(\S+)", content, re.MULTILINE)
        print(f"\n[OK] Gefundene Materialien ({len(materials)}):")

        expected_materials = {"road_surface", "road_slope", "terrain"}
        found_materials = set(materials)

        for mat in materials:
            print(f"  • {mat}")
            if mat in expected_materials:
                print(f"    [OK] Erwartetes Material")
            else:
                print(f"    [!] Unerwartetes Material")

        # Prüfe auf fehlende Materialien
        missing = expected_materials - found_materials
        if missing:
            print(f"\n❌ FEHLER: Fehlende Materialien: {missing}")
            return False

        # Prüfe jedes Material auf notwendige Properties
        required_props = ["Ns", "Ka", "Kd", "Ks", "Ni", "d", "illum"]

        for mat in materials:
            mat_start = content.find(f"newmtl {mat}")
            next_mat = content.find("newmtl", mat_start + 1)
            mat_block = content[
                mat_start : next_mat if next_mat != -1 else len(content)
            ]

            missing_props = []
            for prop in required_props:
                if not re.search(rf"^{prop}\s+", mat_block, re.MULTILINE):
                    missing_props.append(prop)

            if missing_props:
                print(f"\n[!] Material '{mat}' fehlen Properties: {missing_props}")
            else:
                print(f"  [OK] Alle Properties vorhanden fuer '{mat}'")

        print(f"\n✅ MTL-Datei ist KORREKT")
        return True

    except Exception as e:
        print(f"\n❌ FEHLER beim Lesen: {e}")
        return False


def test_obj_file(obj_path):
    """Testet OBJ-Datei auf Integrität."""
    print(f"\n{'=' * 60}")
    print(f"TESTE OBJ-DATEI: {obj_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(obj_path):
        print(f"❌ FEHLER: Datei nicht gefunden!")
        return False

    file_size = os.path.getsize(obj_path)
    print(f"[OK] Dateigroesse: {file_size:,} Bytes ({file_size / 1024 / 1024:.2f} MB)")

    try:
        vertex_count = 0
        face_count = 0
        objects = []
        materials_used = []
        mtl_file = None

        current_object = None
        current_material = None

        # Statistiken pro Objekt
        object_stats = {}

        print("\n📊 Analysiere Datei (zeige erste 100 Zeilen)...")

        with open(obj_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Zeige erste 100 Zeilen
                if line_num <= 100:
                    print(f"  {line_num:4}: {line[:80]}")

                if not line or line.startswith("#"):
                    continue

                # MTL-Bibliothek
                if line.startswith("mtllib "):
                    mtl_file = line.split(None, 1)[1]

                # Objekt-Definition
                elif line.startswith("o "):
                    current_object = line.split(None, 1)[1]
                    objects.append(current_object)
                    object_stats[current_object] = {
                        "vertices": 0,
                        "faces": 0,
                        "material": None,
                    }
                    print(f"\n  -> Objekt gefunden: '{current_object}'")

                # Material-Verwendung
                elif line.startswith("usemtl "):
                    current_material = line.split(None, 1)[1]
                    if current_material not in materials_used:
                        materials_used.append(current_material)
                    if (
                        current_object
                        and object_stats[current_object]["material"] is None
                    ):
                        object_stats[current_object]["material"] = current_material

                # Vertices
                elif line.startswith("v "):
                    vertex_count += 1

                # Faces
                elif line.startswith("f "):
                    face_count += 1
                    if current_object:
                        object_stats[current_object]["faces"] += 1

        print(f"\n{'=' * 60}")
        print(f"ZUSAMMENFASSUNG")
        print(f"{'=' * 60}")
        print(f"[OK] MTL-Datei referenziert: {mtl_file}")
        print(f"[OK] Vertices gesamt: {vertex_count:,}")
        print(f"[OK] Faces gesamt: {face_count:,}")
        print(f"[OK] Objekte gefunden: {len(objects)}")

        # Erwartete Objekte
        expected_objects = {"road_surface", "road_slope", "terrain"}
        found_objects = set(objects)

        print(f"\n📦 OBJEKTE:")
        for obj in objects:
            stats = object_stats[obj]
            status = "[v]" if obj in expected_objects else "[!]"
            print(f"  {status} {obj}")
            print(f"      Faces: {stats['faces']:,}")
            print(f"      Material: {stats['material']}")

        # Prüfe auf fehlende Objekte
        missing_objects = expected_objects - found_objects
        if missing_objects:
            print(f"\n❌ FEHLER: Fehlende Objekte: {missing_objects}")
            return False

        # Prüfe Material-Verwendung
        print(f"\n🎨 VERWENDETE MATERIALIEN:")
        expected_materials = {"road_surface", "road_slope", "terrain"}
        for mat in materials_used:
            status = "[v]" if mat in expected_materials else "[!]"
            print(f"  {status} {mat}")

        missing_materials = expected_materials - set(materials_used)
        if missing_materials:
            print(f"\n[!] Warnung: Nicht verwendete Materialien: {missing_materials}")

        # Prüfe Vertex-Indizes in Faces (Sample)
        print(f"\n🔍 Pruefe Face-Indizes (erste 1000 Faces)...")
        with open(obj_path, "r", encoding="utf-8") as f:
            face_samples = []
            for line in f:
                if line.startswith("f "):
                    face_samples.append(line.strip())
                    if len(face_samples) >= 1000:
                        break

            invalid_faces = []
            max_index_seen = 0
            for face_line in face_samples:
                parts = face_line.split()[1:]  # Skip 'f'
                for part in parts:
                    idx = int(part.split("/")[0])  # Nur Vertex-Index
                    max_index_seen = max(max_index_seen, idx)
                    if idx < 1 or idx > vertex_count:
                        invalid_faces.append((face_line, idx))
                        break

            print(
                f"  Max Index in Sample: {max_index_seen}, Vertex Count: {vertex_count}"
            )

            if invalid_faces:
                print(f"  ❌ FEHLER: {len(invalid_faces)} ungueltige Faces gefunden:")
                for face, bad_idx in invalid_faces[:10]:
                    print(f"    Index {bad_idx} > {vertex_count}: {face}")
                return False
            else:
                print(f"  [OK] Alle Face-Indizes gueltig (1 bis {vertex_count})")

        # Prüfe auf leere Objekte
        print(f"\n📊 OBJEKT-STATISTIKEN:")
        for obj, stats in object_stats.items():
            if stats["faces"] == 0:
                print(f"  [!] Warnung: Objekt '{obj}' hat keine Faces!")
            else:
                print(f"  [OK] {obj}: {stats['faces']:,} Faces")

        print(f"\n✅ OBJ-Datei ist KORREKT")
        return True

    except Exception as e:
        print(f"\n❌ FEHLER beim Lesen: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Hauptfunktion: Teste beide Dateien."""
    print("\n" + "=" * 60)
    print("OBJ/MTL INTEGRITÄTS-TEST")
    print("=" * 60)

    mtl_ok = test_mtl_file("beamng.mtl")
    obj_ok = test_obj_file("beamng.obj")

    print(f"\n{'=' * 60}")
    print(f"GESAMT-ERGEBNIS")
    print(f"{'=' * 60}")

    if mtl_ok and obj_ok:
        print("✅ BEIDE DATEIEN SIND KORREKT UND INTEGER!")
    else:
        print("❌ FEHLER GEFUNDEN:")
        if not mtl_ok:
            print("  • beamng.mtl hat Probleme")
        if not obj_ok:
            print("  • beamng.obj hat Probleme")

    print(f"{'=' * 60}\n")

    return mtl_ok and obj_ok


if __name__ == "__main__":
    main()
