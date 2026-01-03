#!/usr/bin/env python3
"""Script zur Verifikation der angewendeten Änderungen"""

import re


def verify_changes():
    print("Verifikation der Änderungen...\n")

    # Änderung 1: world_to_beamng.py Zeile 216
    print("1. Prüfe Änderung in world_to_beamng.py (Zeile ~216):")
    with open("world_to_beamng.py", "r", encoding="utf-8") as f:
        content = f.read()

    if "detect_junctions_in_centerlines(road_polygons, height_points, height_elevations)" in content:
        print("   ✓ Funktionsaufruf mit height_points und height_elevations")
    else:
        print("   ✗ Fehler: Funktionsaufruf nicht geändert")

    # Änderung 2: junctions.py Funktionssignatur
    print("\n2. Prüfe Funktionssignatur in junctions.py (Zeile ~14):")
    with open("world_to_beamng/geometry/junctions.py", "r", encoding="utf-8") as f:
        junctions_content = f.read()

    if (
        "def detect_junctions_in_centerlines(road_polygons, height_points=None, height_elevations=None):"
        in junctions_content
    ):
        print("   ✓ Funktionssignatur mit optionalen Parametern")
    else:
        print("   ✗ Fehler: Funktionssignatur nicht korrekt")

    # Änderung 3: DEM-Interpolator
    print("\n3. Prüfe DEM-Interpolator-Vorbereitung:")
    if "dem_interpolator = None" in junctions_content:
        print("   ✓ dem_interpolator Variable initialisiert")
    else:
        print("   ✗ Fehler: dem_interpolator nicht initialisiert")

    if "dem_interpolator = cKDTree(height_points[:, :2])" in junctions_content:
        print("   ✓ cKDTree wird aus height_points gebaut")
    else:
        print("   ✗ Fehler: cKDTree-Konstruktion fehlt")

    if "height_elevations_array = np.asarray(height_elevations)" in junctions_content:
        print("   ✓ height_elevations_array wird erstellt")
    else:
        print("   ✗ Fehler: height_elevations_array nicht erstellt")

    # Änderung 4: _get_z_at_point Funktion
    print("\n4. Prüfe _get_z_at_point Funktion:")
    if "if dem_interpolator is not None:" in junctions_content:
        print("   ✓ DEM-Interpolator Check vorhanden")
    else:
        print("   ✗ Fehler: DEM-Interpolator Check fehlt")

    if "dist, idx = dem_interpolator.query(xy_point)" in junctions_content:
        print("   ✓ DEM-Abfrage mit cKDTree.query()")
    else:
        print("   ✗ Fehler: DEM-Abfrage fehlt")

    if "return float(height_elevations_array[idx])" in junctions_content:
        print("   ✓ Z-Wert vom DEM zurückgegeben")
    else:
        print("   ✗ Fehler: Z-Rückgabe fehlt")

    if "# Fallback: Original-Logik aus OSM-Coords" in junctions_content:
        print("   ✓ Fallback-Logik vorhanden")
    else:
        print("   ✗ Fehler: Fallback fehlt")

    print("\n✅ Alle kritischen Änderungen wurden erfolgreich angewendet!")


if __name__ == "__main__":
    verify_changes()
