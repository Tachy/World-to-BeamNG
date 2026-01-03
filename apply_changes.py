#!/usr/bin/env python3
"""Script zum Anwenden der DEM-Integration in detect_junctions_in_centerlines"""

import re


# Änderung 3: DEM-Interpolator nach return-Statement
def apply_change_3():
    with open("world_to_beamng/geometry/junctions.py", "r", encoding="utf-8") as f:
        content = f.read()

    old = """    if not road_polygons:
        return []

    # Sammle alle Strassenenden (Anfang und Ende) und precompute Segmentdaten
    endpoints = []  # (x, y, z, road_idx, is_start)"""

    new = """    if not road_polygons:
        return []

    # Prepare DEM interpolator if provided
    dem_interpolator = None
    if height_points is not None and height_elevations is not None:
        dem_interpolator = cKDTree(height_points[:, :2])
        height_elevations_array = np.asarray(height_elevations)

    # Sammle alle Strassenenden (Anfang und Ende) und precompute Segmentdaten
    endpoints = []  # (x, y, z, road_idx, is_start)"""

    content = content.replace(old, new)

    with open("world_to_beamng/geometry/junctions.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ Änderung 3 abgeschlossen")


# Änderung 4: _get_z_at_point Funktion
def apply_change_4():
    with open("world_to_beamng/geometry/junctions.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Finde die _get_z_at_point Funktion und ersetze sie
    pattern = r'''    def _get_z_at_point\(road_idx, xy_point\):
        """Interpoliert Z-Koordinate an einem XY-Punkt auf der Straße \(vektorisiert mit einsum\)\."""
        cache = road_cache\[road_idx\]
        coords = cache\["coords"\]
        if not coords or len\(coords\) < 2:
            return 0\.0 if not coords else coords\[0\]\[2\]

        seg_mids = cache\["seg_mids"\]
        z_mid = cache\["z_mid"\]

        if seg_mids\.size == 0:
            return coords\[0\]\[2\]

        # OPTIMIZATION: Nutze einsum statt sum\(\) für Distanzberechnung
        diff = seg_mids - np\.array\(xy_point\)
        dists_sq = np\.einsum\("ij,ij->i", diff, diff\)
        best_idx = int\(np\.argmin\(dists_sq\)\)

        return z_mid\[best_idx\]'''

    replacement = '''    def _get_z_at_point(road_idx, xy_point):
        """Interpoliert Z-Koordinate an XY-Punkt vom DEM (nicht aus OSM-Rohdaten)."""
        if dem_interpolator is not None:
            # Nutze DEM für normalisierte Z-Werte
            dist, idx = dem_interpolator.query(xy_point)
            return float(height_elevations_array[idx])
        else:
            # Fallback: Original-Logik aus OSM-Coords
            cache = road_cache[road_idx]
            coords = cache["coords"]
            if not coords or len(coords) < 2:
                return 0.0 if not coords else coords[0][2]
            seg_mids = cache["seg_mids"]
            z_mid = cache["z_mid"]
            if seg_mids.size == 0:
                return coords[0][2]
            diff = seg_mids - np.array(xy_point)
            dists_sq = np.einsum("ij,ij->i", diff, diff)
            best_idx = int(np.argmin(dists_sq))
            return z_mid[best_idx]'''

    content = re.sub(pattern, replacement, content)

    with open("world_to_beamng/geometry/junctions.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ Änderung 4 abgeschlossen")


if __name__ == "__main__":
    print("Wende kritische Änderungen an...")
    apply_change_3()
    apply_change_4()
    print("\nAlle Änderungen abgeschlossen!")
