"""Test calculate_stop_distance with actual data"""

import sys
import numpy as np

sys.path.insert(0, "C:\\Eigene_Programme\\World-to-BeamNG")

from world_to_beamng.mesh.road_mesh import calculate_stop_distance

# Road 926338154 coords (laut check_jp381: 3 points, last at 0.00m from JP381)
# Junction 381 position: [2463.28088232968, 3372.1154665993527, -46.92]

# Simuliere 3 Punkte, wobei der letzte genau auf Junction liegt
jp_pos = np.array([2463.28088232968, 3372.1154665993527])

# Erstelle Test-Koordinaten:
# - Punkt 0: 10m entfernt
# - Punkt 1: 5m entfernt
# - Punkt 2: 0m entfernt (genau auf Junction)
coords = np.array(
    [
        [2463.28, 3362.12],  # ~10m südlich
        [2463.28, 3367.12],  # ~5m südlich
        [2463.28, 3372.12],  # genau auf Junction
    ]
)

half_width = 3.5
min_edge_distance = 3.5
road_width = 7.0

print("Test 1: Letzter Punkt auf Junction (0.00m)")
print(f"Coords:\n{coords}")
print(f"Junction: {jp_pos}")
print(f"min_edge_distance: {min_edge_distance}m\n")

stop_idx = calculate_stop_distance(
    coords,
    jp_pos,
    road_width=road_width,
    min_edge_distance=min_edge_distance,
)

print(f"stop_idx: {stop_idx}")
print(f"Trimmed coords would be: coords[:{stop_idx}]")
print(f"Number of points after trim: {stop_idx}")
print()

# Test 2: Alle Punkte weit weg
coords2 = np.array(
    [
        [2463.28, 3350.12],  # ~22m entfernt
        [2463.28, 3360.12],  # ~12m entfernt
        [2463.28, 3365.12],  # ~7m entfernt
    ]
)

print("Test 2: Alle Punkte weit genug weg (>3.5m)")
print(f"Coords:\n{coords2}")

stop_idx2 = calculate_stop_distance(
    coords2,
    jp_pos,
    road_width=road_width,
    min_edge_distance=min_edge_distance,
)

print(f"stop_idx: {stop_idx2}")
print(f"Should keep all {len(coords2)} points")
