"""
Straßensperre vor Tunneleinfahrten, deren Tunnel über die Kartengrenze reicht: dort endet die Welt mitten in der
Röhre, also wird die Einfahrt gesperrt - eine Reihe rot-weißer Kunststoff-Barrieren (BeamNG-Standardasset
config.ROADBLOCK_SHAPE) quer über die Zufahrt, kurz vor der Portalebene. Die Röhre selbst liegt dann flach auf
Einfahrtshöhe (geometry/polygon.py::apply_structure_elevation_profiles).
"""

import math
from typing import Dict, List, Sequence, Tuple

from ..geometry.polygon import outside_map


def plan_roadblocks(
    plans: Sequence[Dict],
    bounds: Tuple[float, float, float, float],
    edge_margin: float,
    width_margin: float,
    distance: float,
    side_margin: float,
    spacing: float,
    entrances: Sequence[Tuple[float, float]] = (),
    entrance_tol: float = 0.5,
) -> List[Dict]:
    """
    Barriere-Elemente für alle Tunnel-Pläne (tunnel_portal.plan_tunnels()), die mit genau einem Ende über die
    Kartengrenze reichen: vor dem anderen Portal eine Reihe quer über die Fahrbahn (Fahrbahnbreite = Röhrenbreite -
    width_margin, je Seite side_margin mehr), `distance` vor der Portalebene, Elemente im Abstand `spacing`. Nur an
    echten Einfahrten: das Portal liegt höchstens entrance_tol von einem Punkt aus `entrances` (Endpunkte von
    Oberflächenstraßen) - ein Kettenende an einer Verzweigung im Berg (z.B. Festungsstollen) bekommt keine Sperre.

    Returns:
        [{"name", "xy", "rotation_matrix"}, ...] - Höhe setzt der Aufrufer aus dem Gelände. rotation_matrix dreht
        die lokale Vorwärtsachse (+y) auf die Tunnelachse, die Längsachse der Barriere (lokal +x) liegt damit quer.
    """
    blocks = []
    for plan in plans:
        coords = plan["coords"]
        outside = (outside_map(coords[0], bounds, edge_margin), outside_map(coords[-1], bounds, edge_margin))
        if outside[0] == outside[1]:
            continue
        portal = plan["portals"][1] if outside[0] else plan["portals"][0]
        px, py = portal["xy"]
        if not any(math.hypot(ex - px, ey - py) <= entrance_tol for ex, ey in entrances):
            continue
        ux, uy = portal["axis"]  # ins Tunnelinnere
        across = (uy, -ux)  # rechts in Blickrichtung ins Tunnelinnere
        road_width = plan["tube_width"] - width_margin
        count = max(2, math.ceil((road_width + 2.0 * side_margin) / spacing))
        cx, cy = px - ux * distance, py - uy * distance
        # BeamNG: Zeilen = Bilder der lokalen Achsen (siehe ItemManager._heading_rotation_matrix()) - Zeile 0 = quer
        # (Längsachse der Barriere), Zeile 1 = Tunnelachse, Zeile 2 = oben
        rotation = [across[0], across[1], 0.0, ux, uy, 0.0, 0.0, 0.0, 1.0]
        for i in range(count):
            offset = (i - (count - 1) / 2.0) * spacing
            blocks.append(
                {
                    "name": f"roadblock_{plan['id']}_{portal['label']}_{i}",
                    "xy": (cx + across[0] * offset, cy + across[1] * offset),
                    "rotation_matrix": list(rotation),
                }
            )
    return blocks
