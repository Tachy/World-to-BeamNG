"""
Tunnelröhren abdunkeln: BeamNG beleuchtet die Röhre sonst mit Umgebungs- und Himmelslicht, als stünde sie im Freien.
Die eigenen Levels (west_coast_usa, Utah, italy, ...) legen dafür gedrehte Quader vom Typ `Zone` entlang des Tunnels
(`useAmbientLightColor`, `ambientLightColor` schwarz, `skyLightFactor` 0.05) - genau das passiert hier je Tunnel-Plan
(tunnel_portal.plan_tunnels()). Galerien bleiben hell (talseitig offen).

Wie im Vanilla-Tunnel (jungle_rock_island): alle Zonen einer Röhre teilen sich eine `zoneGroup` (ein zusammenhängender
Innenraum), und an beiden Enden sitzt ein `Portal`-Objekt - die Öffnung zwischen Innenraum und Außenwelt. Ohne beides
(erster Versuch 2026-09-24) flackerte die Helligkeit an den Zonengrenzen, dunkel wurde es nie.
"""

from typing import Dict, List, Sequence

import numpy as np

ZONE_FIELDS = {"useAmbientLightColor": True, "ambientLightColor": [0, 0, 0, 1], "skyLightFactor": 0.05}


def _points_between(coords: np.ndarray, start: float, end: float) -> np.ndarray:
    """Polylinienpunkte (x, y, z) von Bogenlänge `start` bis `end` (Endpunkte interpoliert)."""
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))])

    def at(s):
        return np.array([np.interp(s, cum, coords[:, k]) for k in range(3)])

    inner = coords[(cum > start) & (cum < end)]
    return np.vstack([at(start), inner, at(end)])


def _lateral_deviation(points: np.ndarray, a: int, b: int) -> float:
    """Größter seitlicher Abstand der Punkte zwischen a und b von der Sehne a-b (Grundriss)."""
    if b - a < 2:
        return 0.0
    chord = points[b, :2] - points[a, :2]
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return 0.0
    rel = points[a + 1 : b, :2] - points[a, :2]
    return float(np.max(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / length))


def _segments(points: np.ndarray, max_length: float, max_deviation: float) -> List[tuple]:
    """Gierig möglichst lange Abschnitte (Punktindizes), je höchstens max_length lang und max_deviation seitlich
    von ihrer Sehne entfernt."""
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])))])
    segments, start = [], 0
    while start < len(points) - 1:
        end = start + 1
        while end + 1 < len(points):
            candidate = end + 1
            if cum[candidate] - cum[start] > max_length or _lateral_deviation(points, start, candidate) > max_deviation:
                break
            end = candidate
        segments.append((start, end))
        start = end
    return segments


def _rotation_matrix(forward: np.ndarray) -> List[float]:
    """rotationMatrix, deren ZEILEN die Bilder der lokalen Achsen sind (BeamNG-Konvention, siehe
    ItemManager._heading_rotation_matrix()): x = forward (entlang der Röhre, samt Steigung), y = waagerecht quer,
    z = senkrecht dazu (nicht gekippt)."""
    x_axis = forward / np.linalg.norm(forward)
    y_axis = np.cross([0.0, 0.0, 1.0], x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return [float(v) for v in np.vstack([x_axis, y_axis, z_axis]).reshape(-1)]


def _portal_rotation_matrix(forward: np.ndarray) -> List[float]:
    """rotationMatrix eines Portals nach Vanilla-Konvention: lokale y-Achse entlang des Tunnels (waagerecht), x quer,
    z senkrecht."""
    y_axis = np.array([forward[0], forward[1], 0.0])
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    return [float(v) for v in np.vstack([x_axis, y_axis, z_axis]).reshape(-1)]  # Zeilen = lokale Achsen


def plan_tunnel_zones(
    plans: Sequence[Dict],
    max_length: float,
    max_deviation: float,
    end_overlap: float,
    width_margin: float,
    height_margin: float,
    portal_inset: float,
    portal_depth: float,
) -> List[Dict]:
    """
    Zone-Quader je Tunnel-Plan: die Röhre von portal_inset hinter jedem Portal bis zum anderen in Abschnitte
    (höchstens max_length lang, Achse höchstens max_deviation von der Röhrenachse), je Abschnitt ein Quader:
    Länge + end_overlap je Seite, Breite = Röhrenbreite + width_margin, Höhe = Scheitel + height_margin (je zur
    Hälfte unter dem Boden und über dem Scheitel), entlang der Achse gedreht und mit der Steigung geneigt.

    Dazu je Röhre eine gemeinsame zoneGroup und an beiden Enden (Stirnfläche der Zonenkette) ein Portal:
    Breite/Höhe wie die Zonen, portal_depth tief.

    Returns:
        [{"class" ("Zone" | "Portal"), "name", "position" (x, y, z), "rotation_matrix", "scale", "fields"}, ...]
    """
    zones = []
    group = 0
    for plan in plans:
        coords = np.asarray(plan["coords"], dtype=float)
        total = float(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])).sum())
        if total <= 2.0 * portal_inset:
            continue
        points = _points_between(coords, portal_inset, total - portal_inset)
        group += 1
        width, height = plan["tube_width"] + width_margin, plan["crown"] + height_margin
        for label, at, forward in (("start", points[0], points[1] - points[0]), ("end", points[-1], points[-1] - points[-2])):
            if np.hypot(forward[0], forward[1]) < 1e-9:
                continue
            zones.append(
                {
                    "class": "Portal",
                    "name": f"tunnel_zone_portal_{plan['id']}_{label}",
                    "position": (float(at[0]), float(at[1]), float(at[2] + plan["crown"] / 2.0)),
                    "rotation_matrix": _portal_rotation_matrix(forward),
                    "scale": [width, portal_depth, height],
                    "fields": {},
                }
            )
        for index, (a, b) in enumerate(_segments(points, max_length, max_deviation)):
            forward = points[b] - points[a]
            length = float(np.linalg.norm(forward))
            if length < 1e-6:
                continue
            center = (points[a] + points[b]) / 2.0
            zones.append(
                {
                    "class": "Zone",
                    "name": f"tunnel_zone_{plan['id']}_{index}",
                    "position": (float(center[0]), float(center[1]), float(center[2] + plan["crown"] / 2.0)),
                    "rotation_matrix": _rotation_matrix(forward),
                    "scale": [length + 2.0 * end_overlap, width, height],
                    "fields": {**ZONE_FIELDS, "zoneGroup": group},
                }
            )
    return zones
