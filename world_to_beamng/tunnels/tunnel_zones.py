"""
Darken tunnel tubes: otherwise BeamNG lights the tube with ambient and sky light as if it stood in the open.
The stock levels (west_coast_usa, Utah, italy, ...) place rotated boxes of type `Zone` along the tunnel for this
(`useAmbientLightColor`, `ambientLightColor` black, `skyLightFactor` 0.05) - exactly what happens here for each tunnel
plan (tunnel_portal.plan_tunnels()). Galleries stay bright (open on the valley side).

As in the vanilla tunnel (jungle_rock_island): all zones of a tube share one `zoneGroup` (one contiguous interior),
and a `Portal` object sits at both ends - the opening between interior and outside world. Without both
(first attempt 2026-09-24) the brightness flickered at the zone borders and it never got dark.
"""

from typing import Dict, List, Sequence

import numpy as np

ZONE_FIELDS = {"useAmbientLightColor": True, "ambientLightColor": [0, 0, 0, 1], "skyLightFactor": 0.05}


def _points_between(coords: np.ndarray, start: float, end: float) -> np.ndarray:
    """Polyline points (x, y, z) from arc length `start` to `end` (end points interpolated)."""
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))])

    def at(s):
        return np.array([np.interp(s, cum, coords[:, k]) for k in range(3)])

    inner = coords[(cum > start) & (cum < end)]
    return np.vstack([at(start), inner, at(end)])


def _lateral_deviation(points: np.ndarray, a: int, b: int) -> float:
    """Largest lateral distance of the points between a and b from the chord a-b (plan view)."""
    if b - a < 2:
        return 0.0
    chord = points[b, :2] - points[a, :2]
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return 0.0
    rel = points[a + 1 : b, :2] - points[a, :2]
    return float(np.max(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / length))


def _segments(points: np.ndarray, max_length: float, max_deviation: float) -> List[tuple]:
    """Greedily the longest possible sections (point indices), each at most max_length long and at most
    max_deviation lateral distance from its chord."""
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
    """rotationMatrix whose ROWS are the images of the local axes (BeamNG convention, see
    ItemManager._heading_rotation_matrix()): x = forward (along the tube, including gradient), y = horizontal across,
    z = perpendicular to both (not tilted)."""
    x_axis = forward / np.linalg.norm(forward)
    y_axis = np.cross([0.0, 0.0, 1.0], x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return [float(v) for v in np.vstack([x_axis, y_axis, z_axis]).reshape(-1)]


def _portal_rotation_matrix(forward: np.ndarray) -> List[float]:
    """rotationMatrix of a portal following the vanilla convention: local y axis along the tunnel (horizontal), x across,
    z vertical."""
    y_axis = np.array([forward[0], forward[1], 0.0])
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    return [float(v) for v in np.vstack([x_axis, y_axis, z_axis]).reshape(-1)]  # rows = local axes


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
    Zone boxes per tunnel plan: the tube from portal_inset behind each portal to the other one is split into sections
    (at most max_length long, axis at most max_deviation from the tube axis), one box per section:
    length + end_overlap on each side, width = tube width + width_margin, height = crown + height_margin (half of it
    below the floor and half above the crown), rotated along the axis and tilted with the gradient.

    In addition, one shared zoneGroup per tube and a portal at both ends (end face of the zone chain):
    width/height like the zones, portal_depth deep.

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
