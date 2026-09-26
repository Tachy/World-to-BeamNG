"""
SpotLight fixtures inside tunnel tubes: without them the darkness zones (tunnel_zones.py) leave the interior pitch
black except for the vehicle's own headlights. Pattern and preset copied 1:1 from BeamNG's own italy.zip
(tunnelLight_*: SpotLight, mounted near the ceiling, aimed straight down, ~11 m spacing, castShadows=true).

rotationMatrix convention (rows = local axes, as elsewhere in this project): the beam direction is the local Y axis -
verified against italy.zip's tunnelLight sample, where row 1 (Y) equals world [0, 0, -1] (straight down) and row 2 (Z)
equals cross(row0, row1) exactly for a light aligned along the tunnel's local x axis.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column

DOWN = np.array([0.0, 0.0, -1.0])


def _rotation_matrix(forward_xy: np.ndarray) -> List[float]:
    """row0 (x) = horizontal tunnel direction, row1 (y) = straight down (the beam direction), row2 (z) = cross(x, y)."""
    x_axis = np.array([forward_xy[0], forward_xy[1], 0.0])
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, DOWN)
    return [float(v) for v in np.concatenate([x_axis, DOWN, z_axis])]


def plan_tunnel_lights(
    plans: Sequence[Dict], spacing: float, start_inset: float, ceiling_margin: float, fields: Dict
) -> List[Dict]:
    """
    SpotLight fixtures along every tunnel plan (tunnel_portal.plan_tunnels(), after terrain shaping has set each
    portal's "open" flag - see terrain/tunnel_terrain.py::shape_terrain_for_tunnels()).

    Args:
        spacing: distance between fixtures along the tube axis, in meters
        start_inset: kept clear at an OPEN portal end, so the fixture does not clip into the collar/tube-shell
            structure there; a closed end (buried in the mountain, no portal structure) gets no inset
        ceiling_margin: fixture mounted this far below the crown apex, in meters
        fields: BeamNG SpotLight properties, copied as-is onto every fixture (brightness, color, innerAngle,
            outerAngle, intensity, range, castShadows, useColorTemperature, ...)

    Returns:
        [{"class": "SpotLight", "name", "position", "rotation_matrix", "fields"}, ...]
    """
    lights = []
    for plan in plans:
        coords = np.asarray(plan["coords"], dtype=float)
        xy, z = coords[:, :2], coords[:, 2]
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        total = float(cum[-1])
        start, end = plan["portals"]
        start_gap = start_inset if start.get("open", True) else 0.0
        end_gap = start_inset if end.get("open", True) else 0.0
        if total <= start_gap + end_gap:
            continue

        positions = np.arange(start_gap, total - end_gap + 1e-9, spacing)
        light_xy = np.column_stack([np.interp(positions, cum, xy[:, 0]), np.interp(positions, cum, xy[:, 1])])
        light_z = np.interp(positions, cum, z) + plan["crown"] - ceiling_margin
        directions = np.column_stack([np.gradient(light_xy[:, 0], positions), np.gradient(light_xy[:, 1], positions)]) \
            if len(positions) > 1 else np.array([xy[-1] - xy[0]])

        for index, (pos_xy, pos_z, forward) in enumerate(zip(light_xy, light_z, directions)):
            lights.append({
                "class": "SpotLight",
                "name": f"tunnel_light_{plan['id']}_{index}",
                "position": (float(pos_xy[0]), float(pos_xy[1]), float(pos_z)),
                "rotation_matrix": _rotation_matrix(forward),
                "fields": dict(fields),
            })
    return lights


def build_lamp_mesh(lights: Sequence[Dict], material: str, length: float, width: float, height: float, ceiling_margin: float) -> Optional[Dict]:
    """
    Mesh of the visible lamp bodies for the SpotLights of plan_tunnel_lights(): one flat box per light hanging directly
    under the crown (`ceiling_margin` above the light, 1 cm clear of the tube surface), `length` along the tunnel, `width`
    across it. `material` is the emissive lamp material, so the fixtures glow. None without lights.
    """
    if not lights:
        return None
    builder = MeshBuilder()
    for light in lights:
        cx, cy, light_z = light["position"]
        direction = light["rotation_matrix"][:2]  # local x axis = horizontal direction of the tunnel
        top = light_z + ceiling_margin - 0.01
        bottom = top - height
        add_box_column(builder, cx, cy, bottom, top, length, 1.0, direction=direction, across=width)
        half, half_across = length / 2.0, width / 2.0
        dx, dy = direction[0] / np.hypot(*direction), direction[1] / np.hypot(*direction)
        corners = [
            [cx - dx * half + dy * half_across, cy - dy * half - dx * half_across, bottom],
            [cx + dx * half + dy * half_across, cy + dy * half - dx * half_across, bottom],
            [cx + dx * half - dy * half_across, cy + dy * half + dx * half_across, bottom],
            [cx - dx * half - dy * half_across, cy - dy * half + dx * half_across, bottom],
        ]
        builder.quad(corners, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [0.0, 0.0, -1.0])
    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": {material: builder.faces},
    }
