"""
Trunk-accurate checking of trees.

The exclusion zones (carriageways, buildings) and the height from the heightmap apply to the ORIGIN of a
tree asset. Group assets (`*_group`), however, consist of several trunks that stand up to ~9 m beside the origin:
trunks end up on paths and hang freely in the air on slopes even though the origin fits.

Here the trunk feet are read from the asset's collision model (colmesh) and each instance is checked with its
actual trunk positions against the same exclusion zones and the ground - without changing the spacing. What
does not fit is lowered by up to `max_sink`, otherwise the same point gets a different type
from the forest's pool (e.g. a single trunk) or is dropped.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import shapely

logger = logging.getLogger(__name__)

# Trunk foot: lowest point of a trunk in the collision model. Higher pieces (trunk tips, branches) do not count.
FOOT_MAX_Z = 1.0
FOOT_CELL = 1.0  # collision vertices are grouped into cells of this size per trunk
FOOT_MERGE_DISTANCE = 0.8  # feet closer than this (parts of the same trunk) count once

_ORIGIN_FOOT = np.zeros((1, 3))
_GEOMETRY = re.compile(r'<geometry id="([^"]*)"[^>]*>(.*?)</geometry>', re.S)
_POSITIONS = re.compile(r'<float_array[^>]*id="[^"]*positions[^"]*"[^>]*>([^<]*)<')
_INSTANCE = re.compile(r'<instance_geometry url="#([^"]*)" name="([^"]*)"')


def read_trunk_feet(dae_path) -> np.ndarray:
    """
    Trunk feet (model coordinates, (K, 3)) from the collision model of a .dae.

    Without a readable colmesh (or without points near the ground) a single foot at the origin remains.
    """
    try:
        text = Path(dae_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return _ORIGIN_FOOT.copy()

    geometries = {}
    for geometry_id, body in _GEOMETRY.findall(text):
        match = _POSITIONS.search(body)
        if match:
            geometries[geometry_id] = np.array(match.group(1).split(), dtype=float).reshape(-1, 3)

    points = None
    for url, name in _INSTANCE.findall(text):
        if name.lower().startswith("colmesh") and url in geometries:
            points = geometries[url]
            break
    if points is None or len(points) == 0:
        return _ORIGIN_FOOT.copy()

    # Per cell the lowest vertex = foot of the trunk in this cell
    lowest = {}
    for point in points:
        key = (int(np.floor(point[0] / FOOT_CELL)), int(np.floor(point[1] / FOOT_CELL)))
        if key not in lowest or point[2] < lowest[key][2]:
            lowest[key] = point
    candidates = sorted((p for p in lowest.values() if p[2] < FOOT_MAX_Z), key=lambda p: p[2])

    feet: List[np.ndarray] = []
    for point in candidates:  # lowest first: duplicates of the same trunk are dropped
        if all(np.hypot(point[0] - f[0], point[1] - f[1]) >= FOOT_MERGE_DISTANCE for f in feet):
            feet.append(point)
    return np.array(feet) if feet else _ORIGIN_FOOT.copy()


def load_trunk_feet(registered_trees: Dict[str, Dict], root) -> Dict[str, np.ndarray]:
    """
    Trunk feet of all registered tree types.

    Args:
        registered_trees: type name -> {"dae_path": "levels/<level>/art/shapes/trees/....dae", ...}
        root: directory relative to which dae_path is resolved (BeamNG user folder "current")
    """
    feet = {}
    for name, info in registered_trees.items():
        dae_path = info.get("dae_path")
        feet[name] = read_trunk_feet(Path(root) / dae_path) if dae_path else _ORIGIN_FOOT.copy()
    return feet


class TrunkFitter:
    """Checks tree instances with their actual trunk positions against the exclusion zone and the ground."""

    def __init__(
        self,
        feet_by_type: Dict[str, np.ndarray],
        exclusion=None,
        row_exclusion=None,
        height_at=None,
        max_float: float = 0.5,
        max_sink: float = 1.0,
        max_rounds: int = 8,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            feet_by_type: type name -> trunk feet (K, 3) in model coordinates (see load_trunk_feet)
            exclusion: shapely geometry in which no trunk may stand (the same as for the origins)
            row_exclusion: the same for tree rows (smaller distances)
            height_at: height query of the finished heightmap (x, y) -> z; without it the ground check is skipped
            max_float: how far a trunk foot may stand above the ground after lowering (m)
            max_sink: how far a tree may be lowered at most (m); beyond that the type is changed
            max_rounds: number of type re-rolls before an instance is dropped
        """
        self.feet_by_type = feet_by_type
        self.exclusion = exclusion
        self.row_exclusion = row_exclusion
        self.height_at = height_at
        self.max_float = max_float
        self.max_sink = max_sink
        self.max_rounds = max_rounds
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, instances: List[Dict], pool: Dict[str, float], row: bool = False) -> List[Dict]:
        """
        Returns the instances whose trunks fit; instances that fit remain unchanged.

        Args:
            instances: instances in forest4 format (type, pos, rotationMatrix, scale)
            pool: type name -> weight of this forest's trees (replacement types are drawn from it)
            row: tree row - checks against the row exclusion zone
        """
        if not instances:
            return []
        zone = self.row_exclusion if row else self.exclusion

        types = np.array([inst["type"] for inst in instances], dtype=object)
        pos = np.array([inst["pos"] for inst in instances], dtype=float)
        matrices = np.array([inst["rotationMatrix"] for inst in instances], dtype=float).reshape(-1, 3, 3)
        scales = np.array([inst["scale"] for inst in instances], dtype=float)

        names = [name for name in pool if name in self.feet_by_type] or list(pool)
        weights = np.array([pool[name] for name in names], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(names), 1.0 / len(names))

        original_types = types.copy()
        sinks = np.zeros(len(instances))
        keep = np.zeros(len(instances), dtype=bool)
        pending = np.arange(len(instances))
        for round_no in range(self.max_rounds + 1):
            fits, sink = self._evaluate(types[pending], pos[pending], matrices[pending], scales[pending], zone)
            keep[pending[fits]] = True
            sinks[pending[fits]] = sink[fits]
            pending = pending[~fits]
            if len(pending) == 0 or round_no == self.max_rounds:
                break
            types[pending] = self.rng.choice(names, size=len(pending), p=weights)

        if len(pending):
            logger.debug(f"  [Trunk] {len(pending)} instances without a fitting type discarded")

        result = []
        for i in np.flatnonzero(keep):
            instance = instances[i]
            if types[i] == original_types[i] and sinks[i] == 0.0:
                result.append(instance)
            else:
                result.append(
                    {**instance, "type": str(types[i]), "pos": [instance["pos"][0], instance["pos"][1], float(pos[i, 2] - sinks[i])]}
                )
        return result

    def _evaluate(self, types, pos, matrices, scales, zone):
        """Per instance: does it fit (trunks clear and on the ground) and by how much must it be lowered."""
        fits = np.ones(len(types), dtype=bool)
        sink = np.zeros(len(types))
        for name in np.unique(types):
            sel = np.flatnonzero(types == name)
            feet = self.feet_by_type.get(name, _ORIGIN_FOOT)
            # BeamNG reads the model axes as rows of the matrix: world offset = M^T * foot
            offsets = np.einsum("nji,bj->nbi", matrices[sel], feet) * scales[sel][:, None, None]
            x = pos[sel, 0][:, None] + offsets[:, :, 0]
            y = pos[sel, 1][:, None] + offsets[:, :, 1]
            z = pos[sel, 2][:, None] + offsets[:, :, 2]

            if zone is not None:
                blocked = shapely.intersects_xy(zone, x.ravel(), y.ravel()).reshape(x.shape).any(axis=1)
                fits[sel[blocked]] = False
            if self.height_at is not None:
                excess = (z - np.asarray(self.height_at(x, y))).max(axis=1) - self.max_float
                needed = np.clip(excess, 0.0, None)
                too_steep = needed > self.max_sink
                fits[sel[too_steep]] = False
                sink[sel] = np.where(too_steep, 0.0, needed)
        return fits, sink
