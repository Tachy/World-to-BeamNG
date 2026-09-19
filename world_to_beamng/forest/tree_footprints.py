"""
Stamm-genaue Prüfung der Bäume.

Die Ausschlusszonen (Fahrbahnen, Gebäude) und die Höhe aus der Heightmap gelten für den URSPRUNG eines
Baum-Assets. Gruppen-Assets (`*_group`) bestehen aber aus mehreren Stämmen, die bis ~9 m neben dem Ursprung
stehen: Stämme landen auf Wegen und hängen an Hängen frei in der Luft, obwohl der Ursprung passt.

Hier werden die Stammfüße aus dem Kollisionsmodell (Colmesh) des Assets gelesen und jede Instanz mit ihren
tatsächlichen Stammpositionen gegen dieselben Ausschlusszonen und den Boden geprüft - ohne die Abstände zu
ändern. Was nicht passt, wird bis zu `max_sink` abgesenkt, sonst bekommt derselbe Punkt einen anderen Typ
aus dem Pool des Waldes (z.B. einen Einzelstamm) oder entfällt.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import shapely

logger = logging.getLogger(__name__)

# Stammfuß: tiefster Punkt eines Stamms im Kollisionsmodell. Höher liegende Stücke (Stammspitzen, Äste) zählen nicht.
FOOT_MAX_Z = 1.0
FOOT_CELL = 1.0  # Kollisions-Vertices werden in Zellen dieser Größe je Stamm zusammengefasst
FOOT_MERGE_DISTANCE = 0.8  # Füße näher als das (Teilstücke desselben Stamms) zählen einmal

_ORIGIN_FOOT = np.zeros((1, 3))
_GEOMETRY = re.compile(r'<geometry id="([^"]*)"[^>]*>(.*?)</geometry>', re.S)
_POSITIONS = re.compile(r'<float_array[^>]*id="[^"]*positions[^"]*"[^>]*>([^<]*)<')
_INSTANCE = re.compile(r'<instance_geometry url="#([^"]*)" name="([^"]*)"')


def read_trunk_feet(dae_path) -> np.ndarray:
    """
    Stammfüße (Modellkoordinaten, (K, 3)) aus dem Kollisionsmodell einer .dae.

    Ohne lesbares Colmesh (oder ohne Punkte nahe dem Boden) bleibt ein einzelner Fuß im Ursprung.
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

    # Je Zelle der tiefste Vertex = Fuß des Stamms in dieser Zelle
    lowest = {}
    for point in points:
        key = (int(np.floor(point[0] / FOOT_CELL)), int(np.floor(point[1] / FOOT_CELL)))
        if key not in lowest or point[2] < lowest[key][2]:
            lowest[key] = point
    candidates = sorted((p for p in lowest.values() if p[2] < FOOT_MAX_Z), key=lambda p: p[2])

    feet: List[np.ndarray] = []
    for point in candidates:  # tiefste zuerst: Duplikate desselben Stamms fallen weg
        if all(np.hypot(point[0] - f[0], point[1] - f[1]) >= FOOT_MERGE_DISTANCE for f in feet):
            feet.append(point)
    return np.array(feet) if feet else _ORIGIN_FOOT.copy()


def load_trunk_feet(registered_trees: Dict[str, Dict], root) -> Dict[str, np.ndarray]:
    """
    Stammfüße aller registrierten Baumtypen.

    Args:
        registered_trees: Typname -> {"dae_path": "levels/<level>/art/shapes/trees/....dae", ...}
        root: Verzeichnis, relativ zu dem dae_path aufgelöst wird (BeamNG-Benutzerordner "current")
    """
    feet = {}
    for name, info in registered_trees.items():
        dae_path = info.get("dae_path")
        feet[name] = read_trunk_feet(Path(root) / dae_path) if dae_path else _ORIGIN_FOOT.copy()
    return feet


class TrunkFitter:
    """Prüft Baum-Instanzen mit ihren tatsächlichen Stammpositionen gegen Ausschlusszone und Boden."""

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
            feet_by_type: Typname -> Stammfüße (K, 3) in Modellkoordinaten (siehe load_trunk_feet)
            exclusion: shapely-Geometrie, in der kein Stamm stehen darf (dieselbe wie für die Ursprünge)
            row_exclusion: dasselbe für Baumreihen (kleinere Abstände)
            height_at: Höhenabfrage der fertigen Heightmap (x, y) -> z; ohne sie entfällt die Bodenprüfung
            max_float: so weit darf ein Stammfuß nach dem Absenken über dem Boden stehen (m)
            max_sink: so weit darf ein Baum höchstens abgesenkt werden (m); darüber wird der Typ gewechselt
            max_rounds: Anzahl Typ-Neuwürfe, bevor eine Instanz entfällt
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
        Gibt die Instanzen zurück, deren Stämme passen; Instanzen, die passen, bleiben unverändert.

        Args:
            instances: Instanzen im forest4-Format (type, pos, rotationMatrix, scale)
            pool: Typname -> Gewicht der Bäume dieses Waldes (Ersatztypen kommen daraus)
            row: Baumreihe - prüft gegen die Reihen-Ausschlusszone
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
            logger.debug(f"  [Trunk] {len(pending)} Instanzen ohne passenden Typ verworfen")

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
        """Je Instanz: passt sie (Stämme frei und am Boden) und um wie viel muss sie abgesenkt werden."""
        fits = np.ones(len(types), dtype=bool)
        sink = np.zeros(len(types))
        for name in np.unique(types):
            sel = np.flatnonzero(types == name)
            feet = self.feet_by_type.get(name, _ORIGIN_FOOT)
            # BeamNG liest die Modellachsen als Zeilen der Matrix: Welt-Offset = M^T * Fuß
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
