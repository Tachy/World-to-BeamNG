"""
Debug Network Exporter - Sammelt und exportiert Debug-Visualisierungsdaten.

Singleton-Pattern zum Sammeln von Debug-Daten während des Workflows:
- Junctions (Positionen, Verbindungen)
- Roads (Centerlines, Metadaten)
- Boundary-Polygone (Stitching-Visualisierung)
- Component-Linien (Connected Components aus Stitching)
- Universelle Primitiven (Labels, Kreise, Polygone, Linien, Punkte, Pfeile, Vektoren)
- Grid-Farben (für Viewer)

Usage:
    exporter = DebugNetworkExporter.get_instance()

    # Universelle Primitiven:
    exporter.add_label("Debug Label", position=(100, 100, 0))
    exporter.add_circle(50, center=(100, 100, 0))
    exporter.add_polygon([(0, 0, 0), (10, 0, 0), (10, 10, 0)])
    exporter.add_line([(0, 0, 0), (100, 100, 0)])
    exporter.add_point((100, 100, 0))
    exporter.add_arrow((0, 0, 0), (100, 0, 0))
    exporter.add_vector((50, 50, 0), (1, 0, 0), scale=20)

    exporter.export(cache_dir)
"""

import json
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Utility-Funktionen für Koordinaten-Handling
# ============================================================================


def _normalize_coordinate(coord: Union[Tuple, List, np.ndarray]) -> List[float]:
    """
    Konvertiere verschiedene Koordinaten-Formate zu Liste [x, y, z].

    Args:
        coord: Tuple, Liste oder NumPy-Array mit 3 Koordinaten

    Returns:
        [x, y, z] als Liste von floats
    """
    if hasattr(coord, "tolist"):
        coord = coord.tolist()
    elif not isinstance(coord, (list, tuple)):
        raise TypeError(f"Koordinate muss Tuple/List/Array sein, nicht {type(coord)}")

    if len(coord) != 3:
        raise ValueError(f"Koordinate muss 3 Komponenten haben, nicht {len(coord)}")

    return [float(c) for c in coord]


def _normalize_coordinates(coords: Union[List, np.ndarray]) -> List[List[float]]:
    """
    Konvertiere Liste von Koordinaten zu [[x, y, z], ...].

    Args:
        coords: Liste von Tuples/Listen/Arrays oder NumPy 2D-Array

    Returns:
        Liste von [x, y, z] Listen
    """
    if hasattr(coords, "tolist"):
        # NumPy Array
        coords = coords.tolist()

    if not isinstance(coords, (list, tuple)):
        raise TypeError(f"Koordinaten müssen Liste sein, nicht {type(coords)}")

    result = []
    for coord in coords:
        result.append(_normalize_coordinate(coord))

    return result


def _get_default_color(color_type: str = "standard") -> List[float]:
    """
    Gebe Standard-Farbe für einen Typ zurück.

    Args:
        color_type: "standard" (blau), "positive" (grün), "negative" (rot), etc.

    Returns:
        RGB-Farbe als [r, g, b] mit Werten 0.0-1.0
    """
    colors = {
        "standard": [0.0, 0.0, 1.0],  # Blau
        "positive": [0.2, 0.8, 0.2],  # Grün
        "negative": [1.0, 0.2, 0.2],  # Rot
        "warning": [1.0, 0.8, 0.0],  # Gelb
        "neutral": [0.7, 0.7, 0.7],  # Grau
        "highlight": [1.0, 0.0, 1.0],  # Magenta
        "outline": [0.0, 0.0, 0.0],  # Schwarz
    }
    return colors.get(color_type, colors["standard"])


class DebugNetworkExporter:
    """Sammelt Debug-Daten für Visualisierung im DAE Viewer (Singleton)."""

    _instance: Optional["DebugNetworkExporter"] = None

    def __init__(self):
        """Private Constructor - verwende get_instance() stattdessen."""
        if DebugNetworkExporter._instance is not None:
            raise RuntimeError("DebugNetworkExporter ist ein Singleton - verwende get_instance()")

        self.primitives: List[Dict[str, Any]] = []  # Labels, Circles, Polygons, Lines, etc.

        # Grid-Farben für Viewer (standardmäßig)
        self.grid_colors = self._get_default_grid_colors()

    @staticmethod
    def _get_default_grid_colors() -> Dict[str, Any]:
        """Gebe Standard Grid-Farben zurück."""
        return {
            "terrain": {
                "face": [0.8, 0.95, 0.8],
                "edge": [0.2, 0.5, 0.2],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "road": {
                "face": [1.0, 1.0, 1.0],
                "edge": [1.0, 0.0, 0.0],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "building_wall": {
                "face": [0.95, 0.95, 0.95],
                "edge": [0.3, 0.3, 0.3],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "building_roof": {
                "face": [0.6, 0.2, 0.1],
                "edge": [0.3, 0.1, 0.05],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "junction": {
                "color": [0.0, 0.0, 1.0],
                "opacity": 0.5,
            },
            "centerline": {
                "color": [0.0, 0.0, 1.0],
                "line_width": 2.0,
                "opacity": 1.0,
            },
            "boundary": {
                "color": [1.0, 0.0, 1.0],
                "line_width": 2.0,
                "opacity": 1.0,
            },
            "component_terrain": {
                "color": [0.2, 0.8, 0.2],
                "line_width": 3.0,
                "opacity": 1.0,
            },
            "component_road": {
                "color": [0.8, 0.2, 0.2],
                "line_width": 3.0,
                "opacity": 1.0,
            },
        }

    @classmethod
    def get_instance(cls) -> "DebugNetworkExporter":
        """Hole die Singleton-Instanz (erstellt sie bei Bedarf)."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.primitives = []
            cls._instance.grid_colors = cls._get_default_grid_colors()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Setze Singleton-Instanz zurück (für neuen Export-Lauf)."""
        cls._instance = None

    # ========================================================================
    # UNIVERSELLE PRIMITIVEN - Das Herz des neuen Systems
    # ========================================================================

    def add_label(
        self,
        text: str,
        position: Union[Tuple, List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        size: float = 12.0,
    ) -> None:
        """
        Füge ein Text-Label hinzu.

        Args:
            text: Der anzuzeigende Text
            position: (x, y, z) Koordinate der Label-Position
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Blau
            size: Schriftgröße in Pixeln
        """
        if color is None:
            color = _get_default_color("standard")

        primitive = {
            "type": "label",
            "text": str(text),
            "position": _normalize_coordinate(position),
            "color": list(color),
            "size": float(size),
            "opacity": 1.0,
        }
        self.primitives.append(primitive)

    def add_point(
        self,
        position: Union[Tuple, List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        size: float = 5.0,
    ) -> None:
        """
        Füge einen Punkt hinzu.

        Args:
            position: (x, y, z) Koordinate
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Blau
            size: Punkt-Größe in Pixeln
        """
        if color is None:
            color = _get_default_color("standard")

        primitive = {
            "type": "point",
            "position": _normalize_coordinate(position),
            "color": list(color),
            "size": float(size),
            "opacity": 1.0,
        }
        self.primitives.append(primitive)

    def add_line(
        self,
        coords: Union[List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        width: float = 2.0,
        label: Optional[str] = None,
    ) -> None:
        """
        Füge eine Linie hinzu.

        Args:
            coords: Liste von (x, y, z) Koordinaten
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Blau
            width: Linienbreite in Pixeln
        """
        if color is None:
            color = _get_default_color("standard")

        primitive = {
            "type": "line",
            "coords": _normalize_coordinates(coords),
            "color": list(color),
            "line_width": float(width),
            "opacity": 1.0,
        }
        if label:
            primitive["label"] = str(label)
        self.primitives.append(primitive)

    def add_polygon(
        self,
        polygon: Union[List, np.ndarray, Dict[str, Any]],
        color: Optional[Union[List, Tuple]] = None,
        filled: bool = False,
        outline_width: float = 2.0,
        label: Optional[str] = None,
    ) -> None:
        """
        Füge ein Polygon hinzu.

        Args:
            polygon: Entweder Liste von (x, y, z) oder Polygon-Dict mit Key "coords"
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Blau
            filled: Polygon füllen oder nur Outline?
            outline_width: Breite der Outline in Pixeln
            label: Optionales Label
        """
        if color is None:
            color = _get_default_color("standard")

        # Erkenne Dict-Eingabe mit "coords"
        coords_source = polygon
        if isinstance(polygon, dict):
            if "coords" not in polygon:
                raise ValueError("Polygon-Dict muss 'coords' enthalten")
            coords_source = polygon["coords"]

        normalized_coords = _normalize_coordinates(coords_source)

        if len(normalized_coords) < 3:
            raise ValueError(f"Polygon braucht mindestens 3 Punkte, hat {len(normalized_coords)}")

        primitive = {
            "type": "polygon",
            "coords": normalized_coords,
            "color": list(color),
            "filled": bool(filled),
            "outline_width": float(outline_width),
            "opacity": 0.5 if filled else 1.0,
        }
        if label:
            primitive["label"] = str(label)
        self.primitives.append(primitive)

    def add_circle(
        self,
        radius: float,
        center: Union[Tuple, List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        filled: bool = False,
        segments: int = 32,
    ) -> None:
        """
        Füge einen Kreis hinzu.

        Args:
            radius: Radius des Kreises
            center: (x, y, z) Mittelpunkt
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Grün
            filled: Kreis füllen oder nur Umriss?
            segments: Anzahl der Liniensegmente für Kreis-Approximation
        """
        if color is None:
            color = _get_default_color("positive")

        primitive = {
            "type": "circle",
            "center": _normalize_coordinate(center),
            "radius": float(radius),
            "color": list(color),
            "filled": bool(filled),
            "segments": int(segments),
            "opacity": 0.5 if filled else 1.0,
        }
        self.primitives.append(primitive)

    def add_arrow(
        self,
        start: Union[Tuple, List, np.ndarray],
        end: Union[Tuple, List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        width: float = 2.0,
        head_size: float = 0.1,
    ) -> None:
        """
        Füge einen Pfeil hinzu.

        Args:
            start: (x, y, z) Startpunkt
            end: (x, y, z) Endpunkt
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Rot
            width: Linienbreite in Pixeln
            head_size: Größe der Pfeilspitze relativ zur Länge (0.0-1.0)
        """
        if color is None:
            color = _get_default_color("negative")

        primitive = {
            "type": "arrow",
            "start": _normalize_coordinate(start),
            "end": _normalize_coordinate(end),
            "color": list(color),
            "line_width": float(width),
            "head_size": float(head_size),
            "opacity": 1.0,
        }
        self.primitives.append(primitive)

    def add_vector(
        self,
        origin: Union[Tuple, List, np.ndarray],
        direction: Union[Tuple, List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        scale: float = 1.0,
        width: float = 2.0,
        head_size: float = 0.1,
    ) -> None:
        """
        Füge einen Vektor (Pfeil mit Richtung) hinzu.

        Args:
            origin: (x, y, z) Startpunkt
            direction: (dx, dy, dz) Richtungsvektor (wird normalisiert und skaliert)
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Gelb
            scale: Längen-Skalierungsfaktor
            width: Linienbreite in Pixeln
            head_size: Größe der Pfeilspitze relativ zur Länge
        """
        if color is None:
            color = _get_default_color("warning")

        origin_norm = _normalize_coordinate(origin)
        direction_norm = _normalize_coordinate(direction)

        # Berechne Endpunkt aus Origin + skalierter Richtung
        end = [
            origin_norm[0] + direction_norm[0] * scale,
            origin_norm[1] + direction_norm[1] * scale,
            origin_norm[2] + direction_norm[2] * scale,
        ]

        primitive = {
            "type": "arrow",
            "start": origin_norm,
            "end": end,
            "color": list(color),
            "line_width": float(width),
            "head_size": float(head_size),
            "opacity": 1.0,
        }
        self.primitives.append(primitive)

    def add_sphere(
        self,
        center: Union[Tuple, List, np.ndarray],
        radius: float,
        color: Optional[Union[List, Tuple]] = None,
        wireframe: bool = False,
    ) -> None:
        """
        Füge eine Kugel hinzu.

        Args:
            center: (x, y, z) Mittelpunkt
            radius: Radius der Kugel
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Neutral
            wireframe: Als Drahtmodell darstellen?
        """
        if color is None:
            color = _get_default_color("neutral")

        primitive = {
            "type": "sphere",
            "center": _normalize_coordinate(center),
            "radius": float(radius),
            "color": list(color),
            "wireframe": bool(wireframe),
            "opacity": 0.6,
        }
        self.primitives.append(primitive)

    def add_box(
        self,
        min_point: Union[Tuple, List, np.ndarray],
        max_point: Union[Tuple, List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        wireframe: bool = False,
    ) -> None:
        """
        Füge eine Bounding-Box hinzu.

        Args:
            min_point: (x_min, y_min, z_min)
            max_point: (x_max, y_max, z_max)
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Highlight
            wireframe: Als Drahtmodell darstellen?
        """
        if color is None:
            color = _get_default_color("highlight")

        primitive = {
            "type": "box",
            "min": _normalize_coordinate(min_point),
            "max": _normalize_coordinate(max_point),
            "color": list(color),
            "wireframe": bool(wireframe),
            "opacity": 0.3,
        }
        self.primitives.append(primitive)

    def add_grid(
        self,
        origin: Union[Tuple, List, np.ndarray],
        width: float,
        height: float,
        spacing: float,
        color: Optional[Union[List, Tuple]] = None,
        normal: Union[Tuple, List, np.ndarray] = (0, 0, 1),
    ) -> None:
        """
        Füge ein Grid hinzu.

        Args:
            origin: (x, y, z) Gitter-Ursprung
            width: Breite des Gitters
            height: Höhe des Gitters
            spacing: Abstand zwischen Gitterlinien
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Grau
            normal: (nx, ny, nz) Normale der Gitterebene
        """
        if color is None:
            color = _get_default_color("neutral")

        primitive = {
            "type": "grid",
            "origin": _normalize_coordinate(origin),
            "width": float(width),
            "height": float(height),
            "spacing": float(spacing),
            "normal": _normalize_coordinate(normal),
            "color": list(color),
            "opacity": 0.5,
        }
        self.primitives.append(primitive)

    # ========================================================================
    # BATCH & UTILITY METHODEN
    # ========================================================================

    def merge(self, other: "DebugNetworkExporter") -> None:
        """
        Merge Daten aus einem anderen Exporter (für Multi-Tile Support).

        Args:
            other: Anderer DebugNetworkExporter
        """
        # Primitive direkt hinzufügen
        self.primitives.extend(other.primitives)

    def export(self, cache_dir: str, filename: str = "debug_network.json") -> None:
        """
        Exportiere gesammelte Debug-Daten in JSON-Datei.

        Args:
            cache_dir: Zielverzeichnis für Export
            filename: Dateiname (default: debug_network.json)
        """
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(cache_dir) / filename

        data = {
            "primitives": self.primitives,
            "grid_colors": self.grid_colors,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"  [Debug] Exportiert: {len(self.primitives)} Primitive")
        logger.debug(f"  [Debug] Datei: {output_path}")

    def clear(self) -> None:
        """Lösche alle gesammelten Daten."""
        self.primitives.clear()

    def __repr__(self) -> str:
        return f"DebugNetworkExporter(primitives={len(self.primitives)})"
