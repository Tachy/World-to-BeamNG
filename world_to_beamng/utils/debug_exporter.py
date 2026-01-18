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
    
    # Legacy-Methoden:
    exporter.add_road(road_data)
    exporter.add_junction(junction_data)
    exporter.add_component_line(coords, color, label)
    
    # Universelle Primitiven (neu):
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
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np


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
        "standard": [0.0, 0.0, 1.0],      # Blau
        "positive": [0.2, 0.8, 0.2],      # Grün
        "negative": [1.0, 0.2, 0.2],      # Rot
        "warning": [1.0, 0.8, 0.0],       # Gelb
        "neutral": [0.7, 0.7, 0.7],       # Grau
        "highlight": [1.0, 0.0, 1.0],     # Magenta
        "outline": [0.0, 0.0, 0.0],       # Schwarz
    }
    return colors.get(color_type, colors["standard"])


class DebugNetworkExporter:
    """Sammelt Debug-Daten für Visualisierung im DAE Viewer (Singleton)."""

    _instance: Optional["DebugNetworkExporter"] = None

    def __init__(self):
        """Private Constructor - verwende get_instance() stattdessen."""
        if DebugNetworkExporter._instance is not None:
            raise RuntimeError("DebugNetworkExporter ist ein Singleton - verwende get_instance()")

        self.roads: List[Dict[str, Any]] = []
        self.junctions: List[Dict[str, Any]] = []
        self.boundary_polygons: List[Dict[str, Any]] = []
        self.component_lines: List[Dict[str, Any]] = []
        self.primitives: List[Dict[str, Any]] = []  # Labels, Circles, Polygons, etc.

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
            cls._instance.roads = []
            cls._instance.junctions = []
            cls._instance.boundary_polygons = []
            cls._instance.component_lines = []
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
        self.primitives.append(primitive)

    def add_polygon(
        self,
        coords: Union[List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        filled: bool = False,
        outline_width: float = 2.0,
    ) -> None:
        """
        Füge ein Polygon hinzu.
        
        Args:
            coords: Liste von (x, y, z) Koordinaten (mindestens 3)
            color: RGB-Farbe [r, g, b] 0.0-1.0, default: Blau
            filled: Polygon füllen oder nur Outline?
            outline_width: Breite der Outline in Pixeln
        """
        if color is None:
            color = _get_default_color("standard")
        
        normalized_coords = _normalize_coordinates(coords)
        
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
    # LEGACY-METHODEN - Rückwärtskompatibilität
    # ========================================================================

    def add_road(self, road_data: Dict[str, Any]) -> None:
        """
        Füge eine Road zur Debug-Visualisierung hinzu.

        Args:
            road_data: Dict mit Road-Informationen:
                - road_id: Eindeutige Road-ID
                - coords: Liste von (x, y, z) Centerline-Punkten
                - num_points: Anzahl der Punkte
                - junction_start_id: Junction am Anfang (optional)
                - junction_end_id: Junction am Ende (optional)
                - junction_buffer_start: Buffer-Distanz am Start (optional)
                - junction_buffer_end: Buffer-Distanz am Ende (optional)
        """
        # Konvertiere NumPy Arrays zu Listen für JSON-Serialisierung
        road_copy = road_data.copy()
        if "coords" in road_copy:
            coords = road_copy["coords"]
            if hasattr(coords, "tolist"):  # NumPy Array
                road_copy["coords"] = coords.tolist()
            elif isinstance(coords, list) and len(coords) > 0:
                # Prüfe ob Elemente NumPy Arrays sind
                if hasattr(coords[0], "tolist"):
                    road_copy["coords"] = [c.tolist() if hasattr(c, "tolist") else list(c) for c in coords]

        # Füge Rendering-Eigenschaften hinzu
        road_copy.setdefault("color", [0.0, 0.0, 1.0])
        road_copy.setdefault("line_width", 2.0)
        road_copy.setdefault("opacity", 1.0)

        self.roads.append(road_copy)

    def add_junction(self, junction_data: Dict[str, Any]) -> None:
        """
        Füge eine Junction zur Debug-Visualisierung hinzu.

        Args:
            junction_data: Dict mit Junction-Informationen:
                - position: (x, y, z) Junction-Koordinaten
                - road_indices: Liste von Road-Indizes die hier verbunden sind
                - connection_types: Dict mapping road_idx -> ["start"/"end"]
        """
        # Konvertiere NumPy Arrays zu Listen
        junction_copy = junction_data.copy()
        if "position" in junction_copy:
            pos = junction_copy["position"]
            if hasattr(pos, "tolist"):
                junction_copy["position"] = pos.tolist()

        self.junctions.append(junction_copy)

    def add_boundary(self, boundary_data: Dict[str, Any]) -> None:
        """
        Füge ein Boundary-Polygon zur Debug-Visualisierung hinzu.

        Args:
            boundary_data: Dict mit Boundary-Informationen:
                - type: "boundary_polygon" oder "search_circle"
                - coords: Liste von (x, y, z) Polygon-Punkten
                - centerline_point: (x, y, z) Centerline-Sample-Punkt
                - search_radius: Suchradius in Metern
        """
        # Konvertiere NumPy Arrays zu Listen
        boundary_copy = boundary_data.copy()
        if "coords" in boundary_copy:
            coords = boundary_copy["coords"]
            if hasattr(coords, "tolist"):
                boundary_copy["coords"] = coords.tolist()
            elif isinstance(coords, list) and len(coords) > 0:
                if hasattr(coords[0], "tolist"):
                    boundary_copy["coords"] = [c.tolist() if hasattr(c, "tolist") else list(c) for c in coords]

        if "centerline_point" in boundary_copy:
            pt = boundary_copy["centerline_point"]
            if hasattr(pt, "tolist"):
                boundary_copy["centerline_point"] = pt.tolist()

        self.boundary_polygons.append(boundary_copy)

    def add_component_line(self, coords, color=None, label=None, line_width=3.0):
        """
        Füge eine Connected Component Linie hinzu (z.B. Terrain-Kante oder Straßen-Kante).

        Args:
            coords: Liste von (x, y, z) Koordinaten
            color: RGB-Farbe [r, g, b] (0.0-1.0), default: grün
            label: Optionales Label (z.B. "terrain", "road")
            line_width: Linienbreite in Pixeln
        """
        # Konvertiere NumPy Arrays zu Listen
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
        elif isinstance(coords, list) and len(coords) > 0:
            if hasattr(coords[0], "tolist"):
                coords = [c.tolist() if hasattr(c, "tolist") else list(c) for c in coords]

        # Default-Farbe: Grün
        if color is None:
            color = [0.2, 0.8, 0.2]

        component_data = {
            "type": "component_line",
            "coords": [[float(c[0]), float(c[1]), float(c[2])] for c in coords],
            "color": color,
            "line_width": line_width,
            "opacity": 1.0,
        }

        if label:
            component_data["label"] = label

        self.component_lines.append(component_data)

    def add_roads_batch(self, roads: List[Dict[str, Any]]) -> None:
        """Füge mehrere Roads auf einmal hinzu."""
        for road in roads:
            self.add_road(road)

    def add_junctions_batch(self, junctions: List[Dict[str, Any]]) -> None:
        """Füge mehrere Junctions auf einmal hinzu."""
        for junction in junctions:
            self.add_junction(junction)

    def add_boundaries_batch(self, boundaries: List[Dict[str, Any]]) -> None:
        """Füge mehrere Boundary-Polygone auf einmal hinzu."""
        for boundary in boundaries:
            self.add_boundary(boundary)

    def merge(self, other: "DebugNetworkExporter") -> None:
        """
        Merge Daten aus einem anderen Exporter (für Multi-Tile Support).

        Args:
            other: Anderer DebugNetworkExporter
        """
        road_offset = len(self.roads)
        junction_offset = len(self.junctions)

        # Roads direkt hinzufügen
        self.roads.extend(other.roads)

        # Junctions mit angepassten road_indices hinzufügen
        for junction in other.junctions:
            junction_copy = junction.copy()
            # Passe road_indices an
            if "road_indices" in junction_copy:
                junction_copy["road_indices"] = [idx + road_offset for idx in junction_copy["road_indices"]]
            # Passe connection_types an
            if "connection_types" in junction_copy:
                new_conn_types = {}
                for idx, types in junction_copy["connection_types"].items():
                    new_conn_types[int(idx) + road_offset] = types
                junction_copy["connection_types"] = new_conn_types
            self.junctions.append(junction_copy)

        # Boundary-Polygone direkt hinzufügen
        self.boundary_polygons.extend(other.boundary_polygons)

        # Component-Linien direkt hinzufügen
        self.component_lines.extend(other.component_lines)

        # Primitive direkt hinzufügen
        self.primitives.extend(other.primitives)

    def export(self, cache_dir: str, filename: str = "debug_network.json") -> None:
        """
        Exportiere gesammelte Debug-Daten in JSON-Datei.

        Args:
            cache_dir: Zielverzeichnis für Export
            filename: Dateiname (default: debug_network.json)
        """
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, filename)

        data = {
            "roads": self.roads,
            "junctions": self.junctions,
            "boundary_polygons": self.boundary_polygons,
            "component_lines": self.component_lines,
            "primitives": self.primitives,
            "grid_colors": self.grid_colors,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(
            f"  [Debug] Exportiert: {len(self.roads)} Roads, {len(self.junctions)} Junctions, "
            f"{len(self.boundary_polygons)} Boundaries, {len(self.component_lines)} Component-Linien, "
            f"{len(self.primitives)} Primitive"
        )
        print(f"  [Debug] Datei: {output_path}")

    def clear(self) -> None:
        """Lösche alle gesammelten Daten."""
        self.roads.clear()
        self.junctions.clear()
        self.boundary_polygons.clear()
        self.component_lines.clear()
        self.primitives.clear()

    def __repr__(self) -> str:
        return (
            f"DebugNetworkExporter(roads={len(self.roads)}, "
            f"junctions={len(self.junctions)}, "
            f"boundaries={len(self.boundary_polygons)}, "
            f"components={len(self.component_lines)}, "
            f"primitives={len(self.primitives)})"
        )
